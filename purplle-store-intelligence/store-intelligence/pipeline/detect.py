"""
detect.py — Purplle Store Intelligence Detection Pipeline
=========================================================
Architecture: MOG2 background subtraction + contour-based person detection
              + centroid-based IoU tracker + staff color classifier

Why MOG2 over YOLOv8: No external model download required; MOG2 is a proven
production approach for fixed CCTV cameras with stable backgrounds. It handles
overhead/angled camera views naturally (HOG fails on overhead views; YOLOv8
was unavailable due to network restrictions).

Staff classification: Color analysis of clothing (Purplle staff wear all-black
uniforms). Validated visually on sample frames.

Camera roles (determined by visual inspection):
  CAM_1 = Main floor skincare zone (overhead)
  CAM_2 = Main floor makeup/cosmetics zone (wide angle)
  CAM_3 = Entry/exit threshold (side view with glass panel)
  CAM_4 = Stockroom (staff-only, overhead)
  CAM_5 = Billing counter (overhead-ish)
"""

import cv2
import json
import uuid
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ─── Camera Configuration ────────────────────────────────────────────────────
CAMERA_META = {
    "CAM_1": {
        "role": "MAIN_FLOOR_SKINCARE",
        "zone": "SKINCARE",
        "entry_exit": False,
        "staff_zone": False,
        "fps": 30,
        "desc": "Skincare zone - FarmStay, The Face Shop, Minimalist, Derma Co",
    },
    "CAM_2": {
        "role": "MAIN_FLOOR_MAKEUP",
        "zone": "MAKEUP",
        "entry_exit": False,
        "staff_zone": False,
        "fps": 30,
        "desc": "Makeup zone - Lakmé, Maybelline, Swiss Beauty, L'Oreal",
    },
    "CAM_3": {
        "role": "ENTRY_EXIT",
        "zone": "ENTRY_ZONE",
        "entry_exit": True,
        "staff_zone": False,
        "fps": 30,
        "desc": "Store entry/exit glass door - Purplle signage visible",
        # Entry zone: right portion of frame (outside store)
        # EXIT zone: left portion (inside store, then leaving)
        # Person crossing from RIGHT side into LEFT = ENTRY
        # Person crossing from LEFT to RIGHT (outside) = EXIT
        # Split line at x = 55% of frame width
        "entry_split_x_ratio": 0.55,
        # Only care about lower half of frame (near doormat/floor)
        "detection_roi_y_start_ratio": 0.3,
    },
    "CAM_4": {
        "role": "STOCKROOM",
        "zone": "STOCKROOM",
        "entry_exit": False,
        "staff_zone": True,
        "fps": 25,
        "desc": "Back-of-house stockroom, staff only, Purplle boxes visible",
    },
    "CAM_5": {
        "role": "BILLING_COUNTER",
        "zone": "BILLING_COUNTER",
        "entry_exit": False,
        "staff_zone": False,
        "fps": 25,
        "desc": "POS/billing counter with laptop terminal",
    },
}

# All clips start around 20:10 IST on 10/04/2026 per CCTV timestamp
# CAM_3 visible timestamp: 10/04/2026 20:10:12 → UTC = 14:40:12
CLIP_START_UTC = {
    "CAM_1": datetime(2026, 4, 10, 14, 40, 37, tzinfo=timezone.utc),
    "CAM_2": datetime(2026, 4, 10, 14, 40, 12, tzinfo=timezone.utc),
    "CAM_3": datetime(2026, 4, 10, 14, 40, 12, tzinfo=timezone.utc),
    "CAM_4": datetime(2026, 4, 10, 14, 39, 55, tzinfo=timezone.utc),
    "CAM_5": datetime(2026, 4, 10, 14, 39, 57, tzinfo=timezone.utc),
}


# ─── Staff Classification ─────────────────────────────────────────────────────
def classify_staff(frame: np.ndarray, bbox: tuple, is_staff_zone: bool = False) -> tuple:
    """
    Classify a person bounding box as staff or customer.
    
    Purplle staff wear all-black uniforms (confirmed from CAM_2 at t=60s: 
    staff have solid black top + black pants; customers wear varied colors).
    
    Strategy: Measure darkness ratio of upper + lower body separately.
    Staff = predominantly dark (>50% pixels below brightness 75) in both regions.
    
    Returns: (is_staff: bool, confidence: float)
    """
    if is_staff_zone:
        return True, 0.95

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

    if x2 - x1 < 15 or y2 - y1 < 20:
        return False, 0.5  # Too small to classify reliably

    crop = frame[y1:y2, x1:x2]
    ph = crop.shape[0]

    def dark_ratio(region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.sum(gray < 75) / gray.size)

    # Split into upper body (top 55%) and lower body (bottom 45%)
    upper_dark = dark_ratio(crop[: int(ph * 0.55), :])
    lower_dark = dark_ratio(crop[int(ph * 0.45) :, :])
    
    combined = upper_dark * 0.5 + lower_dark * 0.5
    
    # Staff threshold: >45% dark pixels overall
    STAFF_THRESH = 0.45
    is_staff = combined > STAFF_THRESH
    
    if is_staff:
        confidence = min(0.95, 0.5 + (combined - STAFF_THRESH) * 2)
    else:
        confidence = min(0.95, 0.5 + (STAFF_THRESH - combined) * 2)
    
    return is_staff, round(confidence, 3)


# ─── Centroid Tracker ─────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    track_id: int
    visitor_id: str
    camera_id: str
    centroid: tuple          # (x, y) in frame coordinates
    bbox: tuple              # (x1, y1, x2, y2)
    first_seen: datetime
    last_seen: datetime
    missed_frames: int = 0
    is_staff: bool = False
    staff_votes: list = field(default_factory=list)
    session_seq: int = 0
    zone: Optional[str] = None
    zone_enter_time: Optional[datetime] = None
    last_dwell_emit: Optional[datetime] = None
    entry_emitted: bool = False
    exit_emitted: bool = False
    prev_cx: Optional[float] = None  # for entry/exit direction detection
    centroid_history: list = field(default_factory=list)

    def update_staff_vote(self, vote: bool):
        self.staff_votes.append(vote)
        if len(self.staff_votes) > 20:
            self.staff_votes = self.staff_votes[-20:]
        self.is_staff = sum(self.staff_votes) > len(self.staff_votes) * 0.55

    def next_seq(self) -> int:
        self.session_seq += 1
        return self.session_seq


class CentroidTracker:
    """
    IoU + distance based centroid tracker.
    Assigns track IDs to detected blobs across frames.
    """
    def __init__(self, max_missed: int = 15, iou_threshold: float = 0.2, max_dist: float = 120):
        self.next_id = 1
        self.tracks: dict[int, TrackedPerson] = {}
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.max_dist = max_dist

    def _iou(self, b1: tuple, b2: tuple) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _dist(self, c1: tuple, c2: tuple) -> float:
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def update(self, detections: list, frame_time: datetime, camera_id: str) -> dict[int, TrackedPerson]:
        """
        detections: list of (bbox_x1, y1, x2, y2, confidence)
        Returns: dict of active track_id -> TrackedPerson
        """
        if not detections:
            # Mark all as missed
            to_remove = []
            for tid, track in self.tracks.items():
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return dict(self.tracks)

        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        det_centroids = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in det_bboxes]

        if not self.tracks:
            for bbox, centroid in zip(det_bboxes, det_centroids):
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = TrackedPerson(
                    track_id=tid,
                    visitor_id=f"VIS_{uuid.uuid4().hex[:6].upper()}",
                    camera_id=camera_id,
                    centroid=centroid,
                    bbox=bbox,
                    first_seen=frame_time,
                    last_seen=frame_time,
                )
            return dict(self.tracks)

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        # Build cost matrix: prefer IoU match, fallback to distance
        matched_dets = set(); matched_trks = set()
        matches = []

        # First pass: IoU matching
        cost = np.zeros((len(track_ids), len(det_bboxes)))
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                cost[i, j] = self._iou(tb, db)

        # Greedy matching (high IoU first)
        pairs = sorted([(cost[i,j], i, j) for i in range(len(track_ids)) for j in range(len(det_bboxes))], reverse=True)
        for iou_score, i, j in pairs:
            if iou_score < self.iou_threshold:
                break
            if i in matched_trks or j in matched_dets:
                continue
            matches.append((track_ids[i], j))
            matched_trks.add(i); matched_dets.add(j)

        # Second pass: distance matching for unmatched
        for i, tid in enumerate(track_ids):
            if i in matched_trks:
                continue
            best_j, best_d = None, self.max_dist
            for j in range(len(det_bboxes)):
                if j in matched_dets:
                    continue
                d = self._dist(track_centroids[i], det_centroids[j])
                if d < best_d:
                    best_d = d; best_j = j
            if best_j is not None:
                matches.append((tid, best_j))
                matched_trks.add(i); matched_dets.add(best_j)

        # Update matched tracks
        for tid, j in matches:
            t = self.tracks[tid]
            t.prev_cx = t.centroid[0]
            t.centroid = det_centroids[j]
            t.bbox = det_bboxes[j]
            t.last_seen = frame_time
            t.missed_frames = 0
            t.centroid_history.append(det_centroids[j])
            if len(t.centroid_history) > 30:
                t.centroid_history = t.centroid_history[-30:]

        # Mark unmatched tracks as missed
        for i, tid in enumerate(track_ids):
            if i not in matched_trks:
                self.tracks[tid].missed_frames += 1

        # Create new tracks for unmatched detections
        for j in range(len(det_bboxes)):
            if j in matched_dets:
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = TrackedPerson(
                track_id=tid,
                visitor_id=f"VIS_{uuid.uuid4().hex[:6].upper()}",
                camera_id=camera_id,
                centroid=det_centroids[j],
                bbox=det_bboxes[j],
                first_seen=frame_time,
                last_seen=frame_time,
            )

        # Remove stale tracks
        to_remove = [tid for tid, t in self.tracks.items() if t.missed_frames > self.max_missed]
        for tid in to_remove:
            del self.tracks[tid]

        return dict(self.tracks)


# ─── Person Detection (MOG2) ─────────────────────────────────────────────────
def detect_persons_mog2(fg_mask: np.ndarray, frame_h: int, frame_w: int,
                         min_area: int = 1200, max_area: int = 80000) -> list:
    """
    Extract person bounding boxes from MOG2 foreground mask.
    
    Filters:
    - Area between min_area and max_area
    - Aspect ratio > 0.9 (tall blobs = standing people)
    - Not touching image border strongly (reduces false positives from signs/lights)
    
    Returns: list of (x1, y1, x2, y2, confidence)
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / max(w, 1)

        # Height must be reasonable (not a tiny strip or huge blob)
        if h < 30 or w < 15:
            continue

        # Aspect ratio filter: people are taller than wide (AR > 0.8)
        # But overhead cameras may see people from above (AR can be lower)
        if aspect_ratio < 0.8:
            continue

        # Border penalty (blobs stuck to image edge are likely fixed objects)
        border_margin = 5
        if x <= border_margin and w < 40:
            continue

        # Confidence proportional to how well the blob matches person shape
        ideal_ar = 2.0
        ar_score = 1.0 - min(1.0, abs(aspect_ratio - ideal_ar) / ideal_ar)
        area_score = min(1.0, area / 15000)
        confidence = round(0.4 + 0.4 * ar_score + 0.2 * area_score, 3)

        detections.append((x, y, x + w, y + h, confidence))

    return detections


# ─── Event Builder ────────────────────────────────────────────────────────────
def make_event(store_id: str, camera_id: str, track: TrackedPerson,
               event_type: str, zone_id: Optional[str], dwell_ms: int = 0,
               queue_depth: Optional[int] = None, sku_zone: Optional[str] = None,
               confidence: float = 0.85) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": track.visitor_id,
        "event_type": event_type,
        "timestamp": track.last_seen.isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": track.is_staff,
        "confidence": round(confidence, 3),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone or zone_id,
            "session_seq": track.next_seq(),
        },
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────
class StorePipeline:
    def __init__(self, store_id: str, output_dir: str):
        self.store_id = store_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_events: list = []

    def process_camera(self, video_path: str, camera_id: str,
                        sample_rate: int = 3, warmup_secs: int = 5) -> list:
        """
        Process a single camera clip.
        
        sample_rate: process every Nth frame (3 = 10fps for 30fps clip)
        warmup_secs: how many seconds to use for MOG2 background learning
        """
        meta = CAMERA_META.get(camera_id, {})
        clip_start = CLIP_START_UTC.get(camera_id, datetime(2026, 4, 10, 14, 40, 0, tzinfo=timezone.utc))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {video_path}")
            return []

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or meta.get("fps", 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / actual_fps

        print(f"\n{'='*65}")
        print(f"  {camera_id}  |  {meta.get('role','?')}  |  {meta.get('desc','')}")
        print(f"  Duration: {duration_s:.1f}s  FPS: {actual_fps:.1f}  Frames: {total_frames}")
        print(f"{'='*65}")

        is_entry_cam = meta.get("entry_exit", False)
        is_staff_zone = meta.get("staff_zone", False)
        zone_id = meta.get("zone")
        entry_split_x = meta.get("entry_split_x_ratio", 0.55)
        roi_y_start = meta.get("detection_roi_y_start_ratio", 0.0)

        # MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25 if not is_entry_cam else 20,  # more sensitive for entry
            detectShadows=True,
        )
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        tracker = CentroidTracker(
            max_missed=int(actual_fps * 2),  # 2 seconds of missed frames
            iou_threshold=0.15,
            max_dist=150,
        )

        events: list = []
        billing_zone_occupants: set = set()  # track_ids currently in billing zone
        frame_idx = 0

        # ── Phase 1: MOG2 warmup ──────────────────────────────────────────
        warmup_frames = int(warmup_secs * actual_fps)
        print(f"  Warming up MOG2 for {warmup_secs}s ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (960, 540))
            mog2.apply(small, learningRate=0.02)
            frame_idx += 1

        # ── Phase 2: Detection + tracking ─────────────────────────────────
        print(f"  Processing remaining {total_frames - warmup_frames} frames...")
        prev_active_tids: set = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % sample_rate != 0:
                continue

            frame_h, frame_w = frame.shape[:2]
            frame_time = clip_start + timedelta(seconds=frame_idx / actual_fps)

            # Resize for processing speed
            proc_w, proc_h = 960, 540
            scale_x = frame_w / proc_w
            scale_y = frame_h / proc_h
            small = cv2.resize(frame, (proc_w, proc_h))

            # Apply ROI mask for entry camera (focus on doorway area)
            roi_frame = small.copy()
            if roi_y_start > 0:
                roi_y_px = int(proc_h * roi_y_start)
                roi_frame[:roi_y_px, :] = 0  # blank out top portion

            # MOG2 foreground extraction
            fg = mog2.apply(roi_frame, learningRate=0.003)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # remove shadows (127)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            fg = cv2.dilate(fg, kernel_close, iterations=2)

            # Detect person blobs
            raw_dets = detect_persons_mog2(fg, proc_h, proc_w)

            # Scale detections back to original frame coordinates
            dets = []
            for (x1, y1, x2, y2, conf) in raw_dets:
                dets.append((
                    x1 * scale_x, y1 * scale_y,
                    x2 * scale_x, y2 * scale_y,
                    conf
                ))

            # Update tracker
            active_tracks = tracker.update(dets, frame_time, camera_id)
            current_tids = set(active_tracks.keys())

            # ── Per-track event logic ─────────────────────────────────────
            for tid, track in active_tracks.items():
                cx, cy = track.centroid
                bbox = track.bbox

                # Staff classification
                vote, _ = classify_staff(frame, bbox, is_staff_zone)
                track.update_staff_vote(vote)

                # ── ENTRY/EXIT camera (CAM_3) ─────────────────────────────
                if is_entry_cam:
                    split_x = frame_w * entry_split_x

                    # Person just appeared on the right side = potential entrant
                    if track.prev_cx is None:
                        track.prev_cx = cx

                    # ENTRY: moved from right (outside) to left (inside)
                    if track.prev_cx > split_x and cx <= split_x and not track.entry_emitted:
                        track.entry_emitted = True
                        track.exit_emitted = False
                        ev = make_event(self.store_id, camera_id, track, "ENTRY", None,
                                        confidence=0.85)
                        events.append(ev)
                        print(f"    ENTRY: {track.visitor_id} at t={frame_idx/actual_fps:.1f}s "
                              f"(staff={track.is_staff})")

                    # EXIT: moved from left (inside) to right (outside)
                    elif track.prev_cx <= split_x and cx > split_x and not track.exit_emitted:
                        if track.entry_emitted:
                            track.exit_emitted = True
                            dwell = int((frame_time - track.first_seen).total_seconds() * 1000)
                            ev = make_event(self.store_id, camera_id, track, "EXIT", None,
                                            dwell_ms=dwell, confidence=0.85)
                            events.append(ev)
                            print(f"    EXIT:  {track.visitor_id} at t={frame_idx/actual_fps:.1f}s "
                                  f"dwell={dwell//1000}s")
                            # Re-entry detection: if same track re-enters
                            track.entry_emitted = False

                    track.prev_cx = cx

                # ── Zone camera (CAM_1, CAM_2, CAM_4, CAM_5) ─────────────
                else:
                    # ZONE_ENTER on first detection
                    if not track.entry_emitted and zone_id:
                        track.entry_emitted = True
                        track.zone = zone_id
                        track.zone_enter_time = frame_time
                        track.last_dwell_emit = frame_time
                        ev = make_event(self.store_id, camera_id, track, "ZONE_ENTER",
                                        zone_id, confidence=track.is_staff and 0.9 or 0.8)
                        events.append(ev)

                    # ZONE_DWELL every 30 seconds
                    if track.zone_enter_time and track.last_dwell_emit:
                        secs_since_dwell = (frame_time - track.last_dwell_emit).total_seconds()
                        if secs_since_dwell >= 30.0:
                            dwell_ms = int((frame_time - track.zone_enter_time).total_seconds() * 1000)
                            ev = make_event(self.store_id, camera_id, track, "ZONE_DWELL",
                                            zone_id, dwell_ms=dwell_ms, sku_zone=zone_id)
                            events.append(ev)
                            track.last_dwell_emit = frame_time

                    # BILLING_QUEUE_JOIN for CAM_5
                    if camera_id == "CAM_5" and not track.is_staff:
                        if tid not in billing_zone_occupants:
                            billing_zone_occupants.add(tid)
                            queue_depth = len(billing_zone_occupants) - 1
                            if queue_depth > 0:
                                ev = make_event(self.store_id, camera_id, track,
                                                "BILLING_QUEUE_JOIN", "BILLING_QUEUE",
                                                queue_depth=queue_depth)
                                events.append(ev)

            # ── Handle tracks that disappeared ──────────────────────────
            disappeared_tids = prev_active_tids - current_tids
            for tid in disappeared_tids:
                # Try to find track from previous round
                track = tracker.tracks.get(tid)
                if track is None:
                    continue

                if not is_entry_cam and track.entry_emitted and not track.exit_emitted:
                    dwell_ms = int((track.last_seen - (track.zone_enter_time or track.first_seen)).total_seconds() * 1000)
                    ev = make_event(self.store_id, camera_id, track, "ZONE_EXIT",
                                    track.zone or zone_id, dwell_ms=dwell_ms, confidence=0.75)
                    events.append(ev)
                    track.exit_emitted = True

                if camera_id == "CAM_5":
                    billing_zone_occupants.discard(tid)

            prev_active_tids = current_tids

            if frame_idx % int(actual_fps * 30) == 0:
                t_secs = frame_idx / actual_fps
                print(f"    t={t_secs:.0f}s/{duration_s:.0f}s | "
                      f"active={len(current_tids)} | events={len(events)}")

        # Emit final ZONE_EXIT for any still-active tracks
        for tid, track in tracker.tracks.items():
            if not is_entry_cam and track.entry_emitted and not track.exit_emitted:
                dwell_ms = int((track.last_seen - (track.zone_enter_time or track.first_seen)).total_seconds() * 1000)
                ev = make_event(self.store_id, camera_id, track, "ZONE_EXIT",
                                track.zone or zone_id, dwell_ms=dwell_ms, confidence=0.7)
                events.append(ev)

        cap.release()

        # Save per-camera events
        out_file = self.output_dir / f"events_{camera_id}.jsonl"
        with open(out_file, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        # Print summary
        by_type = defaultdict(int)
        for e in events:
            by_type[e["event_type"]] += 1
        n_staff = sum(1 for e in events if e["is_staff"])
        n_cust = len(events) - n_staff
        unique_vis = len(set(e["visitor_id"] for e in events if not e["is_staff"]))

        print(f"\n  ✓ {camera_id} complete: {len(events)} events → {out_file}")
        for et, c in sorted(by_type.items()):
            print(f"      {et}: {c}")
        print(f"      Customer events: {n_cust} | Staff events: {n_staff}")
        print(f"      Unique customer visitor_ids: {unique_vis}")

        self.all_events.extend(events)
        return events

    def save_merged(self):
        all_out = self.output_dir / "all_events.jsonl"
        sorted_events = sorted(self.all_events, key=lambda e: e["timestamp"])
        with open(all_out, "w") as f:
            for ev in sorted_events:
                f.write(json.dumps(ev) + "\n")

        # Global summary
        by_type = defaultdict(int)
        for e in sorted_events:
            by_type[e["event_type"]] += 1

        print(f"\n{'='*65}")
        print(f"  PIPELINE COMPLETE — {len(sorted_events)} total events → {all_out}")
        print(f"{'='*65}")
        for et, c in sorted(by_type.items()):
            print(f"  {et}: {c}")

        cust_events = [e for e in sorted_events if not e["is_staff"]]
        staff_events = [e for e in sorted_events if e["is_staff"]]
        unique_cust = len(set(e["visitor_id"] for e in cust_events))
        unique_staff = len(set(e["visitor_id"] for e in staff_events))
        print(f"\n  Unique customer visitor_ids: {unique_cust}")
        print(f"  Unique staff visitor_ids:    {unique_staff}")
        print(f"  Total: {len(sorted_events)} events")

        return all_out


def main():
    parser = argparse.ArgumentParser(description="Purplle Store Detection Pipeline")
    parser.add_argument("--clips-dir", default="/mnt/user-data/uploads")
    parser.add_argument("--output-dir", default="./data/events")
    parser.add_argument("--store-id", default="STORE_PURPLLE_001")
    parser.add_argument("--cameras", nargs="+", default=["CAM_3", "CAM_1", "CAM_2", "CAM_5", "CAM_4"])
    parser.add_argument("--sample-rate", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--warmup", type=int, default=5, help="MOG2 warmup seconds")
    args = parser.parse_args()

    pipeline = StorePipeline(args.store_id, args.output_dir)

    clips_dir = Path(args.clips_dir)
    for cam_id in args.cameras:
        video_path = clips_dir / f"{cam_id}.mp4"
        if not video_path.exists():
            print(f"WARNING: {video_path} not found, skipping.")
            continue
        pipeline.process_camera(
            str(video_path), cam_id,
            sample_rate=args.sample_rate,
            warmup_secs=args.warmup
        )

    pipeline.save_merged()


if __name__ == "__main__":
    main()
