# CHOICES.md — Three Key Architecture Decisions

## Decision 1: Detection Model — MOG2 vs YOLOv8

### Options considered

| Option | Accuracy | Speed | Dependencies | Works offline |
|--------|----------|-------|--------------|---------------|
| YOLOv8n | High | Fast (GPU) | ~6MB weights from github | No |
| MOG2 + contours | Medium | Very fast (CPU) | Built into OpenCV | Yes |
| HOG person detector | Low | Medium | Built into OpenCV | Yes |
| VLM (GPT-4V frames) | Very high | Slow + costly | API key + internet | No |

### What AI suggested

Claude (in planning phase) recommended YOLOv8s as the primary detector with ByteTrack for tracking, citing its strong COCO benchmark accuracy (48.7 mAP) and production use in retail CV systems.

### What I chose and why

MOG2 background subtraction. The environment had no access to github.com or pytorch.org (network restrictions returned 403), making YOLOv8 weight downloads impossible. But beyond the constraint, MOG2 is genuinely well-suited here:

1. **Fixed camera assumption holds**: All 5 cameras are ceiling/wall-mounted CCTV with zero pan/tilt. The background is stable. MOG2 was designed for exactly this.
2. **Overhead angles break standard detectors**: YOLOv8 and HOG are trained on frontal/side pedestrian views. From directly overhead, the person silhouette is an oval/blob rather than a standing human — which is exactly what MOG2 + contour analysis detects correctly.
3. **Zero dependencies**: No weights file, no CUDA, runs anywhere OpenCV runs.

The accuracy trade-off: MOG2 struggles when multiple people overlap (merged blobs) and when a person stands still for >30s (absorbed into background model). Both are mitigated by using `learningRate=0.003` during inference (slow background update) and by the IoU tracker which maintains identity across brief static periods.

---

## Decision 2: Event Schema Design

### Design rationale

The schema needed to satisfy three competing constraints simultaneously:
- Queryable for real-time analytics (zone dwell, queue depth)
- Flat enough for simple SQL aggregations
- Self-describing enough to reconstruct a visitor's full session from events alone

### Options considered

**Option A** — Normalized relational: separate `sessions` and `events` tables joined by session_id.
Pro: Clean data model. Con: Requires a session-management service to assign session IDs in real time; adds latency and failure surface.

**Option B** — Flat event log with all session context inline (chosen).
Every event carries `visitor_id`, `is_staff`, `zone_id`, `dwell_ms`, `confidence`. Session is implicitly reconstructed by grouping on `visitor_id`.

**Option C** — Kafka-style byte events (schema registry + Avro).
Pro: production-grade streaming. Con: massive operational overhead for a 5-camera store.

### What AI suggested

Claude recommended Option A (normalized) for "data integrity" reasons, specifically that having `is_staff` duplicated on every event is denormalized. I disagreed: for analytics queries, denormalizing `is_staff` onto every event means `WHERE is_staff=0` filters happen at the storage layer — no join required. The 50-byte overhead per event is trivial.

### What I chose and why

Option B — flat event log. The key design insight: the event stream is an append-only audit log, not a transactional database. Denormalization is correct for append-only logs. `metadata` is stored as individual columns (not JSON blob) for efficient SQL indexing on `queue_depth` and `sku_zone`.

The `session_seq` field was my own addition (not AI-suggested): it encodes the ordinal position of each event in a visitor's session, enabling session reconstruction without a timestamp sort.

---

## Decision 3: API Storage — SQLite vs PostgreSQL

### Options considered

| Option | Setup | Scale limit | Concurrent writes | Ops overhead |
|--------|-------|-------------|-------------------|--------------|
| SQLite (WAL mode) | Zero config | ~10k events/day | Single writer | None |
| PostgreSQL | Docker service | Unlimited | Multiple writers | Medium |
| TimescaleDB | Docker + extension | Unlimited | Multiple writers | High |
| DuckDB | Zero config | Analytics-optimized | No concurrent writes | None |

### What AI suggested

Claude suggested TimescaleDB for "production-readiness" — it's a time-series extension on PostgreSQL ideal for event streams. Technically correct for 40 live stores sending real-time events.

### What I chose and why

SQLite with WAL mode. Three reasons:

1. **Submission scope**: The dataset is 5 cameras × ~2.5 minutes. That's ~52 events. SQLite handles this with zero configuration and zero infrastructure.
2. **`docker compose up` should work**: Adding PostgreSQL means a second Docker service, healthcheck ordering, init scripts, and volume management. The acceptance gate is "no manual steps beyond git clone." SQLite keeps that promise.
3. **WAL mode is the right choice for read-heavy analytics**: The API has many concurrent readers (dashboard polling every 5s) and one writer (event ingest). WAL mode on SQLite is optimized for exactly this: readers don't block writers.

**Upgrade path for 40 stores**: Replace `db.py` with an async SQLAlchemy + PostgreSQL backend. The query layer uses plain SQL (no ORM), so the upgrade is a 2-hour task — swap the connection string and `INSERT OR IGNORE` becomes `INSERT … ON CONFLICT DO NOTHING`.
