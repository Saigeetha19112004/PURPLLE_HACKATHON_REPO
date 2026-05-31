from .db import query
from datetime import datetime, timezone, timedelta

def get_metrics(store_id: str) -> dict:
    # Unique customer visitors (ENTRY events, not staff)
    rows = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id=? AND event_type='ENTRY' AND is_staff=0
    """, (store_id,))
    unique_visitors = rows[0]["cnt"] if rows else 0

    # Visitors who reached billing
    billing = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id=? AND zone_id IN ('BILLING_COUNTER','BILLING_QUEUE') AND is_staff=0
    """, (store_id,))
    billing_visitors = billing[0]["cnt"] if billing else 0

    # Abandonment rate
    abandon = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id=? AND event_type='BILLING_QUEUE_ABANDON' AND is_staff=0
    """, (store_id,))
    abandons = abandon[0]["cnt"] if abandon else 0
    abandonment_rate = round(abandons / billing_visitors, 3) if billing_visitors else 0.0

    # Conversion rate: billing visitors / total unique visitors
    conversion_rate = round(billing_visitors / unique_visitors, 3) if unique_visitors else 0.0

    # Avg dwell per zone (customer only)
    zone_rows = query("""
        SELECT zone_id,
               AVG(dwell_ms) as avg_dwell,
               COUNT(DISTINCT visitor_id) as visitors
        FROM events
        WHERE store_id=? AND event_type IN ('ZONE_EXIT','ZONE_DWELL')
          AND is_staff=0 AND zone_id IS NOT NULL AND dwell_ms > 0
        GROUP BY zone_id
    """, (store_id,))
    avg_dwell_per_zone = {r["zone_id"]: {
        "avg_dwell_ms": round(r["avg_dwell"]),
        "visitors": r["visitors"]
    } for r in zone_rows}

    # Current queue depth (latest queue_depth value in billing)
    qd = query("""
        SELECT queue_depth FROM events
        WHERE store_id=? AND zone_id='BILLING_QUEUE' AND queue_depth IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (store_id,))
    queue_depth = qd[0]["queue_depth"] if qd else 0

    # Last event time
    last_ev = query("""
        SELECT MAX(timestamp) as t FROM events WHERE store_id=?
    """, (store_id,))
    last_event_ts = last_ev[0]["t"] if last_ev else None

    return {
        "store_id": store_id,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "unique_visitors": unique_visitors,
        "conversion_rate": conversion_rate,
        "avg_dwell_per_zone": avg_dwell_per_zone,
        "current_queue_depth": queue_depth,
        "abandonment_rate": abandonment_rate,
        "last_event_timestamp": last_event_ts,
        "data_confidence": "HIGH" if unique_visitors >= 20 else "LOW",
    }

def get_funnel(store_id: str) -> dict:
    entries = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND event_type='ENTRY' AND is_staff=0
    """, (store_id,))[0]["cnt"]

    zone_visits = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND event_type='ZONE_ENTER' AND is_staff=0
          AND zone_id NOT IN ('ENTRY_ZONE','BILLING_COUNTER','BILLING_QUEUE')
    """, (store_id,))[0]["cnt"]

    billing = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND zone_id IN ('BILLING_COUNTER','BILLING_QUEUE') AND is_staff=0
    """, (store_id,))[0]["cnt"]

    # Purchase = was in billing zone and NOT in abandon list
    abandoned_vids = set(r["visitor_id"] for r in query("""
        SELECT DISTINCT visitor_id FROM events
        WHERE store_id=? AND event_type='BILLING_QUEUE_ABANDON' AND is_staff=0
    """, (store_id,)))

    billing_vids = set(r["visitor_id"] for r in query("""
        SELECT DISTINCT visitor_id FROM events
        WHERE store_id=? AND zone_id IN ('BILLING_COUNTER','BILLING_QUEUE') AND is_staff=0
    """, (store_id,)))

    purchases = len(billing_vids - abandoned_vids)

    def drop(a, b):
        return round(1 - b / a, 3) if a else 0.0

    return {
        "store_id": store_id,
        "funnel": [
            {"stage": "Entry",        "visitors": entries,     "drop_off_pct": 0.0},
            {"stage": "Zone browse",  "visitors": zone_visits, "drop_off_pct": drop(entries, zone_visits)},
            {"stage": "Billing queue","visitors": billing,     "drop_off_pct": drop(zone_visits, billing)},
            {"stage": "Purchase",     "visitors": purchases,   "drop_off_pct": drop(billing, purchases)},
        ],
        "conversion_rate": round(purchases / entries, 3) if entries else 0.0,
        "note": "Session is the unit; re-entries not double-counted"
    }

def get_heatmap(store_id: str) -> dict:
    rows = query("""
        SELECT zone_id,
               COUNT(DISTINCT visitor_id) as visits,
               AVG(dwell_ms) as avg_dwell
        FROM events
        WHERE store_id=? AND is_staff=0
          AND zone_id IS NOT NULL
          AND event_type IN ('ZONE_ENTER','ZONE_DWELL','ZONE_EXIT')
        GROUP BY zone_id
    """, (store_id,))

    if not rows:
        return {"store_id": store_id, "zones": [], "data_confidence": "NO_DATA"}

    max_visits = max(r["visits"] for r in rows) or 1
    max_dwell  = max(r["avg_dwell"] or 0 for r in rows) or 1
    total_sessions = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND event_type='ENTRY' AND is_staff=0
    """, (store_id,))[0]["cnt"]

    zones = []
    for r in rows:
        score = 0.5 * (r["visits"] / max_visits) + 0.5 * ((r["avg_dwell"] or 0) / max_dwell)
        zones.append({
            "zone_id": r["zone_id"],
            "visit_count": r["visits"],
            "avg_dwell_ms": round(r["avg_dwell"] or 0),
            "normalised_score": round(score * 100),
        })
    zones.sort(key=lambda z: z["normalised_score"], reverse=True)

    return {
        "store_id": store_id,
        "zones": zones,
        "data_confidence": "HIGH" if total_sessions >= 20 else "LOW",
    }

def get_anomalies(store_id: str) -> dict:
    anomalies = []
    now = datetime.now(timezone.utc)

    # 1. BILLING_QUEUE_SPIKE: queue depth > 3
    qd = query("""
        SELECT queue_depth FROM events
        WHERE store_id=? AND zone_id='BILLING_QUEUE' AND queue_depth IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (store_id,))
    if qd and qd[0]["queue_depth"] and qd[0]["queue_depth"] > 3:
        anomalies.append({
            "type": "BILLING_QUEUE_SPIKE",
            "severity": "CRITICAL" if qd[0]["queue_depth"] > 5 else "WARN",
            "detail": f"Queue depth is {qd[0]['queue_depth']}",
            "suggested_action": "Open additional billing counter or redirect staff",
        })

    # 2. DEAD_ZONE: no visits in a zone for > 30 min
    dead = query("""
        SELECT zone_id, MAX(timestamp) as last_visit FROM events
        WHERE store_id=? AND is_staff=0 AND zone_id IS NOT NULL
          AND event_type='ZONE_ENTER'
        GROUP BY zone_id
    """, (store_id,))
    for row in dead:
        if not row["last_visit"]: continue
        try:
            last_t = datetime.fromisoformat(row["last_visit"])
            if (now - last_t).total_seconds() > 1800:
                anomalies.append({
                    "type": "DEAD_ZONE",
                    "severity": "INFO",
                    "detail": f"No customer visits to {row['zone_id']} in 30+ min",
                    "suggested_action": f"Check {row['zone_id']} display / lighting",
                })
        except: pass

    # 3. HIGH_ABANDONMENT: > 40% abandonment rate
    billing_total = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND zone_id IN ('BILLING_COUNTER','BILLING_QUEUE') AND is_staff=0
    """, (store_id,))[0]["cnt"]
    abandon_total = query("""
        SELECT COUNT(DISTINCT visitor_id) as cnt FROM events
        WHERE store_id=? AND event_type='BILLING_QUEUE_ABANDON' AND is_staff=0
    """, (store_id,))[0]["cnt"]
    if billing_total > 0:
        rate = abandon_total / billing_total
        if rate > 0.4:
            anomalies.append({
                "type": "HIGH_QUEUE_ABANDONMENT",
                "severity": "WARN",
                "detail": f"Abandonment rate {rate*100:.0f}% exceeds 40% threshold",
                "suggested_action": "Reduce checkout wait time; add self-checkout option",
            })

    # 4. STALE_FEED: no events in last 10 min (simulate low data scenario)
    last_ev = query("SELECT MAX(timestamp) as t FROM events WHERE store_id=?", (store_id,))
    if last_ev and last_ev[0]["t"]:
        try:
            last_t = datetime.fromisoformat(last_ev[0]["t"])
            lag_min = (now - last_t).total_seconds() / 60
            if lag_min > 10:
                anomalies.append({
                    "type": "STALE_FEED",
                    "severity": "WARN",
                    "detail": f"Last event {lag_min:.1f} minutes ago",
                    "suggested_action": "Check camera connectivity and pipeline health",
                })
        except: pass

    return {
        "store_id": store_id,
        "as_of": now.isoformat(),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
    }

def get_health(store_id: str = None) -> dict:
    stores = query("SELECT DISTINCT store_id FROM events") if not store_id else [{"store_id": store_id}]
    now = datetime.now(timezone.utc)
    feed_status = {}
    for s in stores:
        sid = s["store_id"]
        cams = query("SELECT DISTINCT camera_id, MAX(timestamp) as last_ts FROM events WHERE store_id=? GROUP BY camera_id", (sid,))
        cam_status = {}
        for c in cams:
            try:
                last_t = datetime.fromisoformat(c["last_ts"])
                lag_s = (now - last_t).total_seconds()
                cam_status[c["camera_id"]] = {
                    "last_event": c["last_ts"],
                    "lag_seconds": round(lag_s),
                    "status": "STALE_FEED" if lag_s > 600 else "OK",
                }
            except:
                cam_status[c["camera_id"]] = {"status": "UNKNOWN"}
        feed_status[sid] = cam_status

    total_events = query("SELECT COUNT(*) as cnt FROM events")[0]["cnt"]
    return {
        "status": "OK",
        "timestamp": now.isoformat(),
        "total_events_ingested": total_events,
        "feed_status": feed_status,
        "version": "1.0.0",
    }
