# PROMPT: "Write pytest tests for a retail store intelligence API. Cover: metrics endpoint
# returning correct unique_visitors, conversion_rate, avg_dwell. Funnel endpoint accuracy.
# Edge cases: empty store, all-staff clip, zero purchases, re-entry deduplication.
# Test idempotency of POST /events/ingest."
# CHANGES MADE: Added actual assert values from our known event fixture (8 visitors, 
# 37.5% conversion), changed fixture to use real visitor_ids from detected events, 
# added the BILLING_QUEUE_ABANDON edge case test which AI missed.

import sys, json, pytest
sys.path.insert(0, "/home/claude/store-intelligence")

from app.db import init_db, insert_events, query, DB_PATH
from app.metrics import get_metrics, get_funnel, get_heatmap, get_anomalies
import tempfile, os

STORE = "STORE_PURPLLE_001"

@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db)
    import app.db as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", db)
    init_db()
    yield db

def load_events():
    path = "/home/claude/store-intelligence/data/events/all_events.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]

def test_metrics_unique_visitors():
    insert_events(load_events())
    m = get_metrics(STORE)
    assert m["unique_visitors"] == 8

def test_metrics_conversion_rate():
    insert_events(load_events())
    m = get_metrics(STORE)
    # 3 visitors reached billing / 8 total = 0.375
    assert m["conversion_rate"] == 0.375

def test_metrics_staff_excluded():
    insert_events(load_events())
    m = get_metrics(STORE)
    # Staff should never appear in unique_visitors (ENTRY events only for customers)
    assert m["unique_visitors"] == 8  # 3 staff not counted

def test_funnel_stages():
    insert_events(load_events())
    f = get_funnel(STORE)
    stages = {s["stage"]: s["visitors"] for s in f["funnel"]}
    assert stages["Entry"] == 8
    assert stages["Billing queue"] == 3
    assert stages["Purchase"] == 2  # V3 abandoned, so only V1+V2 purchased

def test_funnel_conversion():
    insert_events(load_events())
    f = get_funnel(STORE)
    assert f["conversion_rate"] == 0.25  # 2 purchases / 8 visitors

def test_funnel_abandonment_excluded():
    insert_events(load_events())
    f = get_funnel(STORE)
    # V3 abandoned billing, should be in billing count but not purchase
    stages = {s["stage"]: s["visitors"] for s in f["funnel"]}
    assert stages["Billing queue"] == 3
    assert stages["Purchase"] == 2

def test_heatmap_zones():
    insert_events(load_events())
    h = get_heatmap(STORE)
    zone_ids = [z["zone_id"] for z in h["zones"]]
    assert "MAKEUP" in zone_ids
    assert "SKINCARE" in zone_ids

def test_heatmap_normalised_score_range():
    insert_events(load_events())
    h = get_heatmap(STORE)
    for z in h["zones"]:
        assert 0 <= z["normalised_score"] <= 100

def test_idempotent_ingest():
    events = load_events()
    r1 = insert_events(events)
    r2 = insert_events(events)  # same events again
    assert r1["accepted"] == 52
    assert r2["duplicate"] == 52
    assert r2["accepted"] == 0
    total = query("SELECT COUNT(*) as c FROM events")[0]["c"]
    assert total == 52  # no duplicates stored

def test_empty_store():
    m = get_metrics("STORE_NONEXISTENT")
    assert m["unique_visitors"] == 0
    assert m["conversion_rate"] == 0.0

def test_zero_purchases():
    # Insert only entry + zone events, no billing
    events = [e for e in load_events() if e["event_type"] in ("ENTRY", "ZONE_ENTER", "ZONE_EXIT") and (e.get("zone_id") or "") not in ("BILLING_COUNTER","BILLING_QUEUE")]
    insert_events(events)
    f = get_funnel(STORE)
    stages = {s["stage"]: s["visitors"] for s in f["funnel"]}
    assert stages["Billing queue"] == 0
    assert stages["Purchase"] == 0
    assert f["conversion_rate"] == 0.0

def test_all_staff_clip():
    staff_events = [e for e in load_events() if e["is_staff"]]
    insert_events(staff_events)
    m = get_metrics(STORE)
    assert m["unique_visitors"] == 0
    assert m["conversion_rate"] == 0.0

def test_reentry_not_double_counted():
    events = load_events()
    insert_events(events)
    m = get_metrics(STORE)
    # V4 entered at t=39s, exited at t=138s — only 8 unique visitors total
    assert m["unique_visitors"] == 8

def test_anomaly_stale_feed():
    insert_events(load_events())
    a = get_anomalies(STORE)
    types = [x["type"] for x in a["anomalies"]]
    assert "STALE_FEED" in types  # events are from April, now it's May

def test_anomaly_high_abandonment():
    insert_events(load_events())
    a = get_anomalies(STORE)
    # 1/3 = 33% abandonment - below 40% threshold, should NOT trigger
    types = [x["type"] for x in a["anomalies"]]
    assert "HIGH_QUEUE_ABANDONMENT" not in types
