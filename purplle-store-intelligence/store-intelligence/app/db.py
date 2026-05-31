import sqlite3, json, os
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "/home/claude/store-intelligence/data/store.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS events (
        event_id    TEXT PRIMARY KEY,
        store_id    TEXT NOT NULL,
        camera_id   TEXT NOT NULL,
        visitor_id  TEXT NOT NULL,
        event_type  TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        zone_id     TEXT,
        dwell_ms    INTEGER DEFAULT 0,
        is_staff    INTEGER DEFAULT 0,
        confidence  REAL,
        queue_depth INTEGER,
        sku_zone    TEXT,
        session_seq INTEGER,
        ingested_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_store_ts  ON events(store_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_visitor   ON events(visitor_id);
    CREATE INDEX IF NOT EXISTS idx_type      ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_zone      ON events(zone_id);

    CREATE TABLE IF NOT EXISTS pos_transactions (
        transaction_id TEXT PRIMARY KEY,
        store_id       TEXT NOT NULL,
        timestamp      TEXT NOT NULL,
        basket_value   REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pos_store_ts ON pos_transactions(store_id, timestamp);
    """)
    conn.commit()
    conn.close()

def insert_events(rows: list[dict]) -> dict:
    conn = get_conn()
    accepted = rejected = duplicate = 0
    errors = []
    for r in rows:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO events
                (event_id,store_id,camera_id,visitor_id,event_type,timestamp,
                 zone_id,dwell_ms,is_staff,confidence,queue_depth,sku_zone,session_seq)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r["event_id"], r["store_id"], r["camera_id"], r["visitor_id"],
                r["event_type"], r["timestamp"], r.get("zone_id"),
                r.get("dwell_ms", 0), int(r.get("is_staff", False)),
                r.get("confidence", 0.0),
                r.get("metadata", {}).get("queue_depth"),
                r.get("metadata", {}).get("sku_zone"),
                r.get("metadata", {}).get("session_seq", 1),
            ))
            if conn.execute("SELECT changes()").fetchone()[0] == 0:
                duplicate += 1
            else:
                accepted += 1
        except Exception as e:
            rejected += 1
            errors.append(str(e)[:120])
    conn.commit()
    conn.close()
    return {"accepted": accepted, "rejected": rejected, "duplicate": duplicate, "errors": errors}

def query(sql: str, params=()) -> list:
    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
