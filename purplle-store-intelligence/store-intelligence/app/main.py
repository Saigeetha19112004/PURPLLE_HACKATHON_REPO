import uuid, time, logging, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from .db import init_db, insert_events, query
from .models import IngestRequest, IngestResponse
from .metrics import get_metrics, get_funnel, get_heatmap, get_anomalies, get_health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("store_intelligence")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Auto-load events from JSONL if DB is empty
    events_file = Path("/home/claude/store-intelligence/data/events/all_events.jsonl")
    existing = query("SELECT COUNT(*) as cnt FROM events")[0]["cnt"]
    if existing == 0 and events_file.exists():
        logger.info(f"Seeding DB from {events_file}")
        events = []
        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        result = insert_events(events)
        logger.info(f"Seeded: {result}")
    yield

app = FastAPI(
    title="Purplle Store Intelligence API",
    version="1.0.0",
    description="Real-time retail analytics from CCTV event streams",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - t0) * 1000)
    logger.info(json.dumps({
        "trace_id": trace_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
    }))
    response.headers["X-Trace-ID"] = trace_id
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "SERVICE_UNAVAILABLE", "message": "An internal error occurred. Check /health for system status."},
    )

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(payload: IngestRequest):
    if len(payload.events) > 500:
        raise HTTPException(400, "Batch size exceeds 500 events")
    rows = [e.model_dump() for e in payload.events]
    result = insert_events(rows)
    return IngestResponse(**result)

@app.get("/stores/{store_id}/metrics")
async def store_metrics(store_id: str):
    exists = query("SELECT 1 FROM events WHERE store_id=? LIMIT 1", (store_id,))
    if not exists:
        raise HTTPException(404, f"Store {store_id} not found or has no events")
    return get_metrics(store_id)

@app.get("/stores/{store_id}/funnel")
async def store_funnel(store_id: str):
    exists = query("SELECT 1 FROM events WHERE store_id=? LIMIT 1", (store_id,))
    if not exists:
        raise HTTPException(404, f"Store {store_id} not found")
    return get_funnel(store_id)

@app.get("/stores/{store_id}/heatmap")
async def store_heatmap(store_id: str):
    return get_heatmap(store_id)

@app.get("/stores/{store_id}/anomalies")
async def store_anomalies(store_id: str):
    return get_anomalies(store_id)

@app.get("/health")
async def health():
    return get_health()
