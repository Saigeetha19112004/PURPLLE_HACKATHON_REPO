# Store Intelligence

End-to-end retail analytics: raw CCTV → live store metrics API.

## Quick Start (5 commands)

```bash
git clone <repo-url> store-intelligence && cd store-intelligence
cp /path/to/CAM_*.mp4 data/clips/
python3 pipeline/detect.py --clips-dir data/clips --output-dir data/events
docker compose up --build -d
curl http://localhost:8001/stores/STORE_PURPLLE_001/metrics
```

## Detection Pipeline

```bash
python3 pipeline/detect.py \
  --clips-dir /path/to/clips \
  --output-dir data/events \
  --cameras CAM_3 CAM_1 CAM_2 CAM_5 CAM_4 \
  --sample-rate 3 \
  --warmup 5
```

Events are written to `data/events/all_events.jsonl` and per-camera `events_CAM_N.jsonl`.

## Ingest Events into API

```bash
python3 -c "
import json, requests
events = [json.loads(l) for l in open('data/events/all_events.jsonl') if l.strip()]
# Send in batches of 500
for i in range(0, len(events), 500):
    r = requests.post('http://localhost:8001/events/ingest', json={'events': events[i:i+500]})
    print(r.json())
"
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /events/ingest` | Ingest up to 500 events (idempotent by event_id) |
| `GET /stores/{id}/metrics` | Unique visitors, conversion rate, avg dwell, queue depth |
| `GET /stores/{id}/funnel` | Entry → Zone → Billing → Purchase with drop-off % |
| `GET /stores/{id}/heatmap` | Zone visit frequency + dwell, normalised 0–100 |
| `GET /stores/{id}/anomalies` | Active anomalies with severity and suggested action |
| `GET /health` | Service status, last event per camera, STALE_FEED warning |

## Tests

```bash
python3 -m pytest tests/ -v
```

## Dashboard

Live dashboard runs at `http://localhost:8001/dashboard` after `docker compose up`.

## Architecture

See `docs/DESIGN.md` for full architecture.  
See `docs/CHOICES.md` for three key decisions with trade-off reasoning.

## Store ID

Default store: `STORE_PURPLLE_001`
