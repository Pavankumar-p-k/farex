# Windows Python Edge Client

## Features
- Local-first / cloud-fallback route selection
- ArcFace embedding extraction (InsightFace)
- Device heartbeat
- Embedding submit for matching
- Live websocket event stream
- Persistent offline queue (SQLite) for unsent embeddings
- Automatic queue flush after reconnect

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Environment Variables
- `LOCAL_BASE_URL`
- `CLOUD_BASE_URL`
- `CLIENT_USERNAME`
- `CLIENT_PASSWORD`
- `CLIENT_CAMERA_INDEX`
- `WS_LOCAL_URL`
- `WS_CLOUD_URL`
- `CLIENT_QUEUE_DB_PATH`
- `CLIENT_QUEUE_MAX_ITEMS`
- `CLIENT_QUEUE_FLUSH_BATCH`
- `CLIENT_HEARTBEAT_SECONDS`
- `CLIENT_REQUEST_TIMEOUT_SECONDS`
