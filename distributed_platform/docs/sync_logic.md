# Real-Time Sync Logic

## Same WiFi / Office LAN
- Edge clients route to local FastAPI.
- Local backend performs recognition + attendance in < LAN latency.
- WebSocket broadcasts to all local dashboards immediately.

## Different Network / Internet
- Clients fallback to cloud endpoint (Supabase edge/API layer).
- Supabase Realtime broadcasts updates globally.

## Hybrid Bridging (Local to Cloud)
- Local FastAPI writes authoritative local records first.
- Background sync worker pushes unsynced rows to Supabase every interval.
- Synced rows get `synced_at` timestamp to avoid duplicate replication.

## Consistency Model
- Local mode prioritizes availability + low latency.
- Cloud becomes global source for cross-site analytics and remote dashboards.
- Eventual consistency between local sites and cloud, bounded by sync interval.

## Failure Handling
- If cloud unavailable:
  - Local mode continues operating offline.
  - Sync retries on next interval.
- If local backend unavailable:
  - Clients route to cloud mode automatically.
- If WebSocket unstable:
  - Clients can poll `GET /events/recognitions`.

