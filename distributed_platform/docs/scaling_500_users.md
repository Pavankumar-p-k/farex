# Production Scaling Plan (500+ Users, Multi-Device)

## Capacity Targets
- 500+ registered employees
- 20-100 concurrent edge devices across sites
- 10-50 recognition events/sec burst at peak entrances

## Backend Scaling

### Local Site Backend
- Run FastAPI with multiple workers (gunicorn/uvicorn workers).
- Use PostgreSQL connection pooling (PgBouncer recommended).
- Keep recognition match matrix cached in memory with periodic invalidation.
- Move websocket fanout to Redis pub/sub if multi-instance local backend.

### Cloud Layer
- Supabase for global persistence and realtime replication.
- Partition high-volume tables by time (monthly partitions for events).
- Use read replicas for dashboard-heavy analytics.

## Performance Design
- Store normalized embeddings and perform vectorized cosine similarity.
- Batch enrollment updates and invalidate matcher cache once per batch.
- Use cooldown dedupe to prevent repeated attendance spam.

## Reliability
- Local-first write path prevents cloud outages from blocking office operations.
- Background sync retries with backoff + dead-letter queue for persistent failures.
- Health checks and automatic client route fallback (local/cloud).

## Observability
- Required metrics:
  - Recognition latency p50/p95
  - WebSocket active connections
  - Sync lag seconds
  - Match success rate
  - Device heartbeat freshness
- Alerts:
  - Sync lag > threshold
  - Drop in match rate
  - Device offline spike

## Security at Scale
- Rotate JWT signing keys and embedding encryption keys.
- Separate site-level credentials and cloud service keys.
- Enforce least-privilege RLS and audit all admin writes.

## Recommended Upgrades Beyond 500 Users
- Dedicated vector database or pgvector for faster global search.
- Redis cache for hot embedding vectors.
- Kafka/NATS event bus for recognition event fanout across services.
- Blue/green rollout for backend and schema migrations.

