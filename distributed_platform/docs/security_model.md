# Security Model

## Data Handling
- Raw face images never leave edge device in recognition flow.
- Only normalized ArcFace embeddings (512 floats) are transmitted.
- Embeddings encrypted at rest before DB storage.

## Authentication and Authorization
- JWT bearer authentication for all API and WebSocket access.
- Roles:
  - `admin`: full control
  - `manager`: enrollment + operations
  - `operator`: operations and monitoring
  - `device`: restricted ingestion/heartbeat scope
- Supabase RLS mirrors same role model in cloud.

## Transport Security
- Use HTTPS/TLS for local and cloud APIs in production.
- Use WSS for websocket in production.
- Service keys stored in secret manager, never bundled with clients.

## Key Management
- Rotate JWT secret and embedding encryption key periodically.
- Maintain separate keys per environment (dev/stage/prod).
- Restrict Supabase service-role key to backend-only runtime.

## Auditability
- Persist recognition events with timestamp, device_id, and confidence.
- Persist attendance logs separately from raw events.
- Log auth failures and suspicious traffic for SIEM.

