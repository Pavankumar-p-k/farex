# Backend API Endpoints

Base prefix: `/api/v1`

## Health
- `GET /health`
  - Liveness probe for local route detection.

## Auth
- `POST /auth/token`
  - Body: `{ "username": "...", "password": "..." }`
  - Returns JWT bearer token with role claim.

## Devices
- `POST /devices/heartbeat`
  - Registers/updates device status and returns `device_id`.
  - Used by edge clients every N seconds.

## Employees
- `GET /employees` (`admin`, `manager`, `operator`)
- `POST /employees` (`admin`, `manager`)
  - Embedding payload must be 512 float values.
  - Stored encrypted at rest.

## Recognition
- `POST /recognitions/match`
  - Input: `device_id`, `embedding[512]`, optional metadata/location.
  - Behavior:
    - Normalizes embedding.
    - Matches against encrypted employee embeddings.
    - Writes recognition event.
    - Writes attendance with cooldown logic.
    - Broadcasts WebSocket event to all connected clients.

## Attendance
- `GET /attendance?limit=200`

## Recognition Events
- `GET /events/recognitions?limit=200`

## Sync
- `GET /sync/status`
- `POST /sync/trigger` (`admin`, `manager`)

## WebSocket
- `WS /ws/events?token=<JWT>`
  - Channel: global event stream.
  - Message format:
    ```json
    {
      "type": "recognition_event",
      "payload": {
        "event_id": "...",
        "employee_id": "...",
        "employee_name": "Alice",
        "confidence": 0.84,
        "matched": true,
        "device_id": "...",
        "timestamp": "2026-02-21T21:00:00Z",
        "location": null
      }
    }
    ```

