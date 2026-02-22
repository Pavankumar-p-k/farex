# Farex Distributed Real-Time Facial + Object Recognition Platform

Hybrid distributed architecture for:
- Windows desktop edge client (Python + ArcFace embedding)
- Android client (Flutter or React Native)
- Local office backend (FastAPI + PostgreSQL + WebSocket)
- Cloud layer (Supabase Postgres + Realtime + Edge Functions)

## Hybrid Mode

Client startup routing:
1. Try local backend `GET /api/v1/health` with short timeout.
2. If reachable: use local FastAPI (`mode=local`) for low-latency recognition.
3. If unreachable: fallback to cloud endpoints (`mode=cloud`) via Supabase.

This gives LAN-speed behavior in office plus cross-network continuity.

## Face Recognition Flow

1. Edge client captures frame.
2. ArcFace model generates normalized 512-d embedding on-device.
3. Client sends embedding only (`/api/v1/recognitions/match`) with JWT.
4. Backend performs cosine similarity match against encrypted stored embeddings.
5. Backend writes recognition events (and optional attendance-style logs if enabled).
6. Backend broadcasts event to all connected dashboards via WebSocket (`/ws/events`).
7. Sync worker periodically replicates local rows to Supabase.

## Security

- Raw images are never transmitted to backend/cloud.
- Embeddings encrypted at rest (`Fernet`) in local DB and cloud DB.
- JWT auth + role-based authorization (`admin`, `manager`, `operator`, `device`).
- RLS policies for Supabase cloud tables.

## Folder Structure

```text
distributed_platform/
  backend/
    app/
      main.py
      api/
        deps.py
        routes/
          auth.py
          health.py
          employees.py
          devices.py
          recognitions.py
          attendance.py
          events.py
          sync.py
      core/
        config.py
        security.py
      db/
        base.py
        models.py
        session.py
      schemas/
        auth.py
        employee.py
        device.py
        recognition.py
        attendance.py
        event.py
      services/
        encryption.py
        matcher.py
        sync.py
      ws/
        manager.py
    requirements.txt
    .env.example
    run_local.cmd
    run_local.ps1
  clients/
    windows_python/
      main.py
      arcface_embedder.py
      hybrid_api.py
      ws_listener.py
      config.py
      requirements.txt
    android_flutter/
      pubspec.yaml
      lib/
        main.dart
        src/
          hybrid_endpoint.dart
          recognition_service.dart
          ws_service.dart
    android_react_native/
      package.json
      src/
        App.tsx
        hybridClient.ts
        wsClient.ts
  supabase/
    migrations/
      001_schema.sql
      002_rls_policies.sql
      003_realtime.sql
    functions/
      ingest-recognition-event.ts
    README.md
    questions.md
  docs/
    api_endpoints.md
    deployment_local_cloud.md
    scaling_500_users.md
    security_model.md
    sync_logic.md
```

## Quick Start

1. Backend (local):
```bash
cd distributed_platform/backend
pip install -r requirements.txt
copy .env.example .env
run_local.cmd
```

2. Windows edge client:
```bash
cd distributed_platform/clients/windows_python
pip install -r requirements.txt
python main.py
```

3. Supabase:
- Apply SQL in `distributed_platform/supabase/migrations/`
- Configure Realtime and RLS
- Deploy edge function template (optional ingestion path)

Read deployment details in `distributed_platform/docs/deployment_local_cloud.md`.

For a public cloud API endpoint (recommended for cross-network Windows failover), use:
`distributed_platform/backend/DEPLOY_RENDER.md`
