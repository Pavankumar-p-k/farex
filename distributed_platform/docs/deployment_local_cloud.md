# Deployment Guide (Local + Cloud)

## 1) Local Office Deployment

### Prerequisites
- Windows/Linux host for local backend
- PostgreSQL 14+
- Python 3.11+ (3.12 works)
- LAN reachability from edge devices

### Steps
1. Create local DB:
```sql
create database face_platform;
```
2. Configure backend env:
```bash
cd distributed_platform/backend
copy .env.example .env
```
3. Update `.env`:
- `DATABASE_URL`
- `JWT_SECRET`
- `EMBEDDING_CIPHER_KEY`
- `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` (optional for sync)
4. Run backend:
```bash
pip install -r requirements.txt
run_local.cmd
```
5. Open firewall for `9000` on office LAN.

## 2) Supabase Cloud Deployment

### Steps
1. Create Supabase project.
2. Apply migrations in `distributed_platform/supabase/migrations` in order.
3. Enable Realtime for required tables.
4. Deploy edge function template:
```bash
supabase functions deploy ingest-recognition-event --no-verify-jwt
```
5. Set function secrets:
```bash
supabase secrets set SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
```

## 3) Client Configuration

### Windows Client
- Set `LOCAL_BASE_URL` to local backend
- Set `CLOUD_BASE_URL` to cloud endpoint
- Run:
```bash
cd distributed_platform/clients/windows_python
python main.py
```

### Android Flutter / RN
- Build with local and cloud base URLs in app config.
- On startup, client probes local `/health`; fallback to cloud.

## 4) Production Hardening
- Add reverse proxy (Nginx/Caddy) with TLS certificates.
- Store secrets in vault/KMS (not `.env` in repo).
- Enable structured logs and central aggregation.
- Add readiness/liveness probes and process supervisor.

