# Render Deployment Guide (Cloud Backend)

This deploys the distributed FastAPI backend as a public cloud endpoint for global failover.

## 1) Push Project to Git Remote

Render Blueprints require a GitHub/GitLab/Bitbucket repo.

From project root:

```powershell
git init
git add .
git commit -m "chore: add Render deployment config"
git branch -M main
git remote add origin <YOUR_GIT_REMOTE_URL>
git push -u origin main
```

## 2) Create Render Blueprint

1. Open Render Dashboard.
2. Click `New` -> `Blueprint`.
3. Select your repository.
4. Render will detect `render.yaml` at repo root.
5. Click `Apply`.

## 3) Fill Required Secrets

Set these in Render service environment before first successful boot:

- `JWT_SECRET` (long random string)
- `EMBEDDING_CIPHER_KEY` (long random string used as key material)
- `BOOTSTRAP_ADMIN_PASSWORD` (strong admin password)
- `CORS_ORIGINS_RAW` (comma-separated frontend origins)

Notes:

- `DATABASE_URL` is auto-injected from Render PostgreSQL.
- `SYNC_ENABLED` is set to `false` in `render.yaml` for cloud-primary mode.

Optional random secret generation (PowerShell):

```powershell
[Convert]::ToBase64String((1..48 | ForEach-Object { Get-Random -Maximum 256 }))
```

## 4) Verify Cloud API

After deploy is live:

```bash
curl https://<your-render-service>.onrender.com/api/v1/health
```

Expected:

```json
{"ok": true, "service": "distributed-face-platform", ...}
```

## 5) Configure Windows Clients for Failover

Set each Windows edge client:

```powershell
$env:LOCAL_BASE_URL="http://<local-backend-ip>:9000"
$env:CLOUD_BASE_URL="https://<your-render-service>.onrender.com"
$env:WS_CLOUD_URL="wss://<your-render-service>.onrender.com/ws/events"
$env:CLIENT_USERNAME="admin"
$env:CLIENT_PASSWORD="<BOOTSTRAP_ADMIN_PASSWORD>"
```

Run:

```powershell
cd distributed_platform\clients\windows_python
python main.py
```

## 6) Real-World Behavior You Get

- Local backend reachable: client uses local route.
- Local backend down: client auto-switches to cloud route.
- Both down: embeddings are queued locally (SQLite) on device.
- Network/server returns: queued embeddings auto-flush.

## 7) Recommended Hardening (Next)

1. Replace bootstrap admin with dedicated service users per device/site.
2. Restrict CORS to your actual dashboard domains only.
3. Rotate `JWT_SECRET` and `EMBEDDING_CIPHER_KEY` periodically.
4. Add TLS-only dashboard/frontend integrations.
