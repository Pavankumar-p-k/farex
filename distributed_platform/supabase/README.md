# Supabase Cloud Layer

This folder contains the cloud-side assets for hybrid deployment.

## Contents
- `migrations/001_schema.sql`: core tables (`employees`, `devices`, `attendance`, `recognition_events`)
- `migrations/002_rls_policies.sql`: role-based RLS policies
- `migrations/003_realtime.sql`: Realtime publication setup
- `functions/ingest-recognition-event.ts`: edge function template for cloud ingestion
- `questions.md`: production readiness and architecture checklist

## Apply Migrations
```bash
supabase db push
```

Or via SQL editor in order:
1. `001_schema.sql`
2. `002_rls_policies.sql`
3. `003_realtime.sql`

## Deploy Edge Function
```bash
supabase functions deploy ingest-recognition-event --no-verify-jwt
```

Set secrets:
```bash
supabase secrets set SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
```

