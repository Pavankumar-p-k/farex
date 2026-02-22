# Architecture Overview

## Components
- Edge clients:
  - Windows Python app
  - Android Flutter app
  - Android React Native app
- Local backend:
  - FastAPI
  - PostgreSQL
  - WebSocket event broadcast
- Cloud backend:
  - Supabase Postgres
  - Supabase Realtime
  - Optional edge function ingestion

## Data Plane
1. Edge captures frame and computes ArcFace embedding (512 float).
2. Edge sends embedding to local backend (`/recognitions/match`) or cloud fallback.
3. Backend matches embedding against encrypted vector store.
4. Backend records `recognition_events` and optional `attendance`.
5. Backend emits event to WebSocket channel for all connected clients.

## Control Plane
- Device heartbeats update `devices` table.
- JWT-based role access controls API and websocket usage.
- Local sync worker replicates records to Supabase.

## Hybrid Routing
- Local preferred when reachable.
- Cloud fallback when local unavailable.
- This yields low latency on LAN and continuity across different networks.

