# Supabase Setup Questions

Use this checklist before production rollout.

1. Project and Region
- Which Supabase project ID and region will host production data?
- Is the region latency acceptable for all office sites?

2. Auth and JWT Claims
- How will roles (`admin`, `manager`, `operator`, `device`) be assigned?
- Will device tokens be minted by a trusted backend only?
- Do JWTs include `device_id` claim for device-scoped policies?

3. Secrets and Key Handling
- Where will `SUPABASE_SERVICE_ROLE_KEY` be stored (Vault/Secrets Manager)?
- Who can access service role key in CI/CD and runtime?
- What is key rotation policy and emergency revocation process?

4. Database and RLS
- Are RLS policies validated for least privilege?
- Is there a tenant/site segmentation requirement (multi-office isolation)?
- Do we need per-site policies on `employees`, `attendance`, and `recognition_events`?

5. Realtime
- Which tables should publish realtime events in production?
- Is there a need for channel-level filters by site/device?
- Do dashboard clients need historical replay or only live events?

6. Data Governance
- Retention period for `recognition_events` and `attendance`?
- Is encrypted embedding considered biometric data under local law?
- Do we need data residency, right-to-erasure, or audit export workflows?

7. Monitoring and Reliability
- Which metrics/alerts are required (error rate, sync lag, websocket disconnects)?
- What is acceptable sync lag from local server to cloud?
- What is disaster recovery RPO/RTO target?

8. Compliance and Security
- Required compliance baseline (ISO 27001, SOC2, internal controls)?
- Need IP allowlists or mTLS between local servers and cloud?
- Any requirement for hardware-backed key management (HSM/KMS)?

9. Capacity and Cost
- Expected events per minute peak across all sites?
- Required retention window and projected storage growth?
- Budget limits for realtime bandwidth and DB compute?

10. Release Management
- How are SQL migrations promoted across dev/stage/prod?
- Is there a rollback strategy for schema and policy changes?
- Who signs off production migrations?

