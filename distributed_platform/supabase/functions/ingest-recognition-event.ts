// Supabase Edge Function template
// deno-lint-ignore-file no-explicit-any
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const admin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

interface RequestBody {
  employee_id?: string | null;
  device_id: string;
  confidence: number;
  matched: boolean;
  timestamp?: string;
  payload_json?: Record<string, unknown>;
  location?: string | null;
}

Deno.serve(async req => {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), { status: 405 });
  }

  const body = (await req.json()) as RequestBody;
  if (!body.device_id || typeof body.confidence !== "number") {
    return new Response(JSON.stringify({ error: "Invalid payload" }), { status: 400 });
  }

  const eventInsert = await admin.from("recognition_events").insert({
    employee_id: body.employee_id ?? null,
    device_id: body.device_id,
    confidence: body.confidence,
    matched: !!body.matched,
    timestamp: body.timestamp ?? new Date().toISOString(),
    payload_json: body.payload_json ?? {},
  });
  if (eventInsert.error) {
    return new Response(JSON.stringify({ error: eventInsert.error.message }), { status: 500 });
  }

  if (body.matched && body.employee_id) {
    const attendanceInsert = await admin.from("attendance").insert({
      employee_id: body.employee_id,
      device_id: body.device_id,
      timestamp: body.timestamp ?? new Date().toISOString(),
      location: body.location ?? null,
    });
    if (attendanceInsert.error) {
      return new Response(JSON.stringify({ error: attendanceInsert.error.message }), { status: 500 });
    }
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
});

