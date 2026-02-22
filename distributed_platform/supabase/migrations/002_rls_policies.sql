-- Row-level security and role policies
alter table public.employees enable row level security;
alter table public.devices enable row level security;
alter table public.attendance enable row level security;
alter table public.recognition_events enable row level security;

-- Expect JWT claims:
--   role: admin | manager | operator | device
--   device_id: optional uuid for device tokens

create or replace function public.jwt_role()
returns text language sql stable as $$
  select coalesce(auth.jwt() ->> 'role', 'anonymous');
$$;

create or replace function public.jwt_device_id()
returns uuid language sql stable as $$
  select nullif(auth.jwt() ->> 'device_id', '')::uuid;
$$;

drop policy if exists employees_select_policy on public.employees;
create policy employees_select_policy on public.employees
for select using (public.jwt_role() in ('admin', 'manager', 'operator', 'device'));

drop policy if exists employees_write_policy on public.employees;
create policy employees_write_policy on public.employees
for all using (public.jwt_role() in ('admin', 'manager'))
with check (public.jwt_role() in ('admin', 'manager'));

drop policy if exists devices_select_policy on public.devices;
create policy devices_select_policy on public.devices
for select using (public.jwt_role() in ('admin', 'manager', 'operator'));

drop policy if exists devices_write_policy on public.devices;
create policy devices_write_policy on public.devices
for all using (
  public.jwt_role() in ('admin', 'manager', 'operator')
  or (public.jwt_role() = 'device' and id = public.jwt_device_id())
)
with check (
  public.jwt_role() in ('admin', 'manager', 'operator')
  or (public.jwt_role() = 'device' and id = public.jwt_device_id())
);

drop policy if exists attendance_select_policy on public.attendance;
create policy attendance_select_policy on public.attendance
for select using (public.jwt_role() in ('admin', 'manager', 'operator', 'device'));

drop policy if exists attendance_insert_policy on public.attendance;
create policy attendance_insert_policy on public.attendance
for insert with check (
  public.jwt_role() in ('admin', 'manager', 'operator')
  or (public.jwt_role() = 'device' and device_id = public.jwt_device_id())
);

drop policy if exists recognition_events_select_policy on public.recognition_events;
create policy recognition_events_select_policy on public.recognition_events
for select using (public.jwt_role() in ('admin', 'manager', 'operator', 'device'));

drop policy if exists recognition_events_insert_policy on public.recognition_events;
create policy recognition_events_insert_policy on public.recognition_events
for insert with check (
  public.jwt_role() in ('admin', 'manager', 'operator')
  or (public.jwt_role() = 'device' and device_id = public.jwt_device_id())
);
