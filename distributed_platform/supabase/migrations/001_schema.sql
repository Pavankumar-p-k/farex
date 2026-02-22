-- Core distributed face recognition schema
create extension if not exists "uuid-ossp";

create table if not exists public.employees (
  id uuid primary key default uuid_generate_v4(),
  external_id text not null unique,
  name text not null,
  role text not null default 'employee',
  embedding_ciphertext text not null,
  embedding_norm double precision not null default 1.0,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.devices (
  id uuid primary key default uuid_generate_v4(),
  device_name text not null,
  status text not null default 'online',
  device_type text not null,
  network_mode text not null default 'cloud',
  metadata_json jsonb not null default '{}'::jsonb,
  last_seen timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create table if not exists public.attendance (
  id uuid primary key default uuid_generate_v4(),
  employee_id uuid not null references public.employees(id) on delete cascade,
  device_id uuid not null references public.devices(id) on delete cascade,
  timestamp timestamptz not null default now(),
  location text null
);

create table if not exists public.recognition_events (
  id uuid primary key default uuid_generate_v4(),
  employee_id uuid null references public.employees(id) on delete set null,
  device_id uuid not null references public.devices(id) on delete cascade,
  confidence double precision not null,
  matched boolean not null default false,
  payload_json jsonb not null default '{}'::jsonb,
  timestamp timestamptz not null default now()
);

create index if not exists idx_employees_external_id on public.employees(external_id);
create index if not exists idx_attendance_employee_time on public.attendance(employee_id, timestamp desc);
create index if not exists idx_attendance_device_time on public.attendance(device_id, timestamp desc);
create index if not exists idx_recognition_events_time on public.recognition_events(timestamp desc);
create index if not exists idx_recognition_events_employee on public.recognition_events(employee_id);
create index if not exists idx_devices_last_seen on public.devices(last_seen desc);

create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_employees_updated_at on public.employees;
create trigger trg_employees_updated_at
before update on public.employees
for each row execute function public.set_updated_at();
