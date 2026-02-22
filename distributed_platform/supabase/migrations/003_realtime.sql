-- Realtime publications for cross-network updates
alter publication supabase_realtime add table public.employees;
alter publication supabase_realtime add table public.devices;
alter publication supabase_realtime add table public.attendance;
alter publication supabase_realtime add table public.recognition_events;
