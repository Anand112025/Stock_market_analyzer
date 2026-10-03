-- Cloud watchlist storage for Stock Market Analyzer v8
-- Run this once in Supabase SQL Editor.
create table if not exists public.watchlists (
    id text primary key,
    data jsonb not null,
    updated_at timestamptz not null default now()
);

-- Keep this table inaccessible to anonymous clients.
-- The Streamlit server should use the Supabase service-role key stored
-- in Streamlit secrets. Never put that key in source code or GitHub.
alter table public.watchlists enable row level security;

-- Optional trigger to keep updated_at current.
create or replace function public.set_watchlist_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists watchlists_updated_at on public.watchlists;
create trigger watchlists_updated_at
before update on public.watchlists
for each row execute function public.set_watchlist_updated_at();
