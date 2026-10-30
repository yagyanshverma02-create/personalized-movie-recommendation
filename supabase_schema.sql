-- Run once in the Supabase SQL Editor for this MovieMind app.
create table if not exists public.account_profiles (
    profile_id text primary key,
    email text not null unique,
    profile jsonb not null
);

create table if not exists public.account_credentials (
    email text primary key,
    password_salt text not null,
    password_hash text not null,
    password_iterations integer not null,
    updated_at text not null
);

alter table public.account_profiles enable row level security;
alter table public.account_credentials enable row level security;

-- The Streamlit server uses the service role key, which bypasses RLS.
-- Never put that key in browser code or commit it to GitHub.
