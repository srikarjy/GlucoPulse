-- Supabase row-level-security spike for docs/ALERTS_API_SECURITY.md (§11). SYNTHETIC DATA ONLY: demo-a, demo-b.
-- Question: can Postgres RLS + Supabase Auth enforce "a logged-in user sees only the alerts of the patient linked to
-- their account", with enrollment by single-use code, without trusting any input for the patient id?
-- Run in a throwaway project. Part 1 = schema + policies (migration), part 2 = demo data, part 3 = checks.

-- ===== Part 1: schema, RLS, enrollment =========================================================================
create schema if not exists glucopulse;

create table glucopulse.glucose_alerts (
    id              uuid primary key default gen_random_uuid(),     -- opaque, reveals nothing
    patient_id      text        not null,
    time            timestamptz not null,
    tier            text        not null check (tier in ('heads_up', 'loud')),
    direction       text        not null check (direction in ('low', 'high')),
    projected_20min double precision not null check (projected_20min between 40 and 400),   -- capped
    projection_capped boolean   not null default false,
    unique (patient_id, time, tier)
);

-- Which login owns which patient. Written only by redeem_enrollment_code(); nobody can insert or update it directly.
create table glucopulse.patient_links (
    user_id    uuid primary key,                 -- auth.users.id (no FK in the spike: no real users are created)
    patient_id text not null unique,             -- one login per patient, one patient per login
    linked_at  timestamptz not null default now()
);

create table glucopulse.enrollment_codes (
    code_hash  text primary key,                 -- sha256 of the code; the code itself is never stored
    patient_id text not null,
    expires_at timestamptz not null,
    used_at    timestamptz,
    used_by    uuid
);

alter table glucopulse.glucose_alerts   enable row level security;
alter table glucopulse.patient_links    enable row level security;
alter table glucopulse.enrollment_codes enable row level security;   -- no policies: invisible to API roles

-- The one read rule: the patient comes from the verified token (auth.uid()), never from the query.
create policy alerts_own_patient on glucopulse.glucose_alerts
    for select to authenticated
    using (patient_id = (select l.patient_id from glucopulse.patient_links l where l.user_id = (select auth.uid())));

create policy links_own_row on glucopulse.patient_links
    for select to authenticated
    using (user_id = (select auth.uid()));
-- No insert/update/delete policies on any table: API roles can only read, and only their own rows.

grant usage on schema glucopulse to authenticated;
grant select on glucopulse.glucose_alerts, glucopulse.patient_links to authenticated;
revoke all on glucopulse.enrollment_codes from anon, authenticated;

-- Enrollment: single use, expiring, hashed. SECURITY DEFINER with a fixed search_path; callable only when logged in.
create or replace function glucopulse.redeem_enrollment_code(code text)
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare
    uid uuid := auth.uid();
    pid text;
begin
    if uid is null then
        raise exception 'not authenticated' using errcode = '28000';
    end if;
    update glucopulse.enrollment_codes c
       set used_at = now(), used_by = uid
     where c.code_hash = encode(extensions.digest(code, 'sha256'), 'hex')
       and c.used_at is null
       and c.expires_at > now()
    returning c.patient_id into pid;
    if pid is null then
        raise exception 'invalid or expired code' using errcode = '22023';   -- same message for every failure
    end if;
    insert into glucopulse.patient_links (user_id, patient_id) values (uid, pid);
    return 'linked';
end;
$$;
revoke all on function glucopulse.redeem_enrollment_code(text) from public, anon;
grant execute on function glucopulse.redeem_enrollment_code(text) to authenticated;

-- ===== Part 2: synthetic demo data (run as the project owner) ==================================================
insert into glucopulse.glucose_alerts (patient_id, time, tier, direction, projected_20min, projection_capped) values
    ('demo-a', '2026-11-24 10:35+00', 'heads_up', 'low',  40, true),
    ('demo-a', '2026-11-24 10:40+00', 'loud',     'low',  40, true),
    ('demo-b', '2026-11-24 13:30+00', 'heads_up', 'high', 255, false);
insert into glucopulse.enrollment_codes (code_hash, patient_id, expires_at) values
    (encode(extensions.digest('DEMO-A-CODE', 'sha256'), 'hex'), 'demo-a', now() + interval '1 day'),
    (encode(extensions.digest('DEMO-B-CODE', 'sha256'), 'hex'), 'demo-b', now() + interval '1 day'),
    (encode(extensions.digest('EXPIRED-CODE', 'sha256'), 'hex'), 'demo-b', now() - interval '1 minute');

-- ===== Part 3: checks, each run as an API user (role authenticated, claims as Supabase Auth would set) ===========
-- user A = 00000000-0000-0000-0000-00000000000a, user B = ...0b, user C = ...0c (never enrolled).
-- Expected results are in docs/ALERTS_API_SECURITY.md §11 once run.
