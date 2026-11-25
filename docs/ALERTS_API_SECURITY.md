# Alerts for AI systems: API and security design

Status: **DRAFT for review. Nothing here is built.** Implementation starts only after this is approved.

Goal: let two AI systems read the streaming glucose alerts (docs/STREAMING.md) without weakening the guarantees the
project already has.

- **Clinician side:** the internal Clinical Reasoning Agent (`agents/mcp_server`).
- **Patient side:** the public Claude/ChatGPT MCP server (`agents/public_mcp`).

Non-goals: writing or acknowledging alerts, raw glucose readings for the patient side, a patient portal, push
notifications, real patients (see §7).

## 1. Principles

1. **One gate.** Every alert read goes through the FHIR facade (`serving/healthcare`). Neither AI server reads
   `glucose_alerts` directly. The facade already enforces SMART scopes, confines `patient/` scopes to the token's
   `patient` claim (`security.py:59`), fails closed, and audits every call.
2. **The patient identity comes from a verified token claim, never from input.** That covers the AI, the user's
   message and a tool argument. The patient-side tool has no patient parameter at all.
3. **No token passthrough.** A token is only ever presented to the audience it was issued for (the MCP authorization
   spec forbids forwarding). Every hop gets its own credential.
4. **Least data.** The patient side gets alert metadata from the last 24 h only.
5. **Existence is not leaked.** A patient you may not see returns 404, as `_patient_or_404` already does. Alert ids
   are opaque.

## 2. Actors and trust boundaries

```
                 ┌──────────── trust boundary: GlucoPulse backend ─────────────┐
 Clinician ──► Clinical agent MCP ──(facade token, user/ scope)──► FHIR facade ──► glucose_alerts
 (reviewer)    agents/mcp_server                                     │  ▲            (TimescaleDB)
                                                                     │  │ audit_log (hash-chained)
 Patient ──► Claude/ChatGPT ──(MCP token, aud=MCP)──► Public MCP ────┘  │
   │          (AI provider: data leaves us here)      agents/public_mcp ─(service token, system/ scope,
   │                                                                     on-behalf-of = verified patient)
   └──► Auth0 login + enrollment code ──► Enrollment API ──► Auth0 app_metadata.patient_id
```

Untrusted: everything the AI or the user types, including tool arguments and prompt text.
Trusted only after verification: token claims, checked for signature, `iss`, `aud`, `exp` and the pinned algorithm.

## 3. API

### 3.1 FHIR facade: alerts as `Observation` (decided 2026-11-24)

One `Observation` per `glucose_alerts` row. It is told apart from glucose readings by its own category:
`CGM_ALERT = https://glucopulse.dev/fhir/CodeSystem/observation-category|cgm-alert`.

| `glucose_alerts` | `Observation` | Note |
|---|---|---|
| (patient_id, time, tier) | `id` = HMAC-SHA256(key, alert_id), first 32 hex | Opaque: does not reveal patient or time |
| — | `status` = `final`; `category` = `CGM_ALERT`; `code` = GlucoPulse `glucose-trend-alert` | Never the LOINC glucose code, so no client mistakes an alert for a reading |
| patient_id | `subject` = `Patient/{pseudonymous id}` | Same ids as the facade's `Patient` |
| time | `effectiveDateTime` | UTC |
| tier | `valueCodeableConcept`: `heads-up` / `loud` | |
| direction + tier | `interpretation` (HL7 v3): heads_up `L`/`H`, loud `LL`/`HH` | |
| glucose, capped projection | `component`: current glucose and projected-20-min glucose (`valueQuantity` mg/dL); the projection carries a `capped` extension | Capped to 40-400, as the MCP tool shows it |
| — | `derivedFrom` → the glucose reading `Observation` | |
| — | `meta.security` = `HTEST` while data is synthetic or demo | Machine-readable "not real patient data" |

Endpoints (extending the existing `GET /fhir/Observation`; old behaviour unchanged):

| Request | Required scope (any one) | Result |
|---|---|---|
| `GET /fhir/Observation?patient={id}&category=…\|cgm-alert&date=ge{instant}&_count=&_page=` | `patient/Observation.rs?category=…\|cgm-alert` (SMART v2 granular, own patient only), or `patient/`, `user/`, `system/` `Observation.read` | `Bundle` searchset of alerts, newest first; `patient` required (no bulk); `_count` ≤ 100; window ≤ 30 days |
| `GET /fhir/Observation?patient={id}` (no `category`) | as today | glucose readings only, unchanged |
| `GET /fhir/Observation/{id}` | as above, matched to the resource's category | the resource, or 404 |

**Why the granular scope matters.** Readings and alerts share the `Observation` type, so a plain
`patient/Observation.read` would let an alerts-only caller read raw glucose too. The patient-side token therefore
carries only the SMART v2 granular scope restricted to `category=cgm-alert`. The facade has to learn to parse that
one form, `?category=` on `Observation`, and must treat any other query in a scope as no access (fail closed).

Errors follow the facade's `OperationOutcome`:

| Status | When |
|---|---|
| 401 | no or invalid token |
| 403 | the token lacks any `Observation` scope that covers the requested category |
| 404 | the patient or alert is outside the token's reach, or does not exist; the two cases are indistinguishable |
| 400 | bad parameters |
| 501 | no store configured |

Every outcome, including denials, writes an `AuditEvent`. Header `X-Purpose-Of-Use` is recorded, as today.

### 3.2 Clinical agent tool (clinician side)

```
get_recent_alerts(patient_id: str, hours: int = 72)   # hours 1..168; scope alerts:read (new) on clinical-agent role
  -> {"alerts": [{id, effective, tier, direction, glucose, projected_20min, capped}], "source": "Observation"}
```

- **Facade call:** the tool calls the facade through `ehr.sync.EhrClient`, with the agent's existing facade token
  (`user/Observation.read`), like the existing tools.
- **Citations:** findings must cite alert `Observation` ids. The existing citation check rejects any sentence whose
  numbers are not in a cited resource.
- **Review:** a reviewer still decides; the agent role has no `review:decide`.
- **Optional, later:** on a `loud` alert, a draft summary is queued for review. It never notifies the patient.

### 3.3 Public MCP tool (patient side)

```
get_my_recent_alerts(hours: int = 24)    # hours 1..24; NO patient argument; MCP scope alerts:mine (new)
  -> {"alerts": [{identified, tier, direction, projected_20min, capped}], "say_to_person": ..., "disclaimer": ...}
```

The patient is the `patient` claim of the verified MCP token. If the token has no claim (not enrolled), the tool says
how to enroll and returns no data.

- **Fields:** no glucose values. The response carries the tier, the direction and the capped projection, plus the
  existing disclaimer ("not for insulin dosing, confirm with a meter").
- **Facade call (recommended): Auth0 On-Behalf-Of token exchange** (RFC 8693), documented by Auth0 for "MCP servers
  that need to call first-party APIs on the user's behalf"
  (auth0.com/docs/secure/call-apis-on-users-behalf/on-behalf-of-token-exchange). The steps:
  1. The public server presents the user's MCP token as `subject_token` and receives a token for audience
     `glucopulse-fhir`.
  2. That token has the same `sub`, the alerts-only scope `patient/Observation.rs?category=…\|cgm-alert`, and an `act` claim naming the MCP server as actor.
  3. The facade's existing `patient/` rule then confines it to the user's own patient.

  With this, the MCP server never holds a credential that can read other patients, and the audit gets the actor
  from `act`.
- **Fallback if the exchange cannot be enabled:** the server uses its own service token (`system/Observation.rs?category=…\|cgm-alert`
  only) and passes the verified patient, with `X-On-Behalf-Of: <sub>` recorded. This carries the residual risk in §5.

### 3.4 Enrollment API (links a login to a patient)

| Request | Who | Effect |
|---|---|---|
| `POST /enrollment/codes {patient}` | care team: facade token with `user/Enrollment.write` (new) | Returns a code: 10 characters from a 32-letter alphabet, single use, expires after 24 h. It is stored as a hash with the issuer's `sub`. |
| `POST /enrollment/redeem {code}` | patient: Auth0 access token (aud = enrollment API) | Constant-time check. On success, an Auth0 Management call sets `app_metadata.patient_id`, and the code is burned. |
| `DELETE /enrollment/link` | patient, or care team for that patient | Clears `app_metadata.patient_id`; access ends when the current token expires (§4) |

- **Auth0 Action:** a post-login Action copies `app_metadata.patient_id` into the access token as the `patient`
  claim, and adds `alerts:mine` only when it is present.
- **Management credential:** a machine-to-machine client with only `update:users_app_metadata`, kept server-side.
- **Guessing limits:** redeem is rate-limited to 5 tries per account per hour and 20 per IP per hour. After 5 failures
  the account is locked from redeeming for 1 h.

## 4. Tokens, audiences, scopes

| Token | Issuer | Audience | Lifetime | Scopes / claims |
|---|---|---|---|---|
| Patient → public MCP | Auth0 | `https://<mcp host>/mcp` | ≤ 15 min access, refresh rotation on | `glucose:alerts`, `alerts:mine`; claim `patient` (after enrollment) |
| Patient → enrollment API | Auth0 | `https://<api host>/enrollment` | ≤ 15 min | `enrollment:redeem` |
| Public MCP → facade (OBO exchange) | Auth0 | `glucopulse-fhir` | ≤ 15 min | user's `sub`, `patient` claim, `patient/Observation.rs?category=…\|cgm-alert`, `act.sub` = MCP server |
| Public MCP → facade (fallback only) | Auth0 client credentials | `glucopulse-fhir` | ≤ 15 min | `system/Observation.rs?category=…\|cgm-alert` only |
| Clinical agent → facade | existing facade issuer | `glucopulse-fhir` | ≤ 15 min (`GLUCOPULSE_MAX_TOKEN_TTL`) | existing `user/Observation.read` (readings and alerts) |
| Care team → facade | existing facade issuer | `glucopulse-fhir` | ≤ 15 min | `user/Enrollment.write` |

- **Verification:** all tokens use RS256/ES256 through JWKS, with `iss`, `aud` and `exp` checked; `alg: none` and
  HS256 public-key confusion are rejected, as tested today.
- **Unlinking:** unlink takes effect within one access-token lifetime, at most 15 min. A refresh after unlinking
  carries no `patient` claim.

## 5. Threat model (STRIDE)

| Threat | Example | Control | Test |
|---|---|---|---|
| Spoofing | Someone redeems another patient's code | Codes from the care team only; single use; 24 h; hashed; rate-limited; audited with issuer and redeemer | reuse, expiry, brute force (locks), wrong account |
| Spoofing | Forged or other-audience token | Signature, `iss`, `aud`, `exp`, pinned algorithm | existing `test_jwks_*` style tests on each new audience |
| Tampering | AI or user asks for "patient cgmacros-012" | The patient tool has no patient argument; the patient comes only from the claim | prompt and argument injection test: an extra argument is rejected by the schema |
| Tampering / IDOR | Patient A's token reads B through the facade | `patient/` scope confined to the claim; 404 for others | A→B search and read both 404, and both audited |
| Repudiation | "I never accessed that" | Hash-chained audit with `sub`, `azp`, on-behalf-of, purpose, outcome | every endpoint writes exactly one audit row, success or denial |
| Information disclosure | Data leaves to the AI provider | Least data (§3.3), consent text, synthetic data only until a BAA (§7) | response-shape test: no glucose values, no ids of other people |
| Information disclosure | Alert id reveals patient and time | HMAC opaque id | id shows no patient or time; same alert gives the same id |
| Information disclosure | Errors echo input (prompt-injection channel) | Fixed error messages, as the public MCP does today | the existing injection tests are extended to the new tool |
| Denial of service | Scraping or brute force | Per-user and per-IP limits on redeem and the tool; `_count` and window caps | limit tests |
| Elevation of privilege | Alerts-only token reads raw glucose (alerts and readings are both `Observation`) | Granular scope `Observation.rs?category=…\|cgm-alert`; a scope with any other query grants nothing (fail closed) | alerts token gets nothing from readings, `Patient` or other categories; an unknown scope query is denied |
| Elevation of privilege | Compromised public MCP server reads every patient's alerts | With OBO exchange (recommended): the server only ever holds tokens for the user who called it; a stolen exchanged token reaches one patient for ≤ 15 min. **Residual risk only with the fallback service token**: alerts-only scope, rate limits, on-behalf-of audit. | exchanged token for A cannot read B; the server holds no `system/` credential in OBO mode |
| Replay | An exchanged token is intercepted | Bearer tokens are replayable (Auth0 notes this); TLS on every hop, ≤ 15 min lifetime, never logged | logs contain no tokens |

## 6. Audit

Each event records: actor `sub`, client `azp`, on-behalf-of `sub` (patient side), action, resource type, a
pseudonymous patient, outcome and status, purpose of use, request id and IP. Never alert values. This is the existing
`audit.py` schema plus one `on_behalf_of` field. Enrollment issue, redeem (success and failure), and unlink are
audited too.

## 7. Privacy and data rules

- **No real patient data on the patient side until a Business Associate Agreement with the AI provider exists.**
  Until then only synthetic or demo patients are enrollable. They are marked `HTEST`, and the enrollment API refuses
  any patient without that mark.
- **CGMacros:** CGMacros data is not sent to any AI provider (its CC BY-NC-SA licence has not been checked for this;
  docs/AI_CONNECTORS.md). Demo patients use synthetic series.
- **Consent:** the Auth0 consent text says alerts are sent to the AI assistant's provider when you ask.
- **Retention:** alert retention in `glucose_alerts` needs a policy decision (§10). The audit is kept per
  HIPAA_MAPPING.md.

## 8. Rate limits

| Surface | Limit |
|---|---|
| `get_my_recent_alerts` | 30 calls per minute per user (the existing public-MCP limiter, shared store before production) |
| Enrollment redeem | 5 per account per hour, 20 per IP per hour, 1 h lock after 5 failures |
| Facade alert search (`category=cgm-alert`) | `_count` ≤ 100, window ≤ 30 days |

## 9. Test plan (written before the code)

The tests in §5, plus:

- **Contract:** the alert `Observation` validates against R4 and round-trips like readings; a search without
  `category` still returns readings only (no change for existing clients).
- **Fail closed:** no auth config means 401/503, never open.
- **Unlink:** after unlink plus a token refresh, the tool returns the enroll message and no data.
- **Minimal response:** a field allow-list test on the patient-side response.
- **Clinician tool:** a finding without an alert `Observation` citation is dropped by the existing check.

## 10. Decisions needed before building

1. ~~FHIR resource~~ **Decided: `Observation`** with category `cgm-alert` and an alerts-only granular scope (§3.1).
2. **Public MCP → facade:** On-Behalf-Of token exchange (recommended, §3.3). Before relying on it, verify on this
   tenant:
   - (a) the feature is available on the tenant's plan;
   - (b) the post-login Action's `patient` claim (or an equivalent token-exchange Action) is present in the exchanged
     token;
   - (c) the facade runs in JWKS mode with Auth0 as `GLUCOPULSE_ISSUER`.

   Otherwise use the service-token fallback and accept the residual risk in §5.
3. **Who may issue enrollment codes:** a care-team role on the facade's issuer. A demo-admin script is the only
   option until there is a real care team.
4. **Alert retention period** in `glucose_alerts`.
5. **Order:** build §3.1, then §3.2 (internal, lower risk), then §3.3–3.4.

## 11. Alternative tested: Supabase row-level security (spike, run 2026-11-24)

Requested 2026-11-24. Supabase is Postgres plus its own login (Supabase Auth). With **row-level security (RLS)** the
database itself returns only the rows a policy allows for the logged-in user. That would enforce ownership (§3.3–3.4)
below the API, so even a buggy endpoint could not read another patient's alerts.

The spike is in `infra/supabase_spike/spike.sql`. It uses synthetic patients `demo-a` and `demo-b` only, in a
throwaway free project.

- **Read rule:** one policy, `patient_id = the patient linked to auth.uid()`. The patient id comes from the verified
  login, never from the query.
- **Enrollment:** the `redeem_enrollment_code(code)` function. Codes are single use, expiring and hashed, and every
  failure gets the same error message. API roles have no direct write access to any table.

Checks to run (each as an API user, role `authenticated` with that user's claims):

| # | Check | Expected |
|---|---|---|
| 1 | User C (not enrolled) reads alerts | 0 rows |
| 2 | Anonymous reads alerts | permission denied |
| 3 | User A redeems `DEMO-A-CODE`, reads alerts | `demo-a`'s 2 rows only |
| 4 | User A filters `where patient_id = 'demo-b'` | 0 rows: the filter cannot widen the policy |
| 5 | User B redeems `DEMO-A-CODE` (already used) | "invalid or expired code" |
| 6 | User B redeems `EXPIRED-CODE` | the same error |
| 7 | User A inserts an alert or a link, or updates a link to `demo-b` | denied |
| 8 | User A reads `enrollment_codes` | permission denied |
| 9 | Supabase security advisor | no RLS or function warnings |

How it compares with the facade design:

| | FHIR facade + Auth0 (§3) | Supabase RLS |
|---|---|---|
| Ownership enforced | in the API (`Principal.can`) | in the database (policy), plus whatever API sits on top |
| Login | Auth0 (already used by the MCP server) | Supabase Auth, a second identity system, unless Auth0 is configured as a third-party provider |
| FHIR, audit chain, SMART scopes | already built and tested | not provided; would be rebuilt or kept in front |
| Data location | our TimescaleDB | a hosted database: a new place where health data lives (BAA needed on a paid plan for real data) |

### Results (2026-11-24)

Run in schema `glucopulse` of the existing free project (migration `glucopulse_alerts_rls_spike`), synthetic data only.

| # | Result |
|---|---|
| 1 | User C, not enrolled: **0 rows** |
| 2 | Anonymous: **permission denied for schema glucopulse** |
| 3 | User A redeemed `DEMO-A-CODE`: **exactly demo-a's 2 alerts** |
| 4 | User A, `where patient_id = 'demo-b'`: **0 rows**; `where true or …`: still only A's 2 rows; sees only their own link |
| 5 | User B, used code: **"invalid or expired code"** |
| 6 | User B, expired code: **the same message**, so a used code and an expired code look alike |
| 7 | User A inserting an alert, switching their link to demo-b, or user C linking themselves: **all permission denied** |
| 8 | Reading `enrollment_codes`: **permission denied** |
| 9 | Security advisor, `glucopulse` schema: one INFO only (`enrollment_codes` has RLS and no policy, which is intended) |

All 9 behaved as designed.

**Limits of this spike:**

- **No real HTTP or real login.** Users were simulated in SQL: role `authenticated` plus `request.jwt.claims`, as
  Supabase sets them. The schema is not exposed through Supabase's REST API, so no real HTTP request and no real
  Supabase Auth login were tested.
- **No guessing limits.** Redeem attempts are not rate-limited yet.
- **No audit.** There is no audit trail.

Expected outcome: RLS is a strong extra layer for ownership. It does not replace the facade's FHIR, audit and scope
handling. Run the spike, then decide.
