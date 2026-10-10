# HIPAA Security Rule mapping

**What this is:** a map from the HIPAA Security Rule technical safeguards (45 CFR 164.312) to what the FHIR facade actually implements, and what it does not.

**What this is not:** a compliance claim. HIPAA applies to covered entities and business associates handling PHI. The dataset here (AZT1D) is public, de-identified research data, so there is no PHI to protect today. Compliance is an organizational status (risk analysis, policies, workforce training, BAAs); code supplies only some of the technical controls. The honest label is *"implements the technical safeguards a HIPAA-regulated deployment would need, with the gaps listed below."*

Every "Implemented" row has a test in `tests/healthcare/` or a live check noted in the PR; every gap is listed, not omitted.

## Technical safeguards (164.312)

| Requirement | Status | Where / how |
|---|---|---|
| (a)(1) Access control | Implemented | SMART scopes per resource (`patient/`, `user/`, `system/`); `patient/` scopes confined to the token's patient. Fail-closed without auth config. `serving/healthcare/security.py` |
| (a)(2)(i) Unique user identification | Implemented | `sub` + `azp` from the verified token recorded on every request |
| (a)(2)(ii) Emergency access procedure | **Not implemented** | No break-glass flow. Needs a policy decision first (who may invoke it, post-hoc review) |
| (a)(2)(iii) Automatic logoff | Implemented | Tokens with lifetime > `GLUCOPULSE_MAX_TOKEN_TTL` (900s default) rejected |
| (a)(2)(iv) Encryption/decryption at rest | **Partial** | Not done by the app. Needs encrypted volume / managed-DB encryption at the deployment layer. Not configured here |
| (b) Audit controls | Implemented | Every call (success, denied, error) logged: who, resource type, pseudonymous patient, outcome, purpose of use, request id, IP. No bodies or values logged. `audit.py`; exposed as FHIR `AuditEvent` |
| (c)(1) Integrity | Implemented (detective) | Hash-chained audit log + DB triggers blocking UPDATE/DELETE/TRUNCATE on `audit_log`. Tamper-**evident**, not tamper-proof (see gaps) |
| (c)(2) Mechanism to authenticate ePHI | **Not implemented** | No signing of resources (no FHIR `Provenance.signature`) |
| (d) Person or entity authentication | Implemented | JWT verified: signature, `exp`, `iat`, `iss`, `aud`; algorithms pinned (RS256/ES256 via JWKS, HS256 dev only); `alg: none` rejected |
| (e)(1) Transmission security | Implemented (deployment) | Caddy TLS terminator (`docker compose --profile tools up`), HSTS; `Cache-Control: no-store` on all `/fhir` responses. Local CA only; use a real cert beyond localhost |
| (e)(2)(ii) Encryption in transit | Implemented (deployment) | Same; internal hops (serving ↔ DB) are plaintext on the Docker network |

## Other rules touched

| Rule | Status | Notes |
|---|---|---|
| Minimum necessary (164.502(b)) | Implemented | FHIR DB role `glucopulse_fhir`: `SELECT` on readings, `INSERT` only on audit. Verified: it can't write readings or read the audit table. No bulk dump endpoint; `patient` search param required |
| Accounting of disclosures (164.528) | Supported | `GET /fhir/AuditEvent?patient=` (patients can read their own) |
| De-identification (164.514) | **Not claimed** | Ids are HMAC-pseudonymized; inbound strings are screened for direct identifiers. Forecasting needs full timestamps, so the data is not Safe Harbor. See `serving/healthcare/deid.py` |
| Breach notification, BAAs, risk analysis, workforce training, contingency plan | **Out of scope** | Organizational, not code |

## Known gaps worth knowing before anyone puts PHI near this

1. **Audit chain head is local.** Someone with write access to the whole file or DB (and the ability to drop the trigger) can rewrite history and recompute hashes. Fix: periodically ship the head hash to an external append-only store.
2. **HS256 mode** shares a symmetric secret. Use `jwks` mode against a real authorization server for anything non-local.
3. **No authorization server is bundled.** The facade is a SMART *resource server*; it validates tokens but does not issue them or run the launch flow. `/fhir/.well-known/smart-configuration` advertises the endpoints you configure.
4. **No encryption at rest, no emergency access, no resource signing** (rows above).
5. **Audit logging is synchronous and best-effort to Postgres**: the JSONL file (fsync'd) is authoritative; a Postgres outage is logged and the request still proceeds.
6. **Not validated against the HL7 CGM IG** profiles — only base R4 via HAPI's validator.
7. **The model is a research forecaster, not a medical device**, and every forecast Observation says so.
