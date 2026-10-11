# Security and privacy

**Wording rule:** the platform *implements technical safeguards* relevant to HIPAA's Security Rule. It is not, and must not be described as, "HIPAA compliant":
compliance is organisational (policies, BAAs, risk analysis, training) and no PHI has been processed here.

| Control | Implementation | Test |
|---|---|---|
| Authentication | JWT bearer, audience-bound (`priorauth-mcp` vs `priorauth-api`), expiry required; a token for one audience is rejected by the other | `test_tokens_audience_bound…`, `test_agent_token_rejected_by_api` |
| Authorisation | Scopes derived server-side from the role; a forged `scope` claim is ignored. Tool scopes checked on every call | `test_scope_enforcement…` |
| Least privilege | 5 agent roles, each with the minimum tools; physicians only for denial/physician-tier | `test_role_tool_lists_are_least_privilege` |
| Idempotency | Mutating tools demand a key; replay returns the stored result; key reuse with other args is an error | `test_idempotent_replay…`, `test_same_key_different_args…` |
| Audit | Hash-chained, append-only log; every tool call/denial/replay/error and every human action. Tamper detection via `/v1/audit/verify`. Postgres advisory lock prevents chain forks | `test_audit_requires_role_and_chain_verifies_and_detects_tampering`, `test_concurrent_audit_appends…` |
| Minimum necessary | Audit entries and span attributes carry ids/codes/flags only (test asserts no note text appears). Evidence is cited by reference | `test_audit_never_contains_clinical_text` |
| Trace content | Prompts/completions are **off** in traces unless `PRIORAUTH_TRACE_CONTENT=1`; enable only against a BAA-covered backend | – |
| FHIR | Read-only client; never writes to the EHR | `test_http_source_is_read_only` |
| Secrets | Env/Secrets Manager (KMS); fail-fast if `PRIORAUTH_JWT_SECRET` is missing/short; nothing in git | compose refuses to start without it |
| Container | non-root (uid 10001), read-only rootfs, caps dropped, no-new-privileges, pinned deps, Trivy in CI | run verified locally |
| Network (AWS) | private subnets, SG-to-SG rules only, MCP reachable from workers only, ALB HTTPS-only with TLS1.3 policy + CIDR allow-list, `/metrics` blocked at the ALB; K8s default-deny NetworkPolicies | static validation only |
| Data at rest (AWS) | RDS + secrets + logs encrypted with a rotating KMS CMK; RDS `force_ssl` | static validation only |

## Known gaps

No SMART-on-FHIR/OAuth handshake (a static token is used for FHIR); HS256 shared secret; no rate limiting or WAF on the API; no emergency-access ("break glass") flow;
no token revocation list; the reviewer identity is a token subject, not tied to an IdP/MFA; no data-retention/purge job; audit chain head is not anchored to an external store.
