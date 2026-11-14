# Security

**Wording rule, same as everywhere else in this repo:** this project *implements technical safeguards*. It is not, and must not be described as, "HIPAA compliant" — compliance is organizational (policies, BAAs, risk analysis, workforce training), and no PHI has ever been processed here. All patient data across every component is synthetic (AZT1D is public, de-identified research data; the Clinical Reasoning Agent's eval patients are hand-built fixtures).

This file is a consolidated index. The detailed, per-component writeups are the source of truth:

- [`docs/HIPAA_MAPPING.md`](docs/HIPAA_MAPPING.md) — the FHIR facade's Security Rule (164.312) mapping, control by control
- [`docs/FHIR.md`](docs/FHIR.md) — resource mapping, endpoints, the EHR round trip
- [`docs/CLINICAL_AGENT.md`](docs/CLINICAL_AGENT.md) — the agent's reuse map, eval numbers, and verified-vs-not table
- [`docs/priorauth/SECURITY.md`](docs/priorauth/SECURITY.md) — the separate prior-authorization platform's own controls and gaps

## Controls, by surface

| Surface | Authentication | Authorization | Audit | Notes |
|---|---|---|---|---|
| `/fhir/*` facade | SMART-scoped bearer JWT, fail-closed (no "disabled" switch) | Per-resource scopes (`patient/`, `user/`, `system/`); `patient/` confined to the token's own patient | Every call — success, denial, error — hash-chained, append-only | `serving/healthcare/security.py`, `audit.py` |
| `/predict` | Enforced when a deployment configures auth (same primitives as the facade); open by design when it's the standalone public demo image | — (stateless, caller supplies all data) | Logged to the same hash-chained audit log when auth is configured | Also: real `model_version` (artifact hash, not a hardcoded label), staleness and quantile-sanity checks before returning |
| Clinical Reasoning Agent MCP server | HS256 JWT, own namespace/secret (can't replay against `priorauth`'s MCP server) | Role-scoped tools; `submit_review_decision` granted only to `reviewer`, never `clinical-agent`, enforced server-side | Every tool call, draft, and review decision in the same chain as the facade's | `agents/mcp_server/auth.py` |
| `priorauth/` | See its own [SECURITY.md](docs/priorauth/SECURITY.md) | — | — | Separate platform, separate secrets, separate `.venv-pa` |

## Specific mitigations worth naming

- **Prompt injection:** any note/document text is quarantined *inside the tool* before the agent ever sees it — `agents/quarantine.py` (ported from `priorauth/criteria.py::find_injections`) and `priorauth`'s own equivalent. A quarantined note returns empty text, not optionally-filtered text.
- **Hallucinated citations:** every sentence in a Clinical Reasoning Agent finding needs a matching, textual evidence citation or it is dropped before a human sees it (`agents/citation.py::ground_finding`). Known gap: this checks the citation is textually present, not that the cited FHIR resource id was actually retrieved this session — measured at 87.5% in the eval harness, not 100%; see `docs/CLINICAL_AGENT.md`.
- **Forecast-as-measurement confusion:** model forecasts written back to an EHR carry a distinct category/tag from real measurements, and a previously-written forecast is filtered out before being sent back to the model as if it were new data (`ehr/sync.py`) — this was a real, found-and-fixed bug, not a preventive measure added speculatively.
- **Known-CVE dependencies:** `priorauth-ui`'s Next.js is kept off versions with the unauthenticated RCE in Image Optimization (GHSA-2464-fhr4-w9xw) and related advisories.
- **Secrets:** nothing is hardcoded; every component fails closed (refuses to start, or refuses to authenticate) without its required secret set via environment. `.env` files are gitignored; `.env.example` documents required variables with empty/fail-closed defaults.

## Known gaps

1. **No bundled authorization server.** The FHIR facade and the Clinical Reasoning Agent's MCP server are resource servers; they validate tokens, they don't issue them. HS256 shared-secret mode is for local dev/test only — use `jwks` mode against a real issuer for anything beyond that.
2. **No encryption at rest, no emergency-access ("break glass") flow, no resource signing** — all deployment-layer or policy decisions, not addressed in code. See `docs/HIPAA_MAPPING.md` for the full list on the FHIR facade specifically.
3. **Citation grounding doesn't verify resource-id provenance** (above) — a model could in principle cite a plausible-looking id it never actually retrieved, and the current check wouldn't catch it. Documented, not silently hidden.
4. **Audit chain head is local**, not anchored to an external append-only store — someone with full write access to the log file/DB could, in principle, rewrite history and recompute hashes.
5. **No rate limiting or WAF** on any HTTP surface in this repo.

## Reporting a vulnerability

This is a portfolio/research project with no production deployment handling real patient data. If you find a real issue, open a GitHub issue on this repo, or reach the maintainer via the contact on the GitHub profile. Please don't open a public issue for anything that would be meaningfully exploitable against a real deployment of this code (there currently isn't one) — email first in that case.
