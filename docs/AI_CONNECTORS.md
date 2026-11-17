# Using GlucoPulse from ChatGPT and Claude

GlucoPulse exposes a small, read-only **MCP server** (`agents/public_mcp/`) that AI assistants can call:

| Tool | What it does |
|---|---|
| `get_glucose_alert(readings, unit, mode)` | Checks recent CGM readings (sent by the caller) for a coming low (< 70) or high (> 250) with the tested two-tier trend alert. Returns none / `heads_up` / `loud`, the direction, a plain-language reason, the 20-min projection, the validated population and a disclaimer. |
| `about_glucopulse_alerts()` | The tested accuracy and limits, so an assistant quotes real numbers (docs/TRIAGE.md §9, §15-19). |

Nothing is stored, no stored patient data can be read, and no tool changes anything. Forecasts and triage levels from
the research are **not** exposed (they are not validated for this use).

## 1. Local: Claude Code (no accounts needed)

```bash
export GLUCOPULSE_MCP_DEV_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
.venv-pa/bin/python -m agents.public_mcp.server            # http://127.0.0.1:8770/mcp, dev tokens, localhost only
TOKEN=$(.venv-pa/bin/python -m agents.public_mcp.auth --dev-token)    # valid 1 h
claude mcp add --transport http glucopulse http://127.0.0.1:8770/mcp --header "Authorization: Bearer $TOKEN"
```

Then ask Claude, for example: *"Here are my last hour of CGM readings (mmol/L) … am I heading low?"*

Dev mode refuses to bind anywhere but localhost.

The dev token expires after 1 h, but `claude mcp add` stores the header as written. For longer sessions, configure a
`headersHelper` that mints a fresh token instead (Claude Code docs, "dynamic headers"), or re-run the two lines above.

## 2. Hosted: ChatGPT and claude.ai (needs an identity provider and HTTPS)

Both connect only to **remote HTTPS** MCP servers and sign users in with **OAuth 2.1 + PKCE**. GlucoPulse is the
*resource server*; an identity provider (IdP) such as Auth0 issues the tokens. As of 2026-11:
- ChatGPT supports custom MCP servers on Business, Enterprise and Edu workspaces, or through OpenAI's plugin flow.
- ChatGPT needs the IdP to advertise `S256` PKCE, and either client ID metadata documents or dynamic client
  registration.
- Enable `offline_access` so ChatGPT can refresh tokens.
- claude.ai custom connectors use the same OAuth discovery.

Steps:
1. **IdP:** create an API whose identifier is your public MCP URL (e.g. `https://mcp.example.org/mcp`).
   - Add the permission `glucose:alerts`.
   - Enable dynamic client registration (or client metadata documents) and refresh tokens.
2. **Host** the server behind HTTPS and set:
   ```bash
   GLUCOPULSE_MCP_AUTH_MODE=jwks
   GLUCOPULSE_MCP_RESOURCE=https://mcp.example.org/mcp
   GLUCOPULSE_MCP_ISSUER=https://YOUR-TENANT.auth0.com/
   GLUCOPULSE_MCP_JWKS_URL=https://YOUR-TENANT.auth0.com/.well-known/jwks.json
   GLUCOPULSE_MCP_HOST=0.0.0.0
   ```
3. **Connect:**
   - ChatGPT: Settings → Apps/Connectors → add a custom MCP server with the URL.
   - claude.ai: Settings → Connectors → Add custom connector with the URL.

   Both read `/.well-known/oauth-protected-resource/mcp`, send the user to your IdP, and call the tools with the
   returned token.

### Hosting on Vercel (chosen host)

`api/index.py` is the Vercel function. Each request builds a fresh stateless server, because serverless platforms may
not run ASGI startup events. `vercel.json` routes every path to it and sets HSTS, nosniff and no-store.

**Deploy only from a staged copy.** `data/` in this repository holds participant data, and `vercel` uploads the folder
it runs in:
```bash
scripts/vercel_stage.sh /tmp/glucopulse-mcp-deploy   # copies 12 allowlisted files; refuses data-like files
cd /tmp/glucopulse-mcp-deploy && vercel deploy --prod
```
`.vercelignore` is a second, allowlist-style guard.

**Project environment variables** (Production; never commit them):

| Variable | Value |
|---|---|
| `GLUCOPULSE_MCP_AUTH_MODE` | `jwks` |
| `GLUCOPULSE_MCP_RESOURCE` | `https://<project>.vercel.app/mcp`: the stable production domain, which must also be the IdP API identifier |
| `GLUCOPULSE_MCP_ISSUER` | the IdP issuer exactly, e.g. `https://<tenant>.us.auth0.com/` |
| `GLUCOPULSE_MCP_JWKS_URL` | e.g. `https://<tenant>.us.auth0.com/.well-known/jwks.json` |
| `GLUCOPULSE_MCP_AUDIT_KEY` | 32+ random characters (keyed hash of user ids in logs) |
| `GLUCOPULSE_MCP_AUDIT_LOG` | `stdout` (Vercel's log stream; there is no writable disk) |

**Auth0 as the IdP.** Check the current setting names in Auth0's docs when doing this; they change.
1. Create an API whose identifier is the `GLUCOPULSE_MCP_RESOURCE` URL, signed with RS256.
   - Add the permission `glucose:alerts`.
   - Enable RBAC and "add permissions in the access token".
   - Allow offline access, so refresh tokens work for ChatGPT.
2. Tenant settings:
   - Enable dynamic client registration, so ChatGPT and claude.ai can register themselves.
   - Make Auth0 accept the OAuth `resource` parameter as the audience. MCP clients send `resource`, not `audience`.
   - Promote a login connection to domain level, so dynamically registered clients can use it.
3. Give the users who should have access the `glucose:alerts` permission.

## 3. Security and safety controls

| Threat | Control | Test |
|---|---|---|
| Unauthenticated use | Every request needs a bearer token; 401 with `WWW-Authenticate` → protected-resource metadata (RFC 9728) | `test_no_token_401_points_to_resource_metadata` |
| Tokens meant for another service (token passthrough / confused deputy) | Audience must equal this server's resource URL; issuer checked; the internal clinical-agent token is rejected | `test_bad_tokens_rejected`, `test_internal_clinical_agent_token_rejected` |
| Forged or expired tokens | Signature (JWKS RS256/ES256 in production), `exp`/`iat` required, `alg=none` refused, dev tokens <= 12 h | `test_unsigned_and_wrongly_signed_tokens_rejected` |
| Over-broad access | Required scope `glucose:alerts`; only two read-only tools; no stored data, no writes | `test_schema_and_read_only_annotations` |
| DNS rebinding from a browser | Host and Origin allowlists | `test_dns_rebinding_blocked` |
| Abuse / cost | 256 KB request cap, checked before the handler, also for chunked uploads; per-user rate limit (30 calls/min) | `test_oversized_body_rejected`, `test_chunked_oversized_body_cut_off`, `test_rate_limit` |
| Prompt injection through tool output | Inputs are numbers and timestamps only; **error messages never echo input** (SDK validation errors that quote the bad value are replaced with a fixed message) | `test_text_injection_attempt_cannot_reach_output`, `test_injection_never_echoed_from_any_field` |
| Unsafe medical use | Descriptions and every result say: screening only, not for insulin dosing, confirm with a meter; validated population stated; stale data refused unless explicitly `retrospective` (and then labelled not current); units explicit, mmol/L sent as mg/dL caught, answers in the caller's unit | `test_alert_on_a_fall_with_disclaimer`, `test_stale_data_refused_unless_retrospective`, `test_mmol_converted_and_mgdl_mistake_caught`, `test_answer_in_callers_unit` |
| A silent "no alert" that is really "could not check" | Timestamps snapped to the 5-min grid (real exports drift by seconds); fewer than 4 consecutive 5-min readings (e.g. 15-min Libre history) is an error that says it is NOT an all-clear | `test_cgm_export_drift_still_gives_a_trend`, `test_fifteen_minute_data_is_refused_not_all_clear` |
| Health data in logs | Audit log is metadata only (tool, outcome, count, keyed HMAC of the user id; set `GLUCOPULSE_MCP_AUDIT_KEY`); no glucose values or times | `test_audit_is_metadata_only` |
| Forged production tokens | RS256/ES256 only; HS256 "signed" with the public key (algorithm confusion) refused; other keys refused; metadata lists the IdP issuer exactly | `test_jwks_*` (offline, with a generated RSA key) |
| Dev credentials in production | Dev mode only binds to localhost; jwks mode requires an https resource URL | `test_dev_mode_refuses_public_bind` |

All tests: `.venv-pa/bin/python -m pytest tests/agents/test_public_mcp.py` (38 tests, including an end-to-end call
from a real MCP client over HTTP and the production token path with a generated RSA key).

## 4. Not done yet

- **Production auth is tested offline only** (generated RSA key). It has not run against a real IdP.
- **No real ChatGPT or claude.ai connection yet:** it needs a public HTTPS host.
- **Before any real-patient use:** a data-protection review (the audit log is metadata only, but readings pass through
  the assistant provider) and a clinical review of the alert wording.
- **Rate limits are per process,** in memory, which means per function instance on Vercel. Before real traffic, move
  the limiter to a shared store (e.g. Upstash Redis) or rely on Vercel's firewall rate limiting.
- **CGMacros data must not be sent through this server.** Its CC BY-NC-SA licence has not been checked for third-party
  LLM use.
