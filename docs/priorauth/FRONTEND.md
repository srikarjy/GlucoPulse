# Frontend: review workbench and playground

`priorauth-ui/` is a Next.js 16 (App Router, React 19, TypeScript, Tailwind 4) app with Cache Components enabled. It has two jobs:

* **Reviewer workbench** (`/queue`, `/review/[taskId]`, `/audit`): the human side of the system. Queue by tier and priority, the AI draft with cited evidence
  and flags, claim, decide, audit trail with hash-chain verification.
* **Playground** (`/playground`): pick a synthetic scenario, optionally edit it (or add a prompt-injection attempt), run it through the real five-agent
  pipeline and watch the stages complete, then see the draft, the server-chosen routing, and a comparison with the synthetic ground truth.

## Run it locally (no Docker)

```bash
# terminal 1: API + worker on SQLite, synthetic data, prints one paste-ready token per role
source .venv-pa/bin/activate && python -m priorauth.scripts.demo_server          # http://127.0.0.1:8080

# terminal 2
cd priorauth-ui && npm ci && npm run dev                                         # http://localhost:3000
```

Open `/login` and paste a token. Roles: `playground` (submit + read drafts, cannot decide), `submitter` (submit, status only), `reviewer-rn`,
`reviewer-md` (physician: may deny and take physician-tier tasks), `auditor`. Point at another API with `PRIORAUTH_API_URL`.
For a real LLM backend use the compose stack (`PRIORAUTH_BACKEND=bedrock`); the demo server runs the deterministic agents.

## Design decisions

* **The JWT never reaches page scripts.** Login posts the token to `/api/session`, which verifies it against the API and stores it in an
  `httpOnly`, `SameSite=Strict` cookie. The browser only talks to `/api/pa/*`, a server-side pass-through that attaches the bearer token, forwards only
  `Idempotency-Key`, allow-lists API areas (`me|requests|reviews|audit|demo`), rejects `..` path tricks, caps body size, and refuses state-changing requests
  whose `Origin` is not this app. Strict CSP (`frame-ancestors 'none'`, no third-party origins), `no-store`, `nosniff`.
* **Pages are thin client shells** that fetch through the proxy; only route handlers read the cookie. That keeps Cache Components happy
  (reading `cookies()` in a server component would otherwise have to stream behind Suspense) and means nothing authenticated is ever cached.
* **The UI cannot present a draft as a decision.** Every draft sits under an "AI-generated draft, advisory only" banner and badges read "Draft: …".
  Synthetic ground truth uses a different label ("Expected label: …") so an expected answer can never be mistaken for a draft.
* **Anti-anchoring in the decision form.** No option is preselected; the draft is shown separately; overriding the draft's implied outcome, or denying,
  requires a written reason (10+ characters); a confirm step records the final decision under the reviewer's identity. Denial and physician-tier claims are
  disabled with an explanation for non-physicians, and the server enforces the same rules independently.
* **Server-side facts are shown as such.** Routing tier and reasons are labelled "chosen by server policy, not by the model". Confidence is shown as a
  heuristic score with a tooltip, never as a probability.
* **Failure states are visible.** A request that exhausted its retries shows a clear "needs manual handling" message instead of a spinner.

## Backend additions this UI needed

`GET /v1/me`, `GET /v1/requests/{id}/timeline` (sanitised audit view: who did what and when, no clinical content; backed by a new indexed `audit_log.subject` column, which also fixed
a latent bug where per-request audit queries only searched the newest 200 rows), `GET /v1/reviews/{task_id}`, a demo-only `GET /v1/demo/scenarios`
(mounted only when `PRIORAUTH_DEMO=1`), and a `playground` role that can submit and read drafts but can never claim or decide.

## Tests

`npm run typecheck`, `npm run lint`, `npx playwright test` (15 tests against the real stack: demo API + worker + production build). They cover the full human workflow
(RN blocked from physician tasks, denial needs a reason, final decision recorded and audited, chain verified), injection handling in the UI, the proxy's security
properties (httpOnly cookie, origin check, allow-list, headers, expired token), and axe WCAG 2.x A/AA scans of the queue, playground and audit pages.

## Limits

* Login is paste-a-token: there is no IdP, MFA or session revocation. A token stays valid until it expires (12 h cookie, token's own `exp` governs the API).
* Automated axe scans find only a fraction of accessibility problems. No manual screen-reader, keyboard-only or colour-blindness review has been done.
* Polling (1-5 s) instead of push; fine for this scale, not for hundreds of concurrent reviewers.
* Only the deterministic agents were exercised through the UI. The Bedrock backend works through the same API but was not driven from the browser.
* Desktop layout reviewed on screenshots; not tested on mobile.
* No deployment configuration yet (the app is Vercel-compatible; the API it points at must be reachable from it and its network allow-list must permit that).
