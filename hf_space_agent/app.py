"""
Clinical Reasoning Agent -- hosted demo for Hugging Face Spaces.

Runs the SAME demo flow as the main repo's agents/demo.py (click-through
synthetic scenarios, not a live chat), served as a simple web page. This
is the real source code (agents/, ehr/, serving/healthcare/, copied in by
the Dockerfile, not re-implemented) running the real FHIR facade, the
real MCP server, and the real deterministic citation/missing-data
grounding and quarantine logic.

HONEST LIMIT, stated here and on the page itself: there is no live LLM.
A ScriptedModel stands in for Bedrock Claude -- this demo proves the
agent loop, FHIR tool wiring, citation/missing-data grounding, quarantine,
and the hash-chained audit trail are REAL and RUNNING, not that a live
model's reasoning is being showcased. Same limit documented throughout
the main repo (docs/CLINICAL_AGENT.md).

All patients and notes are synthetic. Secrets are generated fresh at
process startup, never committed to the image.
"""
from __future__ import annotations

import os
import secrets

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel

from agents.eval.run_eval import ScriptedModel, build_environment, build_scenarios
from agents.mcp_server.auth import Principal
from agents.mcp_server.tools import LocalGateway
from agents.reasoning_agent import ask, live_model_available
from serving.healthcare.audit import verify_chain

app = FastAPI(title="Clinical Reasoning Agent -- Demo", version="1.0.0")

_state: dict = {}


@app.on_event("startup")
def _startup() -> None:
    gw, mcp_url, ids, cleanup = build_environment()
    scenarios = build_scenarios(gw, ids)
    by_kind: dict[str, list] = {}
    for s in scenarios:
        by_kind.setdefault(s.kind, []).append(s)
    _state.update(gw=gw, mcp_url=mcp_url, by_kind=by_kind, cleanup=cleanup)


@app.on_event("shutdown")
def _shutdown() -> None:
    _state["cleanup"]()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Clinical Reasoning Agent -- Demo</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffe69c; padding: .75rem 1rem; border-radius: 6px; margin: 1rem 0; }}
  button {{ margin: .25rem; padding: .5rem 1rem; cursor: pointer; }}
  pre {{ background: #f6f8fa; padding: .75rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }}
  #out {{ margin-top: 1rem; }}
</style></head>
<body>
<h1>Clinical Reasoning Agent -- live demo</h1>
<div class="warn">
  <b>Clinical decision support, never decision making.</b> Synthetic patients only; nothing here has
  been used with PHI. No live LLM is called -- a scripted stand-in replays fixed model outputs so this
  demo proves the agent loop, FHIR tool wiring, citation/missing-data grounding, quarantine, and the
  hash-chained audit trail are REAL and RUNNING, not that a live model's reasoning is being showcased.
  Details: <a href="https://github.com/srikarjy/GlucoPulse/blob/main/docs/CLINICAL_AGENT.md">docs/CLINICAL_AGENT.md</a>.
</div>
<p>Click a step. Each one hits the real FHIR facade / real MCP server / real audit log running in this container.</p>
<button onclick="run('hallucination')">1. Ask a question -- watch citation grounding drop an uncited sentence</button><br>
<button onclick="run('no_data')">2. Ask about a patient with no data -- watch it abstain</button><br>
<button onclick="run('injection')">3. Show a note quarantine (prompt-injection attempt)</button><br>
<button onclick="run('review')">4. Have a human reviewer approve the grounded finding</button><br>
<button onclick="run('audit')">5. Show the hash-chained audit log + verify it</button><br>
<hr>
<p><b>Ask an arbitrary question</b> (needs a live model configured on this Space -- see <code>/api/status</code> for whether one is):</p>
<input id="q" type="text" style="width:70%" value="What was this patient's most recent glucose reading, and should anything be flagged?">
<button onclick="ask()">Ask</button><br>
<div id="out"></div>
<script>
async function run(step) {{
  const out = document.getElementById('out');
  out.innerHTML = '<p>Running...</p>';
  const r = await fetch('/api/' + step);
  const j = await r.json();
  out.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
}}
async function ask() {{
  const out = document.getElementById('out');
  out.innerHTML = '<p>Asking the live model...</p>';
  const r = await fetch('/api/ask', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
                                     body: JSON.stringify({{question: document.getElementById('q').value}})}});
  const j = await r.json();
  out.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
}}
</script>
</body></html>"""


class AskRequest(BaseModel):
    question: str
    patient_id: str | None = None


@app.get("/api/status")
def api_status():
    provider = "OpenAI" if os.getenv("OPENAI_API_KEY") else "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else None
    return JSONResponse({"live_model_available": live_model_available(), "provider": provider,
                         "note": (f"A live {provider} model is configured; /api/ask runs a REAL model "
                                  "against a real tool call." if provider else
                                  "No OPENAI_API_KEY/ANTHROPIC_API_KEY configured on this Space -- /api/ask "
                                  "will return an explanatory error rather than silently faking a response.")})


@app.post("/api/ask")
def api_ask(req: AskRequest):
    if not live_model_available():
        return JSONResponse(
            {"error": "No live model configured on this Space (OPENAI_API_KEY/ANTHROPIC_API_KEY unset). "
                      "The fixed demo buttons above use a scripted model instead -- see docs/CLINICAL_AGENT.md "
                      "for why that distinction is stated honestly rather than hidden."}, status_code=503)
    s = _first("grounded_valid")
    patient_id = req.patient_id or s.patient_id
    try:
        finding, dropped = ask(_state["mcp_url"], f"The patient id is {patient_id}. {req.question}")
    except Exception as exc:  # the live model call can fail for reasons a visitor didn't cause
        # (bad/expired key, rate limit, transient API error) -- never leak a stack trace
        # to a public endpoint; report the failure type without internals.
        return JSONResponse({"error": f"Live model call failed: {type(exc).__name__}: {exc}"},
                            status_code=502)
    return JSONResponse({
        "patient": patient_id, "question": req.question,
        "grounded_finding": finding.model_dump(), "dropped_sentences": dropped,
        "note": "This ran a REAL model call, not a scripted replay.",
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def _first(kind: str):
    s = _state["by_kind"].get(kind)
    if not s:
        raise HTTPException(500, f"no scenario of kind {kind}")
    return s[0]


@app.get("/api/hallucination")
def api_hallucination():
    s = _first("hallucination")
    finding, dropped = ask(_state["mcp_url"], s.question, model_factory=lambda: ScriptedModel(s.turns))
    return JSONResponse({
        "patient": s.patient_id, "question": s.question,
        "raw_drafted_json": s.turns[1][1],
        "after_grounding": finding.model_dump(),
        "dropped_sentences": dropped,
        "explanation": "The uncited second sentence was removed before any human saw it.",
    })


@app.get("/api/no_data")
def api_no_data():
    s = _first("no_data")
    finding, _ = ask(_state["mcp_url"], s.question, model_factory=lambda: ScriptedModel(s.turns))
    return JSONResponse({
        "patient": s.patient_id, "question": s.question,
        "finding": finding.finding, "missing_data": finding.missing_data,
        "explanation": "Zero CGM Observations on file -- correct abstention, not a guess.",
    })


@app.get("/api/injection")
def api_injection():
    s = _first("injection")
    gw: LocalGateway = _state["gw"]
    notes = gw.invoke("fhir_search_notes", patient=s.patient_id, count=100)
    return JSONResponse({
        "patient": s.patient_id, "notes": notes["notes"],
        "explanation": ("The injected note returns empty text and quarantined=true -- the agent's "
                        "tool call never receives the raw instruction-like content at all."),
    })


@app.get("/api/review")
def api_review():
    s = _first("hallucination")
    gw: LocalGateway = _state["gw"]
    finding, _ = ask(_state["mcp_url"], s.question, model_factory=lambda: ScriptedModel(s.turns))
    reviewer_gw = LocalGateway(gw.box, Principal(subject="dr-smith", role="reviewer",
                                                  scopes=frozenset({"audit:read", "review:decide"})))
    decision = reviewer_gw.invoke(
        "submit_review_decision", patient=s.patient_id, reviewer="dr-smith", decision="approve",
        finding=finding.model_dump(), comment="Grounded reading confirmed; approved.",
        idempotency_key=f"demo-review-{secrets.token_hex(4)}")
    return JSONResponse({
        "reviewer": "dr-smith", "patient": s.patient_id, "decision": decision,
        "explanation": "Only a reviewer-scoped principal can call this tool; the agent itself cannot.",
    })


@app.get("/api/audit")
def api_audit():
    gw: LocalGateway = _state["gw"]
    rows = gw.box.audit.read(20, None)
    ok, bad_line = verify_chain(gw.box.audit.path)
    return JSONResponse({
        "recent_entries": rows[-8:], "chain_ok": ok, "first_bad_line": bad_line,
        "explanation": "Every tool call, draft, and review decision is a link in one hash chain.",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
