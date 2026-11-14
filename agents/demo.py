"""
Runs the demo script from docs/CLINICAL_AGENT.md live, against the real
facade/MCP server/ToolBox (no live Bedrock -- ScriptedModel stands in for
the LLM, same honesty limit as everywhere else in this project), and
prints each beat so it can be read or screen-recorded directly.

Run (needs .venv-pa -- strands/mcp aren't in the main .venv):
    .venv-pa/bin/python -m agents.demo
"""
from __future__ import annotations

import json

from agents.citation import ground_finding
from agents.eval.run_eval import ScriptedModel, build_environment, build_scenarios
from agents.mcp_server.auth import Principal
from agents.mcp_server.tools import LocalGateway
from agents.reasoning_agent import ask
from agents.schemas import ClinicalFinding
from serving.healthcare.audit import verify_chain


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    gw, mcp_url, ids, cleanup = build_environment()
    try:
        scenarios = build_scenarios(gw, ids)
        by_kind = {}
        for s in scenarios:
            by_kind.setdefault(s.kind, []).append(s)

        _section("1-4. Grounded patient, ask the demo's own question, show a dropped sentence")
        hallu = by_kind["hallucination"][0]
        print(f"Patient: {hallu.patient_id}")
        print("Question: \"Summarize glycemic control over the last 30 days.\"")
        raw_json = hallu.turns[1][1]
        print(f"\nAgent's RAW drafted JSON (before grounding):\n  {raw_json}")
        finding, dropped = ask(mcp_url, hallu.question, model_factory=lambda: ScriptedModel(hallu.turns))
        print(f"\nAfter ground_finding():")
        print(f"  finding   = {finding.finding!r}")
        print(f"  evidence  = {[e.claim for e in finding.evidence]}")
        print(f"  dropped   = {dropped}")
        print(f"  red_flags = {finding.red_flags}")
        print("\n-> the uncited sentence was visibly removed before any human saw it.")

        _section("5. No-data patient: abstain, don't guess")
        nodata = by_kind["no_data"][0]
        print(f"Patient: {nodata.patient_id}  (zero CGM Observations on file)")
        nodata_finding, _ = ask(mcp_url, nodata.question, model_factory=lambda: ScriptedModel(nodata.turns))
        print(f"  finding      = {nodata_finding.finding!r}")
        print(f"  missing_data = {nodata_finding.missing_data}")
        print("\n-> correct abstention: no finding asserted, the gap is named explicitly.")

        _section("6. Injected note: quarantined before the agent ever sees it")
        inject = by_kind["injection"][0]
        notes = gw.invoke("fhir_search_notes", patient=inject.patient_id, count=100)
        for n in notes["notes"]:
            shown = n["text"] if not n["quarantined"] else "<empty -- quarantined>"
            print(f"  note {n['note_id']}: quarantined={n['quarantined']!s:<5} text={shown!r}")
        print("\n-> the agent's tool call for this patient never receives the injected text at all.")

        _section("7. Human reviewer approves the grounded finding")
        reviewer_gw = LocalGateway(gw.box, Principal(subject="dr-smith", role="reviewer",
                                                      scopes=frozenset({"audit:read", "review:decide"})))
        decision = reviewer_gw.invoke(
            "submit_review_decision", patient=hallu.patient_id, reviewer="dr-smith", decision="approve",
            finding=finding.model_dump(), comment="Grounded reading confirmed against chart; approved.",
            idempotency_key="demo-review-001")
        print(f"Reviewer dr-smith -> decision={decision['detail']['decision']!r} "
              f"for patient={decision['patient']!r}")

        _section("8. Audit log: one continuous hash chain")
        rows = gw.box.audit.read(50, None)
        for r in rows[-6:]:
            print(f"  [{r['action']:<20}] patient={r.get('patient')!s:<28} "
                  f"outcome={r['outcome']:<8} hash={r['entry_hash'][:12]}...")
        ok, bad_line = verify_chain(gw.box.audit.path)
        print(f"\nverify_chain() -> ok={ok}, first_bad_line={bad_line}")
        print("-> every tool call, the draft, and the review decision are links in the SAME chain.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
