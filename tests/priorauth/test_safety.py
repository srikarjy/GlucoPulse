import ast
import pathlib

import pytest

from priorauth.agents.orchestrator import Orchestrator
from priorauth.safety import (
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    calibrate_confidence,
    route,
)
from priorauth.schemas import (
    CriterionResult,
    DraftRecommendation,
    Recommendation,
    ReviewTier,
)
from tests.priorauth.conftest import by_profile

ROOT = pathlib.Path(__file__).resolve().parents[2] / "priorauth"


def _draft(rec, conf, flags=()):
    return DraftRecommendation(request_id="r", policy_id="p", recommendation=rec, confidence=conf, rationale="",
                               criteria_summary=[], flags=list(flags))


def _code_references(path, name):
    """True if `name` appears as an identifier/attribute/import in real code (docstrings and comments ignored)."""
    for n in ast.walk(ast.parse(path.read_text())):
        if (isinstance(n, ast.Name) and n.id == name) or (isinstance(n, ast.Attribute) and n.attr == name):
            return True
        if isinstance(n, ast.ImportFrom) and any(a.name == name for a in n.names):
            return True
    return False


def test_agent_code_never_references_final_decision():
    """Structural guard: FinalDecision may only be used by the human path (api/store/schemas)."""
    allowed = {"api.py", "store.py", "schemas.py"}
    offenders = [str(f) for f in ROOT.rglob("*.py")
                 if f.name not in allowed and "eval" not in f.parts and _code_references(f, "FinalDecision")]
    assert not offenders, offenders


def test_no_agent_module_calls_record_decision():
    for f in (ROOT / "agents").rglob("*.py"):
        assert not _code_references(f, "record_decision")
    assert not _code_references(ROOT / "mcp_server" / "tools.py", "record_decision")


@pytest.mark.parametrize("rec,conf,flags,tier", [
    (Recommendation.SUPPORTS_APPROVAL, 0.97, [], ReviewTier.STANDARD),
    (Recommendation.SUPPORTS_APPROVAL, 0.80, [], ReviewTier.PRIORITY),
    (Recommendation.SUPPORTS_APPROVAL, 0.50, [], ReviewTier.PHYSICIAN),
    (Recommendation.DOES_NOT_SUPPORT, 0.99, [], ReviewTier.PHYSICIAN),          # denial-leaning is always MD
    (Recommendation.SUPPORTS_APPROVAL, 0.97, ["prompt_injection"], ReviewTier.PHYSICIAN),
    (Recommendation.SUPPORTS_APPROVAL, 0.97, ["stale_evidence"], ReviewTier.PRIORITY),
])
def test_routing_matrix(rec, conf, flags, tier):
    assert route(_draft(rec, conf, flags)).tier == tier


def test_highest_confidence_still_requires_a_human():
    r = route(_draft(Recommendation.SUPPORTS_APPROVAL, 1.0))
    assert "agent_draft_always_requires_human" in r.reasons


def test_expedited_gets_top_priority():
    assert route(_draft(Recommendation.SUPPORTS_APPROVAL, 0.9), "expedited").priority == 1


def test_thresholds_are_sane():
    assert 0 < LOW_CONFIDENCE < HIGH_CONFIDENCE < 1


def test_confidence_never_reaches_one_and_injection_caps_it():
    met = [CriterionResult(criterion_id=str(i), met=True, evidence=["e"]) for i in range(5)]
    assert calibrate_confidence(Recommendation.SUPPORTS_APPROVAL, met, []) <= 0.97
    assert calibrate_confidence(Recommendation.SUPPORTS_APPROVAL, met, ["prompt_injection"]) <= 0.5


def test_every_scenario_ends_in_an_open_review_task(box, submit, scenarios):
    orch = Orchestrator(box)
    for s in scenarios[:60]:
        rid = submit(s)
        orch.run(rid)
        t = box.store.get_task(f"TASK-{rid}")
        assert t and t.state == "open"
        assert box.store.get_request(rid)[1] == "awaiting_review"      # never 'decided'
        assert box.store.get_draft(rid).status == "DRAFT"


def test_pipeline_fails_closed_to_physician_review(box, submit):
    class Boom:
        name = "boom"
        def run(self, state): raise RuntimeError("model outage")
    orch = Orchestrator(box)
    orch.agents[2] = Boom()
    rid = submit(by_profile("meets")[0])
    orch.run(rid)
    t = box.store.get_task(f"TASK-{rid}")
    assert t.tier == ReviewTier.PHYSICIAN and "flag:agent_failure" in t.reasons
    assert t.draft.recommendation == Recommendation.INSUFFICIENT_INFO


def test_rerunning_pipeline_is_idempotent(box, submit):
    rid = submit(by_profile("fail")[0])
    orch = Orchestrator(box)
    orch.run(rid); orch.run(rid)
    assert len(box.store.list_tasks("open")) == 1
