from priorauth.criteria import INJECTION_RE, evaluate_policy, overall
from priorauth.fhir import facts_to_fhir, parse_chart
from priorauth.synth.policies import BY_ID
from tests.priorauth.conftest import by_profile


def _eval(s):
    pid = s["request"]["patient_ref"]
    chart = parse_chart(pid, facts_to_fhir(pid, s["facts"]))
    return evaluate_policy(BY_ID[s["expected"]["policy_id"]], chart, s["request"]["free_text"]), BY_ID[s["expected"]["policy_id"]]


def test_clean_scenarios_support_approval():
    for s in by_profile("meets"):
        if s["paraphrased_evidence"]:
            continue
        (res, flags), pol = _eval(s)
        assert overall(pol, res) == "supports_approval", s["scenario_id"]


def test_borderline_values_count_as_met():
    for s in by_profile("borderline"):
        if s["paraphrased_evidence"]:
            continue
        (res, _), pol = _eval(s)
        assert overall(pol, res) == "supports_approval", s["scenario_id"]


def test_contradicted_criterion_is_unmet_not_unknown():
    for s in by_profile("fail"):
        (res, _), pol = _eval(s)
        r = {x.criterion_id: x for x in res}[s["expected"]["broken_criterion"]]
        assert r.met is False


def test_absent_evidence_is_undeterminable_not_unmet():
    for s in by_profile("missing"):
        (res, _), pol = _eval(s)
        assert {x.criterion_id: x for x in res}[s["expected"]["broken_criterion"]].met is None


def test_injection_is_quarantined_and_cannot_create_evidence():
    for s in by_profile("injection"):
        (res, flags), pol = _eval(s)
        assert "prompt_injection" in flags
        assert overall(pol, res) == "does_not_support"        # injection did not flip the outcome


def test_conflicting_same_day_labs_are_flagged_not_averaged():
    for s in by_profile("conflicting"):
        (res, flags), _ = _eval(s)
        assert "conflicting_data" in flags


def test_stale_labs_flagged():
    for s in by_profile("stale"):
        (_, flags), _ = _eval(s)
        assert "stale_evidence" in flags


def test_evidence_refs_never_contain_note_text():
    for s in by_profile("meets")[:10]:
        (res, _), _ = _eval(s)
        for r in res:
            for e in r.evidence:
                assert len(e) < 80 and "Assessment" not in e


def test_injection_regex_does_not_fire_on_normal_clinical_text():
    for t in ["Patient to ignore swelling and continue ice.", "Previous MRI normal.", "Plan: follow up in 6 weeks."]:
        assert not INJECTION_RE.search(t)
