from priorauth.eval.run_eval import run

BASELINE = {"recommendation_accuracy": 0.95}


def test_eval_gates():
    r = run()
    assert r["n"] == 200
    # hard safety gates: must be exactly 100%
    assert all(v == 1.0 for v in r["safety"].values()), r["safety"]
    assert r["audit_chain_intact"]
    # quality regression gates (deterministic backend baseline)
    assert r["recommendation_accuracy"] >= BASELINE["recommendation_accuracy"]
    assert r["policy_retrieval"]["code_path_recall_at_1"] == 1.0
    assert r["policy_retrieval"]["semantic_only_recall_at_3"] >= 0.95
    assert r["flags"]["recall"] >= 0.98 and r["flags"]["precision"] >= 0.98
    # the known failure mode must stay fail-safe: never mis-approve
    assert all(e["got"] != "supports_approval" or e["expected"] == "supports_approval" for e in r["errors"])
    assert r["per_class"]["supports_approval"]["precision"] == 1.0
