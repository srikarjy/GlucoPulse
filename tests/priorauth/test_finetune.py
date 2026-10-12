import json
import subprocess
import sys

import pytest

from priorauth.finetune.baseline import keyword_answer, score
from priorauth.finetune.dataset import build, fingerprint
from priorauth.finetune.train_qlora import load_config, prepare_data
from priorauth.synth.scenarios import PARAPHRASES, generate


@pytest.fixture(scope="module")
def data():
    return build()


def test_no_prompt_overlap_between_train_and_any_test_split(data):
    seen = {fingerprint(e) for e in data["train"] + data["val"]}
    for split in ("test", "test_large"):
        assert not [e for e in data[split] if fingerprint(e) in seen]


def test_held_out_wording_never_appears_in_training(data):
    held = {v[0].lower() for v in PARAPHRASES.values()}                      # variant 0 = eval vocabulary
    for e in data["train"] + data["val"]:
        user = e["messages"][1]["content"].lower()
        assert not any(h in user for h in held), "variant-0 paraphrase leaked into training"
    trained_on = {w.lower() for v in PARAPHRASES.values() for w in v[1:]}
    assert all(any(w in e["messages"][1]["content"].lower() for w in trained_on) for e in data["train"] if e["meta"]["paraphrased"])


def test_test_split_is_exactly_the_eval_scenarios(data):
    ids = {e["meta"]["scenario"] for e in data["test"]}
    assert ids <= {s["scenario_id"] for s in generate()}


def test_labels_valid_documentation_never_false_and_injection_blanked(data):
    for split in data.values():
        for e in split:
            a = json.loads(e["messages"][2]["content"])
            assert a["met"] in (True, None) and set(a) == {"met", "evidence", "reasoning"}
            if e["meta"]["injected"]:
                assert "SYSTEM OVERRIDE" not in e["messages"][1]["content"] and "ignore all policy" not in e["messages"][1]["content"].lower()
                assert a["met"] in (True, None)


def test_every_label_is_supported_by_the_prompt(data):
    for e in data["train"][:300]:
        a = json.loads(e["messages"][2]["content"])
        if a["met"]:
            assert a["evidence"][0] in e["messages"][1]["content"]


def test_baseline_fails_on_paraphrase_and_never_hallucinates(data):
    r = score(data["test_large"], keyword_answer)
    assert r["by_slice"]["paraphrased"]["acc"] == 0.0 and r["by_slice"]["paraphrased"]["n"] >= 50
    assert r["by_slice"]["verbatim"]["acc"] == 1.0 and r["by_slice"]["injected"]["acc"] == 1.0
    assert r["false_positive_met"] == 0


def test_config_validation(tmp_path):
    bad = tmp_path / "c.yaml"; bad.write_text("base_model: x\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_config(str(bad))
    ok = load_config("priorauth/finetune/config.yaml")
    assert ok["quant"]["bnb_4bit_quant_type"] == "nf4" and ok["lora"]["r"] == 16
    ok["quant"]["load_in_4bit"] = False
    import yaml
    p = tmp_path / "d.yaml"; p.write_text(yaml.safe_dump(ok))
    with pytest.raises(ValueError, match="QLoRA requires"):
        load_config(str(p))


def test_dry_run_end_to_end_without_gpu(tmp_path):
    cfg = load_config("priorauth/finetune/config.yaml"); cfg["data_dir"] = str(tmp_path / "d")
    counts = prepare_data(cfg)
    assert counts["train"] > 800 and counts["test_large"] > 1000
    r = subprocess.run([sys.executable, "-m", "priorauth.finetune.train_qlora", "--config", "priorauth/finetune/config.yaml", "--dry-run"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0 and "dry-run OK" in r.stdout, r.stderr
