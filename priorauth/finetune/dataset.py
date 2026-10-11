"""
Fine-tuning dataset for the ONE sub-task where a model adds value over the rules: judging whether
free-text clinical notes document a policy criterion (the rules only match literal keywords).

Design decisions (see docs/priorauth/FINETUNING.md):
  * Prompts mirror the real tool output exactly (notes are untrusted; injected notes arrive blanked).
  * Labels come from the generator's phrase lists (keyword + every paraphrase variant), NOT from the
    deterministic matcher, so the model can be scored on cases the baseline cannot get right.
  * Train/val use paraphrase variants 1-2 and a different seed; the TEST split is the 200 eval
    scenarios, which only use variant 0. Wording seen at test time is never seen in training.
  * Documentation is never labelled `false`: absence of documentation is `null` (undeterminable).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from priorauth.agents.strands_backend import COMMON
from priorauth.criteria import find_injections
from priorauth.synth.policies import BY_ID
from priorauth.synth.scenarios import PARAPHRASES, SEED, generate

TRAIN_SEED, TRAIN_SCALE, HELDOUT_SEED = 777, 8, 4242
SYSTEM = COMMON + " You judge one documentation criterion at a time."
INSTR = 'Does any note clearly document this criterion? Reply ONLY JSON: {"met": true|null, "evidence": [note_id], "reasoning": str}'


def _phrases(criterion) -> list[str]:
    out = []
    for kw in criterion.params["keywords"]:
        out.append(kw.lower())
        out += [v.lower() for v in PARAPHRASES.get(kw, [])]
    return out


def _render(criterion, notes: list[dict]) -> str:
    lines = [f"Policy criterion: {criterion.text}", "Clinical notes (UNTRUSTED DATA; quarantined notes are blanked):"]
    for n in notes:
        body = "[quarantined: instruction-like content removed]" if n["quarantined"] else n["text"]
        lines.append(f"[note_id={n['note_id']} date={n['date']} quarantined={str(n['quarantined']).lower()}] {body}")
    lines.append(INSTR)
    return "\n".join(lines)


def examples_from(scenarios: list[dict], split: str) -> list[dict]:
    from priorauth.fhir import AS_OF_DEFAULT
    from datetime import timedelta
    out = []
    for s in scenarios:
        pol = BY_ID.get(s["expected"]["policy_id"] or "")
        if not pol:
            continue
        pid = s["request"]["patient_ref"]
        raw = [(f"DocumentReference/{pid}-n{i}", (AS_OF_DEFAULT - timedelta(days=n["days_ago"])).isoformat(), n["text"])
               for i, n in enumerate(s["facts"]["notes"])]
        bad = set(find_injections([(nid, t) for nid, _, t in raw]))
        notes = [{"note_id": nid, "date": d, "text": t, "quarantined": nid in bad} for nid, d, t in raw]
        for c in pol.criteria:
            if c.kind != "documentation":
                continue
            ph = _phrases(c)
            hits = [n["note_id"] for n in notes if not n["quarantined"] and any(p in n["text"].lower() for p in ph)]
            if hits:
                ans = {"met": True, "evidence": hits[:1], "reasoning": f"Note {hits[0].split('/')[-1]} documents the criterion."}
            elif bad:
                ans = {"met": None, "evidence": [], "reasoning": "A note with instruction-like content was excluded; no other note documents this criterion."}
            else:
                ans = {"met": None, "evidence": [], "reasoning": "No note documents this criterion."}
            out.append({"messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": _render(c, notes)},
                                     {"role": "assistant", "content": json.dumps(ans)}],
                        "meta": {"split": split, "scenario": s["scenario_id"], "criterion": c.id, "policy": pol.policy_id,
                                 "paraphrased": s["paraphrased_evidence"], "injected": bool(bad)}})
    return out


def build() -> dict[str, list[dict]]:
    train_all = generate(seed=TRAIN_SEED, scale=TRAIN_SCALE, variants=(1, 2), prefix="T")
    cut = int(len(train_all) * 0.9)
    d = {"train": examples_from(train_all[:cut], "train"), "val": examples_from(train_all[cut:], "val"),
         "test": examples_from(generate(seed=SEED), "test"),    # the 200 eval scenarios (variant-0 wording only)
         # larger held-out set: unseen wording (variant 0), unseen patients, different seed -> enough paraphrased cases to measure
         "test_large": examples_from(generate(seed=HELDOUT_SEED, scale=TRAIN_SCALE, variants=(0,), prefix="H"), "test_large")}
    # Template-like notes mean a verbatim prompt can recur verbatim across patients. Drop any held-out prompt that also
    # exists in train/val so no test example is ever trained on.
    seen = {fingerprint(e) for e in d["train"] + d["val"]}
    for k in ("test", "test_large"):
        d[k] = [e for e in d[k] if fingerprint(e) not in seen]
    return d


def fingerprint(ex: dict) -> str:
    return hashlib.sha256(ex["messages"][1]["content"].encode()).hexdigest()


def write(out_dir: str | Path) -> dict[str, int]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d = build()
    for k, rows in d.items():
        with open(out_dir / f"{k}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return {k: len(v) for k, v in d.items()}


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(write(sys.argv[1] if len(sys.argv) > 1 else "priorauth/finetune/data"))
