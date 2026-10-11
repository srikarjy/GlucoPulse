"""
Score a tuned adapter (or any OpenAI-compatible endpoint) on the held-out splits and compare to the keyword baseline.

    python -m priorauth.finetune.evaluate_adapter --adapter artifacts/priorauth-doc-judge-qlora            # local GPU
    python -m priorauth.finetune.evaluate_adapter --endpoint http://host:8000/v1 --model NAME              # e.g. vLLM

NOT RUN in this repo's environment (no GPU, no endpoint). It exists so the first real number is produced by the same
scorer that produced the baseline. A model that raises false-positive `met` (claims documentation that is absent) is
WORSE than the baseline for this task even if overall accuracy rises: `false_positive_met` is the gate to watch.
"""
from __future__ import annotations

import argparse
import json
import re

from priorauth.finetune.baseline import keyword_answer, score
from priorauth.finetune.dataset import build


def parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)).get("met") if m else None
    except json.JSONDecodeError:
        return None          # unparseable output counts as "undeterminable" -> routes to a human


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter"); ap.add_argument("--endpoint"); ap.add_argument("--model")
    ap.add_argument("--split", default="test_large")
    a = ap.parse_args()
    rows = build()[a.split]
    if a.endpoint:
        import httpx
        def predict(r):
            resp = httpx.post(a.endpoint.rstrip("/") + "/chat/completions", timeout=120,
                              json={"model": a.model, "messages": r["messages"][:2], "temperature": 0, "max_tokens": 200})
            return parse(resp.json()["choices"][0]["message"]["content"])
    else:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(a.adapter, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(a.adapter)
        def predict(r):
            ids = tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True, return_tensors="pt").to(model.device)
            out = model.generate(ids, max_new_tokens=200, do_sample=False)
            return parse(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    print(json.dumps({"split": a.split, "keyword_baseline": score(rows, keyword_answer), "model": score(rows, predict)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
