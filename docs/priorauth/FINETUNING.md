# QLoRA fine-tuning

## Should you fine-tune at all?

Challenge it first. The rules already evaluate numeric/coded criteria exactly. The one place a model adds value is **judging whether free-text notes document a criterion**
when the wording isn't a policy keyword. The keyword baseline scores **0 / 114** on that slice of the held-out set (and 1.00 elsewhere), so there is a measurable gap worth closing.
A model that closes it by hallucinating documentation is *worse* than the baseline, so the gate that matters is `false_positive_met = 0`, not accuracy.

## Decision (current): do not fine-tune

The gap that motivated this (rules score 0/114 on paraphrased documentation) was measured on the keyword baseline. A live run of a **stock** Claude Haiku 4.5 through the pipeline
resolved all 10 paraphrased eval scenarios with no hallucinated approvals once an evidence-handoff bug was fixed (see EVALUATION.md, "First live LLM run").
Fine-tuning would add a GPU training cost, an adapter to host and re-validate, and drift risk, to fix a problem that no longer shows up.
Revisit if: a larger or real-world eval shows a paraphrase/documentation gap the stock model cannot close, per-call cost or latency becomes the constraint (a tuned small model could help),
or data cannot leave your boundary. The pipeline below is kept, tested, and ready for that case. It has never been trained.

## Pipeline

| Step | Command | Status |
|---|---|---|
| Build dataset | `python -m priorauth.finetune.dataset` | tested |
| Baseline | `python -m priorauth.finetune.baseline` | tested; numbers above |
| Validate config + data, no GPU | `python -m priorauth.finetune.train_qlora --config priorauth/finetune/config.yaml --dry-run` | tested in CI |
| Train (4-bit NF4 + double-quant, LoRA r=16/α=32 on all linear layers, bf16, grad-checkpointing) | `python -m priorauth.finetune.train_qlora --config …` | **not run: needs CUDA** |
| Score adapter vs baseline | `python -m priorauth.finetune.evaluate_adapter --adapter …` | **not run** |

Base model: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0). Install `priorauth/finetune/requirements-train.txt` on a GPU box only.

## Data and leakage controls

* Task prompts mirror the real tool output exactly, including quarantined notes arriving blanked. Documentation is never labelled `false`; absence is `null`.
* Train/val: seed 777, 1,600 scenarios, paraphrase **variants 1–2**. Test: the 200 eval scenarios plus a 1,200-example held-out set (seed 4242), **variant-0 wording only**.
  Wording and patients seen at test time never appear in training (`test_held_out_wording_never_appears_in_training`).
* Template-like notes can repeat verbatim across patients, so held-out prompts that also occur in train/val are **dropped**; `prepare_data` re-checks and refuses to train on any overlap.

## Limits

The synthetic vocabulary is small (19 criteria phrases × 3 wordings). A model that scores well here has shown the *pipeline* works and generalises across rephrasings of the same ideas; it has **not** shown clinical competence.
Before any real use: de-identified real notes under IRB/BAA, clinician-labelled, with a model-risk review. Prefer serving the adapter behind the same grounding layer, which caps what a wrong judgement can do.
