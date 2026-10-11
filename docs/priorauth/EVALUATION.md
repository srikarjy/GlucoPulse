# Evaluation

`python -m priorauth.eval.run_eval` runs 200 synthetic scenarios through the full pipeline (store, MCP tools, agents, routing) and exits non-zero if any
hard safety gate is below 100% or accuracy is below the floor. It runs in CI. Raw output: `priorauth/eval/baseline_results.json`.

## Design

Scenarios (`priorauth/synth/scenarios.py`, seed 20260115, committed as `scenarios_200.jsonl` and checked for exact reproducibility) cover nine profiles:
clean (55), contradicted criterion (35), missing evidence (30), code mismatch (15), prompt injection (15), conflicting labs (15), stale labs (15), unknown service (10), borderline thresholds (10).
**Labels come from which mutation the generator applied, not from running the evaluator.** Ten "clean" scenarios document criteria in clinically equivalent wording that contains no policy keyword.

## Results (deterministic backend)

| Metric | Value |
|---|---|
| Recommendation accuracy | **0.95** (190/200) |
| Precision / recall: supports_approval | 1.00 / 0.846 |
| Precision / recall: does_not_support | 1.00 / 1.00 |
| Precision / recall: insufficient_info | 0.895 / 1.00 |
| Flag precision / recall | 1.00 / 1.00 |
| Retrieval recall@1: via service code | 1.00 |
| Retrieval recall@1 / @3: description only (no code) | 0.82 / 1.00 |
| **Safety gates** (human review, no agent decisions, injection resistance, physician-on-denial) | **1.00 each** |
| Audit chain intact | yes |
| Confidence ECE (heuristic) | 0.18 |

**All 10 errors are the paraphrased-evidence scenarios, and every one fails safe** (`supports_approval → insufficient_info`: the human sees "documentation not found" rather than a wrong approval).
No error ever favours approval.

## How to read this (the limits)

* **Largely self-referential.** The generator and the rules were written by the same author from one spec, so high scores on structured criteria prove the code matches the spec, not that the spec is clinically right.
  The paraphrase slice is the only part that tests something the implementation does *not* already encode, and it is where the baseline scores 0/10.
* **No LLM numbers exist.** The Strands/Bedrock backend has not been scored. The first real evaluation of it is `python -m priorauth.eval.run_eval --backend bedrock --limit 10` (needs AWS credentials and Bedrock model access; costs money, so start small; `--min-accuracy` only gates the accuracy floor, the safety gates always apply); expect the paraphrase slice to move and watch that safety gates stay at 1.0.
* A bug found by this eval: the bariatric policy lacked a diagnosis criterion, so a wrong-diagnosis request got a 0.97-confidence approval draft. Fixed by adding the criterion; the generator now exercises it.
* Synthetic ≠ real: real notes are longer, messier, and contradictory; real policies have exceptions and clinical judgement the rules cannot encode.
* The hash-based embedder is lexical. Semantic-only recall@1 of 0.82 is a floor; use a real embedding model (`BedrockEmbedder`) and re-measure.

## Regression tests

`tests/priorauth/test_eval_regression.py` fails if any safety gate drops below 1.0, accuracy < 0.95, retrieval/flag recall regress, any error becomes a mis-approval, or the committed scenarios stop matching the generator.
