# Triage panel: multi-source data → what-if forecasts → agents that discuss → a reviewed triage

Clinical decision **support**, never decision making. Every report requires a clinician's review.
Research data (CGMacros: adults with prediabetes, type 2 diabetes or no diabetes; no insulin).

```
sources/ (Fitbit, Apple Health, CGMacros, any app)  ->  typed records
        │
model/cgmacros/   forecaster with meals as KNOWN-FUTURE inputs  ->  what-if forecasts (+30/+60 min, 80% interval)
        │
agents/triage/    case packet (history only)  ->  tools on the existing MCP ToolBox (scope triage:read)
        │
   rule floor ─┐   round 1: forecaster · lifestyle · data quality · skeptic (independent)
               │   round 2: each sees the others' grounded claims and may revise
               │   moderator (no tools) synthesizes; may not go below the most cautious panelist
               └─► report: level = max(rule, panel) · grounded claims · scenario table · audit log · human review
```

```bash
python -m model.cgmacros.dataset                       # 5-min grid from sources/ output
EPOCHS=8 BATCHES=300 python -m model.cgmacros.train all
python -m model.cgmacros.evaluate                      # ablation + what-if probes -> model/results/cgmacros_ablation.json
python -m model.cgmacros.cases --per-label 40 --cap 10 # packets.json (history) + labels.json (outcome), kept apart
python -m agents.triage.demo                           # watch the panel discuss one case   (.venv-pa)
python -m agents.triage.eval --model openai            # rule vs single agent vs panel        (.venv-pa, OPENAI_API_KEY)
```

## 1. Forecaster ablation (held-out participants)

7 participants never seen in training (33 train / 5 validation, split by person, seed 42). Only forecast windows
with no CGM gap in the forecast hour are scored. RMSE in mg/dL; "meal hour" = a logged meal inside the next 60 min.

| Model | T+30 | T+60 | T+60, meal hour | 80% interval coverage (T+60) |
|---|---|---|---|---|
| Persistence (last value) | 19.53 | 26.24 | 34.91 | — |
| A. glucose history only | 19.19 | 25.48 | 34.31 | 0.819 |
| B. + Fitbit (HR, METs, calories) | 18.83 | 25.45 | 34.22 | 0.789 |
| C. + meals as known-future inputs | 18.00 | 24.21 | 30.21 | 0.747 |
| **D. C + planned activity as known-future** | **17.62** | **23.07** | **28.49** | 0.778 |

**Read with care:** C and D are *oracle* results. In this table they are given the meals that were actually logged
in the forecast hour. Triage never has that; its forecasts assume no new meal (below), so this advantage does not
carry over to triage. Fitbit history helps slightly (B vs A).

**What-if probes** on 300 quiet held-out windows (no meal for 2 h before or during the forecast):

| Scenario | Model | Mean change at +60 | Forecast rises in |
|---|---|---|---|
| +60 g carb meal now | D | +31.6 mg/dL | 99.3% of windows |
| 30-minute walk (4 METs) | D | +0.3 mg/dL | — |

Meals move the forecast the right way on average. Planned activity does not: D learned no usable activity effect,
so **walk scenarios are not offered**. Per case, the meal what-if is not always consistent: in 160 of 196 meal
scenarios the meal forecast dips below the no-meal forecast at some step. Those scenarios carry a
`consistency_warning` the agents and the clinician can see. Model-based associations on observational data,
not causal effects.

## 2. The panel

| Piece | What it does |
|---|---|
| Case packet | Summaries of the 2 h before `as_of` + forecasts: `no_new_meal`, `meal_30g_now`, `meal_60g_now` (model D) and `history_only` (model A, no meal assumption). The outcome lives in a file agents never load. Imputed heart rate is flagged, never shown as measured. |
| Tools | `glucose_summary`, `lifestyle_summary`, `data_quality`, `list_scenarios`, `forecast_scenarios` on the existing `ToolBox`: same auth, scope check and per-case patient scoping. |
| Grounding | A claim is dropped unless every cited ref was returned by a tool, every number in it is a numeric *field value* of a cited resource (never an id, timestamp or description) or a threshold (54/70/180/250), a percentage is the 80% interval, a duration is a horizon or a time field, and it contains no treatment, dosing, medication or diagnostic language. Dissent and concerns go through the same checks. |
| Rule floor | More severe of the rule on `no_new_meal` and on `history_only`: urgent if current < 70 or > 250, or the median goes below 70 or above 250 anywhere in the hour; monitor if current > 180, the median exceeds 180, or the interval reaches below 70 or above 250. Final level = max(rule, panel). |
| Fail-safe | A panelist that fails keeps its previous assessment; model/provider errors are contained; the report carries `panel_incomplete` and the errors. The moderator cannot go below the most cautious panelist. |
| Output | `TriageReport` with grounded claims, dropped claims and reasons, dissent, a scenario table copied from tool output, `requires_human_review: true`. Each draft and its claims are appended to the hash-chained audit log. |

## 3. Leak guardrails and the red-team gate

| Direction | Guard |
|---|---|
| In | Untrusted text in tool results is scanned with the clinical agent's injection detector and replaced before any model sees it; the report lists what was quarantined. |
| Out | Claims, concerns and dissent are dropped if they contain direct identifiers (`deid.scan_text`), any patient other than the case's own, injected instructions, or treatment/diagnosis language. |
| Audit trail | Removed claims are recorded only as `[withheld]` + reason + who said it: the trail never repeats removed text or its numbers. |
| Tools | Per-case patient scoping, a fixed scenario menu (no outcome or future data reachable), and a budget of 8 tool calls per agent. |

`python -m agents.triage.redteam` plays 11 scripted attacks (injection, privacy, scope, grounding, safety, future)
plus a future-leak scan of the real case packets. **12/12 blocked.** It runs in CI as `tests/agents/test_redteam.py`,
so a change that weakens any guard fails the build. Building it found a real leak: removed claims were stored
verbatim, so blocked emails, other patients' ids and an invented low still appeared in the report's audit trail.

## 4. Triage evaluation

Outcome in the next 60 minutes, **sustained for ≥ 15 min** (3 readings): urgent = < 70 or > 250 mg/dL;
monitor = > 180, or any reading < 70; routine otherwise (thresholds need clinical sign-off for this population).
At most 10 cases per person per outcome; case ids are shuffled. 98 cases: 40 routine, 40 monitor, 18 urgent
(only 3 of 7 held-out people had a sustained urgent hour). True pool: 2,880 routine, 322 monitor, 42 urgent.

| Method | Accuracy (sample) | Urgent recall [95% CI by person] | Urgent specificity | PPV / NPV at true prevalence (1.3%) | Under-triage |
|---|---|---|---|---|---|
| Rule floor | 0.694 | 0.333 [0.00, 0.40] | 0.988 | 0.267 / 0.991 | 0.265 |
| Single agent | needs a live run | | | | |
| Panel | needs a live run | | | | |

The rule rarely alarms falsely but misses most urgent hours, and the interval is wide because urgent events come
from three people. Whether a panel that weighs context does better is what the live run measures; nothing here
claims it does. A plausible reason for the misses (unlogged meals) is a hypothesis, not a measurement.
`--model scripted` runs the full loop without an API key; scripted agents replay the rule, so those rows prove the
plumbing, not LLM judgment. The panel prompt states the same level definitions the labels use.

## Limits

- Population: adults without insulin therapy; nothing here transfers to type 1 diabetes.
- Triage forecasts assume no new meal; an unlogged meal is invisible (the `history_only` forecast partly offsets this).
- CGMacros timestamps are randomly shifted by the dataset authors, so one time zone (`UTC`) is used.
- Sending case packets to a third-party LLM should be checked against the CGMacros licence (CC BY-NC-SA) and any data-use terms first.
- No live-LLM results yet; until they exist the README does not claim the panel beats anything.
