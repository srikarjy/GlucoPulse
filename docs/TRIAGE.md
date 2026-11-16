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

Live run: `gpt-4o-mini`, 2026-11-15, all 98 cases, one run (`python -m agents.triage.eval --model openai`).

| Method | Accuracy (sample) | Routine recall | Urgent recall [95% CI by person] | Urgent specificity | PPV / NPV at true prevalence (1.3%) | Under-triage | Over-triage |
|---|---|---|---|---|---|---|---|
| Rule floor | **0.694** | 0.925 | 0.333 (6/18) [0.00, 0.40] | 0.988 | 0.267 / 0.991 | 0.265 | 0.041 |
| Single agent (live) | 0.531 | 0.225 | 0.444 (8/18) [0.33, 0.50] | 0.950 | 0.104 / 0.992 | 0.112 | 0.357 |
| Panel (live) | 0.480 | 0.100 | **0.500 (9/18)** [0.33, 0.60] | 0.925 | 0.080 / 0.993 | **0.092** | 0.429 |
| *Baseline: max(rule, monitor), no LLM* | 0.459 | 0.000 | 0.333 (6/18) [0.00, 0.40] | 0.988 | 0.267 / 0.991 | 0.122 | 0.418 |

**What this shows, and what it does not:**

- **The LLMs mostly answer "monitor".** They call only 10% (panel) and 22% (single) of routine hours routine.
  Much of their drop in under-triage comes from that default, not from reading the case: the last row is one line
  of code with no LLM, and it gets most of the way there (under-triage 0.122 vs the panel's 0.092).
- **Against that baseline, neither LLM method is a measured win.** Paired per case (exact McNemar test on
  correctness): panel vs baseline 7 cases only the panel got right, 5 only the baseline, p = 0.77; single agent vs
  baseline 11 / 4, p = 0.12; panel vs single 3 / 8, p = 0.23. The panel is worse than the plain rule on accuracy
  (17 / 38, p = 0.006), because it over-alarms.
- **Urgent hours:** the panel catches 3 more than the rule (9 vs 6 of 18) at the cost of 6 urgent false alarms,
  versus 4 for the single agent. At the real 1.3% prevalence that drops urgent PPV from 0.27 to 0.08. The 18 urgent
  hours come from 3 people, so the intervals overlap and the extra 3 catches are not evidence.
- **The panel does not beat a single agent.** It costs 9 model calls per case (882 total, about 53 s per case) vs 1
  (about 8 s), and is no better on any metric that has an interval.
- **Grounding is doing real work:** it removed 160 of 359 single-agent claims (45%) and 375 of 821 panel claims (46%)
  because a number was not in the cited tool output or the wording broke a rule. Panel parse errors: 6 of 882 calls,
  contained by the fail-safe.
- **Run-to-run noise:** an earlier interrupted run of the same code and model covered 82 of these cases. Verdicts
  matched on 74/82 (single) and 76/82 (panel); the remaining 8–10% changed with no change in input.

The rule's misses may come from unlogged meals; that is a hypothesis, not a measurement. The panel prompt states the
same level definitions the labels use. `--model scripted` runs the full loop without an API key; scripted agents
replay the rule, so those rows prove the plumbing, not LLM judgment. The eval writes one checkpoint line per case
(`eval_<model>.cases.jsonl`) and resumes after a kill; `--fresh` starts over.

## 5. Is there signal the rule misses? A learned ceiling, no LLM

`python -m model.cgmacros.ceiling` asks whether the packet holds information beyond the rule. If a classifier
trained on the packet's numbers cannot beat the rule, no prompt or panel design will. Every 30-min moment of the
12 people the forecaster never trained on (7 test + 5 validation; 5,590 moments, 256 urgent) becomes the same
packet the agents see, and its numeric fields are the features (60; the outcome is used only for the label). The rule is
the real `agents/triage/rules.py`. Logistic regression and gradient-boosted trees, leave-one-participant-out. The
comparison is at the rule's own specificity, not on accuracy (82% of the pool is routine; 89% of the test people's
moments). Moments already below 70 or above 250 (177) are kept, as triage must handle them.

**Pre-registered criterion:** "clearly beats the rule" only if, on the 7 test people, the 95% participant-bootstrap
interval of (classifier − rule) urgent recall at matched specificity excludes 0.

| 7 test people, at the rule's specificity | AUROC | Rule recall | Classifier recall | Difference, 95% CI by person |
|---|---|---|---|---|
| Urgent (42, from 3 people), trees | 0.935 | 0.548 | 0.643 | [0.000, 0.333] |
| Urgent, logistic | 0.878 | 0.548 | 0.524 | [−0.059, 0.333] |
| At least monitor vs routine (364), trees | 0.878 | 0.670 | 0.728 | [0.003, 0.136] |
| At least monitor vs routine, logistic | 0.878 | 0.670 | 0.761 | [0.041, 0.131] |

- **Urgent: the pre-registered criterion is not met.** The trees' interval touches 0. On all 12 people the trees
  catch fewer urgent hours than the rule (0.574 vs 0.625, interval [−0.14, 0.12]), and one person (cgmacros-013)
  drops from 7/8 to 2/8. So there is **no evidence** the packet supports beating the rule on urgent. That is not
  evidence of no gain: with 42 urgent hours from 3 people, this test cannot rule out a gain of up to ~30 points.
- **Routine vs monitor (exploratory, not pre-registered): a modest signal the rule may miss.** At the rule's
  specificity the classifiers flag 6–9 points more of the hours that needed attention. Of the 8 intervals computed
  (2 models × with/without cgmacros-035 in training × test/12-person pool), 5 exclude 0, 1 touches 0 and 2 include
  it. cgmacros-035 is a validation person, so on the test rows "without 035" removes it from training only. This is
  the boundary where the LLMs failed (they called only 10–22% of routine hours routine). Treat it as a hypothesis to
  confirm on new people, not a finding.
- **On the 98 LLM-eval cases** (out-of-fold, thresholds read at the rule's specificity on the test pool; the rebuilt
  features and rule levels match the agents' packets exactly), the trees score accuracy 0.714, urgent 10/18, routine
  recall 0.85, under-triage 0.204, over-triage 0.082, beside the rule's 0.694 / 6/18 / 0.925 / 0.265 / 0.041 and the
  panel's 0.480 / 9/18 / 0.10 / 0.092 / 0.429. The urgent gain looks larger here (6/18 → 10/18, +22 points) than on
  all 42 urgent test hours (0.548 → 0.643, +10 points), so the 98-case figure should not be quoted alone.

Caveats: the thresholds are read off the evaluation pool (a ROC comparison, not a deployable setting); a deployed
threshold must be fixed on people outside it. The 5 validation people were used for the forecaster's early stopping,
so the 12-person rows are mildly optimistic. Results: `model/results/cgmacros_ceiling.json`.

**Reading (a hypothesis):** if the packet holds more than the rule uses, it is most likely at the routine/monitor
boundary, where a small classifier showed a modest edge and the LLMs over-alarmed instead. Neither the LLMs nor the
classifier showed a reliable urgent gain. That points to a design where a rule or classifier sets the level and
the LLM explains it with grounded claims, the same split argued for in recent work (`docs/TRIAGE_LITERATURE.md`).

## 6. Level-setter: the rule keeps urgent, a classifier decides routine vs monitor

`python -m model.cgmacros.levelsetter` (pre-registration in its docstring). Primary policy: urgent if the rule
says urgent; otherwise a logistic model's P(at least monitor) against a threshold. Nested leave-one-person-out over
the same 12 people: each person's model *and* threshold come from the other 11. Shipped as plain coefficients in
`agents/triage/levelsetter_model.json`, scored in pure Python (matches sklearn to 3e-16 on the 98 packets), and
available in the pipeline only as an opt-in: `run_panel(..., level_source="levelsetter")`. In that mode the LLM
cannot raise the level, a disagreeing LLM level is shown to the clinician, and any level-setter failure falls back
to max(rule, LLM) with the error recorded. The default is unchanged.

| 7 test people (3,244 moments, 42 urgent) | Under-triage | Over-triage | **Urgent hours called routine** |
|---|---|---|---|
| Rule | 4.16% | 11.71% | **4** |
| Level-setter (primary, logistic) | 2.87% | 13.44% | **6** |
| Same policy, boosted trees (secondary) | 2.99% | 15.69% | 4 |
| max(rule, classifier), never downgrades (secondary) | 2.53% | 21.15% | 3 |

- **The pre-registered criterion is met:** under-triage falls by 1.3 points (95% interval by person 0.46–2.26),
  over-triage rises 1.7 (limit 5).
- **But the most dangerous error gets worse.** Of 236 hours the level-setter moved from the rule's "monitor" to
  "routine", 9 needed attention and 3 were urgent. Urgent hours called routine go from 4 to 6 on the test people,
  and from 15 to 22 on all 12. The criterion counted every miss equally, so it did not see this. That was a flaw in
  the criterion, not a reason to ignore the result.
- The policy that never downgrades has the fewest severe misses (3; 15 → 9 on all 12), but raises over-triage to
  21%, beyond the limit.
- Same 12 people as where the signal was found: this shows a deployable version keeps it; it does not confirm it.

**Status of this unguarded version:** not recommended. A guard designed on these 12 people was then tested on new
people (section 7).

## 7. Confirmation on 9 new people: the guarded level-setter

The unguarded downgrades that turned urgent hours into "routine" were almost all hours whose lowest reading in the
past 2 hours was near 70. The **guard** therefore blocks any monitor→routine downgrade when that 2-hour minimum is
below 80 mg/dL. Chosen on the 12 people above (post hoc there), so it was tested on people nothing had been
designed on:

- **Split** (`CGMACROS_SPLIT=confirm`): 9 people drawn at random (seed 2026) from the forecaster's original
  *training* people; the forecaster was retrained on 31 others (5 more for early stopping only).
- **Pre-registration** (`model/cgmacros/confirm.py`) committed in `b4b033a` *before* any forecast or outcome of the
  9 people was computed: the policy, the model, the protocol (nested leave-one-person-out over the 14 people the new
  forecaster never trained on) and three criteria, all required.

| 9 new people (4,176 moments; 88 urgent hours from 6 people) | Under-triage | Over-triage | Urgent hours called routine |
|---|---|---|---|
| Rule | 4.38% | 9.60% | 7 |
| **Guarded level-setter** | **2.97%** | 11.69% | **4** |
| Unguarded (replication of section 6) | 3.02% | 10.68% | 6 |

**All three pre-registered criteria are met:** under-triage falls 1.4 points (95% interval by person 0.74–2.27),
over-triage rises 2.1 points (limit 5), and urgent hours called routine fall from 7 to 4. On all 14 people the
guarded policy calls 10 urgent hours routine against the rule's 17; unguarded, it is 18. The guard costs about one
point of over-triage and removes most of the dangerous downgrades.

Read with care:
- Counts of urgent hours called routine are small (7 vs 4) and have no interval; criterion 3 says "no worse",
  not "proven better".
- The shipped model (`agents/triage/levelsetter_model.json`) was trained on the primary 12 with the primary
  forecaster; this confirms the *policy*, retrained the same way, on new people. The guard is now part of the
  shipped scorer.
- One dataset (CGMacros, adults without insulin). The thresholds are not clinically validated.

**Status:** the guarded level-setter is confirmed on people it was not designed on, and is available with
`level_source="levelsetter"`. It is **not** the default: fewer misses overall for ~2 points more alarms is a trade
for a clinician to accept. In that mode the LLM explains and cannot raise the level.

## Limits

- Population: adults without insulin therapy; nothing here transfers to type 1 diabetes.
- Triage forecasts assume no new meal; an unlogged meal is invisible (the `history_only` forecast partly offsets this).
- CGMacros timestamps are randomly shifted by the dataset authors, so one time zone (`UTC`) is used.
- Sending case packets to a third-party LLM should be checked against the CGMacros licence (CC BY-NC-SA) and any data-use terms first.
- One live model (`gpt-4o-mini`), one run, 98 cases, 18 urgent from 3 people. On this evidence the panel does not
  beat the rule, a single agent, or the max(rule, monitor) baseline. A stronger model, calibrated prompts for the
  "routine" level, or more urgent cases could change that; none has been tested.
