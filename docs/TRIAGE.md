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

## 8. Event model: predict the urgent event directly (not confirmed)

Why: on the 98 eval cases the forecaster's 60-min median barely continued a rise (of 20 cases rising >= 20 mg/dL per
15 min, 14 had a median at or below the current reading; the other 6 rose by at most 16 mg/dL), and the rule calls urgent only when that median crosses 250.
The label asks whether the path crosses at any time, which the median understates by construction.

`python -m model.cgmacros.event` trains a gradient-boosted classifier on the urgent event itself, from glucose history
only (levels, 5-60 min changes, slope, acceleration, 2-h range, meals, heart rate), on the forecaster's 31 training
people. The threshold comes from their out-of-fold predictions (target specificity 0.985), never from test labels.
Policy: urgent if the rule says urgent or the event model fires; otherwise the rule's level. Comparator fixed in
advance: urgent if the current reading + 4 x the 15-min change is outside 70-250. The protocol was committed
(`e4c1958`) before any output for the test people was computed.

| Urgent hours (rule floor kept) | Rule recall | Event recall | Gain, 95% CI by person | False alarms rule -> event | Trend recall / false alarms |
|---|---|---|---|---|---|
| **9 new people (pre-registered)**, 88 urgent from 6 people | .705 | .739 | +3.4 pts [0.0, 10.0] | 1.1% -> 1.9% | .784 / 11.8% |
| All 14 confirm held-out people (secondary), 277 urgent | .650 | .726 | +7.6 pts [1.6, 9.7] | 1.5% -> 2.5% | .765 / 12.7% |
| Primary held-out 12 (exploratory), 256 urgent | .625 | .723 | +9.8 pts [6.3, 17.2] | 1.5% -> 2.3% | .777 / 13.4% |

- **Pre-registered result: not confirmed.** Criterion 2 (false alarms at most 2 points higher) passed; criterion 1
  failed. Because the policy is "rule OR model" it can only add urgent calls, so the gain is never negative and a
  lower bound of exactly 0 is not a near miss: in at least 2.5% of person-resamples there was no gain at all, because
  the gain sits in few people. The pre-run power check (which spread extra hits evenly across people, so it was
  optimistic) said about +15 points was needed; the observed gain was 3.4.
- **The trade, in hours:** on the 9 new people the model catches **3** more urgent hours for **30** more false
  alarms (of 4,088 non-urgent moments); on all 14 people, **22** more for **58** more (of 6,225). On its own the
  event model ranks well (test AUROC .96-.98), but most of what it catches the rule already catches.
- **The trend comparator is not usable:** a little more recall for ten times the false alarms (12-13%).
- **It does not fix the severe misses.** Urgent hours called routine: 7 -> 7 on the 9 new people, 17 -> 17 on all
  14. Checked row by row: all 7 on the 9 new people (15 of 17 on all 14) are highs that started at 117-167 mg/dL
  with little trend and then peaked at 258-400, most likely meals eaten after the moment, which no history-only
  model can see. The other 2 of 17 are lows.
- Hour-of-day features may be weak: CGMacros timestamps are shifted by the dataset authors.
- The 5 early-stopping people in the all-14 row are part of the primary held-out 12, whose eval cases were used to
  explore the trend thresholds; that mainly flatters the trend comparator.

**Status:** not shipped. The evidence is a small, consistent gain that this dataset is too small to confirm. The
largest remaining error, a spike from a quiet baseline, needs new information (an upcoming-meal prompt or re-triage
when a meal is logged), not a better model of the past. (Section 9 qualifies this: checking every 5 min, rather than
every 30, warns of most of these spikes before they cross 250.)

## 9. Five-minute replay: published CGM alerts (alert works; adding it to the rule is not confirmed)

`model/cgmacros/replay.py` replays each person's data reading by reading, as a live CGM system would. The protocol
was committed (`3eba163`) before any algorithm was run on CGMacros:
- **Event:** at least 3 readings in a row above 250 or below 70.
- **Warning:** an alarm while the reading is still in the 70-250 range, at most 60 min before onset. "Timely" means at
  least 15 min ahead.
- **False alarm:** an alarm episode (30-min lockout) with no event in the next 60 min. Counted per day of monitored
  readings.
- **Random baseline:** each stream is also compared with random alarms at the same per-person rate (20 seeds). A
  stream that alarms more often warns more by chance.

No algorithm uses parameters fitted to CGMacros.

**Primary 1 (all 45 people, 223 events from 33 people, 439 days): passed.**

| Algorithm | Timely | Random | Warned at all | Median lead | False alarms/day |
|---|---|---|---|---|---|
| alert20 (15-min slope projected 20 min) | .677 | .316 | .982 | 20 min | 2.37 |
| rate2 (\|slope\| >= 2 mg/dL/min) | .587 | .301 | | | 3.33 |
| GRID (Harvey 2014 tuning) | .426 | .213 | | | 1.91 |
| GRID, healthy-adult tuning (Hochsmann 2026) | .596 | .441 | | | 5.09 |

- alert20 beats its random baseline by 36 points, 95% CI by person [28, 47].
- The 20-min median lead is set by design: alert20 projects 20 min ahead.
- alert20 warns timely on .64 of highs and .83 of lows. GRID detects rises only, so it warns on almost no lows (.02).
- GRID as a meal detector, against logged meals:
  - Harvey tuning: 44% sensitivity, 0.6 false positives/day, 30 min to detection.
  - Healthy-adult tuning: 75%, 1.95/day, 30 min. The published figures are 70%, 0.26/day, 37 min.
  - Our false positives are higher, probably because of unlogged snacks and because, unlike the paper, there is no
    night exclusion. The implementation looks right.

**Primary 2 (the 14 people the confirm forecaster never saw): rule + alert20 vs the rule, both checked every 5 min.**
The rule's level was computed at 98.7% of readings.

| Stream | 14 people: timely (random) | False alarms/day | 9 new people: timely (random) | False alarms/day |
|---|---|---|---|---|
| Rule | .395 (.180) | 0.78 | .385 (.150) | 0.82 |
| **Rule + alert20** | **.741** (.415) | **2.38** | .769 (.304) | 2.05 |
| alert20 alone | .691 (.344) | 2.14 | .731 (.238) | 1.74 |
| Event model alone, secondary | .728 (.313) | 0.93 | .731 (.196) | 0.82 |

There are 81 events from 10 people (70 highs, 11 lows) over 138 days. The 9 new people have 26 events from 7 people
over 89 days.

- **Pre-registered result: not confirmed.**
  - Criterion (a) passed: the timely share rose by 35 points, 95% CI by person [28, 50].
  - Criterion (b) failed: false alarms rose by **1.60 a day**, over the limit of 1.0.
  - On the 9 new people, false alarms rose by 1.23 a day, which also fails.
  - The alert's parameters were not retuned after seeing this. Any tuned version would be exploratory.
- **The trade:**
  - All 14 people: about **28 more events warned in time** for about **221 more false alarms**, roughly 8 per extra
    warning.
  - 9 new people: about 10 more for about 109 more.
- **Checking every 5 min is what helps.**
  - The rule alone, checked every 5 min, warns timely on .37 of highs.
  - With alert20 added, that rises to .71.
  - So many of the spikes §8 called invisible become visible once they start rising, about 15-20 min before they
    cross 250. That fits Dassau 2008, where meals showed on the CGM about 30 min after eating began.
  - This does not show that those exact §8 misses are among the newly warned events; that was not checked.
- **Secondary: the event model checked every 5 min.** This is the §8 model alone, trained on the 31 training people,
  with its pre-registered threshold, without the rule.
  - Its timely share is close to rule + alert20 (.728 vs .741), at the rule's false-alarm rate (0.93 vs 0.78 a day).
  - It has the largest margin over its random baseline: 42 points on all 14 people, vs 33 for rule + alert20.
  - It is weaker on lows: .545 timely, the same as the rule. It warns at all on only .89 of events.
  - No criterion was set for it, so it is a candidate for the next pre-registered test, not a finding.
- **Lows:** 11 events (6 on the 9 new people). That is too few for any claim.

**Status:** not shipped. A 20-min predictive alert gives real early warning on its own, about 2 false alarms a day,
in line with consumer CGM alerts. Adding it to the rule failed the false-alarm limit set in advance. Next:
pre-register the 5-min event model, or the rule OR the event model, with the same two criteria.

## 10. Hybrid meal what-if: the no-meal forecast plus a physiological meal curve (passed)

**The problem.** The forecaster learned meals from data alone. Its meal what-ifs were not physiologically
consistent:
- A meal forecast dipped below the no-meal forecast in 44% of the windows below.
- In case-021 a 30 g meal forecast higher than a 60 g meal.

**The idea.** From Apple's Miller, Foti and Fox (2020, arXiv:2008.02852): keep a physiological model's structure and
let a learned model set its parameters. CGMacros has no insulin data, so only the meal part is used.

`model/cgmacros/hybrid.py` builds each meal forecast as:
- **The no-meal forecast**, unchanged.
- **Plus the meal's carbs passed through the Hovorka gut-absorption curve** (time constant 40 min), with a
  glucose-clearance rate fitted on training people. The fitted rate is 0.02/min.
- **Scaled by a non-negative gain**, set from the 2 hours of history.

By construction a bigger meal never forecasts lower, and a meal never forecasts below no meal
(`tests/model_cgmacros/test_hybrid.py`).

**The test.** The protocol was committed (`8139885`) before any held-out output existed:
- **Windows:** every logged meal, with `as_of` 5 min before it.
- **Inputs:** both arms get that hour's logged meals, through the decoder triage uses (no oracle activity).
- **Fitting:** on the 31 confirm-split training people.
- **Primary:** on the 9 new people, the hybrid is non-inferior to the forecaster at T+60 if the upper 95%
  person-bootstrap bound of the RMSE difference is below +2.0 mg/dL.

| Meal windows | Forecaster (D) T+60 RMSE | Hybrid | Hybrid, personalized | Hybrid - D, 95% CI | No-meal forecast |
|---|---|---|---|---|---|
| **9 new people (pre-registered)**, 327 meals | 39.9 | **35.7** | 33.2 | -4.1 [-8.4, +0.6] | 48.0 |
| All 14 held-out people (secondary), 485 meals | 41.4 | 36.4 | 35.1 | -4.9 [-8.9, -1.0] | 51.5 |

- **Pre-registered result: passed (non-inferior).** The upper bound, +0.6, is below the +2.0 margin.
- **Better, not just non-inferior?** On the 9 new people the interval still includes 0. The secondary results lean
  that way:
  - Averaged over the hour: hybrid 25.3 vs D 27.4, CI [-3.9, -0.3].
  - On all 14 people the T+60 interval excludes 0.
- **Personalized gain (secondary).** This scales the gain by the person's response to earlier meals, using only meals
  whose hour had ended. It reached 33.2, CI vs D [-11.5, -3.1]. That is the closest analogue of the Apple paper's
  per-person dynamics, but no criterion was set for it, so it is a candidate, not a finding.
- **What-if consistency:**
  - The hybrid's meal forecast fell below no-meal in 0% of windows; D's did in 44% (9 new) and 48% (14).
  - On a live window for a new person: D forecast 110.9 for a 30 g meal and 110.8 for 60 g at +30 min. The hybrid
    gave 105.7 and 109.8.
- **Calibration of the rise over no-meal at +60 (9 new people):**
  - 50-70 g meals: observed +47, hybrid +45, D +25. D under-predicts large meals by half.
  - 20-40 g meals: observed +20, hybrid +13, D +16. The hybrid is linear in carbs, so it under-predicts small meals.
- **Read with care:**
  - D's T+60 RMSE here (40) is higher than in §1 (28.5). These windows start right at a meal and use the triage
    decoder, not the oracle activity of §1.
  - The cross-validation loss over the clearance-rate grid was not smooth: 0.01 scored worse than 0 and 0.02. The
    fit may have local optima.
  - The gain was fitted on residuals from people the forecaster trained on.
  - The p10-p90 band is shifted with the median. Its width is not modelled.

**Verification after the run.** These checks were made after the run, so they are not part of the pre-registration.

Checks that passed:
- A refit from the cache reproduces k and every RMSE.
- No test person is in the fitting set.
- For all 1,553 windows, these were recomputed from the raw grid and matched: the future glucose, the meal carbs (each
  meal at decoder step 1), and the gain features (from history ending at `as_of`).
- The meal and no-meal arms differ in 98% of windows.
- On all 327 held-out windows, forecasts never decrease as carbs go from 0 to 120 g.
- The hybrid beats D for 8 of the 9 new people. The exception is cgmacros-009: 45.8 vs 36.1.

What the comparison is worth: T+60 RMSE on the 9 new people for simpler baselines.

| Method | T+60 RMSE |
|---|---|
| Persistence | 49.0 |
| No-meal forecast | 48.0 |
| D (forecaster with the meal) | 39.9 |
| No-meal + the training people's average post-meal residual (ignores meal size) | 38.3 |
| No-meal + a straight-line rise per gram | 37.5 |
| No-meal + Hovorka curve, one constant gain | 37.5 |
| **Hybrid (Hovorka curve, learned gain)** | **35.7** |

So:
- **D is a weak bar on these windows.** Adding the average post-meal rise to the no-meal forecast, ignoring the meal
  size, already beats it.
- **At T+60 the Hovorka shape adds no accuracy over a straight line.** Both predict about +26 mg/dL on average at 60
  min; the shape only differs before then.
- **The learned gain adds about 1.8 mg/dL.**

The case for the hybrid is therefore its consistency (bigger meals always forecast higher, and it is built on a
published absorption curve), plus a modest accuracy gain from the gain model. It is not that the physiology beats data
at 60 min. That matches the Apple paper, where the physiological structure paid off mainly at 2-6 h.

**Status:** available, not the default. `python -m model.cgmacros.cases --meal-model hybrid` builds packets whose meal
what-ifs come from the hybrid (`scenarios.HybridMeal`). The default stays the forecaster, because the level-setter
(§6) takes the meal scenarios as features (`agents/triage/features.py`). Switching would change triage levels and
needs the level-setter retrained and re-tested. Triage levels from the rule (no-meal and history-only forecasts) are
unaffected either way.

## 11. Better forecasts: boosted trees with personal history + Chronos-2 (confirmed on 9 locked people)

**How the model was found.** By trial and error in `model/cgmacros/lab.py`, under a protocol fixed before the first
trial:
- **Dev set:** the primary held-out 12 people.
- **Target:** forecasts made with no future-meal information, which is what triage uses.
- **Bar:** the TFT A forecaster.
- **Log:** every trial is in `model/results/lab_log.jsonl`. There were 14 trials, covering the five families in
  TRIAGE_LITERATURE.md §8.

**Dev results** (T+60 RMSE; TFT A scored 27.7):

| Trial | T+60 |
|---|---|
| Boosted trees, 2-h features | 25.7 |
| + 24-h context | 23.9 |
| + personal history | 24.3 |
| + both (`hgb_long_person`) | 23.6 |
| + retrieval of similar past windows | 23.6 (no gain) |
| Chronos-2, zero-shot | 26.0 |
| Fixed 70/30 blend of `hgb_long_person` and Chronos-2 | 23.3 |

Leak checks passed: recomputing every new feature with all later data deleted (or the outcomes replaced with garbage)
left it unchanged.

**Locked test** (`model/cgmacros/lab_confirm.py`, pre-registered in `7a9e36f`):
- **Primary model:** the blend, refit on the 31 confirm-training people.
- **Test set:** the 9 confirm-test people, 25,030 windows. This is their fifth confirmatory use.
- **Criterion:** a T+60 gain over the confirm TFT A whose 95% person-bootstrap interval stays above 0.

| 9 locked people | T+30 | T+60 | Gain vs A, 95% CI | Rising windows T+60 | Meal next hour T+60 | 80% coverage | Better than A |
|---|---|---|---|---|---|---|---|
| TFT A (current) | 18.5 | 26.96 | — | 41.6 | 35.7 | — | — |
| **Blend (pre-registered)** | **15.7** | **22.46** | **+4.5 [3.3, 5.6]** | **35.8** | 31.3 | 0.773 | **9 of 9** |
| `hgb_long_person` alone | 15.8 | 22.89 | +4.1 [2.4, 5.4] | 36.2 | 30.8 | 0.859 | 8 of 9 |
| Chronos-2 alone (zero-shot) | 17.7 | 25.47 | +1.5 [0.9, 2.0] | 40.8 | 34.8 | 0.801 | 8 of 9 |

- **Pre-registered result: confirmed.**
  - T+60 RMSE fell from 26.96 to 22.46 (-17%), and the interval excludes 0.
  - The band covered 77.3% of outcomes, inside the pre-set 0.75-0.85.
  - The blend beat A for every one of the 9 people.
  - The locked gain (+4.5) matched dev rather than shrinking.
  - For context: §1's TFT D, which is *told* the future meals, scored 23.07.
- **Where the gain comes from:** context the TFT never saw. Its 2-h window is normalized per window, so it lacks the
  last 24 h of glucose, the person's usual level, and how they responded to earlier meals. Chronos-2 adds calibrated
  bands and a little accuracy.
- **Rising windows** (the spikes behind most urgent misses) improved from 41.6 to 35.8. That is real but still large:
  a spike from a flat start remains hard without knowing about the meal.
- **Not tested: triage levels.**
  - The rule's thresholds and the level-setter were built on TFT outputs.
  - Swapping in this forecaster changes urgent and monitor calls. That needs its own pre-registered test (urgent
    recall and false alarms) before deployment.
  - The 9 people have now been used five times, so that test needs new people or a new dataset.
- **Operational cost:**
  - Chronos-2 runs in its own environment (`.venv-chronos`, `chronos-forecasting==2.0.1`, revision pinned in
    `chronos_run.py`).
  - It took about 17 minutes on CPU for 25k windows, about 40 ms per forecast.
  - `hgb_long_person` alone keeps most of the gain (+4.1) with no extra dependency, but its band is too wide (0.86).

## 12. Reliability: the accurate model warned less at the extremes; a safety floor fixes that (confirmed)

`model/cgmacros/reliability.py` audited the §11 blend. It checked:
- the Clarke error grid;
- error by the glucose level that occurred;
- whether the median and band reach actual lows (< 70) and highs (> 250);
- cold start;
- each person.

**Finding:** the blend's average gain came partly from predicting closer to the person's usual level, which is the
wrong direction for a warning system.
- On dev, the median reached 3% of actual lows (TFT A: 31%) and the p10 reached 43% (A: 69%).
- On the locked 9, the median reached 12% of actual highs (A: 21%).
- Clarke A+B was fine (99.3% dev, 99.5% locked, vs A's 99.1% and 99.2%).
- Day 1 of a person's data is the weakest (dev 27.2), but still better than A.

**Fix, chosen on dev** (9 more trials, 23 in all; `rel_*` in `lab_log.jsonl`):
- *Learning the band directly* (quantile trees) made the tails worse.
- *Up-weighting extreme training labels* helped highs but not lows.
- *Kept: a safety floor.* Where TFT A's median is outside 70-250 and the blend's is not, use A's median. The band is
  the union of both bands. Nothing is refit, and the forecaster never warns less than the current one.

**Locked test** (pre-registered in `7eb9df9`, the 9 people's sixth use):

| 9 locked people, T+60 | TFT A (current) | Blend (§11) | **Blend + floor** |
|---|---|---|---|
| RMSE | 26.96 | 22.46 | **22.87** (gain +4.1 [3.0, 5.1]) |
| Rising windows RMSE | 41.6 | 35.8 | 36.2 |
| Actual highs reached by median / p90 | 21% / 57% | 12% / 52% | **22% / 62%** |
| Actual lows reached by median / p10 | 81% / 97% | 80% / 96% | **84% / 97%** |
| p90 > 250 when the actual stayed <= 250 | 3.9% | — | 4.2% |
| 80% band coverage | 0.79 | 0.77 | 0.89 (conservative) |
| People better than A | — | 9 of 9 | 9 of 9 |

- **Confirmed.** Both criteria passed:
  - The RMSE gain interval excludes 0.
  - Monitor false alarms rose 0.3 points, within the 1-point limit.
- **The cost:** the floor gives back 0.4 of the RMSE gain, and the band is wider than it needs to be.

**Still weak: big spikes.**
- When glucose actually went above 250, every model predicted far too low (bias -70 for the floored model).
- The median reached only 22% of those highs.
- These are mostly meals the forecaster cannot see. Better history models cannot fix that; knowing about the meal
  (a logged or prompted meal, §10) or checking every 5 min (§9) can.

**Status:**
- **Reliable as a forecaster:** more accurate for every locked person, and never less alarming than the current one at
  the extremes.
- **Still untested for triage levels:** that needs new people.

## 13. Spikes: what meal information can and cannot do (dev only, exploratory)

**Question.** Can the forecaster catch more spikes (actual > 250 within the hour)?

**Headroom check first.** Training people, windows that cross 250 by T+60 from <= 250 now:

| Situation | Share |
|---|---|
| Meal logged in the next hour (a prompt could help) | 26% |
| Meal already logged in the 60 min before `as_of` (the model already had it) | 48% |
| Neither | 28% |

Dev: 20% / 33% / 46%. §8's "unlogged meals after the moment" was therefore only part of the story: many spikes follow
a meal the model knew about.

**Trials** (protocol written before the first trial; dev 12 only, blend + safety floor in every arm):

| Dev, T+60 | Highs reached by median | by p90 | RMSE, meal next hour | RMSE |
|---|---|---|---|---|
| Blend + floor (§12) | 46.4% | 78.2% | 31.0 | 23.66 |
| + Hovorka curve for meals already eaten | 48.1% | 78.6% | 31.0 | 23.68 |
| + meal prompt answered every time (upper bound) | 49.8% | 81.4% | **26.2** | 22.60 |
| + prompt answered half the time | 47.6% | 79.5% | 29.4 | 23.29 |

The meal features are monotone, so more carbs never forecasts lower.

**Result: a negative result.**
- **What meal information does:** it sharpens the forecast around meals. With a perfectly answered prompt, the error in
  the meal hour falls from 31.0 to 26.2.
- **What it doesn't do:** catch many more spikes. Highs reached by the median rose at most 3.4 points.
- **Why:** a 60-min forecast made before a spike begins cannot see how high it will go.
- **What works instead:** checking every 5 min. §9's trend alarm warned of about 70% of highs at least 15 min ahead.
- **Recommendation:** run the forecaster and the alarm every 5 min, rather than adding meal prompts for accuracy.

## 14. External test on 73 new people: more accurate forecasts, but they must not replace the triage forecaster

**Data** (people from two other studies, never used for any choice), read through `sources/external_cgm.py` into the
same 5-min grid:
- **BIG IDEAs** (PhysioNet, doi:10.13026/aw6y-fc44, ODC-By): 16 adults, Dexcom G6, food logs.
- **Hall et al. 2018** (doi:10.1371/journal.pbio.2005143, CC BY): 57 adults including type 2 diabetes, Dexcom G4.

**Gaps and fixes:**
- Heart rate and METs are resting defaults; Hall has no meals.
- One adapter fix was needed: BIG IDEAs logs one row per food *item*, so items with the same start time are summed into
  one meal. Without that, normalization quarantined 3,701 rows.

**Test design.** The protocol (`model/cgmacros/external.py`) was committed in `bd4bcde` before any forecast was made.
- **Models:** frozen.
- **Moments:** 18,294, at 30-min steps.
- **Urgent events:** 819 urgent moments from 48 people, almost all lows. Highs cannot be tested here: there were 3
  events in all.
- **Arms:**
  - Current: rule(TFT D no-meal, TFT A).
  - New: rule(new forecaster, TFT A).

| 73 people | Urgent recall | Predictive recall (in range now) | Urgent false alarms | T+60 RMSE |
|---|---|---|---|---|
| Current rule (D + A) | .615 | .234 | 2.28% | D 19.27 / A 19.23 |
| **New rule (new + A), pre-registered** | .585 (-3.0 [-4.2, -1.6]) | .173 | 1.79% | **new 17.53** (gain vs A +1.7 [1.3, 2.1]) |
| *All three slots (exploratory, after the result)* | *.618* | *.238* | *2.29%* | — |

- **Pre-registered result: passed on paper, but not a win.**
  - Both criteria passed: recall was non-inferior at the -5-point margin, and false alarms did not rise.
  - Recall nonetheless fell by 3 points, with the interval entirely below 0.
  - On lows still ahead (the moment is in range now), recall fell from 23.4% to 17.3%.
  - The new rule traded sensitivity for fewer false alarms. For a warning system, that is the wrong trade.
- **Why** (exploratory, by slot): the new forecaster's median almost never goes below 70. On its own it catches 3.6%
  of upcoming lows, against 21.2% for D and 16.1% for A. This is the §12 shrinkage again. Replacing D removed D's
  catches, and the A slot kept only some of them.
- **Accuracy holds up externally.**
  - T+60 RMSE is 17.5 vs 19.2 overall.
  - The gain held in both datasets: BIG IDEAs +2.6 [2.3, 3.0], Hall +1.2 [0.8, 1.7].
  - It held for new users with no personal history.
- **Decision:**
  - Use the new forecaster for the forecasts people see. It is more accurate, including on new datasets.
  - Do not let it replace D or A in the urgent decision.
  - If it is added to the rule, add it as a third slot, which by construction can only add urgent calls. Exploratory
    numbers above: recall .618, false alarms unchanged, monitor calls +0.6 points. That variant has not been
    pre-registered.

## 15. A 5-minute monitoring loop, tested on the 73 external people

`model/cgmacros/monitor.py`:
- **How it works:** feed one reading at a time and it returns an alarm with its reasons. A 30-min lockout and a reset
  after gaps are the same as in every replay test.
- **Parity check:** it reproduces the offline scorer's alarm episodes exactly (272 of 272 on 12 people, 0
  differences).
- **Unit tests:** `tests/model_cgmacros/test_monitor.py`.

**Which alarm should it use?** Pre-registered in `7994f4e`, the 73 people's second use, lows only. The event model
(§8), checked every 5 min, was compared with alert20 (§9) using `replay.py`'s scoring unchanged:

| 73 external people, 308 events (305 lows) from 51 people | Timely (>= 15 min) | Its random baseline | Median lead | False alarms/day |
|---|---|---|---|---|
| Event model | .188 | .121 | 10 min | 0.63 |
| **alert20** | **.724** | .381 | **25 min** | 3.18 |
| Event model, daytime only | .172 | .123 | 10 min | 0.63 |
| alert20, daytime only | .762 | .418 | 25 min | 3.44 |

- **Pre-registered result: not confirmed.**
  - The event model has fewer false alarms: 2.6/day fewer, CI [2.2, 2.9].
  - But its timely share is far below non-inferior: -54 points, CI [-59, -48].
  - It was trained on CGMacros, where urgent moments are mostly highs, and it does not carry over to lows. On CGMacros
    lows it was also the weakest (§9).
- **alert20 carries over:**
  - Timely .724 vs its random .381, in both datasets (BIG IDEAs .79, Hall .72) and with nights excluded (.76).
  - On CGMacros it was .677 (§9, all 45 people).
  - The cost is about 3 false alarms a day. That is the standing trade-off: §9's attempt to lower it by combining with
    the rule failed its false-alarm limit.
- **Decision:** the loop's default policy is alert20. The event model is an opt-in second policy, and it is not trusted
  for lows.

## 16. Highs on new people: the trend alarm in type 2 diabetes without insulin (ShanghaiT2DM, 15-min CGM)

**Data.** ShanghaiT2DM (Zhao et al., Sci Data 2023, doi:10.1038/s41597-023-01940-7, CC BY 4.0).
- Downloaded through the browser, because figshare refuses scripted downloads. The zip MD5 matched figshare's.
- Kept: recordings with no insulin anywhere (medication summary and every insulin column of the recording). That is 44
  recordings from 42 people, 548 days.
- Readings come every 15 min and are used as they are; nothing is interpolated. Values beyond the sensor's limits
  (39.6 mg/dL floor) are clipped to 40/400.

**Test design.** Pre-registered in `d24a972` (`model/cgmacros/shanghai.py`), the first and only use of these people.
- **Scoring:** `replay.py`'s scoring with the same clock durations at 15-min spacing. An event is >= 2 readings
  (>= 15 min) beyond 250 or 70; the warning window is 60 min; the lockout is 30 min; "timely" means >= 15 min ahead.
- **Alarm:** alert20 uses the last two readings' slope, projected 20 min ahead.
- **Event count:** 97 highs (the docstring's 81 counted 3-reading runs, not the pre-registered 2) and 235 lows.

| 42 people, 548 days | Events (people) | Timely (>= 15 min) | Its random baseline | Warned >= 30 min ahead | False alarms/day |
|---|---|---|---|---|---|
| **Highs** | 97 (20) | **.969** | .098 | .40 | 1.13 (all alarms) |
| Lows | 235 (30) | .847 | .175 | .45 | |
| Lows, excluding 2 people stuck at the sensor floor | 190 (28) | .826 | .139 | | |

- **Pre-registered result: confirmed.** On highs, timely beat the random baseline by 87 points, CI [85, 92].
- **This is the first external evidence on highs.** In type 2 diabetes without insulin, highs build slowly (meals). One
  15-min step before crossing 250, a rising trend projected 20 min ahead almost always crosses, so the alarm nearly
  always fires before the high.
- **Read with care:**
  - **Short leads:** at 15-min spacing the smallest possible lead is 15 min, and the median was exactly that. Only 40%
    of highs were warned >= 30 min ahead.
  - **Not comparable to the 5-min results:** "timely" at 15-min spacing means "warned at the previous reading", so these
    numbers cannot be compared directly with CGMacros (.636 on highs at 5-min spacing).
  - **False alarms** were lower (1.1/day), partly because there are a third as many readings to alarm on.
  - **Not tested here:** the forecasters and the triage rule, which are trained on 5-min data.
- **Combined with §9 and §15:** the trend alarm is the one warning method that has now held up on CGMacros (5 min) and
  on three external datasets: lows in BIG IDEAs and Hall, and highs and lows in ShanghaiT2DM.

## 17. Fewer false alarms: simple rules barely move the trade-off

`model/cgmacros/alarms.py`:
- **Protocol:** committed before the first trial.
- **Dev set:** all 45 CGMacros people. The rules have no fitted parameters, so nobody needs to be held out.
- **Log:** 11 trials, including one crash (a bug on very short segments, fixed and rerun), in
  `model/results/alarm_log.jsonl`.
- **Target:** <= 1 false alarm a day while staying >= 60% timely.

| CGMacros, 45 people | Timely | False alarms/day |
|---|---|---|
| alert20 (current) | .677 | 2.37 |
| **gate** (high alarm only if glucose >= 200, low only if <= 100) | .637 | 1.88 |
| 2 readings in a row | .511 | 1.68 |
| wider margins (+10 / -5 mg/dL) | .561 | 1.66 |
| 2 in a row + gate | .475 | 1.30 |
| 2 in a row + gate + margins | .350 | 0.89 |
| 3 in a row | .269 | 0.96 |
| 2 in a row, projected 30 min | .628 | 3.24 |

**Dev result: no variant met the target.**
- Every rule built only from the glucose stream trades timeliness for false alarms at about the same rate.
- Confirmation over consecutive readings is the costliest, because each extra reading delays the alarm 5 min.

**Locked test of the best trade** (`gate`, pre-registered in `12fa05b`): confirmed, but small.

| | Timely, alert20 -> gate | False alarms/day | Criteria |
|---|---|---|---|
| 73 external people (lows) | .724 -> .711 (diff CI [-.033, 0]) | 3.18 -> 3.00 (saved 0.18 [0.14, 0.22]) | both passed |
| ShanghaiT2DM, 15 min (highs + lows) | .883 -> .883 | 1.13 -> 1.10 | both passed |

- **Why the saving is small:** most of the gate's dev saving was on highs (CGMacros is high-heavy). The low gate (<= 100)
  rarely binds, because lows almost always start from below 100.
- **Decision:** the gate is a safe, tiny improvement, about 6-20% fewer false alarms depending on the population. The
  false-alarm burden stays at about 2-3 a day.
- **Next ideas:** getting to about 1 a day needs something other than stream rules, for example:
  - per-person thresholds learned from the person's own first days;
  - two alert tiers (a quiet "heads-up" and a loud alarm);
  - pairing the alarm with the triage forecast. Note that §9's version of that pairing failed its false-alarm limit.

## 18. Per-person thresholds and two alert tiers

Both were run under a protocol committed before any trial, on CGMacros dev (`pers_*`, `tier_*` in `alarm_log.jsonl`;
7 trials). Scoring was on post-calibration days for every arm.

**Per-person thresholds: dropped on dev.**
- How they worked: the alarm fires only if the current trend is at least as steep as the person's own 75th percentile
  from their first 2 days.
- They did not beat the trade-off: .599 timely at 1.98/day, against the gate's .637 at 1.88. Nothing to test.

**What a "false" alarm really is** (descriptive, CGMacros). After alert20's false alarms:

| What followed | Share |
|---|---|
| Near-high (220-250) | 25% |
| Near-low (70-85) | 30% |
| A large move | 14% |
| Nothing much | 31% |

For the strict tier, "nothing much" was 14%.

**Two tiers, locked test** (pre-registered in `38a7c49`; the 73 external people's fourth use):
- **Heads-up** = alert20.
- **Loud** = gate + 2 readings in a row.

| 73 external people | Timely | Random | False alarms/day | After a false alarm: near-low / nothing much |
|---|---|---|---|---|
| Heads-up (alert20) | .724 | .390 | 3.18 | 79% / 17% |
| **Loud** | .526 | .264 | 2.13 | 84% / 11% |

- **Pre-registered result: not confirmed.**
  - The loud tier's timely share passed (.526 >= .40).
  - Its false alarms did not: 2.13/day against the 1.5 limit.
  - The limit was set from CGMacros dev, where the loud tier ran at 1.19/day.
- **Why: these "false" alarms are mostly near-lows.**
  - In these mostly normoglycemic people, glucose often dips to 70-85 without three readings below 70.
  - 84% of the loud tier's false alarms were followed by such a dip.
  - Only 11% (about 0.24/day) were followed by nothing much.
- **ShanghaiT2DM** (15 min, loud = gate only): no change, 1.10/day and timely .883.
- **Product implication:**
  - Count only alarms followed by nothing much: about 0.5/day for heads-up and 0.24/day for loud on the external
    people.
  - Wording should say "glucose is heading toward low", not "low glucose".
  - The strict event definition (3 readings below 70) labels a correct heads-up about a near-low as false. That is a
    choice of definition, not a model error.

## 19. In the app: `POST /alerts`

`serving/alerts.py` (numpy only, because the serving image ships nothing from `model/`) and `POST /alerts` in
`serving/api.py`:
- **Request:** the patient's recent readings, oldest first, 5 min apart (at least 4; ideally the last 2 h).
- **Response:** the alert at the latest reading (`heads_up`, `loud` or none), its direction, a plain-language reason
  ("Glucose 98 mg/dL and falling; on this trend it would be about 74 mg/dL in 20 min"), the 20-min projection, the
  alerts earlier in the window, and a disclaimer.
- **Wording:** alerts say where glucose is heading, not that a low or high has happened. Most "false" alerts are
  near-misses (§18).
- **Same auth, audit trail and stale-data refusal as `/predict`.** Readings out of time order are rejected.
- **Stateless:** the 30-min quiet period is rebuilt from the readings sent.

The rules are the tested ones:
- **heads-up** = alert20 (§9, §15, §16);
- **loud** = level gate + 2 readings in a row (§17-18).

`tests/model_cgmacros/test_serving_alerts.py` checks that the API copy fires exactly where the research code fires on
CGMacros (every in-range reading of 15 people), plus synthetic streams and the endpoint itself.

Not wired into the app (from §11-14):
- The new forecaster stays research-only. The API's `/predict` still serves the original model.
- Swapping it in needs a serving export of the boosted-tree + Chronos-2 blend, and Chronos-2 needs its own service.

## 20. ChatGPT and Claude can use the alerts: a public MCP server

`agents/public_mcp/` serves two read-only MCP tools, `get_glucose_alert` and `about_glucopulse_alerts`. Setup and the
full threat model are in `docs/AI_CONNECTORS.md`.
- **Auth:** it is an OAuth resource server for an external IdP, with protected-resource metadata and audience-bound
  tokens. It is separate from the internal clinical-agent server, whose tokens are rejected.
- **Safety in the tool contract:**
  - stateless, with caller-supplied readings only;
  - units are explicit;
  - stale data is refused unless asked for as retrospective;
  - "could not check" is an error, never a silent "no alert";
  - error messages never echo input;
  - every result carries the validated population and a not-for-dosing disclaimer.
- **Bugs found while building it:**
  - SDK validation errors quoted injected text back to the model, so all errors are now sanitized.
  - Real CGM exports drift by seconds, which silently broke the 5-min history, so timestamps are now snapped to the
    grid. This also fixes `/alerts`.
  - Chunked oversized uploads crashed the size guard, so bodies are now checked before the handler.
- **Tests:** 38 for this server, including an end-to-end MCP client call and offline checks of the production token
  path.
- **Still needed:** an IdP and a public HTTPS host before ChatGPT or claude.ai can connect.

## 21. Fast falls reach the loud tier sooner

A review of the MCP tool by a second Claude session found that the loud tier needs glucose already at or below 100
(or at or above 200) on two readings in a row, so a fast fall from higher up got only a heads-up even with a deeper
projection than a slow fall that was already loud. Protocol written before any trial (model/cgmacros/alarms.py,
FAST-FALL ESCALATION PROTOCOL): loud = the existing rule OR, from any level, projection out of range and |15-min slope|
>= 1.5 mg/dL/min on two readings in a row.

| | timely (>= 15 min) | false alarms/day |
|---|---|---|
| Dev, 45 CGMacros: existing loud | .475 | 1.30 |
| Dev: + projection < 55 / > 300 | .480 | 1.35 |
| Dev: + fast move (chosen) | .511 | 1.67 |
| Dev: both | .511 | 1.67 (tie; protocol had no tie-break, simpler arm taken) |
| **Test, 73 external people: existing loud** | **.526** | **2.13** |
| **Test: + fast move** | **.542** (gain 95% CI .003 to .033) | **2.26** (+0.13) |

Both pre-registered criteria were met (gain lower bound > 0; at most +0.5 false alarms/day), so `serving/alerts.py`
uses it (parity with the research code is tested). The gain is small. Because a rule needs two readings in a row, a
2 mg/dL/min fall still becomes loud at about 90 rather than at 100; falls of 3 mg/dL/min or faster become loud a
reading sooner. The 73 have been used five times now; the next change to the alarm needs new external data.

## 22. Person-level glycemic state from days of data (negative result)

Idea borrowed from Google's cuffless hypertension work (2026): judge a person's lasting state from data averaged over
days, not one reading. Their waveform method is not replicated (no raw PPG/accelerometer in our development data).
Protocol written before any feature met a label: `model/cgmacros/a1c.py`. Dev = 45 CGMacros (lab A1c); external =
BIG IDEAs 16 + ShanghaiT2DM 84 (first recording per person, >= 5 days CGM).

**Arm A, CGM over days -> HbA1c.** Baseline GMI = 3.31 + 0.02392 x mean glucose (Bergenstal 2018,
doi:10.2337/dc18-1581).

| MAE, A1c points | GMI | LIN (GMI recalibrated, chosen on dev) |
|---|---|---|
| Dev, 45 CGMacros (LOO) | .63 (bias +.55) | .45 (gain CI .05 to .31) |
| **External, 100 pooled (primary)** | **2.04** | **2.44** (GMI - LIN CI -.46 to -.35: FAIL) |
| BIG IDEAs 16 (secondary) | .37 | .34 |
| ShanghaiT2DM 84 (secondary) | 2.36 (bias -2.28) | 2.84 |

The pre-registered primary failed: the recalibrated formula is worse than GMI on the external people. Post hoc (not
pre-registered): in Shanghai the GMI miss grows with A1c (bias -0.36 below 7.3 %, -1.44 at 7.3-9.7 %, -4.99 above
9.7 %), the pattern GMI's own paper describes when glucose has recently been lowered, so the A1c reflects months of
higher glucose than the CGM days show. LIN's steeper slope, learned on steady-state CGMacros people, makes that worse.
Short CGM windows cannot recover an A1c set by the months before them.

**Arm B, wearable heart rate alone -> A1c >= 5.7 (feasibility).** Dev LOO AUC: heart rate .48 (CI .30-.66), age + BMI +
sex .78 (.60-.92), both .75; heart rate adds nothing (gain CI -.09 to +.02). It failed the pre-registered gate, so it
was not run externally.

Nothing ships: the alert tool is unchanged. Deviation logged in the module: the A1c unit guard was widened from 15 to
20 % after it stopped on a real 15.4 % (144.8 mmol/mol) value, before any external prediction was scored.

**Physiology check (post hoc, `model/cgmacros/a1c_kinetic.py`).** A red-cell kinetic model (Xu, Dunn, Ajjan 2021,
doi:10.1177/1932296819897613; K_M 464 mg/dL, reference AGR 65.1 mL/g, lifespan 105 days; reconstructed and tested
against the published worked numbers, doi:10.1111/dom.14763) has a time constant of about 100 days. Simulated: months
at 260 mg/dL, then treatment brings glucose to 140; ten days later the lab A1c is still 9.5 % while GMI on those days
reads 6.7 % (gap 2.8 points; 1.6 after 60 days, 0.4 after 180). In Shanghai's top A1c tertile (median 11.8 %) the lab
A1c implies a steady glucose of about 366 mg/dL, but the CGM recorded a median of 157: the CGM days do not contain the
glucose that set the A1c, so no estimator built on them can recover it (the kinetic model misses Shanghai as badly as
GMI: MAE 2.49 vs 2.36). In the two populations without that history problem, the kinetic model with the
no-diabetes reference AGR, fitted to nothing here, was closer than GMI (CGMacros MAE .53 vs .63; BIG IDEAs .32 vs .37,
bias -.10 vs +.32). Seen after the fact on data already used, so this is a hypothesis for new data, not a result.

## 23. Alarm round 3: Libre (15-min) data, and fewer loud alarms (not found)

Protocol written before any trial: `model/cgmacros/alarms_r3.py`. Serving and the MCP tool now accept 15-minute
readings (e.g. Libre history): when there is no 5-minute run they use a run of readings 13-17 min apart, exactly the
rule tested on ShanghaiT2DM (parity with the research code is tested on all of ShanghaiT2DM).

**B. Loud tier at 15 min.** Dev on CGMacros' own Libre Pro stream (45 people, 15-min readings recovered by
`sources/cgmacros.py`). One arm (gate or fast move on one reading) turned out identical to the heads-up: at 15-min
spacing a projection out of range from >= 100 already implies >= 1.5 mg/dL/min, a design flaw seen only in the dev run.
The other (the same on two readings in a row) met the dev rule and was tested once on ShanghaiT2DM (42 non-insulin):

| 15-min data, ShanghaiT2DM | timely (>= 15 min) | random baseline | false alarms/day |
|---|---|---|---|
| heads-up (shipped since §16) | .883 | .164 | 1.13 |
| **loud, 2 readings in a row** | **.361** (gain CI .245-.365) | .058 | **0.37** |

Confirmed, so it is served. It is a late confirmation (15 min after the heads-up), weaker than the 5-min loud tier;
the tool tells people to act on the heads-up.

**A. Fewer loud false alarms (5-min).** New external data: Jaeb CGMND (healthy adults, Dexcom; 107 adults, 869 days,
433 low events, no highs), kept sealed for this. Arms on CGMacros dev against the shipped loud rule (.511 timely at
1.67/day, reproduced exactly): projection 10 mg/dL past the line .345 at 0.92/day; three readings in a row .269 at
0.95/day. Both cut ~0.7 alarms/day but lose far more than the allowed 0.03 timely, so none was eligible: the loud rule
is unchanged and CGMND remains unused, available for a future attempt. Simple stateless filters trade warnings for
quiet almost one for one; fewer false alarms needs more information than the trend (meals, time of day, history).

## Limits

- Population: adults without insulin therapy; nothing here transfers to type 1 diabetes.
- Triage forecasts assume no new meal; an unlogged meal is invisible (the `history_only` forecast partly offsets this).
- CGMacros timestamps are randomly shifted by the dataset authors, so one time zone (`UTC`) is used.
- Sending case packets to a third-party LLM should be checked against the CGMacros licence (CC BY-NC-SA) and any data-use terms first.
- One live model (`gpt-4o-mini`), one run, 98 cases, 18 urgent from 3 people. On this evidence the panel does not
  beat the rule, a single agent, or the max(rule, monitor) baseline. A stronger model, calibrated prompts for the
  "routine" level, or more urgent cases could change that; none has been tested.
