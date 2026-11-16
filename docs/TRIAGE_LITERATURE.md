# Triage panel: related work

What the literature says about the questions this project's results raised. "Read" means the passages cited were
read from the full text; "abstract" means only the abstract was seen. Searched 2026-11-16.

## 1. Can a classifier predict a glycemic event 60 minutes ahead?

- **Duckworth et al., J Diabetes Sci Technol (2022), doi:10.1177/19322968221103561.** *Read.* Hypoglycemia (< 70)
  and hyperglycemia (> 270) up to 60 min ahead, young adults with type 1 diabetes. A threshold heuristic reached
  AUROC 0.811 / 0.733; logistic regression 0.930 / 0.862; XGBoost 0.998 / 0.989. Current glucose was the most
  important feature, then time of day and the largest recent rise/drop. Moments already below or above the
  thresholds were removed so they could not inflate the metrics. *Relevance:* trees beat thresholds and linear models,
  as here; but in T1D with week-long history features, and apparently not split by person, so the AUROCs are not
  comparable to our leave-one-person-out 0.88–0.94. Unlike them, our triage pool keeps moments already out of range
  (177 of 5,590): a triage system must handle them, but both the rule and the classifier get them right for free,
  which narrows any difference between the two.
- **Yang et al., JMIR (2026), doi:10.2196/100653.** *Abstract.* Meta-analysis of 270 glycemic-prediction studies:
  pooled event-prediction AUROC 0.926, but 24 of 28 assessed units at high risk of bias; external validation in 4%,
  calibration in 4.8%, decision-curve analysis in 2.6%. *Relevance:* a high AUROC in this field is common and weak
  evidence on its own; matched-operating-point comparisons and per-person intervals are the unusual part.
- **Chou, Soriano, Lumbreras (2026), pmid:42019516.** *Abstract only (full text unavailable).* CGM slope used to
  classify impending hyper/hypoglycemia in 16 adults. Supports `change_last_15min` as a core feature.

## 2. Work on CGMacros itself

- **Das et al., Scientific Data (2025), doi:10.1038/s41597-025-05851-7.** *Abstract.* The dataset paper: 45 adults
  (15 healthy, 16 prediabetes, 14 T2D), two CGMs, Fitbit, meal macros and photos, ~10 days each.
- **Sun et al., "GlucoAgent" (2026), arXiv:2611.04126.** *Read (appendices).* An LLM agent that writes and runs
  analysis code per forecast. On 9 held-out CGMacros participants (seeded split by person, 1,903 origins one hour
  apart), RMSE averaged over 30–120 min is 22.5 mg/dL vs 23.5 for Chronos-2; at 60 min 22.1. Its gain from meal and
  activity context is concentrated at 60 min after meals. *Relevance:* the closest published comparison for our
  forecaster (T+60 RMSE 23.07 on 7 held-out people), but with different people, origins and history length
  (GlucoAgent uses up to the full multi-day history; ours uses 2 h), so the numbers are context, not a ranking. It
  shows an LLM helping when it *computes* over the data, not when it judges a summary.
- **Zhang et al. (2026), arXiv:2609.11872.** *Abstract.* Time-series foundation models on CGM; dietary context from
  CGMacros cut post-meal RMSE ~15%. Same direction as our ablation (meals help most in the meal hour).

## 3. LLM triage: direction of errors matters more than accuracy

- **Linzmayer et al. (2026), doi:10.64898/2026.03.18.26348695.** *Read.* Nine frontier models on a 960-vignette
  triage benchmark: under-triage 0–12.3%, over-triage 9.4–36.9%, and **under-triage was uncorrelated with accuracy**
  (Spearman ρ = −0.05). Modal answer over 10 samples per vignette. *Relevance:* matches what we saw (our panel cut
  under-triage while its accuracy fell) and supports reporting under- and over-triage separately, plus repeated
  sampling (our run-to-run disagreement was 7–10%).
- **Wong & Wong, JMIR (2026), doi:10.2196/82026.** *Abstract.* Retrieval of guidelines plus similar past cases cut
  LLM over-triage from 28.8% to 12.7% without raising under-triage. *Relevance:* a candidate fix for over-alarming,
  if similar past moments of the same person can be retrieved without leaking the future.

## 4. Do multi-agent panels help?

- **Wang et al., "Silence is Not Consensus" (2025), arXiv:2505.21503.** *Read (intro, §3).* In two published
  medical multi-agent frameworks, agents converged without real debate in over 61% of cases, and this "silent
  agreement" caused 62–91% of their errors. *Relevance:* agents built on the same model share errors; four
  `gpt-4o-mini` panelists are not four independent opinions. Our "most cautious panelist wins" rule turns that
  correlation into systematic over-triage.
- **Yuan et al. (2026), arXiv:2603.04421.** *Abstract.* Mixed-vendor panels beat single-vendor ones in diagnosis,
  attributed to complementary rather than shared biases. *Relevance:* if a panel is kept, use different models.
- **Kim et al., "MDAgents" (2024), arXiv:2404.15155** and **Misaghi et al. (2026), arXiv:2603.21447.** *Abstracts.*
  Report panel gains on medical QA and ophthalmology vignettes (multiple-choice or diagnosis), not on outcome-labelled
  triage of time series. Gains in those settings do not transfer by default; ours did not.

## 5. Let deterministic code decide; let the LLM explain

- **Bateni & Abdolali, "Language Models as Interfaces, Not Oracles" (2026), arXiv:2606.19183.** *Read.* Pediatric
  appendicitis. An LLM extracts features, deterministic checks validate them, XGBoost decides. Internally it missed 2
  cases vs 7–40 for the frontier and medical end-to-end LLMs (o3-pro, GPT-5.5, Claude Opus 4.7, MedGemma); one weak
  model missed only 1 by calling 98% of children positive. End-to-end LLMs had unstable
  sensitivity/specificity trade-offs, were sensitive to sentence order, and sometimes broke the output format.
- **Cheng, Han, Sun, "Think Fast, Talk Smart" (2026), arXiv:2605.29652.** *Read.* Wearable sleep insights over 280
  user-nights and six models. Handing one analysis layer to an LLM raised numeric error from 0.7% to 16.9%
  (comparison) or unsupported attribution to 24.5%; the rule they draw is "let code own recurring analysis, and let
  LLMs express verified facts within bounded interfaces." *Relevance:* the same split our results point to, and the
  same mechanism as our grounding check (which removed ~45% of agent claims).

## 6. Warn early from the CGM stream: published alert and meal-detection algorithms

- **Harvey, Dassau, Zisser, Seborg, Doyle, J Diabetes Sci Technol (2014), doi:10.1177/1932296814523881.** *Read
  (equations, from PMC).* The Glucose Rate Increase Detector (GRID): noise-spike filter (3 mg/dL/min), first-order
  low-pass filter, 3-point Lagrangian derivative, detection if filtered glucose > 130 mg/dL and the last 3 rates
  > 1.5 or the last 2 > 1.6 mg/dL/min (tau_F = 6 min). Tuned on type 1 diabetes. *Relevance:* implemented exactly in
  `model/cgmacros/replay.py`; the trigger for a meal prompt or re-triage.
- **Hochsmann et al., Scientific Reports (2026), doi:10.1038/s41598-026-50699-5.** *Read (methods, results).* Nine
  published CGM-only meal detectors on 16 healthy adults, free-living, with a shared protocol: 120-min match
  window, 120-min lockout after a true positive. GRID: 70.4% sensitivity, 0.26 false positives/day, 37 min to
  detection; rate-of-change detectors are fastest but least sensitive. *Relevance:* our matching rules and our
  reference numbers; GRID's healthy-adult tuning is our secondary.
- **Dassau, Bequette, Buckingham, Doyle, Diabetes Care (2008), doi:10.2337/dc07-1293.** *Abstract only.* The first
  CGM meal-detection algorithm (voting on rate and acceleration): meals detected about 30 min after eating began,
  when glucose was ~21 mg/dL above baseline. *Relevance:* the time scale: a meal is visible on the CGM within ~30 min,
  long before a spike peaks, which is why checking every 5 min matters more than a better one-off model.

## 7. Physiology-structured forecasting (the meal what-if)

- **Miller, Foti, Fox (Apple), MLHC 2020, arXiv:2008.02852, "Learning Insulin-Glucose Dynamics in the Wild."**
  *Read (methods, results).*
  - **Method:** the UVA/Padova type 1 diabetes simulator, with its parameters (insulin sensitivity, endogenous glucose
    production) allowed to vary over time, driven by a latent sequence model ("DTD-Sim").
  - **Data:** 2 people with type 1 diabetes, CGM plus pump data from HealthKit, about 150 days each.
  - **Accuracy:** LSTM and ARMA were best at 5-30 min. DTD-Sim beat the LSTM by 5% at 1 h, 27% at 2 h and 36% at
    6 h.
  - **What-if forecasts:** the LSTM ignored insulin; DTD-Sim responded plausibly to both carbs and insulin.
  - *Relevance:* our meal what-ifs have the LSTM's problem (meal forecasts dipping below no-meal; 30 g above 60 g).
    `model/cgmacros/hybrid.py` keeps only the meal part, since CGMacros has no insulin. Expect a fix to the what-if
    forecasts, not much accuracy gain at 60 min.
- **Hovorka et al., Physiol Meas (2004), doi:10.1088/0967-3334/25/4/010.** *Abstract; equations via Pompa et al.
  2021, doi:10.1371/journal.pone.0257789 (read).* Gut absorption is a two-compartment chain:
  `U_G(t) = D2(t) / tau_D`, `tau_D = 40 min`, carbohydrate bioavailability `A_G = 0.8`. This is the fixed meal-curve
  shape in the hybrid.

## 8. Better 30-60 min forecasts: what has worked elsewhere (searched 2026-11-16, abstracts unless noted)

- **Zhang et al. (2026), arXiv:2609.11872.** *Abstract.* Eight CGM datasets (T1D, T2D, no diabetes).
  - Zero-shot foundation models did not beat strong task-specific baselines (Elastic Net, PatchTST).
  - Lightly fine-tuned Chronos-Bolt cut RMSE 8.6-18.2% in the non-diabetes/T2D cohort.
  - On CGMacros, residual fusion of meal context cut post-meal RMSE about 15%.
- **GlucoFM-Bench (2026), arXiv:2606.06881.** *Abstract.* 15 datasets, 1,117 people. Chronos-2 and TimesFM zero-shot
  came within 5% of the best fully trained model. With plenty of data, a lightweight LSTM was best, by 4-21%.
- **GlucoAgent (2026), arXiv:2611.04126** (section 2). On CGMacros: Chronos-2 averaged 23.5 RMSE over 30-120 min.
  Generated code that retrieves similar past meals and back-tests per-person models reached 22.5; post-meal 60-min
  RMSE went from 31.0 to 26.5.
- **Retrieval of similar past episodes:**
  - GlyRAG (2026), arXiv:2601.05353, *abstract*: 60-min RMSE on OhioT1DM went from 23.1 (PatchTST) to 20.2.
  - RAFT / "Retrieval Mechanisms Surpass Long-Context Scaling" (2026), arXiv:2605.08217, *abstract*: longer raw
    context hurt (inverse scaling); retrieving relevant past segments helped.
- **Personal traits are learnable quickly.**
  - doi:10.64898/2026.09.14.26362991, *abstract*: 50,463 free-living meals from 992 adults without diabetes. Two
    stable per-person response axes; 3 days of data approximated 14.
  - Incrementally retrained LSTM (PMC11999170), *abstract*: adapting to each person's own data cut 30-min RMSE
    14.6 -> 10.2 (T1D).
- **Reproducibility warning:** PMC13541153, *abstract*. 8 deep models re-run on 128 people: errors depended strongly on
  each person's glycemic control, and generalization across datasets was limited.
- **Ranking for this project (cheapest, best-supported first):**
  1. Strong simple baselines: ridge/Elastic Net or boosted trees, direct multi-horizon.
  2. Per-person adaptation from the person's own earlier days. Our personalized meal gain already helped (§10).
  3. Chronos-2 / Chronos-Bolt, zero-shot then lightly fine-tuned.
  4. Retrieval of the person's similar past windows.
  5. Ensembles.

## What this suggests for the panel

1. The level should come from a rule or a small classifier (section 5 of `TRIAGE.md`), not from the LLM vote.
2. The LLM's job is the grounded explanation for the clinician; grounding already enforces the bounded interface.
3. If agents keep any say in the level, an escalation must be backed by a grounded claim that meets the level's
   definition, and same-model panelists should not be treated as independent votes.
4. Report under- and over-triage separately, at matched operating points, with per-person intervals and repeated
   sampling.
