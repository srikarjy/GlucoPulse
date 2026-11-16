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

## What this suggests for the panel

1. The level should come from a rule or a small classifier (section 5 of `TRIAGE.md`), not from the LLM vote.
2. The LLM's job is the grounded explanation for the clinician; grounding already enforces the bounded interface.
3. If agents keep any say in the level, an escalation must be backed by a grounded claim that meets the level's
   definition, and same-model panelists should not be treated as independent votes.
4. Report under- and over-triage separately, at matched operating points, with per-person intervals and repeated
   sampling.
