# Question Log

Running log of questions asked during GlucoPulse development, and the answers/decisions that came out of them. Purpose: avoid re-litigating a decision already made, and give a paper trail for interview prep ("why did you choose X" should point here).

Add new entries at the top. Keep answers short — link to code/commit instead of restating it.

Format:

```
## YYYY-MM-DD — Question
**Context:** what prompted it
**Answer/Decision:** what was decided and why
**Status:** open | resolved
```

---

## 2026-07-13 — Phase 3 feature set, recompute strategy, and data-quality gate design
**Context:** Before writing the PySpark batch job, three things needed deciding against real numbers, not guessed at: what features it computes, whether it recomputes full history or processes incrementally, and what the data-quality gate actually checks. Pulled the real gap distribution from all 3 loaded patients (34,276 intervals): median/p95/p99 gap is exactly 5 min (normal cadence), only 27 intervals exceed 60 min, and two gaps stand distinctly apart from everything else -- 1,384 min (Subject 3) and 600 min (Subject 1) -- while a cluster of 125-260 min gaps recurs routinely across all three patients (~1 every 4-5 patient-days, consistent with routine sensor changes, not failures). All full calendar weeks have >=93.8% of expected readings (>=1892/2016).
**Answer/Decision:**
- **Feature set:** resample each patient's series to the 5-min grid first, then compute rolling windows on the resampled series (avoids two different notions of "window" -- raw-log windows vs. grid windows -- and matches the forecasting cadence). Computes: delta from previous reading, rolling mean/std of glucose, and `time_since_last_bolus`/`time_since_last_carb`. Cold-start (no prior event yet in a patient's history) is represented as an explicit `has_prior_bolus`/`has_prior_carb` boolean flag plus NaN for the time value -- not a sentinel constant, since sentinels get silently learned as spurious signal by tree models and corrupt loss for NN models if a preprocessing step forgets to mask them.
- **Recompute strategy: full history, every run.** At the current scale (N=3 patients, ~40 days each, a few hundred thousand rows), a full Spark recompute finishes in seconds to low minutes -- incremental batching with a lookback buffer solves a cost/time problem this project doesn't have yet, at the price of real correctness surface area (state tracking, buffer-window sizing) that can't be properly stress-tested at this scale anyway. Full recompute pairs naturally with upsert-on-conflict (`ON CONFLICT (patient_id, time) DO NOTHING` style, same discipline as the consumer) -- no append-only or dedup-pass logic needed. **Scaling note (see docs/PROBLEMS.md):** at production scale this becomes incremental with a lookback buffer sized to the max observed sensor gap (currently 23h/1384min, Subject 3) plus margin, to correctly handle late/backfilled data -- not implemented here, a deliberate scope decision, not an oversight.
- **Features table key: `(patient_id, time)`** -- same grain as `cgm_readings`, not a separate `window_end_ts` (redundant once features are computed on the same resampled grid as the raw table) and does **not** include `model_version` (that's a Phase 4 serving/prediction-logging concern -- multiple model versions read the same features, versioning the features themselves by model would be backwards).
- **Data-quality gate, per patient-week (not per-patient-full-history, so one bad week doesn't permanently disqualify a patient's other weeks under full recompute):**
  - Single-gap threshold: **180 minutes.** Chosen specifically because it's the real break in the distribution -- only the two genuine outliers (1384, 600 min) exceed it; the routine 125-260 min sensor-change gaps stay under it. This is the "undetected sensor gap" scenario from README's opening problem statement, deliberately distinct from a normal sensor change.
  - Count threshold: **<85% of expected readings for that week**, prorated by days-actually-observed for partial boundary weeks (a patient's first/last calendar week is naturally partial because their data starts/ends mid-week, not real dropout -- excluded from being flagged for that reason alone).
  - A patient-week failing either check is **excluded from that run's feature output and logged, not raised** -- mirrors the DLQ separation-of-concerns principle (a single patient's bad week is a localized, expected-to-happen data issue, not a systemic pipeline failure, the same reasoning that kept `cgm-dlq` from hard-blocking the whole ingestion consumer).
  - **Circuit breaker:** a downstream check counts the exclusion rate across the run; only *that* check is allowed to hard-fail the Airflow DAG task (e.g. if more than half the patient-weeks in a run are excluded, that's a systemic signal -- schema change, broken ingestion upstream -- not 25 independent bad sensor days). The specific fraction is a placeholder tied to fleet size (arbitrary at N=3 patients), to be revisited once the patient count is large enough for a percentage to mean something -- documented as a known placeholder, not silently hardcoded.
**Status:** resolved

---

## 2026-07-13 — What does the ingestion consumer (`ingest.py`) actually write, and what's the full DLQ policy?
**Context:** `docs/BLUEPRINT.md`'s original Phase 2 plan said the consumer "computes the real feature set (delta, rolling stats)" — but `docs/PROBLEMS.md` already flagged that as risky for a plain Python consumer (per-patient rolling state lives in-process, lost on restart unless recomputed from TimescaleDB). Separately, `docs/PROBLEMS.md`/`docs/BLUEPRINT.md` both left the DLQ policy only partially decided: conflicting-timestamp groups were resolved (route to `cgm-dlq`), but malformed messages and out-of-range glucose values were explicitly deferred to "when the consumer is actually written."
**Answer/Decision:**
- **Consumer scope: raw-only.** `ingest.py` writes raw glucose + covariates to `cgm_readings`, computing no derived features. This keeps the consumer stateless — every message's DB write is independent of any other message, which is also what makes `ON CONFLICT DO NOTHING` sufficient for at-least-once dedup. Delta/rolling-stat computation is deferred entirely to Phase 3's PySpark batch job over the full raw history, consistent with the stack table's justification for PySpark ("batch feature engineering across full patient history is a legitimate distributed workload") — computing it twice (inline in the consumer, then again in Spark) would duplicate that justification, not support it.
- **Offset commit strategy: commit only after a successful DB write**, at per-message granularity (not batched). Rationale: commit-then-write risks silent, unrecoverable data loss if the consumer crashes between the offset commit and the DB write — unacceptable for a clinical pipeline where a lost glucose reading has no replay path. Commit-after-write means at-least-once (occasional duplicate reprocessing on crash/rebalance), made safe by `ON CONFLICT (patient_id, time) DO NOTHING` (the existing `PRIMARY KEY (patient_id, time)` already provides the uniqueness). Per-message (not batched) commit granularity was chosen because ingestion isn't the throughput bottleneck (Spark's batch stage already owns heavy processing) — batching commits would only enlarge the crash-replay window for no real throughput benefit here.
- **DLQ policy — now fully decided, three separate topics/failure classes** (deliberately not conflated, since each answers a different question and implies a different downstream consumer/intent):
  | Topic | Failure class | Trigger | Consumer intent |
  |---|---|---|---|
  | `cgm-parse-errors` | Structural — message doesn't parse | malformed JSON, missing/wrong-typed field | alerting (engineering bug / upstream schema drift) |
  | `cgm-dlq` | Ambiguous — two valid values disagree at the same timestamp | conflicting same-second groups (existing, unchanged from the 2026-07-10 decision above) | human review, no auto-coalescing |
  | `cgm-implausible` | Invalid — single well-formed value outside physiological range | CGM < 40 or > 400 mg/dL | archival/monitoring, not urgent alerting |

  The 40–400 mg/dL gate isn't an arbitrary threshold — it matches the Dexcom G6's own documented measurement range (the sensor itself doesn't report values outside that band), so it's anchored to the hardware, not invented.
**Status:** resolved

---

## 2026-07-10 — Are all 25 AZT1D CSVs the same schema?
**Context:** Before writing the producer's parser, checked column headers across all 25 files.
**Answer/Decision:** No — 24 of 25 share identical columns/order (`EventDateTime,DeviceMode,BolusType,Basal,CorrectionDelivered,TotalBolusInsulinDelivered,FoodDelivered,CarbSize,CGM`), but **Subject 14** has a different column order and names its glucose column `Readings (CGM / BGM)` instead of `CGM`. Checked whether this might mix continuous-monitor and fingerstick values (the name suggests it): values are in a plausible continuous range (40–237 mg/dL) on the same 5-minute cadence as every other subject, no distinguishing marker in the row — treated as equivalent to `CGM` for parsing purposes. Parser reads by column name (not position) and aliases `Readings (CGM / BGM)` → `CGM` specifically for Subject 14. Confirmed across all 306,712 rows (all 25 subjects, with this alias applied): glucose value is never blank.
**Status:** resolved

---

## 2026-07-10 — How should the producer handle timestamp-groups where two rows share the exact same second but disagree on value?
**Context:** Inspecting all 25 real AZT1D CSVs (Phase 2 step 1, before writing any parser) found `EventDateTime` is a per-second event log, not a clean 5-min grid — CGM ticks, basal updates, and bolus events interleave, and 823 (patient, timestamp) groups have more than one row. Of those: 429 are pure exact duplicates (safe to silently collapse, ~1,705 redundant rows), but **394 have genuinely conflicting values** — e.g. Subject 15 at `2024-01-21 14:06:09` has CGM `226` in one row and `239` in another; Subject 15 at `2024-02-02 17:55:59` has `Basal 0.708` vs `0.75`.
**Answer/Decision:** Conflicting groups are **not** silently resolved (no keep-first/keep-last coalescing). They get routed to `cgm-dlq` as a data-quality anomaly, same as any other malformed record — visibility over guessing which value is "right." Exact duplicates (all fields identical) are silently deduplicated, since there's no information loss in doing so.
**Status:** resolved

---

## 2026-07-10 — Why did we switch from OhioT1DM to AZT1D?
**Context:** OhioT1DM requires a gated institutional request (~1 week turnaround, email to razvan.bunescu@charlotte.edu). A single OhioT1DM file was found re-hosted on Kaggle (unofficial mirror, outside the dataset's own access process) — using it would mean building the ingestion story on data whose provenance we can't honestly stand behind, for a project whose whole pitch is "nothing claimed unless real and properly obtained."
**Answer/Decision:** Switched to **AZT1D** (arXiv:2506.14789, Mendeley DOI `10.17632/gk9m674wcx.1`, CC BY 4.0, no application required). 25 T1D patients on automated insulin delivery, Dexcom G6 Pro CGM at the **same 5-minute interval** as OhioT1DM (so the existing T+30/T+60 forecasting task framing didn't need to change), CSV format. Considered HUPA-UCM first (also open, Mendeley) but rejected it — its 15-minute CGM interval would have broken the 5-minute-grid forecasting task already written into README/memory.
**Correction (2026-07-10, after actually downloading and measuring the data):** the paper's abstract cites 320,488 CGM readings and "26 days average" duration — neither holds up against the real files. Actual measured totals: **306,712 CGM rows**, duration **28–49 days per patient, avg ~42 days (~6 weeks)** — the "26 days" figure was this assistant's misreading of the paper's "26,707 total hours" stat, not a real per-patient average (26,707 ÷ 25 ÷ 24 ≈ 44.5 days, consistent with the measured range). Net effect: AZT1D's per-patient window is much closer to OhioT1DM's 8 weeks than originally represented — the "shorter history" tradeoff was overstated. Lesson: don't repeat a paper's abstract numbers into project docs without measuring the actual downloaded files. The previously-cited persistence-baseline RMSE (~15–25 mg/dL at T+30) was measured on OhioT1DM specifically and still does not carry over — re-measure once AZT1D is actually flowing through the pipeline.
**Status:** resolved

---

## 2026-07-10 — Why does TimescaleDB map to host port 5544 instead of 5432?
**Context:** Phase 1 `docker compose up` failed with `address already in use` on 5432, then again on 5433.
**Answer/Decision:** The dev machine has two native PostgreSQL installs running outside Docker (v17 on 5432, v18 on 5433 — `/Library/PostgreSQL/17` and `/Library/PostgreSQL/18`, confirmed via `netstat -anv`). Rather than stop/reconfigure those existing services, the TimescaleDB container's host-side port mapping was changed to `5544:5432` — the container's internal port stays 5432 (other containers, e.g. Grafana, still reach it at `timescaledb:5432` inside the Docker network), only the host-exposed port changed. Any host-side client (psql, a GUI tool) must connect on `localhost:5544`.
**Status:** resolved

---

*(no earlier questions — first entry above this line)*
