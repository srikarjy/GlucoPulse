# FlowCast — Cardinal Rules

These rules are locked. Do not revisit, redesign, or expand scope without explicitly flagging the specific rule being broken and why.

## 1. No new architecture without a working failure first
The only valid reason to change the design is a real bug or limitation hit while building, not a new idea that sounds more impressive on paper.

## 2. Nothing claimed unless it's actually running on real data
No metric, no percentage, no "it detects X" goes into code comments, README, commit messages, or anywhere else unless it came out of an actual run against a real `trace.txt` or `multiqc_data.json`. Not estimated. Not "should work."

## 3. Stack is frozen for v1
Go, standard library parsing only, rule based classifier (not ML), Claude API with structured JSON output. Rust, FFI, local inference, vector databases, and Docker are not up for debate for v1. If one becomes genuinely necessary later, it needs a real measured reason (an actual latency number that's actually a problem), not a preference.

## 4. Every classifier rule traces back to the reasoning document
If a rule can't be traced to a specific line in the one page causal biology/technical reasoning document, the rule doesn't exist yet. No inventing thresholds.

## 5. Every narrator claim carries a confidence tag, and "Unknown" is a valid, expected, frequent output
The narrator must never guess at causation it hasn't measured. If it stops saying Unknown when it should, that's a regression, not progress.

## 6. One real end to end run before any supporting infrastructure
No eval harness, no CI, no OpenTelemetry, no observability work until one full real diagnosis exists on real data. Building infrastructure around something unproven is how this kind of project dies quietly.

## 7. This project is one signal, not a silver bullet
FlowCast does not get to be "the project that gets every biotech callback." It's one well built, narrow, honestly scoped piece of a larger portfolio. Keep it scoped to what it actually is.

## v1 Locked Scope (for reference)
A Go CLI that parses Nextflow's `trace.txt` and MultiQC's `multiqc_data.json` from one real nf-core/rnaseq run, applies a rule based failure classifier built from real QC fields, and feeds a Claude API narrator (structured JSON output: claim, confidence_tag, evidence_source) that only makes Observed/Reported/Unknown tagged claims and refuses unmeasured causal claims.

**Explicitly not competing with:** nf-prov / BCO / WRROC provenance capture, or AWS HealthOmics. FlowCast's differentiator is the honest, confidence tagged failure narration layer, not provenance capture itself.

**Explicitly excluded from v1:** Rust, FFI, local model inference, weblog live streaming, vector DB, Docker, resource prediction, recommendation engine.

## After v1 ships
Once v1 is complete and demonstrably working end to end on real data, scope for further additions (beyond the v1 exclusions above) can be discussed — but only after v1 is done, and only following the same rules above (real failure first, real data, traceable rules, confidence tags, no premature infrastructure).
