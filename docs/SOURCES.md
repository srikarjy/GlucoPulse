# Multi-source ingestion (`sources/`)

Turns health data from different apps and devices, in different formats, into
one typed record and then into FHIR R4 Observations. Every value either becomes
a normalized measurement or lands in a quarantine report with a reason. Nothing
is silently fixed, merged or dropped.

```
Fitbit JSON ─┐
Apple Health XML ─┤                                     ┌─► FHIR R4 Bundle (Observations)
CGMacros CSV ─┼─► adapter ─► RawMeasurement ─► normalize() ─┼─► typed MeasurementRecord (NDJSON)
Any CSV/JSON + TOML ─┘   (structure only)     (units, time,     └─► quarantine report (reason per record)
mapping                                      types, dedup)
```

```bash
python -m sources.ingest --format cgmacros data/cgmacros/csv/CGMacros-001.csv --tz UTC --out bundle.json
python -m sources.ingest --format fitbit heart.json --patient p-07 --tz America/Chicago --out b.json
python -m sources.ingest --format apple export.xml --patient p-07 --out b.json
python -m sources.ingest --format mapped --mapping sources/mappings/example_logbook_csv.toml export.csv \
    --patient p-07 --tz Europe/Berlin --out b.json --records-out records.ndjson --quarantine-out q.json
```

Exit codes: `0` ok, `2` input/usage error, `3` nothing usable (every record quarantined).

## Formats

| Format | Adapter | Notes |
|---|---|---|
| Fitbit Web API JSON | `fitbit.py` | Intraday heart rate and steps, sleep log v1.2 stages (`levels.data`), SpO2 minutes. Times have no offset, so `--tz` is required. |
| Apple Health `export.xml` | `apple_health.py` | Streamed with `defusedxml` (entity attacks refused, memory bounded). 11 record types. Source names are mapped to device types, never copied (Apple's default is "<Owner Name>'s iPhone"). |
| CGMacros CSV (PhysioNet) | `cgmacros.py` | Two CGMs, Fitbit, meals. Nine header layouts handled. Glucose interpolation undone (see below). |
| Any app's CSV or JSON | `mapped.py` + a TOML mapping | No code per app. Column/JSON path to kind, unit and device. See `sources/mappings/`. |

## Measurement types

Every kind declares a value type. Quantities have a canonical UCUM unit and a plausible
range; codes have an allowed list plus synonyms; period kinds also carry an end time.

| Kind | Code | Value |
|---|---|---|
| CGM glucose / blood glucose | LOINC 99504-3 / 2339-0 | mg/dL (mmol/L converted) |
| Heart rate, respiratory rate | LOINC 8867-4, 9279-1 | /min |
| HRV (SDNN) | LOINC 80404-7 | ms |
| SpO2 | LOINC 59408-5 | % |
| Blood pressure systolic / diastolic | LOINC 8480-6 / 8462-4 | mm[Hg] |
| Body weight | LOINC 29463-7 | kg (lb converted) |
| Steps / calories burned | LOINC 55423-8 / 41981-2 | {steps} / kcal |
| METs | local | {MET} |
| Activity intensity | local code | sedentary, light, moderate, vigorous |
| Sleep stage | local code, period | awake, light, deep, rem, asleep, restless, in_bed |
| Insulin bolus / basal | local | [IU] |
| Meal type | local code | breakfast, lunch, dinner, snack |
| Meal energy, carbs, protein, fat, fiber, sugar, % eaten | local | kcal, g, % |

Local codes (`urn:glucopulse:code`) are used where no LOINC term means the same thing.
Every LOINC macronutrient intake code is a 24-hour total (`g/(24.h)`), which a single
logged meal is not. Insulin delivery would ideally be a FHIR `MedicationAdministration`;
it is an Observation here so every source lands in one uniform stream.

## Normalization rules

| Problem | Rule | Quarantine reason |
|---|---|---|
| Text that isn't a plain decimal (`"1e2"`, `"7_2"`, booleans, non-ASCII digits) | Rejected, never coerced | `non_numeric` |
| Unknown unit | Rejected, never guessed | `unsupported_unit` |
| Unknown category label | Rejected after synonym mapping | `unknown_code` |
| Naive timestamp and no `--tz` | Never assumed UTC | `missing_timezone` |
| Local time in a DST gap / overlap | Never shifted | `nonexistent_local_time` / `ambiguous_local_time` |
| Time outside 1971-2100, or that overflows on conversion | Rejected | `invalid_timestamp` |
| Value outside the kind's range | Rejected | `implausible_value` |
| Period with no end, or end not after start, or longer than 24 h | Rejected | `missing_end` / `invalid_period` |
| Same device, same sample, different values | Both quarantined, never pick one | `conflicting_values` |
| Same device, same sample, same value | Kept once | (counted as `duplicates_dropped`) |

"Same sample" means within half the device's native interval (Fitbit 30 s, Dexcom 150 s,
Libre 450 s), measured from the first reading of the cluster, so it does not depend on
clock alignment. Devices with no known interval, and all period kinds, only match on an
identical start time. Two different devices disagreeing (Dexcom vs Libre) is not a
conflict; they are separate streams.

## Recovering real CGM readings from CGMacros

The published CGMacros glucose columns are linearly interpolated to one value per minute.
Interpolated values are not measurements, and treating them as readings turned 1,178,865
records into false conflicts on the first real run.

Between two real readings, every interpolated point lies on a straight line, so its second
difference (`v[i-1] - 2·v[i] + v[i+1]`) is zero. `recover_native_readings` keeps the
points where the slope changes ("knots") plus each run's endpoints, then restores readings
that happened to lie on a straight stretch by taking the series value at every native
interval between two knots whose gap is an exact multiple of that interval.

Measured on all 45 CGMacros participants (v1.0.0):

| | Before | After |
|---|---|---|
| Quarantined records | 1,178,865 | 1,050 |
| Interpolated points set aside | — | 1,143,301 |
| Recovered readings: certain (slope changes, run ends) | — | 146,625 |
| Recovered readings: refilled on straight stretches | — | 25,103 |
| Dexcom G6 readings per day (sensor: 288) | — | 280 |
| Libre Pro readings per day (sensor: 96) | — | 94 |
| Most common gap between recovered readings | — | 5 min (Dexcom), 15 min (Libre) |

Remaining quarantines: 889 glucose `conflicting_values` (knots 2-4 minutes apart, about
0.6% of readings, left quarantined rather than tie-broken) and 161 `implausible_value`
(131 meals logged as more than 100% eaten, 24 fiber values above 150 g, and a few others).

A refilled reading is indistinguishable from an interpolated value when the sensor
skipped a reading inside a straight stretch, so the 25,103 refilled readings are an upper
bound on true readings in those stretches; they are counted separately for that reason.

The recovery is only applied to CGMacros glucose columns, where the interpolation was
verified. A genuinely per-minute sensor would lose collinear points if it were applied there.

## Mapping any app (`--format mapped`)

```toml
name = "example-logbook-csv"
format = "csv"                       # or "json" (then: records = "data.entries")

[time]
columns = ["Date", "Time"]           # joined with a space
format = "%d.%m.%Y %H:%M"            # or: epoch = "seconds" | "milliseconds"

[[fields]]
column = "Glucose (mmol/L)"          # CSV header, or dotted path in a JSON record
kind = "glucose_blood"
unit = "mmol/L"                      # must be convertible for the kind
device = "meter"                     # stream id, [a-z0-9-] only

[[fields]]
column = "Meal"
kind = "meal_type"                   # code kind: no unit
[fields.values]
"frühstück" = "breakfast"            # source label -> allowed code
```

Mappings are validated when loaded: unknown keys, unknown kinds, units that can't be
converted, label targets that aren't allowed codes, free-text device ids, and missing
`end_column` on period kinds all fail before any data is read. A mapped column missing
from the file is an error, not an empty stream.

## Typed contract

`sources/schema.py::MeasurementRecord` is what downstream consumers (the forecaster,
the triage agents) receive. It re-checks everything normalization guarantees: value type
matches the kind, codes are allowed, quantities are in range and in the canonical unit,
times are timezone-aware, period kinds have a valid end and point kinds have none, and
the patient id is pseudonymous-shaped. All 235,889 records from five real CGMacros
participants validate against it.

JSON Schemas are committed in `sources/schema/` (`measurement.schema.json`,
`mapping.schema.json`). A test fails if they drift from the code; regenerate with
`python -m sources.schema`.

## Known limits

- CGMacros meal macros are stored as given. The data dictionary doesn't say whether they
  are already scaled by "Amount Consumed", so "% eaten" is kept as its own measurement.
- CGMacros timestamps are randomly shifted by the dataset authors; pass `--tz UTC`.
- Apple "Core" sleep is mapped to `light`. Fitbit `shortData` wake events overlap the main
  stage timeline and are counted, not ingested.
- Participant-level CGMacros data (`bio.csv`: A1c, fasting labs, fingersticks) is not ingested yet.
