"""Canonical record for every health data source GlucoPulse ingests.

Adapters emit `RawMeasurement`s exactly as the source wrote them (source units or
labels, possibly naive timestamps). `normalize()` turns those into `Measurement`s
(canonical unit or code, UTC) or `Quarantined` records. Nothing is silently
fixed: anything that can't be normalized unambiguously is quarantined with a
reason, the same philosophy as the CGM pipeline's cgm-dlq.

Value types
  quantity  a number in the kind's canonical UCUM unit, range-checked
  code      one of the kind's allowed codes; source labels are mapped through synonyms
  period    (flag, combinable with either) the record covers [ts, end), e.g. a sleep stage
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

LOINC = "http://loinc.org"
LOCAL = "urn:glucopulse:code"


class Kind(str, Enum):
    GLUCOSE_CGM = "glucose_cgm"
    GLUCOSE_BLOOD = "glucose_blood"
    HEART_RATE = "heart_rate"
    HRV_SDNN = "hrv_sdnn"
    SPO2 = "spo2"
    RESPIRATORY_RATE = "respiratory_rate"
    BP_SYSTOLIC = "bp_systolic"
    BP_DIASTOLIC = "bp_diastolic"
    BODY_WEIGHT = "body_weight"
    STEPS = "steps"
    ACTIVE_CALORIES = "active_calories"
    METS = "mets"
    ACTIVITY_INTENSITY = "activity_intensity"
    SLEEP_STAGE = "sleep_stage"
    INSULIN_BOLUS = "insulin_bolus"
    INSULIN_BASAL = "insulin_basal"
    MEAL_TYPE = "meal_type"
    MEAL_CALORIES = "meal_calories"
    MEAL_CARBS = "meal_carbs"
    MEAL_PROTEIN = "meal_protein"
    MEAL_FAT = "meal_fat"
    MEAL_FIBER = "meal_fiber"
    MEAL_SUGAR = "meal_sugar"
    MEAL_PORTION_EATEN = "meal_portion_eaten"


@dataclass(frozen=True)
class KindSpec:
    code: str
    system: str
    display: str
    category: str                                   # FHIR observation-category code
    value_type: str = "quantity"                    # "quantity" | "code"
    unit: Optional[str] = None                      # canonical UCUM unit (quantity)
    low: Optional[float] = None                     # plausible range, inclusive (quantity)
    high: Optional[float] = None
    codes: tuple[tuple[str, str], ...] = ()         # (code, display) allowed values (code)
    synonyms: dict[str, str] = field(default_factory=dict)   # normalized source label -> code
    period: bool = False                            # record covers [ts, end)
    max_period_hours: float = 24.0


def _q(code, system, display, category, unit, low, high) -> KindSpec:
    return KindSpec(code, system, display, category, "quantity", unit, low, high)


def _c(code, display, category, codes, synonyms, period=False) -> KindSpec:
    syn = {c: c for c, _ in codes}
    syn.update(synonyms)
    return KindSpec(code, LOCAL, display, category, "code", codes=tuple(codes), synonyms=syn, period=period)


# LOINC codes verified against loinc.org. Local codes (urn:glucopulse:code) are used where no
# LOINC term means the same thing: e.g. every LOINC macronutrient intake code is a 24-hour
# total (g/(24.h)), which a single logged meal is not; insulin delivery would ideally be a
# FHIR MedicationAdministration, kept as an Observation here for one uniform stream.
KINDS: dict[Kind, KindSpec] = {
    Kind.GLUCOSE_CGM: _q("99504-3", LOINC, "Glucose [Mass/volume] in Interstitial fluid", "laboratory", "mg/dL", 40, 400),
    Kind.GLUCOSE_BLOOD: _q("2339-0", LOINC, "Glucose [Mass/volume] in Blood", "laboratory", "mg/dL", 20, 600),
    Kind.HEART_RATE: _q("8867-4", LOINC, "Heart rate", "vital-signs", "/min", 25, 250),
    Kind.HRV_SDNN: _q("80404-7", LOINC, "R-R interval.standard deviation (Heart rate variability)", "vital-signs", "ms", 1, 500),
    Kind.SPO2: _q("59408-5", LOINC, "Oxygen saturation in Arterial blood by Pulse oximetry", "vital-signs", "%", 50, 100),
    Kind.RESPIRATORY_RATE: _q("9279-1", LOINC, "Respiratory rate", "vital-signs", "/min", 4, 60),
    Kind.BP_SYSTOLIC: _q("8480-6", LOINC, "Systolic blood pressure", "vital-signs", "mm[Hg]", 50, 260),
    Kind.BP_DIASTOLIC: _q("8462-4", LOINC, "Diastolic blood pressure", "vital-signs", "mm[Hg]", 20, 160),
    Kind.BODY_WEIGHT: _q("29463-7", LOINC, "Body weight", "vital-signs", "kg", 1, 400),
    Kind.STEPS: _q("55423-8", LOINC, "Number of steps in unspecified time Pedometer", "activity", "{steps}", 0, 100_000),
    Kind.ACTIVE_CALORIES: _q("41981-2", LOINC, "Calories burned", "activity", "kcal", 0, 100),
    Kind.METS: _q("mets-1min", LOCAL, "Metabolic equivalent of task, 1-minute estimate", "activity", "{MET}", 0.5, 25),
    Kind.ACTIVITY_INTENSITY: _c(
        "activity-intensity", "Activity intensity level", "activity",
        [("sedentary", "Sedentary"), ("light", "Lightly active"), ("moderate", "Fairly active"), ("vigorous", "Very active")],
        {"0": "sedentary", "1": "light", "2": "moderate", "3": "vigorous"}),   # Fitbit intensity levels 0-3
    Kind.SLEEP_STAGE: _c(
        "sleep-stage", "Sleep stage", "activity",
        [("awake", "Awake"), ("light", "Light / core sleep"), ("deep", "Deep sleep"), ("rem", "REM sleep"),
         ("asleep", "Asleep, stage unspecified"), ("restless", "Restless"), ("in_bed", "In bed")],
        {"wake": "awake", "core": "light",
         "hkcategoryvaluesleepanalysisawake": "awake", "hkcategoryvaluesleepanalysisasleepcore": "light",
         "hkcategoryvaluesleepanalysisasleepdeep": "deep", "hkcategoryvaluesleepanalysisasleeprem": "rem",
         "hkcategoryvaluesleepanalysisasleep": "asleep", "hkcategoryvaluesleepanalysisasleepunspecified": "asleep",
         "hkcategoryvaluesleepanalysisinbed": "in_bed"},
        period=True),
    Kind.INSULIN_BOLUS: _q("insulin-bolus", LOCAL, "Insulin bolus delivered", "therapy", "[IU]", 0, 100),
    Kind.INSULIN_BASAL: _q("insulin-basal", LOCAL, "Basal insulin delivered", "therapy", "[IU]", 0, 100),
    Kind.MEAL_TYPE: _c(
        "meal-type", "Meal type", "social-history",
        [("breakfast", "Breakfast"), ("lunch", "Lunch"), ("dinner", "Dinner"), ("snack", "Snack")],
        {"snacks": "snack", "supper": "dinner"}),
    Kind.MEAL_CALORIES: _q("meal-energy", LOCAL, "Energy in logged meal", "social-history", "kcal", 0, 5000),
    Kind.MEAL_CARBS: _q("meal-carbohydrate", LOCAL, "Carbohydrate in logged meal", "social-history", "g", 0, 500),
    Kind.MEAL_PROTEIN: _q("meal-protein", LOCAL, "Protein in logged meal", "social-history", "g", 0, 300),
    Kind.MEAL_FAT: _q("meal-fat", LOCAL, "Fat in logged meal", "social-history", "g", 0, 300),
    Kind.MEAL_FIBER: _q("meal-fiber", LOCAL, "Fiber in logged meal", "social-history", "g", 0, 150),
    Kind.MEAL_SUGAR: _q("meal-sugar", LOCAL, "Sugars in logged meal", "social-history", "g", 0, 300),
    Kind.MEAL_PORTION_EATEN: _q("meal-portion-eaten", LOCAL, "Portion of logged meal eaten", "social-history", "%", 0, 100),
}

# Native sampling interval per device. Readings within half of it are treated as the same sample;
# devices not listed here only match on an identical timestamp (see normalize.py).
DEVICE_SLOT_SECONDS: dict[str, int] = {"fitbit": 60, "dexcom-g6": 300, "libre-pro": 900}


@dataclass(frozen=True)
class RawMeasurement:
    """As the source wrote it. `ts`/`end` may be naive; `unit` is the source's own unit string."""
    patient: str
    kind: Kind
    ts: datetime
    value: float | str   # raw cell text is allowed; normalize() converts or quarantines it
    unit: str
    source: str          # format/adapter, e.g. "fitbit", "apple_health", "cgmacros", "mapped:<name>"
    device: str          # physical stream, e.g. "dexcom-g6", "libre-pro", "fitbit", "iphone"
    raw_ref: str         # where it came from, e.g. "CGMacros-001.csv:42"
    end: Optional[datetime] = None


@dataclass(frozen=True)
class Measurement:
    """Normalized: canonical unit (quantity) or allowed code (code), timezone-aware UTC times."""
    patient: str
    kind: Kind
    ts: datetime
    value: float | str
    source: str
    device: str
    raw_ref: str
    end: Optional[datetime] = None


@dataclass(frozen=True)
class Quarantined:
    reason: str
    raw_ref: str
    source: str
    detail: dict[str, Any] = field(default_factory=dict)
