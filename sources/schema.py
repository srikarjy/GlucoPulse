"""Typed contract for normalized measurements (and for mapping files).

`MeasurementRecord` is what downstream consumers (the forecaster, the triage
agents, other services) receive. It re-validates everything normalize()
guarantees, so a record that reaches a consumer is internally consistent:
  * value type matches the kind (quantity vs code), codes are allowed values,
    quantities are inside the kind's plausible range and in its canonical unit;
  * times are timezone-aware; period kinds have an end after the start and
    point kinds have no end; the patient id is pseudonymous-shaped.

The JSON Schemas are committed under sources/schema/ and a test fails if they
drift from these models. Regenerate with:  python -m sources.schema
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sources.mapped import MappingSpec
from sources.model import KINDS, Kind, Measurement

SCHEMA_DIR = Path(__file__).parent / "schema"
_KIND_VALUES = frozenset(k.value for k in Kind)


class MeasurementRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    patient: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: Kind
    value_type: Literal["quantity", "code"]
    value: Union[float, str]
    unit: Optional[str] = Field(None, description="canonical UCUM unit for quantity kinds, null for code kinds")
    ts: AwareDatetime
    end: Optional[AwareDatetime] = None
    source: str
    device: str
    raw_ref: str

    @model_validator(mode="before")
    @classmethod
    def _fill_value_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "value_type" not in data:
            kind = data.get("kind")
            if isinstance(kind, (str, Kind)) and str(getattr(kind, "value", kind)) in _KIND_VALUES:
                data = dict(data, value_type=KINDS[Kind(kind)].value_type)
        return data

    @model_validator(mode="before")
    @classmethod
    def _no_bool_values(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("value"), bool):
            raise ValueError("value must be a number or a code, not a boolean")
        return data

    @model_validator(mode="after")
    def _consistent(self):
        spec = KINDS[self.kind]
        if self.value_type != spec.value_type:
            raise ValueError(f"{self.kind.value} is a {spec.value_type} kind")
        if spec.value_type == "code":
            if not isinstance(self.value, str) or self.value not in dict(spec.codes):
                raise ValueError(f"value must be one of {[c for c, _ in spec.codes]}")
            if self.unit is not None:
                raise ValueError("code kinds carry no unit")
        else:
            if isinstance(self.value, str):
                raise ValueError("quantity kinds need a numeric value")
            if not (spec.low <= self.value <= spec.high):
                raise ValueError(f"value outside the plausible range [{spec.low}, {spec.high}]")
            if self.unit != spec.unit:
                raise ValueError(f"unit must be the canonical {spec.unit!r}")
        if spec.period:
            if self.end is None or self.end <= self.ts:
                raise ValueError("period kinds need an end after the start")
        elif self.end is not None:
            raise ValueError("point-in-time kinds have no end")
        return self

    @classmethod
    def from_measurement(cls, m: Measurement) -> "MeasurementRecord":
        spec = KINDS[m.kind]
        return cls(patient=m.patient, kind=m.kind, value_type=spec.value_type, value=m.value,
                   unit=spec.unit if spec.value_type == "quantity" else None, ts=m.ts, end=m.end,
                   source=m.source, device=m.device, raw_ref=m.raw_ref)


def json_schemas() -> dict[str, dict]:
    return {"measurement": MeasurementRecord.model_json_schema(), "mapping": MappingSpec.model_json_schema()}


def main() -> int:
    SCHEMA_DIR.mkdir(exist_ok=True)
    for name, schema in json_schemas().items():
        (SCHEMA_DIR / f"{name}.schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {SCHEMA_DIR / f'{name}.schema.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
