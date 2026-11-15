"""Config-driven adapter: map any app's CSV or JSON export with a TOML file, no code.

A mapping names the time column(s) and how to parse them, and for each value
column: the measurement kind, the source unit (quantities) or a label map
(codes), and a device id. The mapping itself is validated strictly
(`MappingSpec`, unknown keys rejected) so a typo fails at load time instead of
silently dropping a column. See sources/mappings/*.toml for examples, and
sources/schema/mapping.schema.json for the published spec.
"""
from __future__ import annotations

import csv
import json
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sources._common import AdapterResult, checked_patient, clip
from sources.model import KINDS, Kind, Quarantined, RawMeasurement

SLUG = r"^[a-z0-9][a-z0-9-]{0,47}$"


class TimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str] = Field(min_length=1, description="column(s) joined with a single space before parsing")
    format: Optional[str] = Field(None, description="strptime format; include %z if the text carries an offset")
    epoch: Optional[Literal["seconds", "milliseconds"]] = Field(None, description="numeric Unix time (UTC)")

    @model_validator(mode="after")
    def _one_way(self):
        if (self.format is None) == (self.epoch is None):
            raise ValueError("time needs exactly one of 'format' or 'epoch'")
        return self


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str = Field(min_length=1, description="CSV header, or dotted path inside a JSON record")
    kind: Kind
    unit: str = Field("", description="source unit for quantity kinds; empty for code kinds")
    device: str = Field("app", pattern=SLUG, description="stream id; free text is not allowed (privacy)")
    values: dict[str, str] = Field(default_factory=dict, description="source label -> allowed code (code kinds)")
    end_column: Optional[str] = Field(None, description="end time column, required for period kinds")
    end_format: Optional[str] = Field(None, description="strptime format for end_column when it differs from the "
                                      "start format; a time-only format takes the start's date (next day if earlier)")

    @field_validator("values")
    @classmethod
    def _lower_keys(cls, v: dict[str, str]) -> dict[str, str]:
        lowered: dict[str, str] = {}
        for k, code in v.items():
            key = k.strip().lower()
            if key in lowered and lowered[key] != code:
                raise ValueError(f"labels {k!r} and another differ only by case but map to different codes")
            lowered[key] = code
        return lowered

    @model_validator(mode="after")
    def _consistent_with_kind(self):
        from sources.normalize import _unit_factor   # local: avoid an import cycle at module load

        spec = KINDS[self.kind]
        if spec.value_type == "quantity":
            if not self.unit or _unit_factor(self.kind, self.unit) is None:
                raise ValueError(f"unit {self.unit!r} cannot be converted to {spec.unit} for {self.kind.value}")
            if self.values:
                raise ValueError("'values' label maps are only for code kinds")
        else:
            if self.unit:
                raise ValueError(f"{self.kind.value} is a code kind and takes no unit")
            allowed = {c for c, _ in spec.codes}
            bad = sorted(set(self.values.values()) - allowed)
            if bad:
                raise ValueError(f"label map targets {bad} are not codes of {self.kind.value}: {sorted(allowed)}")
        if spec.period and not self.end_column:
            raise ValueError(f"{self.kind.value} covers a period and needs 'end_column'")
        if not spec.period and self.end_column:
            raise ValueError(f"{self.kind.value} is a point-in-time kind; remove 'end_column'")
        if self.end_format and not self.end_column:
            raise ValueError("'end_format' needs 'end_column'")
        return self


class MappingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=SLUG)
    format: Literal["csv", "json"]
    records: Optional[str] = Field(None, description="JSON only: dotted path to the list of records")
    time: TimeSpec
    fields: list[FieldSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _records_path(self):
        if self.format == "json" and not self.records:
            raise ValueError("json mappings need 'records' (dotted path to the record list)")
        if self.format == "csv" and self.records:
            raise ValueError("'records' only applies to json mappings")
        streams = [(f.kind, f.device) for f in self.fields]
        dupes = sorted({f"{k.value}/{d}" for k, d in streams if streams.count((k, d)) > 1})
        if dupes:
            raise ValueError(f"two columns feed the same stream {dupes}; give them different 'device' ids")
        return self


def load_mapping(path: str | Path) -> MappingSpec:
    with open(path, "rb") as fh:
        return MappingSpec.model_validate(tomllib.load(fh))


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _parse_time(spec: TimeSpec, parts: list[Any]) -> Optional[datetime]:
    if any(p is None or (isinstance(p, str) and not p.strip()) for p in parts):
        return None
    try:
        if spec.epoch:
            from sources.normalize import _to_number   # plain decimals only: no "1e9", "1_000", booleans
            number = _to_number(parts[0]) if len(parts) == 1 else None
            if number is None:
                return None
            return datetime.fromtimestamp(number / (1000 if spec.epoch == "milliseconds" else 1), tz=timezone.utc)
        return datetime.strptime(" ".join(str(p).strip() for p in parts), spec.format)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _parse_end(time: TimeSpec, end_format: Optional[str], text: Any, start: datetime) -> Optional[datetime]:
    if not end_format:
        return _parse_time(time, [text])
    if text is None or not str(text).strip():
        return None
    try:
        end = datetime.strptime(str(text).strip(), end_format)
    except ValueError:
        return None
    if not any(d in end_format for d in ("%Y", "%y", "%d", "%j")):   # time-only: borrow the start's date
        end = datetime.combine(start.date(), end.time(), tzinfo=start.tzinfo)
        if end <= start:                                            # e.g. 23:30 -> 00:10 crosses midnight
            end += timedelta(days=1)
    return end


def parse_mapped(path: str | Path, spec: MappingSpec, patient: str) -> AdapterResult:
    patient = checked_patient(patient)
    path = Path(path)
    out = AdapterResult()
    source = f"mapped:{spec.name}"
    for i, record in _records(path, spec, out):
        rref = f"{path.name}:{i}"
        ts = _parse_time(spec.time, [record.get(c) for c in spec.time.columns])
        if ts is None:
            out.quarantined.append(Quarantined("malformed_record", rref, source,
                                               {"time": [clip(record.get(c)) for c in spec.time.columns]}))
            continue
        for f in spec.fields:
            value = record.get(f.column)
            if value is None or (isinstance(value, str) and not value.strip()):
                out.stats["skipped_empty_cell"] += 1
                continue
            if f.values and isinstance(value, (str, int, float)) and not isinstance(value, bool):
                value = f.values.get(str(value).strip().lower(), value)
            end = None
            if f.end_column:
                end = _parse_end(spec.time, f.end_format, record.get(f.end_column), ts)
                if end is None:
                    out.quarantined.append(Quarantined("malformed_record", rref, source,
                                                       {"end": clip(record.get(f.end_column))}))
                    continue
            out.raws.append(RawMeasurement(patient=patient, kind=f.kind, ts=ts, value=value, unit=f.unit,
                                           source=source, device=f.device, raw_ref=rref, end=end))
    return out


def _records(path: Path, spec: MappingSpec, out: AdapterResult):
    """Yield (record number, flat dict of column -> raw value) for the mapped columns."""
    wanted = list(dict.fromkeys(spec.time.columns + [f.column for f in spec.fields] +
                                [f.end_column for f in spec.fields if f.end_column]))
    if spec.format == "json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        records = _dig(doc, spec.records)
        if not isinstance(records, list):
            raise ValueError(f"{path.name}: '{spec.records}' is not a list of records")
        for i, rec in enumerate(records):
            yield i, ({c: _dig(rec, c) for c in wanted} if isinstance(rec, dict) else {})
        return
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{path.name}: mapping columns not in the file: {missing}")
        out.stats["unmapped_columns"] = len([h for h in header if h and h not in wanted])
        idx = {c: header.index(c) for c in wanted}
        while True:
            try:
                row = next(reader)
            except StopIteration:
                return
            except csv.Error as exc:
                out.quarantined.append(Quarantined("malformed_record", f"{path.name}:{reader.line_num}",
                                                   f"mapped:{spec.name}", {"why": clip(str(exc))}))
                continue
            yield reader.line_num, {c: (row[j] if j < len(row) else "") for c, j in idx.items()}
