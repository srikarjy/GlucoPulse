"""Ingest one health-data file into a FHIR R4 Bundle plus a quarantine report.

    python -m sources.ingest --format cgmacros data/CGMacros-001.csv --tz UTC --out bundle.json
    python -m sources.ingest --format fitbit heart.json --patient p-07 --tz America/Chicago --out b.json
    python -m sources.ingest --format apple export.xml --patient p-07 --out b.json
    python -m sources.ingest --format mapped --mapping sources/mappings/example_logbook_csv.toml export.csv \
        --patient p-07 --tz Europe/Berlin --out b.json --records-out records.ndjson

Prints a JSON summary to stdout. Exit codes:
  0  at least one measurement accepted (or the input was simply empty)
  2  usage / input error (unknown tz, bad patient id, unreadable or oversized file, --out == input)
  3  nothing usable: every record was quarantined (e.g. missing --tz, unsafe XML)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from serving.healthcare.fhir_mapping import bundle, observation_from_measurement
from sources.apple_health import parse_apple_health
from sources.cgmacros import parse_cgmacros
from sources.fitbit import parse_fitbit
from sources.mapped import load_mapping, parse_mapped
from sources.normalize import normalize
from sources.schema import MeasurementRecord

MAX_JSON_BYTES = 100 * 1024 * 1024   # JSON docs are read whole; cap them


def _adapt(fmt: str, path: Path, patient: str | None, mapping: Path | None):
    if fmt == "cgmacros":
        return parse_cgmacros(path, patient)
    if not patient:
        raise ValueError(f"--patient is required for --format {fmt}")
    if fmt in ("fitbit", "mapped") and path.stat().st_size > MAX_JSON_BYTES and path.suffix.lower() == ".json":
        raise ValueError(f"{path.name}: larger than {MAX_JSON_BYTES // 2**20} MB")
    if fmt == "fitbit":
        return parse_fitbit(json.loads(path.read_text()), patient=patient, ref=path.name)
    if fmt == "mapped":
        if mapping is None:
            raise ValueError("--mapping is required for --format mapped")
        return parse_mapped(path, load_mapping(mapping), patient=patient)
    return parse_apple_health(path, patient=patient)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sources.ingest")
    ap.add_argument("--format", required=True, choices=["cgmacros", "fitbit", "apple", "mapped"])
    ap.add_argument("--mapping", type=Path, help="TOML mapping file for --format mapped (see sources/mappings/)")
    ap.add_argument("path", type=Path)
    ap.add_argument("--patient", help="pseudonymous patient id (required unless the file is CGMacros-NNN.csv)")
    ap.add_argument("--tz", help="IANA time zone for sources whose timestamps carry no offset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quarantine-out", type=Path)
    ap.add_argument("--records-out", type=Path, help="also write typed MeasurementRecords as NDJSON")
    args = ap.parse_args(argv)

    try:
        outputs = [p.resolve() for p in (args.out, args.quarantine_out, args.records_out) if p]
        if args.path.resolve() in outputs:
            raise ValueError("refusing to overwrite the input file")
        adapted = _adapt(args.format, args.path, args.patient, args.mapping)
        result = normalize(adapted.raws, tz=args.tz)
        quarantined = adapted.quarantined + result.quarantined
        args.out.write_text(json.dumps(bundle([observation_from_measurement(m) for m in result.measurements]),
                                       indent=1))
        if args.quarantine_out:
            args.quarantine_out.write_text(json.dumps([asdict(q) for q in quarantined], indent=1, default=str))
        if args.records_out:
            with args.records_out.open("w") as fh:
                for m in result.measurements:
                    fh.write(MeasurementRecord.from_measurement(m).model_dump_json() + "\n")
    except (ValueError, OSError, RecursionError, OverflowError, csv.Error) as exc:
        print(f"error: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return 2

    print(json.dumps({
        "format": args.format,
        "accepted": len(result.measurements),
        "quarantined": len(quarantined),
        "duplicates_dropped": result.stats["duplicates_dropped"],
        "reasons": dict(Counter(q.reason for q in quarantined)),
        "adapter_stats": dict(adapted.stats),
    }))
    return 3 if not result.measurements and quarantined else 0


if __name__ == "__main__":
    sys.exit(main())
