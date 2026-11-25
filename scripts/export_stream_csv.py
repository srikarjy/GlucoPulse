"""Export CGMacros Dexcom readings as plain `timestamp,glucose` CSVs for the Kafka replay producer.

Why a separate step: the producer's Docker image contains only replay_sensor.py, not the `sources` package. This
script runs on the host (in .venv), reuses the tested CGMacros adapter (which recovers the native 5-min Dexcom
readings from the dataset's per-minute interpolation), and writes one file per person that the producer can stream
with `--source csv`.

Why CGMacros and not AZT1D: the alert rule is validated only in adults NOT using insulin. AZT1D is type 1 diabetes on
insulin, so streaming alerts must run on CGMacros. Files go to data/stream/ (gitignored); patient ids stay the
CGMacros pseudonyms (cgmacros-NNN).

Usage:
    .venv/bin/python scripts/export_stream_csv.py                  # all 45 people
    .venv/bin/python scripts/export_stream_csv.py --patients 1 21  # cgmacros-001, cgmacros-021
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sources.cgmacros import parse_cgmacros  # noqa: E402
from sources.model import Kind  # noqa: E402

DEVICE = "dexcom-g6"                      # 5-min readings; the Libre Pro stream is 15-min
TS_FORMAT = "%Y-%m-%d %H:%M:%S"           # same format the AZT1D replay uses


def dexcom_readings(csv_path: Path) -> tuple[str, list[tuple[str, float]], int]:
    """(patient id, sorted unique (timestamp, mg/dL) readings, rows dropped as non-numeric or duplicate)."""
    res = parse_cgmacros(csv_path)
    patient = csv_path.stem.lower()
    by_ts: dict[str, float] = {}
    dropped = 0
    for m in res.raws:
        if m.kind is not Kind.GLUCOSE_CGM or m.device != DEVICE:
            continue
        try:
            value = float(m.value)
        except (TypeError, ValueError):
            dropped += 1
            continue
        ts = m.ts.strftime(TS_FORMAT)
        if ts in by_ts:                    # one value per timestamp; the adapter already resolved conflicts
            dropped += 1
            continue
        by_ts[ts] = value
    return patient, sorted(by_ts.items()), dropped


def write_stream_csv(readings: list[tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "glucose"])
        w.writerows((ts, f"{g:g}") for ts, g in readings)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "cgmacros" / "csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "stream"))
    p.add_argument("--patients", type=int, nargs="*", help="CGMacros numbers, e.g. 1 21 (default: all)")
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = sorted(src.glob("CGMacros-*.csv"))
    if args.patients:
        wanted = {f"CGMacros-{n:03d}.csv" for n in args.patients}
        files = [f for f in files if f.name in wanted]
    if not files:
        sys.exit(f"no CGMacros CSVs found in {src}")
    for f in files:
        patient, readings, dropped = dexcom_readings(f)
        write_stream_csv(readings, out / f"{patient}.csv")
        print(f"{patient}: {len(readings)} readings ({dropped} dropped) -> {out / (patient + '.csv')}")


if __name__ == "__main__":
    main()
