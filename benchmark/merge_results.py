"""
Concatenate per-sample benchmark outputs (benchmark/results/results_<sample>.csv
and *_phasing.csv, written by `run_benchmark.py --only <sample>`) into
benchmark/results.csv and benchmark/results_phasing.csv, in paper sample order.

Usage:
    python benchmark/merge_results.py
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = ["CA-0098", "PM1284", "852_T1_33_23905"]


def merge(suffix: str) -> None:
    rows, fields = [], None
    for s in SAMPLES:
        f = HERE / "results" / f"results_{s}{suffix}.csv"
        with f.open(newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            fields = fields or r.fieldnames
            rows.extend(r)
    out = HERE / f"results{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows")


if __name__ == "__main__":
    merge("")
    merge("_phasing")
