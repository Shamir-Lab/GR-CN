"""
Thin driver: run main.py's exhaustive pipeline (every tied-optimal phasing x
loose-pairing option, full CSV deliverable per option) across all 3 named
samples, and roll the results into one cross-sample overview table.

For each sample, the raw (undeduplicated) tied-phasing enumeration always
runs first. If it finds more than one raw tied phasing, a second,
topology-deduplicated ("canonical") pass also runs, since the raw count
includes pure haplotype/copy-label relabeling duplicates alongside genuinely
distinct reconstructions (see README.md). If the raw phasing is unique (no
ambiguity at all -- e.g. CA-0098), the second pass is skipped as redundant
and the single result is written without a _raw/_canonical suffix.

Output layout:
  analysis/haplotype_options_report/<sample>/<option_id>/...            (unambiguous samples)
  analysis/haplotype_options_report/<sample>_raw/<option_id>/...        (ambiguous samples)
  analysis/haplotype_options_report/<sample>_canonical/<option_id>/...  (ambiguous samples)
  analysis/haplotype_options_report/<sample...>/summary.csv
  analysis/haplotype_options_report/samples_overview.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import main as gr_cn_main  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent

SAMPLES = [
    ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23], "CA-0098_5_18_22_X"),
    ("jabba_data/graphs_all/PM1284.json", [3, 11, 12], "PM1284_3_11_12"),
    ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14], "852_T1_33_23905_6_9_12_14"),
]


def _write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_sample(path: str, chroms, label: str, cpu_n: int, n_workers: int) -> dict:
    raw = gr_cn_main.run_exhaustive(
        jabba_path=path, L=2, chromosomes=chroms, dedupe=False,
        cpu_n=cpu_n, n_workers=n_workers,
        out_dir=str(OUT_ROOT / f"{label}_raw"), label=f"{label}_raw",
    )

    if raw["n_raw_tied_phasings"] > 1:
        canonical = gr_cn_main.run_exhaustive(
            jabba_path=path, L=2, chromosomes=chroms, dedupe=True,
            cpu_n=cpu_n, n_workers=n_workers,
            out_dir=str(OUT_ROOT / f"{label}_canonical"), label=f"{label}_canonical",
        )
        n_canonical = canonical["n_canonical_tied_phasings"]
    else:
        # No ambiguity at all (raw IS canonical) -- skip the redundant
        # duplicate pass/folder and drop the _raw suffix.
        (OUT_ROOT / f"{label}_raw").rename(OUT_ROOT / label)
        n_canonical = 1

    return {
        "sample": label,
        "n_subgraphs": raw["n_subgraphs"],
        "n_raw_tied_phasings": raw["n_raw_tied_phasings"],
        "n_canonical_tied_phasings": n_canonical,
        "raw_enumeration_capped": raw["raw_enumeration_capped"],
        "n_loose_pairing_variants": raw["n_loose_pairing_variants"],
        "distinct_optimal_max_chromosomes": raw["distinct_optimal_max_chromosomes"],
        "objective_is_stable_across_options": raw["objective_is_stable_across_options"],
    }


def main(cpu_n: int = 1, n_workers: int = 1) -> None:
    overview = [run_sample(path, chroms, label, cpu_n, n_workers) for path, chroms, label in SAMPLES]

    _write_csv(OUT_ROOT / "samples_overview.csv", overview, [
        "sample", "n_subgraphs", "n_raw_tied_phasings", "n_canonical_tied_phasings",
        "raw_enumeration_capped", "n_loose_pairing_variants",
        "distinct_optimal_max_chromosomes", "objective_is_stable_across_options",
    ])
    print("\nDone. samples_overview.csv written.", flush=True)


if __name__ == "__main__":
    main()
