"""
Main driver: for each sample, enumerate every tied-optimal haplotype-labeled
phasing option (exact 3-stage lexicographic optimum, see enumerate_options.py)
crossed with every tied-optimal loose-end pairing, write a full CSV folder
per option, and write one per-sample summary.csv comparing all options.

Output layout:
  analysis/haplotype_options_report/<sample>/<option_id>/genome_a.csv
  analysis/haplotype_options_report/<sample>/<option_id>/genome_b.csv
  analysis/haplotype_options_report/<sample>/<option_id>/ebg_structure.csv
  analysis/haplotype_options_report/<sample>/<option_id>/ebg_components_summary.csv
  analysis/haplotype_options_report/<sample>/<option_id>/scenarios_summary.csv
  analysis/haplotype_options_report/<sample>/summary.csv
  analysis/haplotype_options_report/samples_overview.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gr_cn.io import jabba, reference as reference_mod  # noqa: E402
from enumerate_options import enumerate_full_options  # noqa: E402
from write_csvs import write_option_folder  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent

SAMPLES = [
    ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23], "CA-0098_5_18_22_X"),
    ("jabba_data/graphs_all/PM1284.json", [3, 11, 12], "PM1284_3_11_12"),
    ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14], "852_T1_33_23905_6_9_12_14"),
]


def _write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_sample(path: str, chroms, label: str, dedupe: bool = False) -> dict:
    t0 = time.time()
    print(f"=== {label} ===", flush=True)
    graph = jabba.load(path, chromosomes=chroms)
    ref = reference_mod.build(graph)

    full_options, meta, equivalence_classes = enumerate_full_options(graph, ref, L=2, dedupe=dedupe)
    print(f"  {meta}", flush=True)

    sample_dir = OUT_ROOT / label

    if dedupe:
        # Transparency file: which raw (arc-level) tied phasings collapsed
        # into each canonical (downstream-cycle-distinct) representative.
        eq_rows = []
        for rep_label, raw_labels in equivalence_classes.items():
            eq_rows.append({
                "canonical_phasing_label": rep_label,
                "n_raw_ties_collapsed": len(raw_labels),
                "raw_phasing_labels": ";".join(raw_labels),
            })
        eq_rows.sort(key=lambda r: -r["n_raw_ties_collapsed"])
        _write_csv(sample_dir / "phasing_equivalence_classes.csv", eq_rows, [
            "canonical_phasing_label", "n_raw_ties_collapsed", "raw_phasing_labels",
        ])
    summary_rows = []
    for opt in full_options:
        opt_dir = sample_dir / opt["option_id"]
        stats = write_option_folder(graph, ref, opt["phasing"], opt["extra_pairs"], opt_dir, L=2)
        summary_rows.append({
            "option_id": opt["option_id"],
            "phasing_label": opt["phasing_label"],
            "loose_pairing_label": opt["loose_label"],
            "n_components": stats["n_components"],
            "n_mixed_haplotype_edges": stats["n_mixed_haplotype_edges"],
            "n_ebg_cycles": stats["n_cycles"],
            "n_ebg_nodes": stats["n_ebg_nodes"],
            "n_possible_scenarios": stats["n_scenarios"],
            "scenarios_enumeration_truncated": stats["truncated"],
            "optimal_max_chromosomes": stats["optimal_objective"],
            "n_optimal_scenarios": stats["n_optimal_scenarios"],
            "n_full_scenarios_bruteforce": stats["n_full_scenarios"],
            "bnb_expanded_states": stats["bnb_expanded_states"],
            "bnb_matches_full_enum": stats["bnb_matches_full_enum"],
            "bnb_speedup_factor": stats["bnb_speedup_factor"],
            "n_real_chromatids_genome_a": stats["n_real_chromatids_genome_a"],
            "n_real_chromatids_genome_b": stats["n_real_chromatids_genome_b"],
        })
        print(f"    {opt['option_id']} ({opt['phasing_label']}, {opt['loose_label']}): "
              f"cycles={stats['n_cycles']} scenarios={stats['n_scenarios']} "
              f"optimal_max_chr={stats['optimal_objective']} "
              f"bnb_states={stats['bnb_expanded_states']}/{stats['n_full_scenarios']}", flush=True)

    _write_csv(sample_dir / "summary.csv", summary_rows, [
        "option_id", "phasing_label", "loose_pairing_label", "n_components",
        "n_mixed_haplotype_edges", "n_ebg_cycles", "n_ebg_nodes",
        "n_possible_scenarios", "scenarios_enumeration_truncated",
        "optimal_max_chromosomes", "n_optimal_scenarios",
        "n_full_scenarios_bruteforce", "bnb_expanded_states",
        "bnb_matches_full_enum", "bnb_speedup_factor",
        "n_real_chromatids_genome_a", "n_real_chromatids_genome_b",
    ])

    distinct_objs = sorted({r["optimal_max_chromosomes"] for r in summary_rows})
    elapsed = time.time() - t0
    print(f"  {len(full_options)} option(s) written in {elapsed:.1f}s. "
          f"Distinct optimal_max_chromosomes across options: {distinct_objs}", flush=True)

    return {
        "sample": label,
        "n_subgraphs": meta["n_subgraphs"],
        "n_raw_tied_phasings": meta["n_raw_phasing_options"],
        "n_canonical_tied_phasings": meta["n_canonical_phasing_options"],
        "raw_enumeration_capped": meta["truncated"] or any(meta["per_subgraph_hit_solution_cap"]),
        "n_loose_pairing_variants": meta["n_loose_pairing_variants"],
        "n_full_options_written": len(full_options),
        "distinct_optimal_max_chromosomes": ";".join(str(v) for v in distinct_objs),
        "objective_is_stable_across_options": len(distinct_objs) == 1,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    overview = []
    for path, chroms, label in SAMPLES:
        overview.append(run_sample(path, chroms, label))

    _write_csv(OUT_ROOT / "samples_overview.csv", overview, [
        "sample", "n_subgraphs", "n_raw_tied_phasings", "n_canonical_tied_phasings",
        "raw_enumeration_capped", "n_loose_pairing_variants", "n_full_options_written",
        "distinct_optimal_max_chromosomes", "objective_is_stable_across_options", "elapsed_s",
    ])
    print("\nDone. samples_overview.csv written.", flush=True)


if __name__ == "__main__":
    main()
