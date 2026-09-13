"""
GR_CN — main pipeline entry point.

Default behavior is exhaustive: enumerate every tied-optimal haplotype
phasing and every tied loose-end pairing for a sample, run the full EBG +
scenario-search + CSV pipeline on each one, and write a CSV deliverable
folder per option (see gr_cn.phasing.enumerate / gr_cn.report).

Pass --quick to instead run once with a single (arbitrary, but still
provably optimal) phasing and loose-end pairing, and just find/print one
best scenario -- the fast path, with no CSV output unless --out is given.

Usage:
    python main.py <jabba_json_path> [options]

Options:
    --L INT                  Number of haplotypes (default: 2)
    --chromosomes INT [INT]  Restrict to these chromosome numbers (e.g. --chromosomes 1 5 7 22)
    --quick                  Run once with a single optimal phasing instead of every tied option
    --method STR              Scenario search: dp | branch_and_bound (default: dp)
    --timing STR              Cost timing: before|after|both (default: before)
    --cpu INT                 CP-SAT search workers used *within* each solve (default: 1)
    --workers INT             Parallel OS processes across independent subgraphs / options (default: 1)
    --dedupe                  (exhaustive mode) collapse pure haplotype/copy-label relabelings
    --out-dir PATH             (exhaustive mode) where to write the CSV deliverable
    --time-limit FLOAT        (quick mode) per-subgraph phasing ILP time limit in seconds (default: 200)
    --merge-haplotypes         (quick mode) merge same-topology cycles across haplotypes into one event
    --out PATH                 (quick mode) write the scenario result to this JSON file
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

from gr_cn import report
from gr_cn.io import jabba, reference
from gr_cn.phasing import ilp, loose
from gr_cn.phasing import enumerate as penum
from gr_cn.ebg import builder, cycles as ebg_cycles
from gr_cn.scenario import enumeration
from gr_cn.scenario.clustering import cycle_chromosomes


def _print_result(result: enumeration.ScenarioResult) -> None:
    print(f"Status:          {result.status}")
    print(f"Method:          {result.method}")
    print(f"Exact:           {result.exact}")
    print(f"Max chromosomes: {result.objective_value}")
    print(f"Steps:           {len(result.partition)}")
    for i, (part, cost, curr) in enumerate(
        zip(result.partition, result.step_costs, result.step_num_chromosomes), 1
    ):
        print(f"  Step {i}: cycles={part}, touched={cost}, total_chr={curr}")
    if result.message:
        print(f"Note: {result.message}")


def _write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_quick(
    jabba_path: str,
    L: int = 2,
    time_limit_s: float = 200.0,
    cpu_n: int = 1,
    n_workers: int = 1,
    method: str = "dp",
    timing: str = "before",
    chromosomes: Optional[List[int]] = None,
    merge_haplotypes: bool = False,
    out_path: Optional[str] = None,
) -> enumeration.ScenarioResult:
    """
    Fast path: one arbitrary (but still provably optimal) phasing and
    loose-end pairing, one best scenario. Use run_exhaustive() (the default)
    to see every tied-optimal option instead.
    """
    chr_label = f" (chromosomes {chromosomes})" if chromosomes else ""
    print(f"[1/5] Loading JABBA graph: {jabba_path}{chr_label}", flush=True)
    graph = jabba.load(jabba_path, chromosomes=chromosomes)
    print(f"      {len(graph.intervals)} intervals, "
          f"{len(graph.alt_adjacencies)} ALT, "
          f"{len(graph.ref_adjacencies)} REF, "
          f"{len(graph.loose_adjacencies)} LOOSE adjacencies")

    print(f"[2/5] Building reference genome (Genome A)...", flush=True)
    ref = reference.build(graph)
    n_deleted_copies = sum(info.n_deleted for info in ref.deleted_intervals)
    n_gained_copies = sum(info.n_gained for info in ref.gained_intervals)
    print(f"      Deleted segments: {n_deleted_copies}, "
          f"gained segments: {n_gained_copies}")

    print(f"[3/5] Running phasing ILP (L={L}, time_limit={time_limit_s}s, "
          f"cpu={cpu_n}, workers={n_workers})...", flush=True)
    phasing_result = ilp.phase(graph, ref, L=L, time_limit_s=time_limit_s,
                                cpu_n=cpu_n, n_workers=n_workers)
    print(f"      Status: {phasing_result.status}, "
          f"components: {phasing_result.num_components}, "
          f"mixed-haplotype edges: {phasing_result.num_mixed_haplotype_edges}")

    if graph.loose_adjacencies:
        try:
            extra_pairs = loose.resolve_post_phasing(graph, phasing_result)
            if extra_pairs:
                phasing_result.grey_pairs.extend(extra_pairs)
                print(f"      Connected {len(extra_pairs)} open loose-end pair(s) post-phasing.")
        except loose.TooManyLooseEnds as e:
            print(f"      WARNING: {e}", flush=True)
            print("      Proceeding with unresolved loose ends.", flush=True)

    print(f"[4/5] Building EBG and extracting cycles...", flush=True)
    raw_nodes = builder.build(graph, ref, phasing_result, L=L)
    ebg_nodes = builder.build(graph, ref, phasing_result, L=L)
    ebg_cycles.remove_shared_adjacencies(ebg_nodes)
    raw_cycles = ebg_cycles.extract_cycles(ebg_nodes)
    if merge_haplotypes:
        all_cycles = ebg_cycles.merge_partner_cycles(raw_cycles)
        merged_count = len(raw_cycles) - len(all_cycles)
        print(f"      EBG nodes: {len(ebg_nodes)}, raw cycles: {len(raw_cycles)}, "
              f"events (after merging {merged_count} partner copies): {len(all_cycles)}")
    else:
        all_cycles = raw_cycles
        print(f"      EBG nodes: {len(ebg_nodes)}, raw cycles: {len(raw_cycles)}, "
              f"events: {len(all_cycles)}")

    cycles_on_raw = ebg_cycles.remap_cycles_to_nodes(all_cycles, raw_nodes)

    print(f"[5/5] Searching for optimal scenario (method={method}; "
          f"cycles are always split into independent chromosome clusters first)...", flush=True)
    result, per_cluster = enumeration.find_best_scenario_clustered(
        cycles=cycles_on_raw, all_nodes=raw_nodes, timing=timing, method=method,
    )

    if len(per_cluster) > 1:
        print(f"\n      Found {len(per_cluster)} independent chromosome cluster(s):")
        for ci, (idx, cr) in enumerate(per_cluster, 1):
            chrs = set()
            for gi in idx:
                chrs |= cycle_chromosomes(all_cycles[gi])
            print(f"        Cluster {ci}: {len(idx)} cycle(s), chromosomes {sorted(chrs)}, "
                  f"max chromosomes {cr.objective_value}, steps {len(cr.partition)}")

    print(f"\n--- Result ---")
    _print_result(result)

    if out_path:
        out = {
            "status": result.status,
            "method": result.method,
            "exact": result.exact,
            "objective_max_chromosomes": result.objective_value,
            "partition": result.partition,
            "step_costs": result.step_costs,
            "step_num_chromosomes": result.step_num_chromosomes,
        }
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"\nResult written to {out_path}")

    return result


def run_exhaustive(
    jabba_path: str,
    L: int = 2,
    chromosomes: Optional[List[int]] = None,
    dedupe: bool = False,
    cpu_n: int = 1,
    n_workers: int = 1,
    out_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    """
    Default path: enumerate every tied-optimal (phasing x loose-pairing)
    option, run the full downstream pipeline (EBG -> cycles -> scenario
    search) on each, and write a CSV deliverable folder per option plus a
    per-sample summary.csv.

    Returns one row (dict) summarizing the whole sample, suitable for
    collecting into a cross-sample overview table.
    """
    label = label or Path(jabba_path).stem
    print(f"=== {label} ===", flush=True)

    print(f"[1/3] Loading JABBA graph: {jabba_path}"
          f"{f' (chromosomes {chromosomes})' if chromosomes else ''}", flush=True)
    graph = jabba.load(jabba_path, chromosomes=chromosomes)
    ref = reference.build(graph)

    print(f"[2/3] Enumerating every tied-optimal phasing x loose-pairing option "
          f"(cpu={cpu_n}, workers={n_workers}, dedupe={dedupe})...", flush=True)
    options, meta, equivalence_classes = penum.enumerate_full_options(
        graph, ref, L=L, dedupe=dedupe, cpu_n=cpu_n, n_workers=n_workers,
    )
    print(f"      {meta}", flush=True)

    out_root = Path(out_dir) if out_dir else Path("analysis/haplotype_options_report") / label

    if dedupe:
        eq_rows = []
        for rep_label, raw_labels in equivalence_classes.items():
            eq_rows.append({
                "canonical_phasing_label": rep_label,
                "n_raw_ties_collapsed": len(raw_labels),
                "raw_phasing_labels": ";".join(raw_labels),
            })
        eq_rows.sort(key=lambda r: -r["n_raw_ties_collapsed"])
        _write_csv(out_root / "phasing_equivalence_classes.csv", eq_rows, [
            "canonical_phasing_label", "n_raw_ties_collapsed", "raw_phasing_labels",
        ])

    print(f"[3/3] Building EBG + scenario search + writing CSVs for {len(options)} option(s) "
          f"-> {out_root} (workers={n_workers})...", flush=True)
    summaries = report.write_all_options(graph, ref, options, out_root, L=L, n_workers=n_workers)

    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "option_id": s["option_id"],
            "phasing_label": s["phasing_label"],
            "loose_pairing_label": s["loose_pairing_label"],
            "n_components": s["n_components"],
            "n_mixed_haplotype_edges": s["n_mixed_haplotype_edges"],
            "n_ebg_cycles": s["n_cycles"],
            "n_ebg_nodes": s["n_ebg_nodes"],
            "n_possible_scenarios": s["n_scenarios"],
            "scenarios_enumeration_truncated": s["truncated"],
            "optimal_max_chromosomes": s["optimal_objective"],
            "n_optimal_scenarios": s["n_optimal_scenarios"],
            "n_full_scenarios_bruteforce": s["n_full_scenarios"],
            "bnb_expanded_states": s["bnb_expanded_states"],
            "bnb_matches_full_enum": s["bnb_matches_full_enum"],
            "bnb_speedup_factor": s["bnb_speedup_factor"],
            "n_real_chromatids_genome_a": s["n_real_chromatids_genome_a"],
            "n_real_chromatids_genome_b": s["n_real_chromatids_genome_b"],
        })
        print(f"    {s['option_id']} ({s['phasing_label']}, {s['loose_pairing_label']}): "
              f"cycles={s['n_cycles']} scenarios={s['n_scenarios']} "
              f"optimal_max_chr={s['optimal_objective']} "
              f"bnb_states={s['bnb_expanded_states']}/{s['n_full_scenarios']}", flush=True)

    _write_csv(out_root / "summary.csv", summary_rows, [
        "option_id", "phasing_label", "loose_pairing_label", "n_components",
        "n_mixed_haplotype_edges", "n_ebg_cycles", "n_ebg_nodes",
        "n_possible_scenarios", "scenarios_enumeration_truncated",
        "optimal_max_chromosomes", "n_optimal_scenarios",
        "n_full_scenarios_bruteforce", "bnb_expanded_states",
        "bnb_matches_full_enum", "bnb_speedup_factor",
        "n_real_chromatids_genome_a", "n_real_chromatids_genome_b",
    ])

    distinct_objs = sorted({r["optimal_max_chromosomes"] for r in summary_rows})
    print(f"\n{len(options)} option(s) written to {out_root}. "
          f"Distinct optimal_max_chromosomes across options: {distinct_objs}", flush=True)

    return {
        "sample": label,
        "n_subgraphs": meta["n_subgraphs"],
        "n_raw_tied_phasings": meta["n_raw_phasing_options"],
        "n_canonical_tied_phasings": meta["n_canonical_phasing_options"],
        "raw_enumeration_capped": meta["truncated"] or any(meta["per_subgraph_hit_solution_cap"]),
        "n_loose_pairing_variants": meta["n_loose_pairing_variants"],
        "n_full_options_written": len(options),
        "distinct_optimal_max_chromosomes": ";".join(str(v) for v in distinct_objs),
        "objective_is_stable_across_options": len(distinct_objs) == 1,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR_CN: genome rearrangement scenario finder")
    p.add_argument("jabba_path", help="Path to JABBA JSON file")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--chromosomes", type=int, nargs="+", default=None,
                   metavar="CHR", help="Restrict analysis to these chromosome numbers")
    p.add_argument("--quick", action="store_true",
                   help="Run once with a single optimal phasing instead of every tied option")
    p.add_argument("--method", default="dp", choices=["dp", "branch_and_bound"])
    p.add_argument("--timing", default="before", choices=["before", "after", "both"])
    p.add_argument("--cpu", type=int, default=1,
                   help="CP-SAT search workers used within each solve")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel OS processes across independent subgraphs / options")
    # exhaustive mode only
    p.add_argument("--dedupe", action="store_true",
                   help="(exhaustive mode) collapse pure haplotype/copy-label relabelings")
    p.add_argument("--out-dir", default=None,
                   help="(exhaustive mode) where to write the CSV deliverable")
    # quick mode only
    p.add_argument("--time-limit", type=float, default=200.0,
                   help="(quick mode) per-subgraph phasing ILP time limit in seconds")
    p.add_argument("--merge-haplotypes", action="store_true",
                   help="(quick mode) merge same-topology cycles across haplotypes into one event")
    p.add_argument("--out", default=None, help="(quick mode) write the scenario result to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.quick:
        run_quick(
            jabba_path=args.jabba_path,
            L=args.L,
            time_limit_s=args.time_limit,
            cpu_n=args.cpu,
            n_workers=args.workers,
            method=args.method,
            timing=args.timing,
            chromosomes=args.chromosomes,
            merge_haplotypes=args.merge_haplotypes,
            out_path=args.out,
        )
    else:
        run_exhaustive(
            jabba_path=args.jabba_path,
            L=args.L,
            chromosomes=args.chromosomes,
            dedupe=args.dedupe,
            cpu_n=args.cpu,
            n_workers=args.workers,
            out_dir=args.out_dir,
        )
