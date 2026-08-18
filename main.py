"""
GR_CN — main pipeline entry point.

Usage:
    python main.py <jabba_json_path> [options]

Options:
    --L INT                  Number of haplotypes (default: 2)
    --time-limit FLOAT       Per-subgraph phasing ILP time limit in seconds (default: 200)
    --cpu INT                CP-SAT worker threads (default: 1)
    --method STR             Scenario search method: auto|exact|ilp|beam|greedy (default: auto)
    --exact-max INT          Max cycles for exact B&B search in auto mode (default: 16)
    --ilp-max INT            Max cycles for DP-exact ILP in ilp mode (default: 18)
    --timing STR             Cost timing: before|after|both (default: before)
    --chromosomes INT [INT]  Restrict to these chromosome numbers (e.g. --chromosomes 1 5 7 22)
    --no-cluster             Disable independent cluster optimization
    --merge-haplotypes       Merge same-topology cycles across haplotypes into one atomic
                              multi-haplotype event (old default). By default, each haplotype
                              copy of a homozygous rearrangement is scheduled as its own
                              independent scenario step.
    --out PATH               Write scenario result to this JSON file (optional)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from gr_cn.io import jabba, reference
from gr_cn.phasing import ilp, loose
from gr_cn.ebg import builder, cycles as ebg_cycles
from gr_cn.scenario import enumeration
from gr_cn.scenario.clustering import cluster_cycles, cycle_chromosomes


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


def run(
    jabba_path: str,
    L: int = 2,
    time_limit_s: float = 200.0,
    cpu_n: int = 1,
    method: str = "auto",
    exact_max_cycles: int = 16,
    ilp_max_cycles: int = 18,
    timing: str = "before",
    chromosomes: Optional[List[int]] = None,
    use_clustering: bool = True,
    independent_haplotypes: bool = True,
    out_path: Optional[str] = None,
) -> enumeration.ScenarioResult:

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

    print(f"[3/5] Running phasing ILP (L={L}, time_limit={time_limit_s}s)...", flush=True)
    phasing_result = ilp.phase(graph, ref, L=L, time_limit_s=time_limit_s, cpu_n=cpu_n)
    print(f"      Status: {phasing_result.status}, "
          f"components: {phasing_result.num_components}, "
          f"mixed-haplotype edges: {phasing_result.num_mixed_haplotype_edges}")

    # Connect whichever haplotype copies the ILP left open at a genuine
    # (non-telomeric) loose breakpoint.
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
    # Cycle extraction needs its own (mutated) copy -- rebuild rather than
    # deep-copy so raw_nodes stays unstripped for the scenario search below,
    # which needs the untouched backbone connectivity remove_shared_
    # adjacencies deliberately nulls out here.
    ebg_nodes = builder.build(graph, ref, phasing_result, L=L)
    ebg_cycles.remove_shared_adjacencies(ebg_nodes)
    raw_cycles = ebg_cycles.extract_cycles(ebg_nodes)
    if independent_haplotypes:
        all_cycles = raw_cycles
        print(f"      EBG nodes: {len(ebg_nodes)}, "
              f"raw cycles: {len(raw_cycles)}, "
              f"events: {len(all_cycles)} "
              f"(--independent-haplotypes: partner-cycle merging skipped)")
    else:
        all_cycles = ebg_cycles.merge_partner_cycles(raw_cycles)
        merged_count = len(raw_cycles) - len(all_cycles)
        print(f"      EBG nodes: {len(ebg_nodes)}, "
              f"raw cycles: {len(raw_cycles)}, "
              f"events (after merging {merged_count} partner copies): {len(all_cycles)}")

    # all_cycles' nodes come from the same builder.build() call as ebg_nodes,
    # a distinct object set from raw_nodes -- remap onto raw_nodes' objects
    # so the scenario search can see real (unstripped) segment connectivity.
    cycles_on_raw = ebg_cycles.remap_cycles_to_nodes(all_cycles, raw_nodes)

    print(f"[5/5] Searching for optimal scenario (method={method})...", flush=True)

    if use_clustering and len(all_cycles) > 1:
        cluster_indices = cluster_cycles(all_cycles)
        if len(cluster_indices) > 1:
            print(f"      Found {len(cluster_indices)} independent chromosome cluster(s):")
            for ci, idx in enumerate(cluster_indices, 1):
                chrs = set()
                for gi in idx:
                    chrs |= cycle_chromosomes(all_cycles[gi])
                print(f"        Cluster {ci}: {len(idx)} cycle(s), "
                      f"chromosomes {sorted(chrs)}")

            result, per_cluster = enumeration.find_best_scenario_clustered(
                cycles=cycles_on_raw,
                all_nodes=raw_nodes,
                timing=timing,
                method=method,
                exact_max_cycles=exact_max_cycles,
                ilp_max_cycles=ilp_max_cycles,
            )

            print(f"\n--- Per-cluster Results ---")
            for ci, (idx, cr) in enumerate(per_cluster, 1):
                chrs = set()
                for gi in idx:
                    chrs |= cycle_chromosomes(all_cycles[gi])
                print(f"\n  Cluster {ci} (chr {sorted(chrs)}, "
                      f"{len(idx)} cycles, method={cr.method}):")
                print(f"    Max chromosomes: {cr.objective_value}  "
                      f"Steps: {len(cr.partition)}  Exact: {cr.exact}")

            print(f"\n--- Combined Result ---")
            _print_result(result)
        else:
            result = enumeration.find_best_scenario(
                cycles=cycles_on_raw,
                all_nodes=raw_nodes,
                timing=timing,
                method=method,
                exact_max_cycles=exact_max_cycles,
                ilp_max_cycles=ilp_max_cycles,
            )
            print(f"\n--- Result ---")
            _print_result(result)
    else:
        result = enumeration.find_best_scenario(
            cycles=cycles_on_raw,
            all_nodes=raw_nodes,
            timing=timing,
            method=method,
            exact_max_cycles=exact_max_cycles,
            ilp_max_cycles=ilp_max_cycles,
        )
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR_CN: genome rearrangement scenario finder")
    p.add_argument("jabba_path", help="Path to JABBA JSON file")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--time-limit", type=float, default=200.0)
    p.add_argument("--cpu", type=int, default=1)
    p.add_argument("--method", default="auto",
                   choices=["auto", "exact", "ilp", "beam", "greedy"])
    p.add_argument("--exact-max", type=int, default=16)
    p.add_argument("--ilp-max", type=int, default=18)
    p.add_argument("--timing", default="before",
                   choices=["before", "after", "both"])
    p.add_argument("--chromosomes", type=int, nargs="+", default=None,
                   metavar="CHR",
                   help="Restrict analysis to these chromosome numbers")
    p.add_argument("--no-cluster", action="store_true",
                   help="Disable independent cluster optimization")
    p.add_argument("--merge-haplotypes", action="store_true",
                   help="Merge same-topology cycles across haplotypes into one atomic "
                        "multi-haplotype event (old default). By default each haplotype "
                        "copy of a homozygous rearrangement is scheduled independently.")
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        jabba_path=args.jabba_path,
        L=args.L,
        time_limit_s=args.time_limit,
        cpu_n=args.cpu,
        method=args.method,
        exact_max_cycles=args.exact_max,
        ilp_max_cycles=args.ilp_max,
        timing=args.timing,
        chromosomes=args.chromosomes,
        use_clustering=not args.no_cluster,
        independent_haplotypes=not args.merge_haplotypes,
        out_path=args.out,
    )
