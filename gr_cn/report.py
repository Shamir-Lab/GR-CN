"""
CSV writers for the EBG cycle data of one phasing:
  genome_a.csv              -- Genome A (diploid reference), identical across options
  genome_b.csv               -- Genome B connectivity for this option
  ebg_structure.csv          -- EBG edges (Definition 4; shared adjacencies stripped)
  ebg_components_summary.csv -- one row per extracted EBG cycle/component
  scenarios_summary.csv      -- one row per possible (k,d,g)-break scenario ordering
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Set

from gr_cn.ebg import builder, cycles as ebg_cycles
from gr_cn.models.node import EBGNode, NodeType
from gr_cn.scenario import enumeration


def cycle_chromosomes(cycle: List[EBGNode]) -> Set[int]:
    """Return the set of reference chromosomes touched by a cycle."""
    return {
        node.ref_chromosome
        for node in cycle
        if node.node_type is NodeType.INTERNAL and node.ref_chromosome > 0
    }


# Enumerating every ordered partition (Fubini number) is only sane up to a
# handful of cycles; above this, fall back to the best-found scenario only
# and note the truncation explicitly rather than silently writing nothing.
MAX_CYCLES_FOR_FULL_ENUMERATION = 7


def write_genome_a_csv(reference, out_path: Path) -> None:
    rows = []
    for info in sorted(reference.cn_info, key=lambda i: (i.interval.chromosome, i.interval.start)):
        iv = info.interval
        rows.append({
            "iid": iv.iid,
            "chromosome": iv.chrom_str,
            "start": iv.start,
            "end": iv.end,
            "length": iv.length,
            "diploid_cn": info.diploid_cn,
            "cancer_cn_in_sample": info.cancer_cn,
            "n_deleted_copies": info.n_deleted,
            "n_gained_copies": info.n_gained,
            "is_telomere_tail": iv.is_telomere_tail,
            "is_telomere_head": iv.is_telomere_head,
            "is_centromere_included": iv.is_centromere_included,
        })
    _write_csv(out_path, rows, [
        "iid", "chromosome", "start", "end", "length", "diploid_cn",
        "cancer_cn_in_sample", "n_deleted_copies", "n_gained_copies",
        "is_telomere_tail", "is_telomere_head", "is_centromere_included",
    ])


def _node_row(n) -> dict:
    return {
        "node_type": n.node_type.value,
        "iid": n.iid,
        "side": n.end.value,
        "haplotype": n.haplotype,
        "copy_index": n.copy_index,
        "chromosome": n.ref_chromosome,
        "position": n.ref_position,
    }


def write_genome_b_csv(raw_nodes, out_path: Path) -> None:
    rows = []
    for n in raw_nodes:
        if n.node_type is not NodeType.INTERNAL:
            continue
        partner = n.grey_neighbor
        row = _node_row(n)
        row["chromosome"] = n.ref_chromosome
        row["is_telomere"] = n.is_telomere
        if partner is None:
            row.update({
                "partner_type": "NONE", "partner_iid": "", "partner_side": "",
                "partner_haplotype": "", "partner_copy_index": "", "edge_type": "UNRESOLVED",
            })
        else:
            row.update({
                "partner_type": partner.node_type.value,
                "partner_iid": partner.iid,
                "partner_side": partner.end.value,
                "partner_haplotype": partner.haplotype,
                "partner_copy_index": partner.copy_index,
                "edge_type": "REF_shared" if n.black_neighbor is n.grey_neighbor else "ALT_rearranged",
            })
        rows.append(row)
    rows.sort(key=lambda r: (r["chromosome"], r["iid"], r["haplotype"], r["copy_index"], r["side"]))
    _write_csv(out_path, rows, [
        "node_type", "iid", "side", "haplotype", "copy_index", "chromosome", "position",
        "is_telomere", "partner_type", "partner_iid", "partner_side", "partner_haplotype",
        "partner_copy_index", "edge_type",
    ])


def write_ebg_structure_csv(ebg_nodes, raw_cycles, out_path: Path) -> Dict[tuple, int]:
    """Returns node.key() -> cycle_id map (for reuse by the components summary)."""
    cycle_of_key: Dict[tuple, int] = {}
    for ci, cyc in enumerate(raw_cycles):
        for n in cyc:
            cycle_of_key[n.key()] = ci

    rows = []
    seen_pairs = set()

    def emit(u, v, color):
        canon = tuple(sorted((u.key(), v.key())))
        edge_key = (canon, color)
        if edge_key in seen_pairs:
            return
        seen_pairs.add(edge_key)
        cyc_id = cycle_of_key.get(u.key(), cycle_of_key.get(v.key(), -1))
        row = {"color": color, "cycle_id": cyc_id}
        ur, vr = _node_row(u), _node_row(v)
        row.update({f"u_{k}": val for k, val in ur.items()})
        row.update({f"v_{k}": val for k, val in vr.items()})
        rows.append(row)

    for n in ebg_nodes:
        if n.black_neighbor is not None:
            emit(n, n.black_neighbor, "black")
        if n.grey_neighbor is not None:
            emit(n, n.grey_neighbor, "grey")

    rows.sort(key=lambda r: (r["cycle_id"], r["color"], r["u_iid"]))
    node_cols = ["node_type", "iid", "side", "haplotype", "copy_index", "chromosome", "position"]
    cols = ["cycle_id", "color"] + [f"u_{k}" for k in node_cols] + [f"v_{k}" for k in node_cols]
    _write_csv(out_path, rows, cols)
    return cycle_of_key


def write_ebg_components_summary_csv(raw_cycles, out_path: Path) -> None:
    rows = []
    for ci, cyc in enumerate(raw_cycles):
        ordered = ebg_cycles.order_cycle(cyc)
        n_black = sum(1 for n in cyc if n.black_neighbor is not None and n.black_neighbor in cyc)
        n_grey = sum(1 for n in cyc if n.grey_neighbor is not None and n.grey_neighbor in cyc)
        chroms = sorted(cycle_chromosomes(cyc))
        haps = sorted({n.haplotype for n in cyc})
        types = sorted({n.node_type.value for n in cyc})
        node_list = ";".join(
            f"{n.node_type.value}:{n.iid}{n.end.value}:h{n.haplotype}:c{n.copy_index}"
            for n in (ordered or cyc)
        )
        rows.append({
            "cycle_id": ci,
            "n_nodes": len(cyc),
            "n_black_edges": n_black // 2,
            "n_grey_edges": n_grey // 2,
            "chromosomes": ";".join(str(c) for c in chroms),
            "haplotypes_involved": ";".join(str(h) for h in haps),
            "node_types": ";".join(types),
            "ordered_nodes": node_list,
        })
    _write_csv(out_path, rows, [
        "cycle_id", "n_nodes", "n_black_edges", "n_grey_edges", "chromosomes",
        "haplotypes_involved", "node_types", "ordered_nodes",
    ])


def write_scenarios_summary_csv(cycles_on_raw, raw_nodes, out_path: Path) -> dict:
    """
    One row per possible scenario (ordered partition of the EBG cycles).
    Returns {"n_cycles", "n_scenarios", "optimal_objective", "n_optimal_scenarios",
             "truncated"} for the caller's per-sample summary table.
    """
    n_cycles = len(cycles_on_raw)
    # 'real' chromatid counts (deleted copies and gained-copy AH bookkeeping
    # excluded, per DynamicGenomeScenario.is_real / _is_ordinary_synteny) at
    # the two structural endpoints: state=0 is Genome A's own connectivity,
    # state=full_mask is Genome B's (every cycle's grey/rearranged edges
    # applied). These are NOT necessarily 2*n_chromosomes -- see the note on
    # n_real_chromatids_genome_b in the caller for why.
    chrom_model = enumeration.DynamicGenomeScenario(cycles_on_raw, raw_nodes, "real")
    n_chromatids_a = chrom_model.n_chromosomes(0)
    n_chromatids_b = chrom_model.n_chromosomes(chrom_model.full_mask)

    if n_cycles == 0:
        _write_csv(out_path, [], ["scenario_id", "n_steps", "objective_max_chromosomes",
                                    "is_global_optimal", "steps", "step_costs",
                                    "step_chromosomes_before"])
        return {"n_cycles": 0, "n_scenarios": 0, "optimal_objective": 0,
                "n_optimal_scenarios": 0, "truncated": False, "n_full_scenarios": 0,
                "bnb_expanded_states": 0, "bnb_objective": 0, "bnb_matches_full_enum": True,
                "bnb_speedup_factor": 1.0,
                "n_real_chromatids_genome_a": n_chromatids_a,
                "n_real_chromatids_genome_b": n_chromatids_b}

    # Exact branch-and-bound, run unconditionally (even when the full
    # enumeration below is also tractable) so expanded_states gives a direct,
    # per-option empirical measurement of B&B's pruning vs brute-force
    # enumerating every Fubini(n_cycles) ordering.
    bnb = enumeration._exact_branch_and_bound(chrom_model, timing="before")

    truncated = n_cycles > MAX_CYCLES_FOR_FULL_ENUMERATION
    rows = []
    if not truncated:
        results = list(enumeration.enumerate_all_scenarios(
            cycles_on_raw, raw_nodes, max_cycles=MAX_CYCLES_FOR_FULL_ENUMERATION,
        ))
        best_obj = min(r.objective_value for r in results)
        for sid, r in enumerate(results, 1):
            rows.append({
                "scenario_id": sid,
                "n_steps": len(r.partition),
                "objective_max_chromosomes": r.objective_value,
                "is_global_optimal": r.objective_value == best_obj,
                "steps": "|".join(",".join(str(c) for c in step) for step in r.partition),
                "step_costs": "|".join(str(c) for c in r.step_costs),
                "step_chromosomes_before": "|".join(str(c) for c in r.step_num_chromosomes),
            })
        n_optimal = sum(1 for r in rows if r["is_global_optimal"])
    else:
        best = enumeration._dp_exact(chrom_model, timing="before")
        rows.append({
            "scenario_id": 1,
            "n_steps": len(best.partition),
            "objective_max_chromosomes": best.objective_value,
            "is_global_optimal": True,
            "steps": "|".join(",".join(str(c) for c in step) for step in best.partition),
            "step_costs": "|".join(str(c) for c in best.step_costs),
            "step_chromosomes_before": "|".join(str(c) for c in best.step_num_chromosomes),
        })
        best_obj = best.objective_value
        n_optimal = 1

    _write_csv(out_path, rows, ["scenario_id", "n_steps", "objective_max_chromosomes",
                                  "is_global_optimal", "steps", "step_costs",
                                  "step_chromosomes_before"])

    # Fubini(n_cycles) = true size of the brute-force search space, regardless
    # of whether the full listing above was actually generated (the truncated
    # case still needs this for the B&B-vs-brute-force comparison).
    n_full_scenarios = _fubini(n_cycles)

    return {
        "n_cycles": n_cycles, "n_scenarios": len(rows), "optimal_objective": best_obj,
        "n_optimal_scenarios": n_optimal, "truncated": truncated,
        "n_full_scenarios": n_full_scenarios,
        "bnb_expanded_states": bnb.expanded_states,
        "bnb_objective": bnb.objective_value,
        "bnb_matches_full_enum": bnb.objective_value == best_obj,
        "bnb_speedup_factor": round(n_full_scenarios / max(1, bnb.expanded_states), 2),
        "n_real_chromatids_genome_a": n_chromatids_a,
        "n_real_chromatids_genome_b": n_chromatids_b,
    }


def _fubini(n: int) -> int:
    """Ordered Bell number a(n) -- number of ordered partitions of an
    n-element set, i.e. the size of the brute-force scenario search space."""
    from math import comb
    if n == 0:
        return 1
    return sum(comb(n, k) * _fubini(n - k) for k in range(1, n + 1))


def _write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_option_downstream(graph, reference, phasing, extra_pairs, L: int = 2) -> dict:
    """Run EBG-build + cycle extraction (independent haplotypes, partner
    cycles not merged) for one phasing option."""
    phasing = replace(phasing, grey_pairs=list(phasing.grey_pairs) + list(extra_pairs))

    raw_nodes = builder.build(graph, reference, phasing, L=L)
    ebg_nodes = builder.build(graph, reference, phasing, L=L)
    ebg_cycles.remove_shared_adjacencies(ebg_nodes)
    raw_cycles = ebg_cycles.extract_cycles(ebg_nodes)
    cycles_on_raw = ebg_cycles.remap_cycles_to_nodes(raw_cycles, raw_nodes)

    return {
        "phasing": phasing,
        "raw_nodes": raw_nodes,
        "ebg_nodes": ebg_nodes,
        "raw_cycles": raw_cycles,
        "cycles_on_raw": cycles_on_raw,
    }


def write_option_folder(graph, reference, phasing, extra_pairs, out_dir: Path, L: int = 2) -> dict:
    dr = build_option_downstream(graph, reference, phasing, extra_pairs, L=L)

    write_genome_a_csv(reference, out_dir / "genome_a.csv")
    write_genome_b_csv(dr["raw_nodes"], out_dir / "genome_b.csv")
    write_ebg_structure_csv(dr["ebg_nodes"], dr["raw_cycles"], out_dir / "ebg_structure.csv")
    write_ebg_components_summary_csv(dr["raw_cycles"], out_dir / "ebg_components_summary.csv")
    scen_meta = write_scenarios_summary_csv(dr["cycles_on_raw"], dr["raw_nodes"],
                                             out_dir / "scenarios_summary.csv")

    return {
        "n_ebg_nodes": len(dr["ebg_nodes"]),
        "n_cycles": len(dr["raw_cycles"]),
        "n_mixed_haplotype_edges": dr["phasing"].num_mixed_haplotype_edges,
        "n_components": dr["phasing"].num_components,
        **scen_meta,
    }
