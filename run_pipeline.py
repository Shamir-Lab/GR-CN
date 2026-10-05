"""
GR-CN pipeline: from JaBbA graphs to the paper's figures and tables.

For each sample:
  1. parse the JaBbA graph (Genome B) and reconstruct the diploid Genome A;
  2. phase Genome B: enumerate every tied-optimal phasing;
  3. for every phasing, build the EBG, extract its cycles and enumerate all
     scenarios (ordered partitions of the cycles), recording the distribution
     of C_max and the search effort of full enumeration vs branch-and-bound;
  4. select the phasing whose scenarios vary most in C_max;
  5. write the EBG cycle data of the selected phasing (Genome A/B, EBG edges,
     cycles, every scenario), its optimum and representative scenarios.
Then, over all samples:
  6. figures: representative scenarios; complete scenarios, search-tree nodes
     and running time of full enumeration vs branch-and-bound;
  7. supplementary tables S1-S3.

Outputs (results/):
  <sample>/phasings.csv        every tied phasing: cycles, C_max distribution, effort
  <sample>/selected_phasing/   EBG cycle data of the selected phasing (CSV)
  <sample>/summary.json        selection, optimum, representative scenarios, effort
  figures/                     PDF + PNG
  tables/                      CSV + LaTeX

Usage:
    python run_pipeline.py                     # everything (~1.5 h, dominated by phasing 852)
    python run_pipeline.py --reuse-phasing     # reuse cached phasings (minutes)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

from gr_cn import analysis, figures, report, tables
from gr_cn.io import jabba, reference
from gr_cn.phasing import enumerate as penum
from gr_cn.scenario import enumeration

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SAMPLES = {  # name: (JaBbA graph, chromosomes of the event; 23 = X)
    "CA-0098": ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23]),
    "PM1284": ("jabba_data/graphs_all/PM1284.json", [3, 11, 12]),
    "852_T1_33_23905": ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14]),
}


def phase(name: str, graph, ref, cpu_n: int, reuse: bool):
    """Stage 2. Every tied-optimal phasing (x tied loose-end pairing).
    Cached, since this is the slow stage."""
    cache = RESULTS / name / ".phasings.pkl"
    if reuse and cache.exists():
        print("  2. phasing: reusing cached phasings", flush=True)
        return pickle.loads(cache.read_bytes())
    t0 = time.perf_counter()
    options, meta = penum.enumerate_full_options(graph, ref, L=2, cpu_n=cpu_n)
    meta["seconds"] = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((options, meta)))
    print(f"  2. phasing: {meta['n_phasings']} tied phasing(s), {len(options)} option(s) "
          f"with loose-end pairings, in {meta['seconds']:.0f}s", flush=True)
    return options, meta


def run_sample(name: str, cpu_n: int, repeats: int, reuse: bool):
    print(f"== {name}", flush=True)
    out = RESULTS / name
    rel, chroms = SAMPLES[name]

    # 1. parse Genome B, reconstruct Genome A
    graph = jabba.load(str(ROOT / rel), chromosomes=chroms)
    ref = reference.build(graph)
    print(f"  1. parsed: {len(graph.intervals)} intervals, {len(graph.alt_adjacencies)} ALT adjacencies",
          flush=True)

    # 2. phasing
    options, meta = phase(name, graph, ref, cpu_n, reuse)

    # 3. EBG cycles, scenarios and search effort of every phasing
    analyses = [analysis.analyze_phasing(graph, ref, opt, repeats) for opt in options]
    print(f"  3. analyzed {len(analyses)} phasing(s)", flush=True)

    # 4. select the phasing whose scenarios vary most in C_max
    chosen = analysis.select_phasing(analyses)
    opt = chosen["option"]
    print(f"  4. selected {opt['phasing_label']}/{opt['loose_label']}: {len(chosen['cycles'])} cycles, "
          f"C_max {dict(sorted(chosen['cmax_distribution'].items()))}", flush=True)

    rows = []
    for a in analyses:
        o, d, e = a["option"], a["cmax_distribution"], a["effort"]
        rows.append({
            "option_id": o["option_id"], "phasing_label": o["phasing_label"],
            "loose_pairing_label": o["loose_label"],
            "n_cycles": len(a["cycles"]), "optimal_cmax": e["optimum"],
            "n_distinct_cmax": len(d), "cmax_entropy": a["cmax_entropy"],
            "cmax_distribution": ";".join(f"{k}:{d[k]}" for k in sorted(d)),
            "enum_complete_scenarios": e["enumeration"]["complete_scenarios"],
            "enum_tree_nodes": e["enumeration"]["tree_nodes"],
            "enum_seconds": e["enumeration"]["seconds"],
            "bnb_complete_scenarios": e["branch_and_bound"]["complete_scenarios"],
            "bnb_tree_nodes": e["branch_and_bound"]["tree_nodes"],
            "bnb_seconds": e["branch_and_bound"]["seconds"],
            "selected": a is chosen,
        })
    with (out / "phasings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 5. EBG cycle data, optimum and representative scenarios of the selected phasing
    report.write_option_folder(graph, ref, opt["phasing"], opt["extra_pairs"],
                               out / "selected_phasing", L=2)
    cycles, nodes = chosen["cycles"], chosen["nodes"]
    optimal = enumeration._dp_exact(enumeration.DynamicGenomeScenario(cycles, nodes), analysis.TIMING)
    reps = analysis.representative_scenarios(analysis.all_scenarios(cycles, nodes), optimal)
    summary = {
        "sample": name,
        "n_tied_phasings": meta["n_phasings"],
        "n_options": len(options),  # tied phasing x tied loose-end pairing
        "phasing_search_capped": bool(meta["truncated"] or any(meta["per_subgraph_hit_solution_cap"])),
        "selected": {"option_id": opt["option_id"], "phasing_label": opt["phasing_label"],
                     "loose_pairing_label": opt["loose_label"], "n_cycles": len(cycles)},
        "cmax_distribution": {str(k): v for k, v in sorted(chosen["cmax_distribution"].items())},
        "optimal_scenario": {"cmax": optimal.objective_value, "partition": optimal.partition,
                             "step_costs": optimal.step_costs},
        "representative_scenarios": reps,
        "effort": chosen["effort"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    e = chosen["effort"]
    print(f"  5. optimum {e['optimum']}; enumeration {e['enumeration']['seconds'] * 1e3:.2f} ms "
          f"vs branch-and-bound {e['branch_and_bound']['seconds'] * 1e3:.2f} ms", flush=True)
    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cpu", type=int, default=1, help="CP-SAT workers per phasing solve (1 = deterministic)")
    p.add_argument("--repeats", type=int, default=5, help="timing repeats per method")
    p.add_argument("--reuse-phasing", action="store_true",
                   help="reuse the phasings cached by an earlier run (skips the slow stage)")
    a = p.parse_args()

    phasings, summaries = {}, []
    for name in SAMPLES:
        rows, summary = run_sample(name, a.cpu, a.repeats, a.reuse_phasing)
        phasings[name] = rows
        summaries.append(summary)

    figures.draw_all(summaries, RESULTS / "figures")   # 6. figures
    tables.write_all(phasings, summaries, RESULTS / "tables")  # 7. tables
    print(f"== done: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
