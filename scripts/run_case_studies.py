"""
Case-study pipeline: from each JaBbA graph to one analyzed phasing.

For each of the three samples:
  1. load the JaBbA graph and reconstruct Genome A;
  2. phase Genome B: enumerate every distinct (canonical) tied-optimal phasing;
  3. for each phasing, build the EBG and enumerate all of its scenarios
     (ordered partitions of the cycles), recording the distribution of C_max;
  4. select the phasing whose scenarios vary most in the objective
     (see select_phasing), and for it
       - write its CSV deliverable (Genome A/B, EBG, every scenario),
       - find the optimum (subset DP) and representative scenarios,
       - measure search effort of full enumeration vs. branch-and-bound:
         complete scenarios generated, search-tree nodes and running time.

Outputs, per sample, in analysis/case_studies/<sample>/:
  phasing_options.csv   one row per distinct phasing, with its C_max distribution
  selected_phasing/     CSV deliverable of the selected phasing (see README)
  case_study.json       selection, distribution, representative scenarios, effort
Figures are drawn from these by scripts/plot_case_studies.py.

Usage:
    python scripts/run_case_studies.py                    # all samples
    python scripts/run_case_studies.py --sample PM1284
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr_cn import report  # noqa: E402
from gr_cn.io import jabba, reference  # noqa: E402
from gr_cn.phasing import enumerate as penum  # noqa: E402
from gr_cn.scenario import enumeration  # noqa: E402

OUT = ROOT / "analysis" / "case_studies"
TIMING = "before"

SAMPLES = {
    "CA-0098": ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23]),
    "PM1284": ("jabba_data/graphs_all/PM1284.json", [3, 11, 12]),
    "852_T1_33_23905": ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14]),
}


def _entropy(dist: Counter) -> float:
    n = sum(dist.values())
    return -sum(c / n * math.log2(c / n) for c in dist.values())


def select_phasing(rows: list) -> dict:
    """
    The phasing whose scenarios vary most in C_max: most distinct C_max
    values, then most cycles (the richer scenario space), then the highest
    entropy of the C_max distribution. Ties keep the first phasing found.
    """
    key = lambda r: (r["n_distinct_cmax"], r["n_cycles"], round(r["cmax_entropy"], 12))
    best = rows[0]
    for r in rows[1:]:
        if key(r) > key(best):
            best = r
    return best


def representative_scenarios(scenarios: list, optimal: enumeration.ScenarioResult) -> list:
    """
    Up to four scenarios with distinct C_max: all at once (one step); the
    worst multi-step scenario with a different C_max (which can be worse
    than all at once); one in between; and the optimum. For each value the
    scenario with fewest steps (first found) is used.
    """
    def fewest_steps(value, min_steps=1):
        cands = [s for s in scenarios if s.objective_value == value and len(s.partition) >= min_steps]
        return min(cands, key=lambda s: len(s.partition)) if cands else None

    joint = next(s for s in scenarios if len(s.partition) == 1)
    picks = [("All at once", joint)]
    multi = sorted({s.objective_value for s in scenarios
                    if len(s.partition) > 1 and s.objective_value != joint.objective_value})
    if multi:
        worst = multi[-1]
        between = [v for v in multi if optimal.objective_value < v < worst]
        picks.append(("Suboptimal", fewest_steps(worst, 2)))
        if between:
            picks.append(("Intermediate", fewest_steps(between[(len(between) - 1) // 2], 2)))
    picks.append(("Optimal", optimal))

    out, used = [], set()
    for name, s in picks:
        if s is None or (s.objective_value in used and name != "Optimal"):
            continue
        out = [o for o in out if o["cmax"] != s.objective_value]  # optimum wins a tie
        used.add(s.objective_value)
        out.append({"name": name, "cmax": s.objective_value,
                    "partition": s.partition, "step_costs": s.step_costs})
    return out


def measure_effort(cycles, nodes, repeats: int) -> dict:
    """Full enumeration vs. branch-and-bound (and DP, for reference), each on
    a fresh cost model per run; time = median over `repeats` runs."""
    def enum_once():
        stats = {}
        t0 = time.perf_counter()
        scen = list(enumeration.enumerate_all_scenarios(cycles, nodes, timing=TIMING,
                                                        max_cycles=len(cycles), stats=stats))
        return time.perf_counter() - t0, scen, stats

    def bnb_once():
        model = enumeration.DynamicGenomeScenario(cycles, nodes)
        t0 = time.perf_counter()
        res = enumeration._exact_branch_and_bound(model, TIMING)
        return time.perf_counter() - t0, res

    def dp_once():
        model = enumeration.DynamicGenomeScenario(cycles, nodes)
        t0 = time.perf_counter()
        res = enumeration._dp_exact(model, TIMING)
        return time.perf_counter() - t0, res

    enum_runs = [enum_once() for _ in range(repeats)]
    bnb_runs = [bnb_once() for _ in range(repeats)]
    dp_runs = [dp_once() for _ in range(repeats)]
    _, scen, stats = enum_runs[0]
    bnb, dp = bnb_runs[0][1], dp_runs[0][1]
    optimum = min(s.objective_value for s in scen)
    assert bnb.objective_value == dp.objective_value == optimum, "search methods disagree"
    return {
        "optimum": optimum,
        "enumeration": {"complete_scenarios": len(scen), "tree_nodes": stats["nodes"],
                        "seconds": statistics.median(r[0] for r in enum_runs)},
        "branch_and_bound": {"complete_scenarios": bnb.n_complete_scenarios,
                             "tree_nodes": bnb.expanded_states,
                             "seconds": statistics.median(r[0] for r in bnb_runs)},
        "dp": {"seconds": statistics.median(r[0] for r in dp_runs)},
        "repeats": repeats,
    }


def run_sample(name: str, cpu_n: int, repeats: int, reuse_phasing: bool = False) -> None:
    rel, chroms = SAMPLES[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"== {name}: phasing", flush=True)
    graph = jabba.load(str(ROOT / rel), chromosomes=chroms)
    ref = reference.build(graph)
    cache = out / ".phasings.pkl"  # phasing is the slow step; reuse it with --reuse-phasing
    if reuse_phasing and cache.exists():
        options, meta, phasing_seconds = pickle.loads(cache.read_bytes())
        print("   reusing cached phasings", flush=True)
    else:
        t0 = time.perf_counter()
        options, meta, _eq = penum.enumerate_full_options(graph, ref, L=2, dedupe=True, cpu_n=cpu_n)
        phasing_seconds = time.perf_counter() - t0
        cache.write_bytes(pickle.dumps((options, meta, phasing_seconds)))
    print(f"   {len(options)} distinct phasing(s) in {phasing_seconds:.0f}s", flush=True)

    rows, downstream = [], {}
    for opt in options:
        dr = report.build_option_downstream(graph, ref, opt["phasing"], opt["extra_pairs"], L=2)
        cycles, nodes = dr["cycles_on_raw"], dr["raw_nodes"]
        scen = list(enumeration.enumerate_all_scenarios(cycles, nodes, timing=TIMING,
                                                        max_cycles=len(cycles)))
        dist = Counter(s.objective_value for s in scen)
        rows.append({
            "option_id": opt["option_id"], "phasing_label": opt["phasing_label"],
            "loose_pairing_label": opt["loose_label"], "n_cycles": len(cycles),
            "n_scenarios": len(scen), "optimal_cmax": min(dist),
            "n_distinct_cmax": len(dist), "cmax_entropy": _entropy(dist),
            "cmax_distribution": ";".join(f"{k}:{dist[k]}" for k in sorted(dist)),
        })
        downstream[opt["option_id"]] = (opt, cycles, nodes, scen, dist)

    chosen = select_phasing(rows)
    for r in rows:
        r["selected"] = r is chosen
    with (out / "phasing_options.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"   selected {chosen['phasing_label']}/{chosen['loose_pairing_label']}: "
          f"{chosen['n_cycles']} cycles, C_max distribution {chosen['cmax_distribution']}", flush=True)

    opt, cycles, nodes, scen, dist = downstream[chosen["option_id"]]
    report.write_option_folder(graph, ref, opt["phasing"], opt["extra_pairs"],
                               out / "selected_phasing", L=2)
    optimal = enumeration._dp_exact(enumeration.DynamicGenomeScenario(cycles, nodes), TIMING)
    effort = measure_effort(cycles, nodes, repeats)
    print(f"   optimum {effort['optimum']}; enumeration "
          f"{effort['enumeration']['seconds'] * 1e3:.2f} ms vs B&B "
          f"{effort['branch_and_bound']['seconds'] * 1e3:.2f} ms", flush=True)

    result = {
        "sample": name,
        "n_distinct_phasings": len(options),
        "n_raw_phasings": meta["n_raw_phasing_options"],
        "phasing_seconds": phasing_seconds,
        "selected": {k: chosen[k] for k in ("option_id", "phasing_label", "loose_pairing_label",
                                            "n_cycles", "n_scenarios", "n_distinct_cmax",
                                            "cmax_entropy")},
        "cmax_distribution": {str(k): dist[k] for k in sorted(dist)},
        "optimal_scenario": {"cmax": optimal.objective_value, "partition": optimal.partition,
                             "step_costs": optimal.step_costs},
        "representative_scenarios": representative_scenarios(scen, optimal),
        "effort": effort,
    }
    (out / "case_study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample", choices=list(SAMPLES), nargs="+", default=list(SAMPLES))
    p.add_argument("--cpu", type=int, default=1, help="CP-SAT workers per phasing solve (1 = deterministic)")
    p.add_argument("--repeats", type=int, default=5, help="timing repeats per method")
    p.add_argument("--reuse-phasing", action="store_true",
                   help="reuse the phasings cached by an earlier run (skips the slow step)")
    a = p.parse_args()
    for s in a.sample:
        run_sample(s, a.cpu, a.repeats, a.reuse_phasing)


if __name__ == "__main__":
    main()
