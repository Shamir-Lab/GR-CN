"""
Search-effort benchmark: brute-force enumeration vs subset DP vs branch-and-bound.

For each of the three paper samples, every distinct tied-optimal phasing
option is enumerated (as in `main.py --dedupe`), its EBG built, and the
ordered-partition search over *all* its cycles (the F_m scenarios counted in
the paper; no cluster decomposition) is solved three ways:

  - brute force:  every ordered partition of the cycles (F_m leaves), keeping
                  only the optimum,
  - DP:           subset dynamic programming (3^m - 2^m transitions),
  - B&B:          gr_cn's exact branch-and-bound,
and additionally timed as
  - full enumeration: enumerate_all_scenarios(), which materializes every
                  scenario with its per-step profile (what the pipeline does to
                  write scenarios_summary.csv).

The first three use the same cost model (DynamicGenomeScenario), so effort is
directly comparable. Per method it records:
  - cost_evals:    calls to step_cost(), i.e. search-tree edges / DP transitions
  - genome_states: distinct intermediate genomes built (union-find passes)
  - seconds:       median wall time over --repeats runs, each on a fresh
                   model (no shared cache)
and checks that all methods reach the same optimum. Per-sample phasing
enumeration metadata is written next to the results (<out>_phasing.csv).

Usage:
    python benchmark/run_benchmark.py [--out benchmark/results.csv] [--only PM1284]
                                      [--repeats 5] [--cpu 1]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from math import comb
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gr_cn import report  # noqa: E402
from gr_cn.io import jabba, reference  # noqa: E402
from gr_cn.phasing import enumerate as penum  # noqa: E402
from gr_cn.scenario import enumeration  # noqa: E402

SAMPLES = [
    ("CA-0098", "jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23]),
    ("PM1284", "jabba_data/graphs_all/PM1284.json", [3, 11, 12]),
    ("852_T1_33_23905", "jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14]),
]
TIMING = "before"


def counted_model(cycles, all_nodes):
    """A fresh cost model whose step_cost() calls are counted."""
    model = enumeration.DynamicGenomeScenario(cycles, all_nodes)
    model.evals = 0
    inner = model.step_cost

    def step_cost(state, step, timing="before"):
        model.evals += 1
        return inner(state, step, timing)

    model.step_cost = step_cost
    return model


def brute_force(model) -> int:
    """Optimum over every ordered partition (same recursion as enumerate_all_scenarios)."""
    full = model.full_mask
    best = [float("inf")]

    def rec(state: int, cur_max: int) -> None:
        if state == full:
            best[0] = min(best[0], cur_max)
            return
        remaining = full ^ state
        sub = remaining
        while sub:
            rec(state | sub, max(cur_max, model.step_cost(state, sub, TIMING)))
            sub = (sub - 1) & remaining

    rec(0, 0)
    return int(best[0])


def _run_once(name: str, cycles, all_nodes):
    if name == "enum":
        t0 = time.perf_counter()
        scen = list(enumeration.enumerate_all_scenarios(cycles, all_nodes, timing=TIMING))
        return {"obj": min(s.objective_value for s in scen), "cost_evals": "",
                "genome_states": "", "n_materialized": len(scen)}, time.perf_counter() - t0
    model = counted_model(cycles, all_nodes)
    t0 = time.perf_counter()
    if name == "brute":
        obj = brute_force(model)
    elif name == "dp":
        obj = enumeration._dp_exact(model, TIMING).objective_value
    else:
        obj = enumeration._exact_branch_and_bound(model, TIMING).objective_value
    dt = time.perf_counter() - t0
    return {"obj": obj, "cost_evals": model.evals, "genome_states": len(model._cache)}, dt


def run_method(name: str, cycles, all_nodes, repeats: int) -> dict:
    """Effort counters from the first run; time = median over `repeats` runs."""
    res, dt = _run_once(name, cycles, all_nodes)
    times = [dt] + [_run_once(name, cycles, all_nodes)[1] for _ in range(repeats - 1)]
    return {**res, "seconds": statistics.median(times)}


def fubini(m: int) -> int:
    a = [1] + [0] * m
    for n in range(1, m + 1):
        a[n] = sum(comb(n, k) * a[n - k] for k in range(1, n + 1))
    return a[m]


def main(out: Path, only: Optional[List[str]], cpu_n: int, repeats: int) -> None:
    fields = ["sample", "option_id", "phasing_label", "loose_pairing_label",
              "n_cycles", "n_ebg_nodes", "n_scenarios", "optimum",
              "brute_cost_evals", "brute_seconds",
              "dp_cost_evals", "dp_genome_states", "dp_seconds",
              "bnb_cost_evals", "bnb_genome_states", "bnb_seconds",
              "enum_n_materialized", "enum_seconds",
              "all_methods_agree"]
    meta_rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, path, chrs in SAMPLES:
            if only and name not in only:
                continue
            print(f"== {name}: enumerating distinct tied phasings...", flush=True)
            graph = jabba.load(str(ROOT / path), chromosomes=chrs)
            ref = reference.build(graph)
            t0 = time.perf_counter()
            options, meta, _eq = penum.enumerate_full_options(
                graph, ref, L=2, dedupe=True, cpu_n=cpu_n)
            meta_rows.append({
                "sample": name, "n_raw_tied_phasings": meta["n_raw_phasing_options"],
                "n_canonical_tied_phasings": meta["n_canonical_phasing_options"],
                "raw_enumeration_capped": meta["truncated"] or any(meta["per_subgraph_hit_solution_cap"]),
                "n_loose_pairing_variants": meta["n_loose_pairing_variants"],
                "n_options": len(options), "phasing_enumeration_seconds": round(time.perf_counter() - t0, 1)})
            print(f"   {len(options)} option(s)", flush=True)
            for opt in options:
                dr = report.build_option_downstream(
                    graph, ref, opt["phasing"], opt["extra_pairs"], L=2)
                cycles, raw_nodes = dr["cycles_on_raw"], dr["raw_nodes"]
                m = len(cycles)
                res = {k: run_method(k, cycles, raw_nodes, repeats)
                       for k in ("brute", "dp", "bnb", "enum")}
                row = {"sample": name, "option_id": opt["option_id"],
                       "phasing_label": opt["phasing_label"],
                       "loose_pairing_label": opt["loose_label"],
                       "n_cycles": m, "n_ebg_nodes": len(dr["ebg_nodes"]),
                       "n_scenarios": fubini(m), "optimum": res["bnb"]["obj"],
                       "enum_n_materialized": res["enum"]["n_materialized"],
                       "all_methods_agree": len({r["obj"] for r in res.values()}) == 1}
                for k, r in res.items():
                    row[f"{k}_seconds"] = round(r["seconds"], 5)
                    if k != "enum":
                        row[f"{k}_cost_evals"] = r["cost_evals"]
                    if k in ("dp", "bnb"):
                        row[f"{k}_genome_states"] = r["genome_states"]
                w.writerow(row)
                f.flush()
                print(f"   {opt['option_id']}: m={m} opt={row['optimum']} "
                      f"t_enum={row['enum_seconds']} t_brute={row['brute_seconds']} "
                      f"t_dp={row['dp_seconds']} t_bnb={row['bnb_seconds']} "
                      f"agree={row['all_methods_agree']}", flush=True)
    meta_out = out.with_name(out.stem + "_phasing.csv")
    with meta_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        w.writeheader()
        w.writerows(meta_rows)
    print(f"\nWritten {out} and {meta_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "benchmark" / "results.csv")
    p.add_argument("--only", nargs="+", default=None, help="restrict to these sample names")
    p.add_argument("--cpu", type=int, default=1,
                   help="CP-SAT workers per phasing solve (1 = deterministic)")
    p.add_argument("--repeats", type=int, default=5, help="timing repeats per method")
    a = p.parse_args()
    main(a.out, a.only, a.cpu, a.repeats)
