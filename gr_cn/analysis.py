"""
Scenario analysis of phased samples: per-phasing scenario statistics,
selection of the phasing to study, representative scenarios, and search
effort of full enumeration vs. branch-and-bound.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from typing import Dict, List

from gr_cn import report
from gr_cn.scenario import enumeration

TIMING = "before"


def _entropy(dist: Counter) -> float:
    n = sum(dist.values())
    return -sum(c / n * math.log2(c / n) for c in dist.values())


def measure_effort(cycles, nodes, repeats: int) -> Dict:
    """
    Full enumeration vs. branch-and-bound, each on a fresh cost model per
    run: complete scenarios generated, search-tree nodes visited and median
    running time over `repeats` runs. Checks that both, and the subset DP,
    reach the same optimum.
    """
    def enum_once():
        stats: Dict = {}
        t0 = time.perf_counter()
        scen = list(enumeration.enumerate_all_scenarios(cycles, nodes, timing=TIMING,
                                                        max_cycles=len(cycles), stats=stats))
        return time.perf_counter() - t0, scen, stats

    def bnb_once():
        model = enumeration.DynamicGenomeScenario(cycles, nodes)
        t0 = time.perf_counter()
        res = enumeration._exact_branch_and_bound(model, TIMING)
        return time.perf_counter() - t0, res

    enum_runs = [enum_once() for _ in range(repeats)]
    bnb_runs = [bnb_once() for _ in range(repeats)]
    _, scen, stats = enum_runs[0]
    bnb = bnb_runs[0][1]
    dp = enumeration._dp_exact(enumeration.DynamicGenomeScenario(cycles, nodes), TIMING)
    optimum = min(s.objective_value for s in scen)
    assert bnb.objective_value == dp.objective_value == optimum, "search methods disagree"
    return {
        "optimum": optimum,
        "enumeration": {"complete_scenarios": len(scen), "tree_nodes": stats["nodes"],
                        "seconds": statistics.median(r[0] for r in enum_runs)},
        "branch_and_bound": {"complete_scenarios": bnb.n_complete_scenarios,
                             "tree_nodes": bnb.expanded_states,
                             "seconds": statistics.median(r[0] for r in bnb_runs)},
        "repeats": repeats,
    }


def all_scenarios(cycles, nodes) -> list:
    return list(enumeration.enumerate_all_scenarios(cycles, nodes, timing=TIMING,
                                                    max_cycles=len(cycles)))


def analyze_phasing(graph, ref, option: Dict, repeats: int) -> Dict:
    """EBG cycles of one phasing, the C_max distribution over all its
    scenarios, and its search effort. (The scenarios themselves are not
    kept: the 852 sample alone has seven phasings with 47,293 each.)"""
    dr = report.build_option_downstream(graph, ref, option["phasing"], option["extra_pairs"], L=2)
    cycles, nodes = dr["cycles_on_raw"], dr["raw_nodes"]
    dist = Counter(s.objective_value for s in all_scenarios(cycles, nodes))
    return {
        "option": option, "cycles": cycles, "nodes": nodes,
        "cmax_distribution": dist, "cmax_entropy": _entropy(dist),
        "effort": measure_effort(cycles, nodes, repeats),
    }


def select_phasing(analyses: List[Dict]) -> Dict:
    """
    The phasing whose scenarios vary most in C_max: most distinct C_max
    values, then most cycles (the richer scenario space), then the highest
    entropy of the C_max distribution. Ties keep the first phasing found.
    """
    key = lambda a: (len(a["cmax_distribution"]), len(a["cycles"]), round(a["cmax_entropy"], 12))
    best = analyses[0]
    for a in analyses[1:]:
        if key(a) > key(best):
            best = a
    return best


def representative_scenarios(scenarios: list, optimal: enumeration.ScenarioResult) -> List[Dict]:
    """
    The scenarios drawn for a sample, in this order:
      1. all at once: every cycle in a single step;
      2. singletons: every cycle in its own step -- the worst-scoring order,
         to show that the order matters;
      3. the optimum, unless 1 or 2 already attains it;
      4. a suboptimal scenario that is neither 1 nor 2 (between 2 and m-1
         steps), preferring three steps and a C_max not shown yet.
    Among equal candidates the first enumerated is used.
    """
    m = max(len(s.partition) for s in scenarios)  # number of cycles
    opt = optimal.objective_value
    joint = next(s for s in scenarios if len(s.partition) == 1)
    picks = [("All at once", joint)]
    if m > 1:
        singles = [s for s in scenarios if len(s.partition) == m]
        worst = max(singles, key=lambda s: s.objective_value)  # max() keeps the first of ties
        picks.append(("Singletons (optimal)" if worst.objective_value == opt else "Singletons", worst))
    if all(s.objective_value != opt for _, s in picks):
        picks.append(("Optimal", optimal))

    shown = {s.objective_value for _, s in picks}
    middle = [s for s in scenarios if 1 < len(s.partition) < m and s.objective_value > opt]
    for pool in ([s for s in middle if len(s.partition) == 3], middle):
        if not pool:
            continue
        values = sorted({s.objective_value for s in pool})
        fresh = [v for v in values if v not in shown] or values
        target = fresh[(len(fresh) - 1) // 2]
        picks.append(("Suboptimal", next(s for s in pool if s.objective_value == target)))
        break

    return [{"name": name, "cmax": s.objective_value, "n_steps": len(s.partition),
             "partition": s.partition, "step_costs": s.step_costs} for name, s in picks]
