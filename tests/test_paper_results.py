"""
Regression tests pinning the numbers reported in the paper for CA-0098
(the one sample whose phasing is unique, so the result is fully determined).

Run with:  pytest            (a few seconds)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from gr_cn import report
from gr_cn.io import jabba, reference
from gr_cn.phasing import enumerate as penum
from gr_cn.scenario import enumeration

ROOT = Path(__file__).resolve().parents[1]
CA0098 = ROOT / "jabba_data" / "graphs_all" / "CA-0098.json"


def fubini(m: int) -> int:
    from math import comb
    a = [1] + [0] * m
    for n in range(1, m + 1):
        a[n] = sum(comb(n, k) * a[n - k] for k in range(1, n + 1))
    return a[m]


def test_fubini_numbers():
    assert [fubini(m) for m in range(1, 8)] == [1, 3, 13, 75, 541, 4683, 47293]


@pytest.fixture(scope="module")
def ca0098():
    graph = jabba.load(str(CA0098), chromosomes=[5, 18, 22, 23])
    ref = reference.build(graph)
    options, meta = penum.enumerate_full_options(graph, ref, L=2, cpu_n=1)
    assert len(options) == 1, "CA-0098 phasing is expected to be unique"
    opt = options[0]
    dr = report.build_option_downstream(graph, ref, opt["phasing"], opt["extra_pairs"], L=2)
    return dr["cycles_on_raw"], dr["raw_nodes"]


def test_ca0098_cycles(ca0098):
    cycles, _ = ca0098
    assert len(cycles) == 7


@pytest.mark.parametrize("method", ["dp", "branch_and_bound"])
def test_ca0098_optimum(ca0098, method):
    cycles, nodes = ca0098
    result = enumeration.find_best_scenario(cycles, all_nodes=nodes, method=method)
    assert result.exact
    assert result.objective_value == 2


def test_ca0098_search_effort_counters(ca0098):
    """Enumeration visits every node of the ordered-partition tree (2 F_m of
    them); branch-and-bound reaches the optimum with far fewer."""
    cycles, nodes = ca0098
    stats = {}
    n = sum(1 for _ in enumeration.enumerate_all_scenarios(cycles, nodes, stats=stats))
    assert n == fubini(7) and stats["nodes"] == 2 * fubini(7)
    res = enumeration._exact_branch_and_bound(
        enumeration.DynamicGenomeScenario(cycles, nodes), "before")
    assert res.objective_value == 2
    assert 1 <= res.n_complete_scenarios <= res.expanded_states < stats["nodes"]


def test_ca0098_scenario_distribution(ca0098):
    """Appendix C, Table S2: C_max over all 47,293 ordered partitions."""
    cycles, nodes = ca0098
    scen = list(enumeration.enumerate_all_scenarios(cycles, nodes))
    assert len(scen) == fubini(7)
    dist = Counter(s.objective_value for s in scen)
    assert dist == {2: 6360, 3: 10410, 4: 22274, 5: 6169, 6: 1827, 7: 217, 8: 36}
