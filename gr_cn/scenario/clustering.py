"""
Cycle cluster decomposition.

Finds groups of EBG cycles that are independent — they involve disjoint sets of
reference chromosomes, so the optimal scenario for each cluster can be found
independently without affecting other clusters.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from gr_cn.models.node import EBGNode, NodeType


def cycle_chromosomes(cycle: List[EBGNode]) -> Set[int]:
    """Return the set of reference chromosomes touched by a cycle."""
    return {
        node.ref_chromosome
        for node in cycle
        if node.node_type is NodeType.INTERNAL and node.ref_chromosome > 0
    }


def cluster_cycles(cycles: List[List[EBGNode]]) -> List[List[int]]:
    """
    Partition cycle indices into independent clusters.

    Two cycles belong to the same cluster if they share at least one chromosome.
    Cycles in different clusters can be optimized independently because their
    rearrangements do not interact.

    Returns clusters sorted largest-first, each cluster is a list of cycle indices
    into the original cycles list.
    """
    n = len(cycles)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    chrom_to_cycles: Dict[int, List[int]] = defaultdict(list)
    for i, cyc in enumerate(cycles):
        for ch in cycle_chromosomes(cyc):
            chrom_to_cycles[ch].append(i)

    for cycle_list in chrom_to_cycles.values():
        for i in range(1, len(cycle_list)):
            union(cycle_list[0], cycle_list[i])

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return sorted(groups.values(), key=lambda g: -len(g))
