"""
Cycle extraction from an EBG node list.

After builder.build() assigns black/grey neighbours, shared adjacencies
(same edge in both A and B) appear as nodes where black_neighbor is grey_neighbor.
These are stripped first, leaving only alternating cycle components per Lemma 1.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set

from gr_cn.models.node import EBGNode


def remap_cycles_to_nodes(
    cycles: Sequence[Sequence[EBGNode]], all_nodes: Sequence[EBGNode],
) -> List[List[EBGNode]]:
    """
    `cycles` and `all_nodes` sometimes come from two separate builder.build()
    calls on the same inputs (e.g. PipelineResult.all_cycles vs raw_nodes),
    so they are equal in structure but distinct Python objects -- match them
    up by EBGNode.key() (their stable logical identity) rather than id().
    """
    by_key = {n.key(): n for n in all_nodes}
    return [[by_key[n.key()] for n in cyc] for cyc in cycles]


def remove_shared_adjacencies(nodes: List[EBGNode]) -> None:
    """
    Remove edges that are shared between A and B (black == grey neighbour).

    Under the paper's formulation these edges cancel out and their nodes
    become isolated vertices (degree 0) in the EBG.
    Modifies nodes in-place.
    """
    for node in nodes:
        if node.black_neighbor is not None and node.black_neighbor is node.grey_neighbor:
            node.black_neighbor = None
            node.grey_neighbor = None


def _neighbors(node: EBGNode) -> List[EBGNode]:
    seen = set()
    result = []
    for nb in (node.black_neighbor, node.grey_neighbor):
        if nb is not None and nb not in seen:
            seen.add(nb)
            result.append(nb)
    return result


def _connected_component(start: EBGNode, visited: Set[EBGNode]):
    stack = [start]
    component = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component.append(node)
        for nb in _neighbors(node):
            if nb not in visited:
                stack.append(nb)
    return component


def _is_alternating_cycle(component: List[EBGNode]) -> bool:
    """True if every node in the component has exactly 2 neighbours (degree 2)."""
    return all(len(_neighbors(n)) == 2 for n in component)


def extract_cycles(nodes: List[EBGNode]) -> List[List[EBGNode]]:
    """
    Extract all alternating cycles from the EBG.

    Non-cycle components (paths, isolated nodes) are silently ignored —
    they correspond to unresolved loose ends or other degenerate cases.

    Returns
    -------
    List of cycles, each a list of EBGNode in an unspecified (unordered) traversal.
    Call order_cycle() to get a canonical traversal order.
    """
    visited: Set[EBGNode] = set()
    cycles = []
    for node in nodes:
        if node not in visited:
            component = _connected_component(node, visited)
            if len(component) > 1 and _is_alternating_cycle(component):
                cycles.append(component)
    return cycles


def _cycle_topology(cycle: List[EBGNode]) -> frozenset:
    """
    Haplotype-agnostic signature for a cycle.

    Two cycles that represent the same structural rearrangement on different
    haplotypes will have identical signatures.  We strip the haplotype field
    and keep (node_type, iid, end, copy_index) for every node.
    """
    from gr_cn.models.node import NodeType
    return frozenset(
        (n.node_type.value, n.iid, n.end.value, n.copy_index)
        for n in cycle
    )


def merge_partner_cycles(cycles: List[List[EBGNode]]) -> List[List[EBGNode]]:
    """
    Merge cycles that represent the same rearrangement on different haplotypes.

    An ALT adjacency with weight k produces k identical EBG cycles, one per
    haplotype copy.  These are "partner" cycles — they share the same structural
    topology and always occur as a single event.  Merging them into one combined
    node list lets the scenario model treat them as one atomic step.

    Returns a list of merged cycles, sorted largest-first by node count.
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for cyc in cycles:
        sig = _cycle_topology(cyc)
        groups[sig].append(cyc)

    merged = []
    for cyc_list in groups.values():
        combined = []
        for c in cyc_list:
            combined.extend(c)
        merged.append(combined)

    merged.sort(key=lambda c: -len(c))
    return merged


def order_cycle(cycle: List[EBGNode]) -> List[EBGNode]:
    """
    Return the nodes of a cycle in alternating black/grey traversal order.

    The traversal starts from cycle[0] and alternates between following the
    black neighbour and the grey neighbour so that the walk forms a valid
    alternating path back to the start.
    """
    if not cycle:
        return []

    start = cycle[0]
    ordered = [start]
    prev: Optional[EBGNode] = None
    current = start

    while True:
        nbs = _neighbors(current)
        candidates = [n for n in nbs if n is not prev]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt is start:
            break
        ordered.append(nxt)
        prev, current = current, nxt

    return ordered
