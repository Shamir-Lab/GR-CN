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


def _effective(node: EBGNode):
    """(black, grey) neighbours with shared adjacencies (black is grey) dropped,
    so this works on stripped and unstripped node lists alike."""
    if node.black_neighbor is not None and node.black_neighbor is node.grey_neighbor:
        return None, None
    return node.black_neighbor, node.grey_neighbor


def cap_open_paths(nodes: List[EBGNode]) -> List[EBGNode]:
    """
    Close every open alternating path of the EBG with TELOMERE vertices.

    In the paper's model chromosome ends are vertices, so every vertex has one
    black and one grey edge and the EBG is a union of cycles (Lemma 1). The
    builder has no explicit telomere vertices, so an extremity that is a
    chromosome end in A but joined in B (grey edge only), or joined in A but a
    chromosome end in B (black edge only), is the end of an open path. Each
    such path is closed here:
      - one end missing grey, the other missing black: one telomere vertex T,
        grey to the first end (a new chromosome end in B) and black to the
        second (a chromosome end in A);
      - both ends missing grey: telomeres T1, T2 with grey edges to the ends
        and a black edge T1-T2 (the two ends were created by the rearrangement);
      - both ends missing black: symmetric, with a grey edge T1-T2.
    Telomere vertices are keyed by the extremity they cap, so repeated builds
    on the same input produce identical keys. Returns the new vertices; the
    caller appends them to its node list.
    """
    from gr_cn.models.node import NodeType

    def eff_neighbors(n):
        return [x for x in _effective(n) if x is not None]

    new: List[EBGNode] = []
    seen: Set[EBGNode] = set()
    for start in nodes:
        if start in seen or not eff_neighbors(start):
            continue
        comp, stack = [], [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(y for y in eff_neighbors(x) if y not in seen)
        ends = sorted((x for x in comp if len(eff_neighbors(x)) == 1), key=lambda x: x.key())
        if not ends:
            continue
        if len(ends) != 2:
            raise ValueError(f"EBG component with {len(ends)} degree-1 vertices is not a path")

        def telomere(end: EBGNode) -> EBGNode:
            t = EBGNode(node_type=NodeType.TELOMERE, iid=end.iid, end=end.end,
                        haplotype=end.haplotype, copy_index=end.copy_index,
                        ref_chromosome=-1, ref_position=-1, is_telomere=True)
            new.append(t)
            return t

        missing_grey = [e for e in ends if _effective(e)[1] is None]
        missing_black = [e for e in ends if _effective(e)[0] is None]
        if len(missing_grey) == 1 and len(missing_black) == 1:
            t = telomere(missing_grey[0])
            t.grey_neighbor, missing_grey[0].grey_neighbor = missing_grey[0], t
            t.black_neighbor, missing_black[0].black_neighbor = missing_black[0], t
        elif len(missing_grey) == 2:
            t1, t2 = telomere(ends[0]), telomere(ends[1])
            t1.grey_neighbor, ends[0].grey_neighbor = ends[0], t1
            t2.grey_neighbor, ends[1].grey_neighbor = ends[1], t2
            t1.black_neighbor, t2.black_neighbor = t2, t1
        else:
            t1, t2 = telomere(ends[0]), telomere(ends[1])
            t1.black_neighbor, ends[0].black_neighbor = ends[0], t1
            t2.black_neighbor, ends[1].black_neighbor = ends[1], t2
            t1.grey_neighbor, t2.grey_neighbor = t2, t1
    return new


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

    Isolated nodes are ignored. Open paths should not occur: builder.build()
    closes them with telomere vertices (cap_open_paths).

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
