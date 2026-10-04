"""
Per-step EBG state for the scenario visualization.

Mirrors the semantics of gr_cn.scenario.enumeration.DynamicGenomeScenario
(same black/grey toggle-per-cycle model) but keeps EBGNode object references
throughout instead of remapping to integer ids, since the drawing code needs
the original node identity (chromosome, haplotype, copy, DC/AH-ness, ...).

A "step" here is a point in the ordered partition timeline:
  step 0            -- before any cycle has been applied (all cycles pending)
  step k (1..len(P)) -- after partition[0..k-1] have been applied

Each pending cycle contributes two things to a step's picture:
  - its BLACK edges, drawn as the "currently real" structure
  - its GREY edges, drawn as a dashed preview labelled with the cycle's tag
Once a cycle is applied its GREY edges become the real (solid) structure and
it no longer needs a label -- the ambiguity is resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from string import ascii_lowercase
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from gr_cn.models.node import EBGNode, End, NodeType

EdgeTriple = Tuple[EBGNode, EBGNode, str]  # (u, v, kind)
# kind in {"base", "shared", "black", "grey-resolved", "grey-pending:<label>"}
# ("base"/"shared" are always-active; "shared" only appears when
# build_viz_steps is given all_nodes -- see there for what it means)


@dataclass
class CycleInfo:
    index: int
    label: str
    nodes: List[EBGNode]
    black_edges: List[Tuple[EBGNode, EBGNode]]
    grey_edges: List[Tuple[EBGNode, EBGNode]]
    # merge_partner_cycles combines the same rearrangement's independent
    # per-haplotype copies into one CycleInfo (they are always applied
    # together as one atomic step), so "T1" alone can under-count what's
    # actually drawn -- node_label gives each *original* raw cycle (each
    # haplotype copy) its own distinguishable tag ("T1a"/"T1b", ...), and
    # sub_labels lists them in a stable order.
    node_label: Dict[int, str] = field(default_factory=dict)
    sub_labels: List[str] = field(default_factory=list)


@dataclass
class VizStep:
    step_index: int
    applied: FrozenSet[int]
    just_applied: List[int]
    nodes: List[EBGNode]
    active_edges: List[EdgeTriple]
    pending_edges: List[EdgeTriple]


def _cycle_edges(cycle: Sequence[EBGNode]) -> Tuple[List[Tuple[EBGNode, EBGNode]], List[Tuple[EBGNode, EBGNode]]]:
    ids = {id(x) for x in cycle}
    black, grey = [], []
    seen_black, seen_grey = set(), set()
    for node in cycle:
        bn = node.black_neighbor
        if bn is not None and id(bn) in ids:
            key = frozenset((id(node), id(bn)))
            if key not in seen_black:
                seen_black.add(key)
                black.append((node, bn))
        gn = node.grey_neighbor
        if gn is not None and id(gn) in ids:
            key = frozenset((id(node), id(gn)))
            if key not in seen_grey:
                seen_grey.add(key)
                grey.append((node, gn))
    return black, grey


def _sub_components(cycle: Sequence[EBGNode]) -> List[List[EBGNode]]:
    """
    Split a (possibly partner-merged) cycle back into the individual
    connected components merge_partner_cycles combined -- i.e. the original
    per-haplotype raw cycles -- using the same black/grey adjacency that
    defines the cycle in the first place.
    """
    ids = {id(n) for n in cycle}
    visited: set = set()
    comps: List[List[EBGNode]] = []
    for start in cycle:
        if id(start) in visited:
            continue
        stack = [start]
        comp: List[EBGNode] = []
        while stack:
            cur = stack.pop()
            if id(cur) in visited:
                continue
            visited.add(id(cur))
            comp.append(cur)
            for nb in (cur.black_neighbor, cur.grey_neighbor):
                if nb is not None and id(nb) in ids and id(nb) not in visited:
                    stack.append(nb)
        comps.append(comp)
    return comps


def segment_base_edges(nodes: Sequence[EBGNode]) -> List[Tuple[EBGNode, EBGNode]]:
    """
    The permanent tail-head pairing for every touched INTERNAL node: the
    physical DNA segment backbone, identical in genome A and genome B, so
    it is always drawn as a plain line regardless of cycle state.

    DC/AH nodes are deliberately excluded: their own T-H pairing is *not*
    symmetric between genomes -- a DC pair's internal edge is a genuine
    black_neighbor edge (real in genome A, absent from B; see
    ebg.builder), and an AH pair's is the opposite (grey_neighbor, real in
    B, absent from A). Treating it as a universal "base" edge would draw
    it in both genomes, which is wrong in whichever one it isn't real EBG
    structure for. Where it *is* real, the normal black/grey active-edge
    path already draws it.
    """
    by_seg: Dict[tuple, Dict[End, EBGNode]] = {}
    for node in nodes:
        if node.node_type is not NodeType.INTERNAL:
            continue
        key = (node.node_type, node.iid, node.haplotype, node.copy_index)
        by_seg.setdefault(key, {})[node.end] = node
    edges = []
    for sides in by_seg.values():
        if End.TAIL in sides and End.HEAD in sides:
            edges.append((sides[End.TAIL], sides[End.HEAD]))
    return edges


def build_cycle_infos(cycles: Sequence[Sequence[EBGNode]]) -> List[CycleInfo]:
    infos = []
    for i, cyc in enumerate(cycles):
        black, grey = _cycle_edges(cyc)
        label = f"T{i + 1}"
        sub_components = _sub_components(cyc)
        node_label: Dict[int, str] = {}
        if len(sub_components) <= 1:
            sub_labels = [label]
            for n in cyc:
                node_label[id(n)] = label
        else:
            sub_labels = []
            for k, comp in enumerate(sub_components):
                suffix = ascii_lowercase[k] if k < len(ascii_lowercase) else str(k + 1)
                sub_label = f"{label}{suffix}"
                sub_labels.append(sub_label)
                for n in comp:
                    node_label[id(n)] = sub_label
        infos.append(CycleInfo(
            index=i,
            label=label,
            nodes=list(cyc),
            black_edges=black,
            grey_edges=grey,
            node_label=node_label,
            sub_labels=sub_labels,
        ))
    return infos


def touched_nodes(cycles: Sequence[Sequence[EBGNode]]) -> List[EBGNode]:
    """All (deduplicated) nodes belonging to some cycle."""
    seen: Dict[int, EBGNode] = {}
    for cyc in cycles:
        for n in cyc:
            seen[id(n)] = n
    return list(seen.values())


def remap_cycles_to_nodes(
    cycles: Sequence[Sequence[EBGNode]], all_nodes: Sequence[EBGNode],
) -> List[List[EBGNode]]:
    """
    `cycles` (from PipelineResult.all_cycles) and `all_nodes` (raw_nodes)
    come from two separate builder.build() calls on the same inputs, so
    they are equal in structure but distinct Python objects -- match them
    up by EBGNode.key() (their stable logical identity) rather than id().
    """
    by_key = {n.key(): n for n in all_nodes}
    return [[by_key[n.key()] for n in cyc] for cyc in cycles]


def build_viz_steps(
    cycles: Sequence[Sequence[EBGNode]],
    partition: Sequence[Sequence[int]],
    all_nodes: Optional[Sequence[EBGNode]] = None,
) -> List[VizStep]:
    """
    Build the step-by-step EBG state timeline for a scenario partition.

    A node's own tail-head "base" edge (the physical DNA segment) is always
    true regardless of any cycle's resolution state -- what a cycle toggles
    is which *adjacency* connects that segment to its neighbours, not
    whether the segment itself exists. So each base edge appears exactly
    once both of its endpoints' owning cycles have been applied (nodes with
    no owning cycle at all -- untouched regions -- count as always
    satisfied). That makes step 0 pure EBG-edges-only, and the final step
    (every cycle applied) reconstruct the full genome-B structure.

    If `all_nodes` is given (pass PipelineResult.raw_nodes), every node in
    the requested region is included -- not just ones some cycle touches --
    so unchanged regions and true chromosome telomeres are drawn too, via a
    "shared" edge kind: whichever real adjacency a node has that isn't
    already represented as some cycle's own black/grey edge (never toggled,
    always active). Most such nodes have black_neighbor is grey_neighbor
    (the identical-in-both-genomes case remove_shared_adjacencies strips
    before cycle extraction); a few have only one of the two set (a real
    adjacency that didn't end up part of any alternating cycle -- e.g. a
    loose end) and are shown the same way. A node with neither set is a
    genuine chromosome-end telomere and gets no shared edge at all --
    draw_step's cap marker takes over there, same as in the genome A/B
    panels. Without all_nodes, the old touched-only behaviour is used, with
    no "shared" edges.

    A node's own tail-head "base" edge is likewise always active at every
    step, matching the fact that the physical segment itself never toggles
    -- only which *adjacency* connects it to its neighbours does.

    Returns len(partition) + 1 VizStep objects: index 0 is the state before
    any cycle is applied, index k is the state after partition[0:k].
    """
    if all_nodes is not None:
        cycles = remap_cycles_to_nodes(cycles, all_nodes)
        nodes = list(all_nodes)
    else:
        nodes = touched_nodes(cycles)

    cycle_infos = build_cycle_infos(cycles)
    base_edges = segment_base_edges(nodes)
    owner: Dict[int, int] = {id(n): info.index for info in cycle_infos for n in info.nodes}

    # Every node's own black/grey pointer that isn't already covered by its
    # owning cycle's own black_edges/grey_edges (i.e. at least one endpoint
    # has no owning cycle, or -- defensively, since a cycle-owned node's
    # pointers can only ever lead back into that same cycle -- both do but
    # to different cycles). These connections never toggle across the
    # scenario timeline.
    shared_edges: List[Tuple[EBGNode, EBGNode]] = []
    if all_nodes is not None:
        seen: set = set()
        for n in nodes:
            for nb in (n.black_neighbor, n.grey_neighbor):
                if nb is None:
                    continue
                n_owner = owner.get(id(n))
                if n_owner is not None and n_owner == owner.get(id(nb)):
                    continue  # already covered by info.black_edges/grey_edges
                key = frozenset((id(n), id(nb)))
                if key not in seen:
                    seen.add(key)
                    shared_edges.append((n, nb))

    steps: List[VizStep] = []
    applied: set = set()
    just_applied: List[int] = []

    for step_index in range(len(partition) + 1):
        active: List[EdgeTriple] = [(u, v, "base") for u, v in base_edges]
        active.extend((u, v, "shared") for u, v in shared_edges)
        pending: List[EdgeTriple] = []

        for info in cycle_infos:
            if info.index in applied:
                active.extend((u, v, "grey-resolved") for u, v in info.grey_edges)
            else:
                active.extend((u, v, "black") for u, v in info.black_edges)
                pending.extend(
                    (u, v, f"grey-pending:{info.node_label[id(u)]}") for u, v in info.grey_edges
                )

        steps.append(VizStep(
            step_index=step_index,
            applied=frozenset(applied),
            just_applied=list(just_applied),
            nodes=nodes,
            active_edges=active,
            pending_edges=pending,
        ))

        if step_index < len(partition):
            just_applied = list(partition[step_index])
            applied = applied | set(just_applied)

    return steps


def step_label(cycle_indices: Sequence[int], cycle_infos: Sequence[CycleInfo]) -> str:
    labels: List[str] = []
    for i in cycle_indices:
        labels.extend(cycle_infos[i].sub_labels)
    return "{" + ", ".join(labels) + "}"
