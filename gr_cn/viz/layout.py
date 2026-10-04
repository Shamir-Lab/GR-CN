"""
2D chain layout for one VizStep.

Real synteny (INTERNAL) nodes are arranged into vertical "chains" -- one
column per connected component of the current active structure, top-to-
bottom in traversal order -- matching the paper figure's node columns.

DC/AH nodes are never placed inline in a chain; they're satellite boxes
drawn beside whichever real synteny node they're currently wired to (an
active edge if the cycle is resolved, otherwise the pending preview edge),
exactly like the small "H_DC / T_DC" boxes beside the main columns in the
reference figure.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from gr_cn.models.node import EBGNode, End, NodeType
from gr_cn.viz.model import EdgeTriple, VizStep


@dataclass
class NodeLayout:
    node: EBGNode
    col: int
    row: int


@dataclass
class ChainLayout:
    col: int
    node_layouts: List[NodeLayout]
    is_circular: bool


@dataclass
class SatelliteBox:
    node_type: NodeType
    key: tuple
    tail_node: EBGNode
    head_node: EBGNode
    anchors: List[EBGNode]
    col: float
    row: float


@dataclass
class StepLayout:
    chains: List[ChainLayout]
    satellites: List[SatelliteBox]
    node_pos: Dict[int, Tuple[float, float]]
    n_cols: int
    n_rows: int
    # Canonical (min(id(u),id(v)), max(id(u),id(v))) -> signed perpendicular
    # offset for edges that would otherwise be drawn exactly on top of each
    # other (same two endpoint positions). 0.0 (the default for any pair not
    # present) means "draw straight".
    edge_curve: Dict[Tuple[int, int], float] = field(default_factory=dict)


def edge_curve_offset(layout: "StepLayout", u: EBGNode, v: EBGNode) -> float:
    """Look up a curve offset for edge (u, v) regardless of argument order."""
    a, b = (id(u), id(v)) if id(u) < id(v) else (id(v), id(u))
    return layout.edge_curve.get((a, b), 0.0)


def _sort_key(node: EBGNode) -> tuple:
    return (node.ref_chromosome, node.haplotype, node.copy_index, node.ref_position, node.end.value)


def _order_component(adj: Dict[int, List[int]], id_to_node: Dict[int, EBGNode],
                      comp_ids: List[int]) -> Tuple[List[EBGNode], bool]:
    """Return (ordered node list, is_circular) for one connected component."""
    degree = {i: len(set(adj[i])) for i in comp_ids}
    endpoints = sorted(
        [i for i in comp_ids if degree[i] <= 1],
        key=lambda i: _sort_key(id_to_node[i]),
    )

    if endpoints:
        start = endpoints[0]
        is_circular = False
    else:
        start = min(comp_ids, key=lambda i: _sort_key(id_to_node[i]))
        is_circular = True

    ordered = [start]
    visited = {start}
    current = start
    while True:
        candidates = [n for n in adj[current] if n not in visited]
        if not candidates:
            break
        nxt = sorted(candidates, key=lambda i: _sort_key(id_to_node[i]))[0]
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt

    return [id_to_node[i] for i in ordered], is_circular


def _align_chain_rows(chains: List["ChainLayout"], edges: Sequence["EdgeTriple"]) -> Dict[int, float]:
    """
    Compute a vertical (row) offset per chain so that cross-chain EBG edges
    connect nodes at, or as close as possible to, the same row -- instead of
    every chain starting at row 0 regardless of where its cycle partners
    sit. Chains are only ever shifted as a rigid block; their own internal
    node order (the linear chromosome direction) is untouched.

    A cycle typically threads through the same pair of chains more than
    once, at different local offsets, so no rigid per-chain shift can
    satisfy every cross-edge exactly -- something has to stay diagonal.
    A least-squares fit spreads that error evenly, which in practice left
    *every* edge a little bit diagonal and nothing readable. Instead this
    takes the majority vote (median) of what each chain's neighbours imply
    its offset should be, iterated to a fixed point: whichever alignment
    most of a chain's cross edges agree on wins outright (drawn straight),
    and only the genuine outliers stay diagonal.
    """
    row_of: Dict[int, Tuple[int, int]] = {}
    for chain in chains:
        for nl in chain.node_layouts:
            row_of[id(nl.node)] = (chain.col, nl.row)

    # (col_a, col_b, delta) meaning: desired row(col_a) - row(col_b) == delta
    cross: List[Tuple[int, int, float]] = []
    for u, v, _kind in edges:
        pu, pv = row_of.get(id(u)), row_of.get(id(v))
        if pu is None or pv is None or pu[0] == pv[0]:
            continue
        cross.append((pu[0], pv[0], float(pu[1] - pv[1])))

    incident: Dict[int, List[Tuple[int, float]]] = {c.col: [] for c in chains}
    for a, b, delta in cross:
        incident[a].append((b, delta))    # offset(a) should be offset(b) + delta
        incident[b].append((a, -delta))   # offset(b) should be offset(a) - delta

    # Seed with a spanning-tree pass (BFS) so the iteration below starts
    # from an already-reasonable layout rather than all zeros.
    offset: Dict[int, float] = {}
    for chain in chains:
        if chain.col in offset:
            continue
        offset[chain.col] = 0.0
        queue = [chain.col]
        while queue:
            cur = queue.pop()
            for nb, delta in incident[cur]:
                if nb in offset:
                    continue
                offset[nb] = offset[cur] + delta
                queue.append(nb)

    for _ in range(25):
        moved = False
        for col in offset:
            implied = sorted(offset[nb] + delta for nb, delta in incident[col])
            if not implied:
                continue
            mid = len(implied) // 2
            new_val = implied[mid] if len(implied) % 2 else (implied[mid - 1] + implied[mid]) / 2
            if abs(new_val - offset[col]) > 1e-9:
                moved = True
            offset[col] = new_val
        if not moved:
            break
    return offset


def compute_layout(step: VizStep) -> StepLayout:
    primary = [n for n in step.nodes if n.node_type is NodeType.INTERNAL]
    satellites_raw = [n for n in step.nodes if n.node_type in (NodeType.DC, NodeType.AH)]

    primary_ids = {id(n) for n in primary}
    id_to_node = {id(n): n for n in primary}

    # Group by active edges only (base scaffold + whichever cycle edge is
    # currently realized). Every primary node has degree <= 2 here (one base
    # neighbour + one active cycle neighbour), which is what the single-path
    # walk in _order_component relies on -- pulling pending edges in too
    # would give some nodes degree 3 and silently strand the unfollowed
    # branch out of the chain entirely.
    adj: Dict[int, List[int]] = {id(n): [] for n in primary}
    for u, v, _kind in step.active_edges:
        if id(u) in primary_ids and id(v) in primary_ids:
            adj[id(u)].append(id(v))
            adj[id(v)].append(id(u))

    visited: set = set()
    components: List[List[int]] = []
    for n in primary:
        nid = id(n)
        if nid in visited:
            continue
        stack = [nid]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(adj[cur])
        components.append(comp)

    ordered_components = []
    for comp_ids in components:
        ordered_nodes, is_circular = _order_component(adj, id_to_node, comp_ids)
        ordered_components.append((ordered_nodes, is_circular))

    ordered_components.sort(key=lambda oc: _sort_key(oc[0][0]))

    # Provisional integer column per chain; node_pos x is finalized below
    # once we know how much extra width each column's DC/AH boxes need.
    chains: List[ChainLayout] = []
    for col, (ordered_nodes, is_circular) in enumerate(ordered_components):
        node_layouts = [NodeLayout(node=node, col=col, row=row) for row, node in enumerate(ordered_nodes)]
        chains.append(ChainLayout(col=col, node_layouts=node_layouts, is_circular=is_circular))

    # Every chain currently starts at row 0 regardless of where its cycle
    # partners sit in their own chains, so a cross-chain EBG edge can cut
    # across many unrelated rows/columns. Shift each chain up/down as a
    # rigid block (never reordering its own nodes -- the chromosome still
    # reads top-to-bottom) so that cross-chain edges land as close to
    # horizontal as possible.
    row_offset = _align_chain_rows(chains, step.active_edges + step.pending_edges)

    node_pos: Dict[int, Tuple[float, float]] = {}
    for chain in chains:
        off = row_offset.get(chain.col, 0.0)
        for nl in chain.node_layouts:
            node_pos[id(nl.node)] = (float(chain.col), float(nl.row) + off)

    rows_all = [y for _, y in node_pos.values()]
    n_rows = int(max(rows_all) - min(rows_all)) + 1 if rows_all else 0

    # ------------------------------------------------------------------
    # Satellite (DC/AH) boxes: group by pair, anchor to whichever real
    # synteny node they're wired to right now (active first, else pending).
    # ------------------------------------------------------------------
    anchor_edges: Dict[int, EBGNode] = {}
    for edges in (step.active_edges, step.pending_edges):
        for u, v, _kind in edges:
            u_is_sat, v_is_sat = u.node_type in (NodeType.DC, NodeType.AH), v.node_type in (NodeType.DC, NodeType.AH)
            if u_is_sat and not v_is_sat:
                anchor_edges.setdefault(id(u), v)
            elif v_is_sat and not u_is_sat:
                anchor_edges.setdefault(id(v), u)

    pairs: Dict[tuple, Dict[End, EBGNode]] = {}
    for n in satellites_raw:
        key = (n.node_type, n.iid, n.haplotype, n.copy_index)
        pairs.setdefault(key, {})[n.end] = n

    # First pass: figure out how many satellite "slots" pile up on any single
    # anchor, and the worst case (max) per column -- that's how much extra
    # width the column needs so its boxes don't run into the next column.
    # A DC/AH pair can have a *separate* real edge from each of its own tail
    # and head (e.g. a deletion breaks both ends of the interval
    # independently) -- when both are wired up, anchor the box between them
    # (average row) instead of snapping to just one, so neither connecting
    # edge has to reach further than the other.
    box_specs = []  # (key, tail_node, head_node, anchors, icol, arow, slot, tail_above)
    slots_used: Dict[Tuple[int, int], int] = {}
    max_slots_per_col: Dict[int, int] = {}
    for key, sides in pairs.items():
        tail_node = sides.get(End.TAIL)
        head_node = sides.get(End.HEAD)
        if tail_node is None or head_node is None:
            continue
        tail_anchor = anchor_edges.get(id(tail_node))
        head_anchor = anchor_edges.get(id(head_node))
        tail_anchor_pos = node_pos.get(id(tail_anchor)) if tail_anchor is not None else None
        head_anchor_pos = node_pos.get(id(head_anchor)) if head_anchor is not None else None
        anchors = [
            a for a, p in ((tail_anchor, tail_anchor_pos), (head_anchor, head_anchor_pos))
            if p is not None
        ]
        if not anchors:
            continue
        positions = [tail_anchor_pos, head_anchor_pos] if tail_anchor_pos and head_anchor_pos else \
            [p for p in (tail_anchor_pos, head_anchor_pos) if p is not None]
        acol = sum(p[0] for p in positions) / len(positions)
        arow = sum(p[1] for p in positions) / len(positions)
        icol, irow = int(round(acol)), int(round(arow))
        slot = slots_used.get((icol, irow), 0)
        slots_used[(icol, irow)] = slot + 1
        max_slots_per_col[icol] = max(max_slots_per_col.get(icol, 0), slot + 1)
        # When both ends are independently wired to real nodes, put whichever
        # sub-box sits nearer its own anchor's row on that same side, so the
        # two connecting lines run side by side instead of crossing into an X.
        tail_above = True
        if tail_anchor_pos is not None and head_anchor_pos is not None:
            tail_above = tail_anchor_pos[1] <= head_anchor_pos[1]
        box_specs.append((key, tail_node, head_node, anchors, icol, arow, slot, tail_above))

    # Cumulative x position per column: a column that needs k satellite
    # slots reserves k * 0.62 extra width before the next column starts.
    col_x: Dict[int, float] = {0: 0.0}
    for col in range(1, len(chains)):
        col_x[col] = col_x[col - 1] + 1.0 + max_slots_per_col.get(col - 1, 0) * 0.62
    for chain in chains:
        x = col_x.get(chain.col, float(chain.col))
        for nl in chain.node_layouts:
            node_pos[id(nl.node)] = (x, node_pos[id(nl.node)][1])

    satellites: List[SatelliteBox] = []
    for key, tail_node, head_node, anchors, icol, arow, slot, tail_above in box_specs:
        acol_x = col_x.get(icol, float(icol))
        box_col = acol_x + 0.62 + slot * 0.62
        satellites.append(SatelliteBox(
            node_type=key[0], key=key, tail_node=tail_node, head_node=head_node,
            anchors=anchors, col=box_col, row=arow,
        ))
        tail_row, head_row = (arow - 0.2, arow + 0.2) if tail_above else (arow + 0.2, arow - 0.2)
        node_pos[id(tail_node)] = (box_col, tail_row)
        node_pos[id(head_node)] = (box_col, head_row)

    rightmost = max(
        [x for x, _ in node_pos.values()] + [col_x.get(len(chains) - 1, 0.0)],
        default=0.0,
    )
    n_cols = int(rightmost) + 2  # +2: fractional slop plus the box's own half-width

    # ------------------------------------------------------------------
    # Curve offsets: edges whose two endpoints sit at the exact same pair
    # of positions as another edge would otherwise be drawn as one
    # indistinguishable line. Only true position coincidences are curved --
    # a cross-diagram edge spanning different rows in the same two columns
    # is left straight, since _align_chain_rows already did its best to
    # make it read as horizontal as possible.
    # ------------------------------------------------------------------
    CURVE_STEP = 0.14
    groups: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)
    for u, v, _kind in list(step.active_edges) + list(step.pending_edges):
        pu, pv = node_pos.get(id(u)), node_pos.get(id(v))
        if pu is None or pv is None:
            continue
        a, b, pa, pb = (id(u), id(v), pu, pv) if id(u) < id(v) else (id(v), id(u), pv, pu)
        key = (tuple(round(c, 3) for c in pa), tuple(round(c, 3) for c in pb))
        pair = (a, b)
        if pair not in groups[key]:
            groups[key].append(pair)

    edge_curve: Dict[Tuple[int, int], float] = {}
    for pairs in groups.values():
        if len(pairs) <= 1:
            continue
        for i, pair in enumerate(pairs):
            k = (i + 1) // 2
            sign = 1 if i % 2 == 1 else -1
            edge_curve[pair] = 0.0 if i == 0 else sign * k * CURVE_STEP

    return StepLayout(chains=chains, satellites=satellites, node_pos=node_pos,
                       n_cols=n_cols, n_rows=n_rows, edge_curve=edge_curve)
