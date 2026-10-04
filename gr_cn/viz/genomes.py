"""
Full (not cycle-restricted) Genome A / Genome B panel construction.

Genome A ("black-only") and Genome B ("grey-if-present") are built straight
from EBGNode.black_neighbor / grey_neighbor on a *fresh, unstripped* node
list -- gr_cn.ebg.cycles.remove_shared_adjacencies() mutates nodes in place
to drop shared (A==B) adjacencies before cycle extraction, which is exactly
the information a full-genome panel needs to keep, so callers must pass in
a builder.build() output that has not been through that step.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from gr_cn.models.node import EBGNode, NodeType
from gr_cn.viz.model import EdgeTriple, VizStep, segment_base_edges


def full_genome_steps(all_nodes: Sequence[EBGNode], use_grey: bool) -> Tuple[VizStep, VizStep]:
    """
    Build the (layout_step, draw_step) pair for a full-genome panel.

    layout_step carries anchor-helper pending edges for DC/AH boxes (so
    compute_layout can place them even when their *displayed* colour has no
    live edge to a real synteny node -- e.g. a DC pair in Genome A, whose
    only active edge is its own {T_DC, H_DC} box pairing).
    draw_step is identical but with pending_edges cleared, since a genome
    panel shows one resolved structure, not a pending choice.
    """
    base_edges = segment_base_edges(all_nodes)
    active: List[EdgeTriple] = [(u, v, "base") for u, v in base_edges]
    seen_active: set = set()
    for node in all_nodes:
        neighbor = node.grey_neighbor if use_grey else node.black_neighbor
        if neighbor is None:
            continue
        key = frozenset((id(node), id(neighbor)))
        if key in seen_active:
            continue
        seen_active.add(key)
        # Only tag an edge as genome-specific (black / grey-resolved) when it
        # actually differs between genomes; an unrearranged connection where
        # black_neighbor is grey_neighbor is "shared" in both panels, same as
        # the EBG/scenario view, so it keeps the same neutral styling instead
        # of being drawn as if this specific edge is what makes A and B differ.
        shared = node.black_neighbor is node.grey_neighbor
        kind = "shared" if shared else ("grey-resolved" if use_grey else "black")
        active.append((node, neighbor, kind))

    pending: List[EdgeTriple] = []
    seen_pending: set = set()
    for node in all_nodes:
        if node.node_type not in (NodeType.DC, NodeType.AH):
            continue
        other = node.black_neighbor if use_grey else node.grey_neighbor
        if other is None:
            continue
        key = frozenset((id(node), id(other)))
        if key in seen_pending:
            continue
        seen_pending.add(key)
        pending.append((node, other, "anchor-only"))

    nodes = list(all_nodes)
    layout_step = VizStep(step_index=0, applied=frozenset(), just_applied=[],
                           nodes=nodes, active_edges=active, pending_edges=pending)
    draw_step = VizStep(step_index=0, applied=frozenset(), just_applied=[],
                         nodes=nodes, active_edges=active, pending_edges=[])
    return layout_step, draw_step
