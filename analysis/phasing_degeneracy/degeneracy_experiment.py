"""
Subgraph construction + tied loose-end pairing enumeration, shared with the
main enumeration pipeline.

analysis/haplotype_options_report/enumerate_options.py imports build_subgraphs,
_grey_pair_signature, and enumerate_loose_pairings from this module -- they
reconstruct the same per-weakly-connected-subgraph CP-SAT inputs
gr_cn.phasing.ilp builds internally (reusing its private _build_model /
_add_alt_usage_constraints / _extract_components helpers rather than
duplicating them), and extend gr_cn.phasing.loose's loose-end pairing logic
to return every tied-minimal pairing instead of just the argmin.

This file used to also contain a standalone degeneracy-measurement experiment
(enumerate every distinct tied phasing/loose-pairing combination and check
whether the final scenario objective is stable across all of them) built on
top of these same helpers. That harness predates
analysis/haplotype_options_report's exhaustive, exact enumeration and isn't
part of the run.py pipeline chain -- it's preserved at
archive/degeneracy_experiment_standalone_harness.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from gr_cn.phasing import loose


# ---------------------------------------------------------------------------
# Rebuild the same per-subgraph CP-SAT inputs ilp.phase() builds internally.
# ---------------------------------------------------------------------------

def build_subgraphs(graph, reference, L=2):
    nodes_to_phase: List[tuple] = []
    block_edges: List[Tuple[tuple, tuple]] = []

    for iv in graph.intervals:
        cn_info = reference.get(iv.iid)
        n_deleted = cn_info.n_deleted if cn_info else 0
        n_gained = cn_info.n_gained if cn_info else 0

        for hap in range(L):
            if hap < n_deleted:
                continue
            t_node = (iv.iid, "t", hap, 0)
            h_node = (iv.iid, "h", hap, 0)
            nodes_to_phase += [t_node, h_node]
            block_edges.append((t_node, h_node))
        for copy_i in range(1, n_gained + 1):
            t_node = (iv.iid, "t", 0, copy_i)
            h_node = (iv.iid, "h", 0, copy_i)
            nodes_to_phase += [t_node, h_node]
            block_edges.append((t_node, h_node))

    strip_to_nodes: Dict[str, List[tuple]] = {}
    for node in nodes_to_phase:
        key = f"{node[0]}_{node[1]}"
        strip_to_nodes.setdefault(key, []).append(node)

    def expand_edge(src_key, snk_key):
        return [(s, t) for s in strip_to_nodes.get(src_key, [])
                for t in strip_to_nodes.get(snk_key, [])]

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes_to_phase)

    for t_node, h_node in block_edges:
        G.add_edge(t_node, h_node, color="black", is_alt=False, adj_id=-1, is_fwd=True)
        G.add_edge(h_node, t_node, color="black", is_alt=False, adj_id=-1, is_fwd=False)

    ref_adjs = [(f"{abs(a.source_iid)}_{a.source_side}", f"{abs(a.sink_iid)}_{a.sink_side}")
                for a in graph.ref_adjacencies if a.sink_iid is not None]
    for src_key, snk_key in ref_adjs:
        for s, t in expand_edge(src_key, snk_key):
            G.add_edge(s, t, color="grey", is_alt=False, adj_id=-1, is_fwd=True)
            G.add_edge(t, s, color="grey", is_alt=False, adj_id=-1, is_fwd=False)

    alt_adj_weights: Dict[int, int] = {}
    for adj_id, a in enumerate(adj for adj in graph.alt_adjacencies if adj.sink_iid is not None):
        src_key = f"{abs(a.source_iid)}_{a.source_side}"
        snk_key = f"{abs(a.sink_iid)}_{a.sink_side}"
        w = max(1, round(a.weight)) if a.weight is not None else 1
        alt_adj_weights[adj_id] = w
        for s, t in expand_edge(src_key, snk_key):
            G.add_edge(s, t, color="grey", is_alt=True, adj_id=adj_id, is_fwd=True)
            G.add_edge(t, s, color="grey", is_alt=True, adj_id=adj_id, is_fwd=False)

    subgraphs = sorted(
        [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)],
        key=len,
    )
    return subgraphs, alt_adj_weights


def _grey_pair_signature(grey_pairs) -> frozenset:
    return frozenset(tuple(sorted((u, v))) for u, v in grey_pairs)


# ---------------------------------------------------------------------------
# Enumerate tied loose-end pairings for a given phasing.
# ---------------------------------------------------------------------------

def enumerate_loose_pairings(graph, phasing) -> List[Tuple[str, list]]:
    """Returns list of (label, extra_pairs) -- all minimal-total-distance
    pairings (ties included). Empty list if nothing to resolve."""
    loose_sides = loose._non_telomeric_loose_sides(graph)
    if not loose_sides:
        return [("no_loose", [])]

    open_ends = []
    for comp in phasing.components:
        if comp["type"] not in ("path", "singleton"):
            continue
        nodes = comp["nodes"]
        for node in (nodes[0], nodes[-1]):
            iid, side, _hap, _copy = node
            if (iid, side) in loose_sides and node not in open_ends:
                open_ends.append(node)

    n = len(open_ends)
    if n == 0:
        return [("no_open_ends", [])]
    if n % 2 != 0 or n > 4:
        return [("unresolved_too_many", [])]
    if n == 2:
        return [("only_pairing", [(open_ends[0], open_ends[1])])]

    a, b, c, d = open_ends

    def pair_distance(u, v):
        chrom_u, pos_u = loose._position(graph, u)
        chrom_v, pos_v = loose._position(graph, v)
        if chrom_u != chrom_v:
            return loose._CROSS_CHR_PENALTY + abs(pos_u - pos_v)
        return abs(pos_u - pos_v)

    candidates = {
        "ab_cd": [(a, b), (c, d)],
        "ac_bd": [(a, c), (b, d)],
        "ad_bc": [(a, d), (b, c)],
    }
    dists = {k: sum(pair_distance(x, y) for x, y in v) for k, v in candidates.items()}
    best = min(dists.values())
    ties = [(k, candidates[k]) for k, dv in dists.items() if dv == best]
    return ties
