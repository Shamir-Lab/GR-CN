"""
Phasing ILP for JABBA graphs.

Given a set of synteny-block nodes and their REF + ALT adjacencies, assign
each block copy to one of L haplotypes such that the resulting paths/cycles
cover all nodes with alternating black (REF) / grey (ALT) edges, minimizing:
  1. Total number of alternating components (paths + cycles + singletons).
  2. Among optimal solutions, minimize cross-haplotype adjacency edges.

ALT adjacencies are mandatory: each ALT adjacency with weight w must appear
exactly round(w) times in the phasing solution.

This is implemented as a two-stage CP-SAT solve.

Entry point: phase(graph, L=2) -> PhasingResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx
from ortools.sat.python import cp_model

from gr_cn.io.jabba import JabbaGraph
from gr_cn.io.reference import ReferenceGenome

# Node tuple: (iid, side, haplotype, copy_index)
# side in {"t", "h"}
PhaseNode = Tuple[int, str, int, int]


@dataclass
class PhasingResult:
    """
    Output of the phasing ILP.

    components: list of alternating path/cycle/singleton components.
      Each component is a dict with keys:
        "type": "path" | "cycle" | "singleton"
        "nodes": list of PhaseNode tuples in traversal order
    grey_pairs: list of (u, v) PhaseNode pairs that need an explicit grey
      edge in the EBG -- every used ALT connection, plus every used REF
      connection, with no exceptions. A same-haplotype, copy-index-0 REF
      pair ends up identical to the corresponding gr_cn.ebg.builder
      unconditional black edge, so set_grey there just confirms
      black_neighbor is grey_neighbor -- exactly the "shared, unchanged"
      state gr_cn.ebg.cycles.remove_shared_adjacencies needs to correctly
      cancel it out. Excluding it instead of setting it left grey_neighbor
      unset, which renders as a broken Genome-B connection even when the
      solver actually used it.
    subgraph_results: per-connected-subgraph ILP stats.
    """
    components: List[Dict] = field(default_factory=list)
    grey_pairs: List[Tuple[PhaseNode, PhaseNode]] = field(default_factory=list)
    subgraph_results: Dict[str, Dict] = field(default_factory=dict)
    num_components: int = 0
    num_mixed_haplotype_edges: int = 0
    status: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_color(c: str) -> str:
    c = str(c).strip().lower()
    if c in {"b", "black"}:
        return "B"
    if c in {"g", "grey", "gray"}:
        return "G"
    raise ValueError(f"Unknown edge color: {c!r}")


def _hap(v: PhaseNode) -> int:
    return v[2]


def _build_model(
    G: nx.MultiDiGraph,
    color_attr: str = "color",
    forbid_2cycles: bool = True,
) -> Dict:
    """Build the CP-SAT model for one weakly-connected subgraph."""
    V = list(G.nodes())
    n = len(V)

    A: List[Tuple] = []
    arc_color: Dict[Tuple, str] = {}
    arc_is_alt: Dict[Tuple, bool] = {}
    arc_adj_id: Dict[Tuple, int] = {}
    arc_is_fwd: Dict[Tuple, bool] = {}
    arc_mixed: Dict[Tuple, int] = {}

    for u, v, k, data in G.edges(keys=True, data=True):
        col = _norm_color(data.get(color_attr, "B"))
        A.append((u, v, k))
        arc_color[(u, v, k)] = col
        arc_is_alt[(u, v, k)] = bool(data.get("is_alt", False))
        arc_adj_id[(u, v, k)] = int(data.get("adj_id", -1))
        arc_is_fwd[(u, v, k)] = bool(data.get("is_fwd", True))
        arc_mixed[(u, v, k)] = 1 if _hap(u) != _hap(v) else 0

    model = cp_model.CpModel()
    x = {arc: model.NewBoolVar(f"x[{arc}]") for arc in A}
    s = {v: model.NewBoolVar(f"s[{v}]") for v in V}
    t = {v: model.NewBoolVar(f"t[{v}]") for v in V}
    z = {v: model.NewBoolVar(f"z[{v}]") for v in V}
    w = {v: model.NewBoolVar(f"w[{v}]") for v in V}
    c = {v: model.NewBoolVar(f"c[{v}]") for v in V}
    u_ord = {v: model.NewIntVar(0, n - 1, f"u[{v}]") for v in V}

    out_arcs: Dict = {v: [] for v in V}
    in_arcs:  Dict = {v: [] for v in V}
    outB: Dict = {v: [] for v in V}
    outG: Dict = {v: [] for v in V}
    inB:  Dict = {v: [] for v in V}
    inG:  Dict = {v: [] for v in V}

    for (i, j, k) in A:
        out_arcs[i].append(x[(i, j, k)])
        in_arcs[j].append(x[(i, j, k)])
        if arc_color[(i, j, k)] == "B":
            outB[i].append(x[(i, j, k)]); inB[j].append(x[(i, j, k)])
        else:
            outG[i].append(x[(i, j, k)]); inG[j].append(x[(i, j, k)])

    for v in V:
        model.Add(s[v] + t[v] + z[v] + w[v] == 1)
        model.Add(sum(out_arcs[v]) == s[v] + z[v])
        model.Add(sum(in_arcs[v])  == t[v] + z[v])

    model.Add(sum(s[v] for v in V) == sum(t[v] for v in V))

    for v in V:
        ib = sum(inB[v])  if inB[v]  else 0
        ig = sum(inG[v])  if inG[v]  else 0
        ob = sum(outB[v]) if outB[v] else 0
        og = sum(outG[v]) if outG[v] else 0
        model.Add(ib == og).OnlyEnforceIf(z[v])
        model.Add(ig == ob).OnlyEnforceIf(z[v])

    for v in V:
        model.Add(c[v] <= z[v])
        model.Add(u_ord[v] == 0).OnlyEnforceIf(s[v])
        model.Add(u_ord[v] == 0).OnlyEnforceIf(w[v])
        model.Add(u_ord[v] == 0).OnlyEnforceIf(c[v])

    for (i, j, k) in A:
        model.Add(u_ord[j] >= u_ord[i] + 1 - n + n * x[(i, j, k)] - n * c[j])

    if forbid_2cycles:
        pair_keys: Dict = {}
        for i, j, k in G.edges(keys=True):
            pair_keys.setdefault((i, j), []).append(k)
        done = set()
        for (i, j), keys_ij in pair_keys.items():
            if (i, j) in done or (j, i) in done:
                continue
            done |= {(i, j), (j, i)}
            keys_ji = pair_keys.get((j, i), [])
            if keys_ji:
                model.Add(
                    sum(x[(i, j, k)] for k in keys_ij) +
                    sum(x[(j, i, k)] for k in keys_ji) <= 1
                )

    num_comp_expr = sum(s[v] for v in V) + sum(w[v] for v in V) + sum(c[v] for v in V)
    mixed_expr = sum(arc_mixed[arc] * x[arc] for arc in A)

    return dict(model=model, V=V, A=A, x=x, s=s, t=t, z=z, w=w, c=c,
                u_ord=u_ord, num_comp_expr=num_comp_expr, mixed_expr=mixed_expr,
                arc_color=arc_color, arc_is_alt=arc_is_alt,
                arc_adj_id=arc_adj_id, arc_is_fwd=arc_is_fwd)


def _add_alt_usage_constraints(d: Dict, alt_adj_weights: Dict[int, int]) -> None:
    """Add equality constraints forcing each ALT adjacency to be used exactly w times.

    Each haplotype pairing for a given adjacency gets its own independent
    forward/reverse arc pair; forbid_2cycles already stops a single pairing
    from using both directions at once, so summing across *all* arcs (both
    directions, every pairing) sharing an adj_id counts each realized
    pairing exactly once. Summing only the forward-labeled arcs (as before)
    is wrong: a pairing realized via its reverse arc is invisible to that
    sum, so the solver can satisfy "==w" while actually routing the same
    physical junction through a second haplotype pairing for free -- e.g. a
    single balanced rearrangement ending up applied to both haplotype
    copies instead of just one.
    """
    A = d["A"]
    x = d["x"]
    arc_is_alt = d["arc_is_alt"]
    arc_adj_id = d["arc_adj_id"]
    model = d["model"]

    # Group ALT arcs (both directions) by adj_id
    adj_arcs: Dict[int, List] = {}
    for arc in A:
        if arc_is_alt[arc]:
            gid = arc_adj_id[arc]
            adj_arcs.setdefault(gid, []).append(arc)

    for gid, arcs in adj_arcs.items():
        w = alt_adj_weights.get(gid, 0)
        if w > 0 and arcs:
            model.Add(sum(x[arc] for arc in arcs) == w)


def _extract_components(V, A, x, s, w, value_fn) -> List[Dict]:
    succ = {v: None for v in V}
    for arc in A:
        if value_fn(x[arc]) == 1:
            i, j = arc[:2]
            succ[i] = j

    used, components = set(), []
    for v in V:
        if value_fn(w[v]) == 1:
            components.append({"type": "singleton", "nodes": [v]})
            used.add(v)
    for v in V:
        if value_fn(s[v]) == 1:
            path, cur = [v], v
            used.add(v)
            while succ[cur] is not None:
                cur = succ[cur]; path.append(cur); used.add(cur)
            components.append({"type": "path", "nodes": path})
    for v in V:
        if v in used or succ[v] is None:
            continue
        cyc, cur = [v], succ[v]
        used.add(v)
        while cur != v:
            cyc.append(cur); used.add(cur); cur = succ[cur]
        components.append({"type": "cycle", "nodes": cyc})
    return components


def _solve_subgraph(
    G: nx.MultiDiGraph,
    time_limit_s: float,
    cpu_n: int,
    alt_adj_weights: Dict[int, int],
) -> Dict:
    if len(G.nodes()) == 0:
        return {"status": "OPTIMAL", "objective_num_components": 0,
                "objective_num_mixed_haplotype_edges": 0, "components": [],
                "grey_pairs": []}

    # Stage 1: minimize components (with ALT usage constraints)
    d1 = _build_model(G)
    _add_alt_usage_constraints(d1, alt_adj_weights)
    d1["model"].Minimize(d1["num_comp_expr"])
    s1 = cp_model.CpSolver()
    s1.parameters.max_time_in_seconds = time_limit_s
    s1.parameters.num_search_workers = cpu_n
    st1 = s1.Solve(d1["model"])
    if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "INFEASIBLE_OR_UNKNOWN"}
    opt_comp = int(s1.ObjectiveValue())

    # Stage 2: fix component count, minimize cross-haplotype edges
    d2 = _build_model(G)
    _add_alt_usage_constraints(d2, alt_adj_weights)
    d2["model"].Add(d2["num_comp_expr"] == opt_comp)
    d2["model"].Minimize(d2["mixed_expr"])
    s2 = cp_model.CpSolver()
    s2.parameters.max_time_in_seconds = time_limit_s
    s2.parameters.num_search_workers = cpu_n
    st2 = s2.Solve(d2["model"])
    if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "INFEASIBLE_OR_UNKNOWN"}

    V, A, x, sv, w = d2["V"], d2["A"], d2["x"], d2["s"], d2["w"]
    arc_color = d2["arc_color"]
    arc_is_alt = d2["arc_is_alt"]
    components = _extract_components(V, A, x, sv, w, s2.Value)
    status = "OPTIMAL" if st1 == cp_model.OPTIMAL and st2 == cp_model.OPTIMAL else "FEASIBLE"

    # Collect every used grey-colored arc that needs an explicit grey edge
    # in the EBG -- every used ALT arc (a real structural rearrangement)
    # and every used REF arc, with no exceptions. A same-haplotype,
    # copy-index-0 "vanilla" REF arc is NOT special-cased out here: even
    # though gr_cn.ebg.builder also adds a black edge for every REF
    # adjacency on every matching haplotype pair at copy_index 0
    # unconditionally (Genome A's structure doesn't depend on what the
    # phasing chose), that black edge alone does not tell the EBG whether
    # this specific connection is *also* real in Genome B -- only an
    # explicit grey edge does that. Skipping it here previously left
    # grey_neighbor unset even when the solver genuinely selected the arc,
    # which rendered the affected nodes as disconnected in Genome B.
    # Deduplicate by canonical (smaller id first).
    grey_pairs = []
    seen: set = set()
    for arc in A:
        if s2.Value(x[arc]) != 1 or arc_color[arc] != "G":
            continue
        u, v = arc[0], arc[1]
        canon = (u, v) if id(u) < id(v) else (v, u)
        if canon not in seen:
            seen.add(canon)
            grey_pairs.append((u, v))

    return {
        "status": status,
        "objective_num_components": opt_comp,
        "objective_num_mixed_haplotype_edges": int(s2.ObjectiveValue()),
        "num_paths": sum(s2.Value(sv[v]) for v in V),
        "num_singletons": sum(s2.Value(w[v]) for v in V),
        "components": components,
        "grey_pairs": grey_pairs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def phase(
    graph: JabbaGraph,
    reference: ReferenceGenome,
    L: int = 2,
    time_limit_s: float = 200.0,
    cpu_n: int = 1,
) -> PhasingResult:
    """
    Run the phasing ILP on a parsed JABBA graph.

    Parameters
    ----------
    graph:
        Parsed JABBA graph (output of io.jabba.load).
    reference:
        Reference genome (Genome A) from io.reference.build(), used to look up
        each interval's n_deleted / n_gained so haplotype copies that will be
        DC-capped (deleted) are excluded from phasing up front — see note below.
    L:
        Number of haplotypes (default 2 for diploid).
    time_limit_s:
        Per-subgraph CP-SAT time limit in seconds.
    cpu_n:
        Number of CP-SAT worker threads.

    Returns
    -------
    PhasingResult
        Phased components with haplotype assignments and ALT adjacency pairs.

    Notes
    -----
    Deleted haplotype copies (n_deleted > 0) are always haplotype indices
    0..n_deleted-1 (both, for a homozygous deletion) — matching the
    convention gr_cn.ebg.builder uses for DC-node capping. Those copies are
    excluded from the phasing graph entirely so the ILP can never route a
    real REF/ALT connection through a copy that doesn't exist in Genome B;
    without this, a real junction and a DC-cap edge can collide on the same
    node, silently corrupting the resulting EBG (see git history for the
    PM1284 chr3/11/12 case that surfaced this).
    """
    nodes_to_phase: List[PhaseNode] = []
    block_edges: List[Tuple[PhaseNode, PhaseNode]] = []

    for iv in graph.intervals:
        cn_info = reference.get(iv.iid)
        n_deleted = cn_info.n_deleted if cn_info else 0
        n_gained = cn_info.n_gained if cn_info else 0

        for hap in range(L):
            if hap < n_deleted:
                continue
            t_node: PhaseNode = (iv.iid, "t", hap, 0)
            h_node: PhaseNode = (iv.iid, "h", hap, 0)
            nodes_to_phase += [t_node, h_node]
            block_edges.append((t_node, h_node))
        for copy_i in range(1, n_gained + 1):
            t_node = (iv.iid, "t", 0, copy_i)
            h_node = (iv.iid, "h", 0, copy_i)
            nodes_to_phase += [t_node, h_node]
            block_edges.append((t_node, h_node))

    strip_to_nodes: Dict[str, List[PhaseNode]] = {}
    for node in nodes_to_phase:
        key = f"{node[0]}_{node[1]}"
        strip_to_nodes.setdefault(key, []).append(node)

    def expand_edge(src_key: str, snk_key: str) -> List[Tuple[PhaseNode, PhaseNode]]:
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

    # ALT adjacencies are mandatory: track their required usage count
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

    all_components: List[Dict] = []
    all_grey_pairs: List[Tuple[PhaseNode, PhaseNode]] = []
    subgraph_results: Dict[str, Dict] = {}
    total_comp = 0
    total_mixed = 0
    overall_status = "OPTIMAL"

    for i, sg in enumerate(subgraphs):
        key = f"subgraph_{i}"
        result = _solve_subgraph(sg, time_limit_s=time_limit_s, cpu_n=cpu_n,
                                 alt_adj_weights=alt_adj_weights)
        subgraph_results[key] = result
        if result["status"] == "INFEASIBLE_OR_UNKNOWN":
            overall_status = "INFEASIBLE_OR_UNKNOWN"
            continue
        if result["status"] == "FEASIBLE":
            overall_status = "FEASIBLE"
        all_components.extend(result.get("components", []))
        all_grey_pairs.extend(result.get("grey_pairs", []))
        total_comp += result.get("objective_num_components", 0)
        total_mixed += result.get("objective_num_mixed_haplotype_edges", 0)

    return PhasingResult(
        components=all_components,
        grey_pairs=all_grey_pairs,
        subgraph_results=subgraph_results,
        num_components=total_comp,
        num_mixed_haplotype_edges=total_mixed,
        status=overall_status,
    )
