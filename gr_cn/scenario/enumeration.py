"""
Dynamic ordered-partition search for EBG cycle scenarios.

The key idea: the chromosome decomposition is not fixed — it changes as cycles
are applied (black - grey).  The cost of a step is evaluated on the intermediate
genome *just before* that step, so the same cycle may have different cost
depending on the order in which earlier cycles were executed.

Two exact search methods are supported (both correct; the graphs in this
project never exceed ~10 cycles, well within reach of either):
  - "dp": subset dynamic programming over all 2^n_cycles states, O(3^n) time.
    Conceptually evaluates every possible ordered partition, just without
    re-deriving shared subproblems.
  - "branch_and_bound": depth-first search with memoization and pruning.
    Typically faster in practice since it never expands provably-suboptimal
    branches, but has no fixed time-complexity guarantee independent of
    instance shape the way the DP does.

Cycles are always first split into independent chromosome clusters
(cluster_cycles) before either method runs — clusters can be ordered fully
independently of one another, so solving them separately is never worse, and
is exactly as correct, as solving the whole cycle set jointly.

Entry point:
    find_best_scenario(cycles, all_nodes, method="dp", ...) -> ScenarioResult
    find_best_scenario_clustered(...) -> (ScenarioResult, per_cluster_detail)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from gr_cn.models.node import EBGNode, NodeType

Edge = Tuple[int, int]
StepMask = int


def _is_ordinary_synteny(node: EBGNode) -> bool:
    """
    True for a synteny node that represents genuine, present-in-both-genomes
    chromosome material -- as opposed to a deleted fragment (DC-capped) or a
    latent gained copy (AH-capped, copy_index > 0).
    """
    if not node.is_real_synteny():
        return False
    if node.copy_index > 0:
        return False  # gained copy -> auxiliary (AH-associated)
    for nb in (node.black_neighbor, node.grey_neighbor):
        if nb is not None and nb.node_type is NodeType.DC:
            return False  # deleted copy -> auxiliary
    return True


# ---------------------------------------------------------------------------
# DSU (union-find) for component detection
# ---------------------------------------------------------------------------

class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    status: str
    objective_value: int               # cost_max value (or cost_sum, depending on objective)
    partition: List[List[int]]         # ordered partition as lists of cycle indices
    step_costs: List[int]              # per-step cost (chromosomes touched)
    step_num_chromosomes: List[int]    # total chromosomes in intermediate genome before each step
    method: str
    n_cycles: int
    expanded_states: int
    exact: bool
    message: str = ""
    # Branch-and-bound only: complete scenarios (leaves) the search reached.
    n_complete_scenarios: int = 0


# ---------------------------------------------------------------------------
# Core simulation class
# ---------------------------------------------------------------------------

def _canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def _bit_indices(mask: int) -> Iterator[int]:
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


class DynamicGenomeScenario:
    """Simulates applying EBG cycles as black-to-grey swaps and evaluates cost."""

    def __init__(
        self,
        cycles: Sequence[Sequence[EBGNode]],
        all_nodes: Optional[Iterable[EBGNode]] = None,
        count_components: str = "real",
    ):
        if count_components not in {"real", "all"}:
            raise ValueError("count_components must be 'real' or 'all'.")
        self.cycles_original = [list(c) for c in cycles]
        self.n_cycles = len(self.cycles_original)
        self.count_components = count_components

        # Build ordered node list
        ordered: List[EBGNode] = []
        seen: Set[int] = set()  # use id() for identity

        def add(x: EBGNode) -> None:
            if id(x) not in seen:
                seen.add(id(x))
                ordered.append(x)

        if all_nodes is not None:
            for x in all_nodes:
                add(x)
        for cyc in self.cycles_original:
            for x in cyc:
                add(x)
                if x.black_neighbor is not None:
                    add(x.black_neighbor)
                if x.grey_neighbor is not None:
                    add(x.grey_neighbor)

        self.nodes = ordered
        self.node_to_id = {id(x): i for i, x in enumerate(ordered)}
        self.is_real = [_is_ordinary_synteny(x) for x in ordered]

        self.cycle_node_ids = [
            tuple(sorted({self.node_to_id[id(x)] for x in cyc}))
            for cyc in self.cycles_original
        ]

        # Extract black and grey edges per cycle
        self.black_edges: List[Tuple[Edge, ...]] = []
        self.grey_edges: List[Tuple[Edge, ...]] = []
        for cyc in self.cycles_original:
            b, g = self._cycle_edges(cyc)
            self.black_edges.append(tuple(sorted(b)))
            self.grey_edges.append(tuple(sorted(g)))

        # Base edges: immutable t-h synteny block edges
        base: Set[Edge] = set()
        by_seg: Dict[tuple, Dict[str, int]] = {}
        for nid, node in enumerate(self.nodes):
            if node.node_type is not NodeType.INTERNAL:
                continue
            key = (node.iid, node.haplotype, node.copy_index)
            by_seg.setdefault(key, {})[node.end.value] = nid
        for sides in by_seg.values():
            if "t" in sides and "h" in sides:
                base.add(_canonical_edge(sides["t"], sides["h"]))

        # Every node with no owning cycle connects to its real neighbour --
        # untouched backbone, or an unresolved/unmodeled structural element
        # outside any extracted cycle (e.g. an open loose-end path per
        # ebg.cycles.extract_cycles's docstring). Without this, distinct
        # segments never merge into the true chromosome components they
        # physically belong to: a cycle's own touched nodes would appear
        # severed from the untouched backbone they're actually still
        # attached to, and two untouched segments would never merge at all
        # -- both wildly overcounting "chromosomes touched" per step.
        # Such a node is never toggled by the scenario, so it always shows
        # its real Genome-B connection (grey_neighbor) -- falling back to
        # black_neighbor only when grey is unset -- rather than adding both
        # pointers as if they were simultaneously real: for a node outside
        # any cycle, black_neighbor and grey_neighbor can genuinely differ
        # (that's exactly what "unresolved" here means), and only one of
        # them is ever true structure.
        # Requires `all_nodes` to be the *unstripped* node list (e.g.
        # PipelineResult.raw_nodes) -- a stripped list (post
        # cycles.remove_shared_adjacencies) has already nulled both
        # pointers on truly-unchanged nodes, so there is nothing left here
        # to recover their connectivity from.
        owner: Dict[int, int] = {
            nid: cyc_idx for cyc_idx, node_ids in enumerate(self.cycle_node_ids) for nid in node_ids
        }
        shared: Set[Edge] = set()
        for nid, node in enumerate(self.nodes):
            if owner.get(nid) is not None:
                continue
            nb = node.grey_neighbor if node.grey_neighbor is not None else node.black_neighbor
            if nb is None or id(nb) not in self.node_to_id:
                continue
            nbid = self.node_to_id[id(nb)]
            if owner.get(nbid) is not None:
                continue  # a cycle-owned neighbour's edges are already covered by that cycle's own black/grey_edges
            shared.add(_canonical_edge(nid, nbid))

        self.base_edges: Tuple[Edge, ...] = tuple(sorted(base | shared))

        self.full_mask = (1 << self.n_cycles) - 1
        self._cache: Dict[int, tuple] = {}

    def _cycle_edges(self, cyc: Sequence[EBGNode]) -> Tuple[Set[Edge], Set[Edge]]:
        cyc_ids = {id(x) for x in cyc}
        black, grey = set(), set()
        for node in cyc:
            uid = self.node_to_id[id(node)]
            if node.black_neighbor is not None and id(node.black_neighbor) in cyc_ids:
                black.add(_canonical_edge(uid, self.node_to_id[id(node.black_neighbor)]))
            if node.grey_neighbor is not None and id(node.grey_neighbor) in cyc_ids:
                grey.add(_canonical_edge(uid, self.node_to_id[id(node.grey_neighbor)]))
        return black, grey

    def _state_info(self, state: int) -> tuple:
        cached = self._cache.get(state)
        if cached is not None:
            return cached

        edges = list(self.base_edges)
        for i in range(self.n_cycles):
            edges.extend(self.grey_edges[i] if state & (1 << i) else self.black_edges[i])

        dsu = _DSU(len(self.nodes))
        for u, v in edges:
            dsu.union(u, v)

        root_to_comp: Dict[int, int] = {}
        comp_of_node = [0] * len(self.nodes)
        for nid in range(len(self.nodes)):
            r = dsu.find(nid)
            if r not in root_to_comp:
                root_to_comp[r] = len(root_to_comp)
            comp_of_node[nid] = root_to_comp[r]

        n_comp = len(root_to_comp)
        comp_is_real = [False] * n_comp
        for nid, comp in enumerate(comp_of_node):
            if self.is_real[nid]:
                comp_is_real[comp] = True

        # Per-cycle component masks
        masks_all, masks_real = [], []
        for node_ids in self.cycle_node_ids:
            ma, mr = 0, 0
            for nid in node_ids:
                comp = comp_of_node[nid]
                ma |= 1 << comp
                if comp_is_real[comp]:
                    mr |= 1 << comp
            masks_all.append(ma)
            masks_real.append(mr)

        info = (comp_of_node, comp_is_real, masks_all, masks_real,
                n_comp, sum(comp_is_real))
        self._cache[state] = info
        return info

    def n_chromosomes(self, state: int) -> int:
        info = self._state_info(state)
        return info[5] if self.count_components == "real" else info[4]

    def step_cost(self, state: int, step: int, timing: str = "before") -> int:
        def touched(s: int) -> int:
            info = self._state_info(s)
            masks = info[3] if self.count_components == "real" else info[2]
            t = 0
            for i in _bit_indices(step):
                t |= masks[i]
            return _popcount(t)

        if timing == "before":
            return touched(state)
        if timing == "after":
            return touched(state | step)
        return max(touched(state), touched(state | step))


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def _exact_branch_and_bound(
    model: DynamicGenomeScenario,
    timing: str,
) -> ScenarioResult:
    """Exact depth-first branch-and-bound over execution states (sets of
    executed cycles), minimizing cost_max (Algorithm 2 of the paper).

    Pruning:
      - bound: a child whose running maximum reaches the incumbent is dropped
        (cost_max never decreases along a branch);
      - dominance: a state already reached with a running maximum <= the
        current one is not expanded again (its completions depend only on
        the state, not on how it was reached);
      - singleton bound: step_cost is monotone in the step, so a step costs at
        least its costliest single cycle; children that cannot beat the
        incumbent are dropped before their cost is evaluated.
    Children are explored cheapest first (larger steps first among equals)
    so that a good incumbent is found early. Only the objective is optimized;
    among optimal scenarios, the one found first is returned.
    """
    best_obj = [float("inf")]
    best = [[], [], []]
    expanded = [0]
    complete = [0]
    seen: Dict[int, int] = {}
    # Monotonicity only holds when cost is taken on the state before the
    # step; with 'after'/'both' the evaluated genome depends on the step.
    use_singleton_bound = timing == "before"

    def dfs(state: int, cur_max: int, parts: List[int], costs: List[int], currs: List[int]) -> None:
        expanded[0] += 1

        if state == model.full_mask:
            complete[0] += 1
            if cur_max < best_obj[0]:
                best_obj[0] = cur_max
                best[0] = [list(_bit_indices(s)) for s in parts]
                best[1] = costs[:]
                best[2] = currs[:]
            return

        old = seen.get(state)
        if old is not None and old <= cur_max:
            return
        seen[state] = cur_max

        remaining = model.full_mask ^ state
        single = {i: (model.step_cost(state, 1 << i, timing) if use_singleton_bound else 0)
                  for i in _bit_indices(remaining)}
        children = []
        sub = remaining
        while sub:
            if max(cur_max, max(single[i] for i in _bit_indices(sub))) < best_obj[0]:
                if use_singleton_bound and sub & (sub - 1) == 0:
                    c = single[sub.bit_length() - 1]
                else:
                    c = model.step_cost(state, sub, timing)
                children.append((c, -_popcount(sub), sub))
            sub = (sub - 1) & remaining
        children.sort()

        for c, _, step in children:
            new_max = max(cur_max, c)
            if new_max >= best_obj[0]:
                continue
            dfs(state | step, new_max, parts + [step], costs + [c],
                currs + [model.n_chromosomes(state)])

    dfs(0, 0, [], [], [])
    return ScenarioResult(
        status="OPTIMAL",
        objective_value=int(best_obj[0]),
        partition=best[0],
        step_costs=best[1],
        step_num_chromosomes=best[2],
        method="exact_branch_and_bound",
        n_cycles=model.n_cycles,
        expanded_states=expanded[0],
        exact=True,
        n_complete_scenarios=complete[0],
    )


# ---------------------------------------------------------------------------
# Full enumeration via subset DP (provably optimal, O(3^k) time, O(2^k) space)
# ---------------------------------------------------------------------------

def _dp_exact(
    model: DynamicGenomeScenario,
    timing: str,
) -> ScenarioResult:
    """
    Optimal ordered-partition search via subset DP.

    Equivalent to solving the scenario ILP exactly. Uses the same cost function
    as the branch-and-bound but fills the DP table bottom-up, guaranteeing
    O(3^k) time with O(2^k) space. Practical for k <= 18.
    """
    k = model.n_cycles
    full = model.full_mask
    INF = float('inf')

    dp_cost = [INF] * (full + 1)
    dp_step = [0] * (full + 1)
    dp_cost[full] = 0.0

    for mask in range(full - 1, -1, -1):
        remaining = full & ~mask
        sub = remaining
        while sub:
            c = float(model.step_cost(mask, sub, timing))
            next_mask = mask | sub
            total = max(c, dp_cost[next_mask])
            if total < dp_cost[mask]:
                dp_cost[mask] = total
                dp_step[mask] = sub
            sub = (sub - 1) & remaining

    # Reconstruct partition
    partition: List[List[int]] = []
    step_costs: List[int] = []
    step_chroms: List[int] = []
    state = 0
    while state != full:
        step = dp_step[state]
        step_costs.append(model.step_cost(state, step, timing))
        step_chroms.append(model.n_chromosomes(state))
        partition.append(list(_bit_indices(step)))
        state |= step

    return ScenarioResult(
        status="OPTIMAL",
        objective_value=int(dp_cost[0]) if dp_cost[0] < INF else 0,
        partition=partition,
        step_costs=step_costs,
        step_num_chromosomes=step_chroms,
        method="ilp_dp",
        n_cycles=k,
        expanded_states=full + 1,
        exact=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_best_scenario_single(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]],
    count_components: str,
    timing: str,
    method: str,
) -> ScenarioResult:
    """Solve one (already-clustered) set of cycles jointly -- no further
    decomposition. Internal helper; use find_best_scenario / _clustered."""
    model = DynamicGenomeScenario(cycles, all_nodes, count_components)

    if model.n_cycles == 0:
        return ScenarioResult(
            status="OPTIMAL", objective_value=0, partition=[], step_costs=[],
            step_num_chromosomes=[], method=method, n_cycles=0,
            expanded_states=0, exact=True,
        )

    if method == "dp":
        return _dp_exact(model, timing)
    if method == "branch_and_bound":
        return _exact_branch_and_bound(model, timing)
    raise ValueError(f"method must be 'dp' or 'branch_and_bound', got {method!r}")


def find_best_scenario(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]] = None,
    count_components: str = "real",
    timing: str = "before",
    method: str = "dp",
) -> ScenarioResult:
    """
    Find the scenario (ordered partition of EBG cycles) that minimises
    the maximum number of genome components touched per step (cost_max).

    Cycles are always first split into independent chromosome clusters and
    solved separately (see find_best_scenario_clustered) -- this is never
    worse than solving them jointly and is exactly as correct.

    Parameters
    ----------
    cycles:
        Alternating cycles from ebg.cycles.extract_cycles().
    all_nodes:
        All EBG nodes (list(nodes)); used to infer synteny block edges.
    count_components:
        'real' counts only components containing real synteny nodes;
        'all' counts every component including DC/AH auxiliaries.
    timing:
        'before' | 'after' | 'both' — when to evaluate chromosome count.
    method:
        'dp' (subset dynamic programming, full enumeration) or
        'branch_and_bound' (depth-first search with pruning). Both exact.
    """
    combined, _per_cluster = find_best_scenario_clustered(
        cycles, all_nodes=all_nodes, count_components=count_components,
        timing=timing, method=method,
    )
    return combined


def find_best_scenario_clustered(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]] = None,
    count_components: str = "real",
    timing: str = "before",
    method: str = "dp",
) -> Tuple["ScenarioResult", List[Tuple[List[int], "ScenarioResult"]]]:
    """
    Find optimal scenario by optimizing each independent chromosome cluster separately.

    Cycles involving completely disjoint chromosome sets can be ordered independently —
    the optimal ordering within one cluster does not affect another cluster's cost.

    Returns
    -------
    combined : ScenarioResult
        All cluster steps concatenated; objective = max per-cluster objective.
    per_cluster : list of (global_cycle_indices, ScenarioResult) per cluster.
    """
    from gr_cn.scenario.clustering import cluster_cycles

    cycles_list = list(cycles)
    clusters = cluster_cycles(cycles_list)
    all_nodes_list = list(all_nodes) if all_nodes is not None else None

    per_cluster: List[Tuple[List[int], ScenarioResult]] = []
    for cluster_idx in clusters:
        cluster_cycs = [cycles_list[i] for i in cluster_idx]
        local = _find_best_scenario_single(
            cluster_cycs,
            all_nodes=all_nodes_list,
            count_components=count_components,
            timing=timing,
            method=method,
        )
        global_partition = [[cluster_idx[li] for li in step] for step in local.partition]
        remapped = ScenarioResult(
            status=local.status,
            objective_value=local.objective_value,
            partition=global_partition,
            step_costs=local.step_costs,
            step_num_chromosomes=local.step_num_chromosomes,
            method=local.method,
            n_cycles=local.n_cycles,
            expanded_states=local.expanded_states,
            exact=local.exact,
            message=local.message,
        )
        per_cluster.append((cluster_idx, remapped))

    all_partition: List[List[int]] = []
    all_costs: List[int] = []
    all_chroms: List[int] = []
    for _, cr in per_cluster:
        all_partition.extend(cr.partition)
        all_costs.extend(cr.step_costs)
        all_chroms.extend(cr.step_num_chromosomes)

    combined_obj = max((cr.objective_value for _, cr in per_cluster), default=0)
    is_exact = all(cr.exact for _, cr in per_cluster)

    combined = ScenarioResult(
        status="OPTIMAL" if is_exact else "FEASIBLE",
        objective_value=combined_obj,
        partition=all_partition,
        step_costs=all_costs,
        step_num_chromosomes=all_chroms,
        method="clustered",
        n_cycles=len(cycles_list),
        expanded_states=sum(cr.expanded_states for _, cr in per_cluster),
        exact=is_exact,
        message=f"{len(clusters)} independent cluster(s) solved separately.",
    )
    return combined, per_cluster


def enumerate_all_scenarios(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]] = None,
    count_components: str = "real",
    timing: str = "before",
    max_cycles: int = 10,
    stats: Optional[dict] = None,
) -> Iterator[ScenarioResult]:
    """
    Yield every ordered partition with its dynamic cost profile.
    Only feasible for small cycle counts (Fubini numbers grow fast).

    stats: if given, stats["nodes"] counts the nodes of the enumeration
    tree (the root, every partial and every complete scenario).
    """
    model = DynamicGenomeScenario(cycles, all_nodes, count_components)
    if model.n_cycles > max_cycles:
        raise ValueError(
            f"{model.n_cycles} cycles exceeds max_cycles={max_cycles}. "
            "Only use for small instances."
        )

    if stats is not None:
        stats["nodes"] = 0

    def rec(state: int, parts: List[int], costs: List[int], currs: List[int]):
        if stats is not None:
            stats["nodes"] += 1
        if state == model.full_mask:
            yield ScenarioResult(
                status="FEASIBLE",
                objective_value=max(costs) if costs else 0,
                partition=[list(_bit_indices(s)) for s in parts],
                step_costs=costs[:],
                step_num_chromosomes=currs[:],
                method="enumeration",
                n_cycles=model.n_cycles,
                expanded_states=0,
                exact=True,
            )
            return
        remaining = model.full_mask ^ state
        sub = remaining
        while sub:
            c = model.step_cost(state, sub, timing)
            yield from rec(state | sub, parts + [sub], costs + [c],
                           currs + [model.n_chromosomes(state)])
            sub = (sub - 1) & remaining

    yield from rec(0, [], [], [])
