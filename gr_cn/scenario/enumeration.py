"""
Dynamic ordered-partition search for EBG cycle scenarios.

The key idea: the chromosome decomposition is not fixed — it changes as cycles
are applied (black - grey).  The cost of a step is evaluated on the intermediate
genome *just before* that step, so the same cycle may have different cost
depending on the order in which earlier cycles were executed.

The search strategies are:
  - exact (branch-and-bound): optimal but exponential in n_cycles
  - beam: heuristic, good for large n_cycles
  - greedy: fast upper bound
  - auto: exact for n_cycles ≤ exact_max_cycles, otherwise beam

Entry point:
    find_best_scenario(cycles, all_nodes, objective=cost_max, ...) -> ScenarioResult
"""
from __future__ import annotations

import math
import random
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

def _greedy_upper_bound(
    model: DynamicGenomeScenario,
    timing: str,
    seed: int = 0,
    restarts: int = 50,
) -> ScenarioResult:
    rng = random.Random(seed)
    best_obj = math.inf
    best_partition: Optional[List[List[int]]] = None
    best_costs: Optional[List[int]] = None
    best_currs: Optional[List[int]] = None

    for _ in range(max(1, restarts)):
        state = 0
        partition, costs, currs = [], [], []
        while state != model.full_mask:
            remaining = list(_bit_indices(model.full_mask ^ state))
            rng.shuffle(remaining)
            singles = sorted(
                [(model.step_cost(state, 1 << i, timing), rng.random(), 1 << i) for i in remaining]
            )
            step = singles[0][2]
            changed = True
            while changed:
                changed = False
                base = model.step_cost(state, step, timing)
                opts = sorted([
                    (model.step_cost(state, step | (1 << i), timing) - base,
                     model.step_cost(state, step | (1 << i), timing), rng.random(), i)
                    for i in remaining if not (step & (1 << i))
                ])
                for inc, c, _, i in opts:
                    if inc <= 0 or (inc == 1 and rng.random() < 0.15):
                        step |= 1 << i
                        changed = True
                        break
            currs.append(model.n_chromosomes(state))
            costs.append(model.step_cost(state, step, timing))
            partition.append(list(_bit_indices(step)))
            state |= step

        obj = max(costs) if costs else 0
        if obj < best_obj or (obj == best_obj and best_partition and len(partition) < len(best_partition)):
            best_obj, best_partition, best_costs, best_currs = obj, partition, costs, currs

    return ScenarioResult(
        status="FEASIBLE",
        objective_value=int(best_obj),
        partition=best_partition or [],
        step_costs=best_costs or [],
        step_num_chromosomes=best_currs or [],
        method="greedy",
        n_cycles=model.n_cycles,
        expanded_states=0,
        exact=False,
    )


def _exact_branch_and_bound(
    model: DynamicGenomeScenario,
    timing: str,
    ub: Optional[ScenarioResult] = None,
) -> ScenarioResult:
    if ub is None:
        ub = _greedy_upper_bound(model, timing)

    # Use lists so inner function can mutate without nonlocal
    best_obj = [ub.objective_value]
    best = [ub.partition, ub.step_costs, ub.step_num_chromosomes]
    expanded = [0]
    seen: Dict[int, int] = {}

    def dfs(state: int, cur_max: int, parts: List[int], costs: List[int], currs: List[int]) -> None:
        expanded[0] += 1

        # Check the finish line before consulting the memo below: `seen` only
        # tracks the best cur_max reached per state, not step count, so a
        # longer path that reaches full_mask first can "claim" the memo slot
        # and cause a later, shorter path landing on the same final cur_max to
        # be pruned before ever reaching the tie-break (fewest steps) below.
        if state == model.full_mask:
            if cur_max < best_obj[0] or (cur_max == best_obj[0] and len(parts) < len(best[0])):
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
        subs = []
        sub = remaining
        while sub:
            subs.append(sub)
            sub = (sub - 1) & remaining
        subs.sort(key=lambda s: (model.step_cost(state, s, timing), -_popcount(s)))

        for step in subs:
            c = model.step_cost(state, step, timing)
            new_max = max(cur_max, c)
            if new_max > best_obj[0]:
                continue
            if new_max == best_obj[0] and len(parts) + 1 >= len(best[0]):
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
    )


def _beam_search(
    model: DynamicGenomeScenario,
    timing: str,
    beam_width: int,
    candidate_limit: int,
    seed: int,
    ub: Optional[ScenarioResult] = None,
) -> ScenarioResult:
    rng = random.Random(seed)
    if ub is None:
        ub = _greedy_upper_bound(model, timing, seed=seed, restarts=100)
    best = ub
    best_obj = ub.objective_value

    beam: Dict[int, tuple] = {0: (0, [], [], [])}
    expanded = 0

    while beam:
        next_beam: Dict[int, tuple] = {}
        for state, (cur_max, parts, costs, currs) in beam.items():
            if state == model.full_mask:
                continue
            expanded += 1
            remaining = list(_bit_indices(model.full_mask ^ state))

            # Candidates: all singletons + greedy packs
            cands: Set[int] = {1 << i for i in remaining}
            for seed_i in remaining:
                step = 1 << seed_i
                improved = True
                while improved:
                    improved = False
                    base = model.step_cost(state, step, timing)
                    opts = sorted([
                        (model.step_cost(state, step | (1 << i), timing) - base, rng.random(), i)
                        for i in remaining if not (step & (1 << i))
                    ])
                    for inc, _, i in opts:
                        if best_obj == math.inf or model.step_cost(state, step | (1 << i), timing) <= best_obj:
                            step |= 1 << i
                            cands.add(step)
                            improved = True
                            break
                cands.add(step)

            ranked = sorted(
                cands,
                key=lambda s: (model.step_cost(state, s, timing), -_popcount(s)),
            )[:candidate_limit]

            for step in ranked:
                c = model.step_cost(state, step, timing)
                new_max = max(cur_max, c)
                if new_max > best_obj + 2:
                    continue
                ns = state | step
                nparts = parts + [step]
                ncosts = costs + [c]
                ncurrs = currs + [model.n_chromosomes(state)]

                if ns == model.full_mask:
                    if new_max < best_obj or (new_max == best_obj and len(nparts) < len(best.partition)):
                        best_obj = new_max
                        best = ScenarioResult(
                            status="FEASIBLE",
                            objective_value=int(new_max),
                            partition=[list(_bit_indices(s)) for s in nparts],
                            step_costs=ncosts,
                            step_num_chromosomes=ncurrs,
                            method="beam_search",
                            n_cycles=model.n_cycles,
                            expanded_states=expanded,
                            exact=False,
                        )
                    continue

                old = next_beam.get(ns)
                if old is None or (new_max, len(nparts)) < (old[0], len(old[1])):
                    next_beam[ns] = (new_max, nparts, ncosts, ncurrs)

        ranked_beam = sorted(
            next_beam.items(),
            key=lambda kv: (kv[1][0], -_popcount(kv[0]), len(kv[1][1])),
        )
        beam = dict(ranked_beam[:beam_width])

    best.expanded_states = expanded
    return best


# ---------------------------------------------------------------------------
# ILP: subset DP (provably optimal, O(3^k) time, O(2^k) space)
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

def find_best_scenario(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]] = None,
    count_components: str = "real",
    timing: str = "before",
    method: str = "auto",
    exact_max_cycles: int = 16,
    ilp_max_cycles: int = 18,
    beam_width: int = 5000,
    candidate_limit: int = 100,
    random_restarts: int = 200,
    seed: int = 0,
) -> ScenarioResult:
    """
    Find the scenario (ordered partition of EBG cycles) that minimises
    the maximum number of genome components touched per step (cost_max).

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
        'auto' | 'exact' | 'ilp' | 'beam' | 'greedy'
    exact_max_cycles:
        Use exact (B&B) search for n_cycles ≤ this in auto mode.
    ilp_max_cycles:
        Use DP-exact ILP for n_cycles ≤ this when method='ilp'; beam above.
    """
    model = DynamicGenomeScenario(cycles, all_nodes, count_components)

    if model.n_cycles == 0:
        return ScenarioResult(
            status="OPTIMAL", objective_value=0, partition=[], step_costs=[],
            step_num_chromosomes=[], method=method, n_cycles=0,
            expanded_states=0, exact=True,
        )

    greedy = _greedy_upper_bound(model, timing, seed=seed, restarts=random_restarts)

    if method == "greedy":
        return greedy
    if method == "exact":
        return _exact_branch_and_bound(model, timing, ub=greedy)
    if method == "beam":
        return _beam_search(model, timing, beam_width, candidate_limit, seed, ub=greedy)
    if method == "ilp":
        if model.n_cycles <= ilp_max_cycles:
            return _dp_exact(model, timing)
        result = _beam_search(model, timing, beam_width, candidate_limit, seed, ub=greedy)
        result.message = (
            f"Beam search used (n_cycles={model.n_cycles} > ilp_max_cycles={ilp_max_cycles})."
        )
        result.method = "ilp_beam_fallback"
        return result

    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'exact', 'ilp', 'beam', or 'greedy', got {method!r}"
        )

    if model.n_cycles <= exact_max_cycles:
        return _exact_branch_and_bound(model, timing, ub=greedy)

    result = _beam_search(model, timing, beam_width, candidate_limit, seed, ub=greedy)
    result.message = (
        f"Beam search used (n_cycles={model.n_cycles} > exact_max_cycles={exact_max_cycles})."
    )
    return result


def find_best_scenario_clustered(
    cycles: Sequence[Sequence[EBGNode]],
    all_nodes: Optional[Iterable[EBGNode]] = None,
    count_components: str = "real",
    timing: str = "before",
    method: str = "auto",
    exact_max_cycles: int = 16,
    ilp_max_cycles: int = 18,
    beam_width: int = 5000,
    candidate_limit: int = 100,
    random_restarts: int = 200,
    seed: int = 0,
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
        local = find_best_scenario(
            cluster_cycs,
            all_nodes=all_nodes_list,
            count_components=count_components,
            timing=timing,
            method=method,
            exact_max_cycles=exact_max_cycles,
            ilp_max_cycles=ilp_max_cycles,
            beam_width=beam_width,
            candidate_limit=candidate_limit,
            random_restarts=random_restarts,
            seed=seed,
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
) -> Iterator[ScenarioResult]:
    """
    Yield every ordered partition with its dynamic cost profile.
    Only feasible for small cycle counts (Fubini numbers grow fast).
    """
    model = DynamicGenomeScenario(cycles, all_nodes, count_components)
    if model.n_cycles > max_cycles:
        raise ValueError(
            f"{model.n_cycles} cycles exceeds max_cycles={max_cycles}. "
            "Only use for small instances."
        )

    def rec(state: int, parts: List[int], costs: List[int], currs: List[int]):
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
