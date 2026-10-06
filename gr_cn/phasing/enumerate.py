"""
Exact enumeration of every tied-optimal haplotype-labeled phasing option.

Optimality is defined by a strict lexicographic hierarchy, solved jointly
(not greedily) at each stage with CP-SAT:

  stage 1: minimize number of phased connected components   (existing ILP obj)
  stage 2: minimize total cross-haplotype ("mixed") edges    (existing ILP obj)
  stage 3: minimize the COUNT of hap1 nodes that touch any mixed edge
           (the hap0-priority rule, solved exactly, not greedily -- a greedy
           per-node lock in iid order was shown to leave a real sample with
           22 "unresolved" nodes when the true joint minimum was only 6)

Every subgraph solution tied at (opt_comp, opt_mixed, opt_bad) is enumerated
via no-good cuts, deduplicated by its grey_pairs signature. Subgraphs combine
by Cartesian product (bounded) and each combined phasing is then crossed with
every tied-minimal loose-end pairing.

This is the exhaustive counterpart to ilp.phase(): phase() returns one
optimal phasing; enumerate_full_options() returns every phasing tied at that
same optimum, crossed with every tied loose-end pairing.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from gr_cn.phasing import ilp, loose
from gr_cn.phasing.ilp import build_subgraphs

# Solve limits are CP-SAT *deterministic* time (a measure of solver work, not
# seconds), so a solve that hits its limit does so identically on every
# machine and under any load: with one worker, phasing enumeration is then
# reproducible. A generous wall-clock limit only guards against runaways.
SOLVE_TIME_LIMIT_S = 120.0
ENUM_SOLVE_TIME_LIMIT_S = 15.0
WALL_CLOCK_GUARD_S = 3600.0
# Ties at the exact 3-stage optimum can genuinely number in the dozens+ for
# larger subgraphs. Enumerating literally all of them is not tractable in
# report-generation time, so this is a reporting cap, not a claim of
# exhaustiveness -- meta["truncated"] records when it was hit.
MAX_SOLUTIONS_PER_SUBGRAPH = 128
MAX_ATTEMPTS_PER_SUBGRAPH = 1500
MAX_TOTAL_PHASING_COMBOS = 128
VERBOSE = True

_CROSS_CHR_PENALTY = loose._CROSS_CHR_PENALTY


def _solver(time_limit: float = SOLVE_TIME_LIMIT_S, workers: int = 1, seed: int = 0) -> cp_model.CpSolver:
    s = cp_model.CpSolver()
    s.parameters.max_deterministic_time = time_limit
    s.parameters.max_time_in_seconds = WALL_CLOCK_GUARD_S
    s.parameters.num_search_workers = workers
    s.parameters.random_seed = seed
    return s


def _solve_or_die(solver: cp_model.CpSolver, model, label: str) -> int:
    """Solve and return the objective value, raising loudly instead of
    silently reading ObjectiveValue() off a timed-out/infeasible solve
    (CP-SAT returns 0 for that, which is indistinguishable from a real
    answer of 0 -- this caused a real bug once: a timed-out stage-1 solve
    silently produced opt_comp=0, cascading into '0 tied options found')."""
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"{label}: CP-SAT solve did not reach OPTIMAL/FEASIBLE "
            f"(status={solver.StatusName(status)}) -- cannot trust ObjectiveValue()."
        )
    return int(solver.ObjectiveValue())


def _node_mixed_terms(d, node):
    terms = []
    for arc in d["A"]:
        u, v, _k = arc
        if (u == node or v == node) and u[2] != v[2]:
            terms.append(d["x"][arc])
    return terms


def _grey_pair_signature(grey_pairs) -> frozenset:
    return frozenset(tuple(sorted((u, v))) for u, v in grey_pairs)


def exact_stage_values(G, alt_adj_weights, cpu_n: int = 1, _retries: int = 3) -> Tuple[int, int, int]:
    """Return (opt_comp, opt_mixed, opt_bad) -- the exact joint optimum.

    Uses cpu_n CP-SAT search workers here (unlike the enumeration loop
    below, which stays single-threaded for reproducible no-good-cut
    ordering) because these three solves just need one fast, certifiably-
    optimal answer.

    CP-SAT with a large worker count on a large model has been observed --
    rarely, non-reproducibly, with no associated exception or bad status --
    to report OPTIMAL with opt_comp=0 for a non-empty graph, which is
    structurally impossible (you cannot cover >0 nodes with 0 components).
    Retrying with a fresh model/solver (and cpu_n forced to 1) has always
    recovered the correct answer, so guard against this rather than trust a
    single solve blindly.
    """
    n_nodes = len(G.nodes())
    for attempt in range(_retries):
        # Vary both worker count and random seed across attempts -- a rare,
        # non-reproducible CP-SAT bug can otherwise return the same
        # impossible result deterministically every time when cpu_n == 1
        # (constant workers=1, seed=0 on every attempt gives byte-identical
        # search behaviour, so a genuine retry needs *something* to differ).
        n_workers = cpu_n if attempt == 0 else 1

        d1 = ilp._build_model(G)
        ilp._add_alt_usage_constraints(d1, alt_adj_weights)
        d1["model"].Minimize(d1["num_comp_expr"])
        s1 = _solver(SOLVE_TIME_LIMIT_S, workers=n_workers, seed=attempt)
        opt_comp = _solve_or_die(s1, d1["model"], "stage1 (min components)")
        if opt_comp > 0 or n_nodes == 0:
            break
        if VERBOSE:
            print(f"    ... stage1 returned impossible opt_comp=0 for {n_nodes}-node graph "
                  f"(attempt {attempt + 1}/{_retries}, workers={n_workers}); retrying", flush=True)
    else:
        raise RuntimeError(
            f"exact_stage_values: stage1 kept returning opt_comp=0 for a {n_nodes}-node "
            f"graph after {_retries} attempts -- not a transient fluke, needs investigation."
        )

    d2 = ilp._build_model(G)
    ilp._add_alt_usage_constraints(d2, alt_adj_weights)
    d2["model"].Add(d2["num_comp_expr"] == opt_comp)
    d2["model"].Minimize(d2["mixed_expr"])
    s2 = _solver(SOLVE_TIME_LIMIT_S, workers=n_workers)
    opt_mixed = _solve_or_die(s2, d2["model"], "stage2 (min mixed edges)")

    d3 = ilp._build_model(G)
    ilp._add_alt_usage_constraints(d3, alt_adj_weights)
    d3["model"].Add(d3["num_comp_expr"] == opt_comp)
    d3["model"].Add(d3["mixed_expr"] == opt_mixed)
    hap1_nodes = sorted({v for v in d3["V"] if v[0] != -1 and v[2] == 1},
                         key=lambda v: (v[0], v[1], v[3]))
    bad = {}
    for node in hap1_nodes:
        terms = _node_mixed_terms(d3, node)
        if not terms:
            continue
        b = d3["model"].NewBoolVar(f"bad_{node}")
        d3["model"].AddMaxEquality(b, terms)
        bad[node] = b
    if bad:
        d3["model"].Minimize(sum(bad.values()))
        s3 = _solver(SOLVE_TIME_LIMIT_S, workers=n_workers)
        opt_bad = _solve_or_die(s3, d3["model"], "stage3 (min bad-node count)")
    else:
        opt_bad = 0

    return opt_comp, opt_mixed, opt_bad


def _build_locked_model(G, alt_adj_weights, opt_comp, opt_mixed, opt_bad):
    d4 = ilp._build_model(G)
    ilp._add_alt_usage_constraints(d4, alt_adj_weights)
    d4["model"].Add(d4["num_comp_expr"] == opt_comp)
    d4["model"].Add(d4["mixed_expr"] == opt_mixed)
    hap1_nodes = sorted({v for v in d4["V"] if v[0] != -1 and v[2] == 1},
                         key=lambda v: (v[0], v[1], v[3]))
    bad = {}
    for node in hap1_nodes:
        terms = _node_mixed_terms(d4, node)
        if not terms:
            continue
        b = d4["model"].NewBoolVar(f"bad_{node}")
        d4["model"].AddMaxEquality(b, terms)
        bad[node] = b
    if bad:
        d4["model"].Add(sum(bad.values()) == opt_bad)
    return d4


def enumerate_subgraph_options(G, alt_adj_weights, cpu_n: int = 1) -> Tuple[List[Dict], bool]:
    """All distinct-signature solutions tied at the exact 3-stage optimum.

    Returns (solutions, exhaustive). exhaustive=True means the no-good-cut
    search genuinely reached INFEASIBLE (a real completeness proof, not a
    cap or a give-up); exhaustive=False means it stopped early (a cap was
    hit, or a solve stayed UNKNOWN after escalating retries) -- solutions is
    then only a lower bound on the true tied-option count.
    """
    if len(G.nodes()) == 0:
        return [{"components": [], "grey_pairs": [],
                  "objective_num_components": 0,
                  "objective_num_mixed_haplotype_edges": 0,
                  "objective_bad_nodes": 0}], True

    opt_comp, opt_mixed, opt_bad = exact_stage_values(G, alt_adj_weights, cpu_n=cpu_n)

    # The locked model (opt_comp/opt_mixed/opt_bad all fixed as equalities)
    # is EXACTLY the constraint set exact_stage_values already proved
    # feasible inside its own stage-3 solve -- so a first-attempt INFEASIBLE
    # here is just as structurally impossible as exact_stage_values'
    # opt_comp=0 case, and has been observed to happen just as rarely/
    # non-reproducibly. Rebuild-and-retry rather than trust it silently.
    #
    # A genuine UNKNOWN (solver hit the time limit without any verdict) is a
    # different situation from a wrongly-returned INFEASIBLE: it just means
    # this attempt's time budget was too tight for this thread count on a
    # large model, not that anything is broken -- so it gets a bigger time
    # budget next attempt rather than only a different seed.
    first_status = None
    time_limit = ENUM_SOLVE_TIME_LIMIT_S
    for build_attempt in range(5):
        d4 = _build_locked_model(G, alt_adj_weights, opt_comp, opt_mixed, opt_bad)
        n_workers = cpu_n if build_attempt == 0 else 1
        solver = _solver(time_limit, workers=n_workers, seed=build_attempt)
        first_status = solver.Solve(d4["model"])
        if first_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        if VERBOSE:
            print(f"    ... locked model impossibly returned "
                  f"{solver.StatusName(first_status)} on first solve "
                  f"(build attempt {build_attempt + 1}/5, workers={n_workers}, "
                  f"time_limit={time_limit:.0f}s); rebuilding", flush=True)
        if first_status == cp_model.UNKNOWN:
            time_limit *= 2
    else:
        raise RuntimeError(
            "enumerate_subgraph_options: locked model (opt_comp/opt_mixed/opt_bad fixed to "
            f"values exact_stage_values already proved feasible: {opt_comp}/{opt_mixed}/{opt_bad}) "
            "kept reporting infeasible/unknown after 5 rebuild attempts with growing time "
            "budgets -- not a transient fluke or a slow solve; needs investigation."
        )

    model, x, A = d4["model"], d4["x"], d4["A"]
    solutions: List[Dict] = []
    seen_sigs = set()
    exhaustive = True  # False if we ever give up on an UNKNOWN rather than
                        # proving genuine INFEASIBLE -- solutions is then a
                        # lower bound, not the true tied-option count.
    attempt_time_limit = ENUM_SOLVE_TIME_LIMIT_S

    for attempt in range(MAX_ATTEMPTS_PER_SUBGRAPH):
        if len(solutions) >= MAX_SOLUTIONS_PER_SUBGRAPH:
            exhaustive = False
            if VERBOSE:
                print(f"    ... stopping: hit MAX_SOLUTIONS_PER_SUBGRAPH={MAX_SOLUTIONS_PER_SUBGRAPH} "
                      f"(not necessarily exhaustive)", flush=True)
            break
        if attempt == 0:
            status = first_status
        else:
            # A genuine UNKNOWN here just means this particular no-good-cut
            # solve was harder than the current time budget -- give it more
            # time and retry rather than silently under-reporting the count
            # as if INFEASIBLE had been proven. Only after several escalating
            # retries still come back UNKNOWN do we give up (and say so).
            for retry in range(4):
                solver.parameters.max_deterministic_time = attempt_time_limit
                status = solver.Solve(model)
                if status != cp_model.UNKNOWN:
                    break
                attempt_time_limit *= 2
                if VERBOSE:
                    print(f"    ... attempt {attempt}: UNKNOWN at time_limit="
                          f"{attempt_time_limit/2:.0f}s, retrying at {attempt_time_limit:.0f}s "
                          f"(retry {retry + 1}/4)", flush=True)
        if VERBOSE and attempt % 10 == 0:
            print(f"    ... attempt {attempt}: found={len(solutions)} status={solver.StatusName(status)}",
                  flush=True)
        if status == cp_model.UNKNOWN:
            exhaustive = False
            if VERBOSE:
                print(f"    ... stopping: solver still UNKNOWN after escalating retries "
                      f"(found={len(solutions)}, NOT proven exhaustive)", flush=True)
            break
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        used_arcs = [a for a in A if solver.Value(x[a]) == 1]
        unused_arcs = [a for a in A if solver.Value(x[a]) == 0]

        components = ilp._extract_components(d4["V"], A, x, d4["s"], d4["w"], solver.Value)
        grey_pairs, seen_pair = [], set()
        for arc in A:
            if solver.Value(x[arc]) != 1 or d4["arc_color"][arc] != "G":
                continue
            u, v = arc[0], arc[1]
            canon = tuple(sorted((u, v)))
            if canon not in seen_pair:
                seen_pair.add(canon)
                grey_pairs.append((u, v))

        sig = _grey_pair_signature(grey_pairs)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            solutions.append({
                "components": components,
                "grey_pairs": grey_pairs,
                "objective_num_components": opt_comp,
                "objective_num_mixed_haplotype_edges": opt_mixed,
                "objective_bad_nodes": opt_bad,
            })

        model.Add(
            sum(x[a].Not() for a in used_arcs) + sum(x[a] for a in unused_arcs) >= 1
        )
    else:
        exhaustive = False
        if VERBOSE:
            print(f"    ... stopping: hit MAX_ATTEMPTS_PER_SUBGRAPH={MAX_ATTEMPTS_PER_SUBGRAPH} "
                  f"(not necessarily exhaustive)", flush=True)

    return solutions, exhaustive


def _enumerate_subgraph_options_task(args: Tuple) -> Tuple[List[Dict], bool]:
    """Picklable single-argument wrapper for process-pool use."""
    sg, alt_adj_weights, cpu_n = args
    return enumerate_subgraph_options(sg, alt_adj_weights, cpu_n=cpu_n)


def enumerate_sample_phasing_options(graph, reference, L: int = 2, cpu_n: int = 1, n_workers: int = 1):
    """
    Returns (phasing_options, meta) where phasing_options is a list of
    (label, ilp.PhasingResult), each a distinct tied-optimal joint phasing,
    and meta carries per-subgraph diagnostics.

    cpu_n:
        CP-SAT search workers used *within* each subgraph's solves.
    n_workers:
        Number of subgraphs enumerated in parallel (separate OS processes).
        Each weakly-connected subgraph's tied-option search is fully
        independent of the others -- this is typically the most expensive
        part of the whole pipeline for samples with many subgraphs, so it's
        the highest-value place to add process-level parallelism. cpu_n and
        n_workers multiply; keep their product within your core count.
    """
    subgraphs, alt_adj_weights = build_subgraphs(graph, reference, L=L)

    if n_workers > 1 and len(subgraphs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(sg, alt_adj_weights, cpu_n) for sg in subgraphs]
        with ProcessPoolExecutor(max_workers=min(n_workers, len(subgraphs))) as ex:
            raw_results = list(ex.map(_enumerate_subgraph_options_task, tasks))
    else:
        raw_results = [enumerate_subgraph_options(sg, alt_adj_weights, cpu_n=cpu_n)
                        for sg in subgraphs]

    per_subgraph_solutions = [r[0] for r in raw_results]
    per_subgraph_exhaustive = [r[1] for r in raw_results]

    meta = {
        "n_subgraphs": len(subgraphs),
        "per_subgraph_n_options": [len(s) for s in per_subgraph_solutions],
        "per_subgraph_stage_values": [
            (s[0]["objective_num_components"], s[0]["objective_num_mixed_haplotype_edges"],
             s[0]["objective_bad_nodes"]) if s else (0, 0, 0)
            for s in per_subgraph_solutions
        ],
        "per_subgraph_hit_solution_cap": [len(s) >= MAX_SOLUTIONS_PER_SUBGRAPH for s in per_subgraph_solutions],
        "per_subgraph_exhaustive": per_subgraph_exhaustive,
        "all_subgraphs_exhaustive": all(per_subgraph_exhaustive),
    }

    total_combos = 1
    for s in per_subgraph_solutions:
        total_combos *= max(1, len(s))
    meta["total_phasing_combos"] = total_combos
    meta["truncated"] = total_combos > MAX_TOTAL_PHASING_COMBOS

    index_ranges = [range(len(s)) if s else range(1) for s in per_subgraph_solutions]
    combos = itertools.product(*index_ranges)

    options = []
    for combo in itertools.islice(combos, MAX_TOTAL_PHASING_COMBOS):
        all_components, all_grey_pairs = [], []
        total_comp = total_mixed = total_bad = 0
        for si, j in enumerate(combo):
            sols = per_subgraph_solutions[si]
            sol = sols[j] if sols else {"components": [], "grey_pairs": [],
                                         "objective_num_components": 0,
                                         "objective_num_mixed_haplotype_edges": 0,
                                         "objective_bad_nodes": 0}
            all_components.extend(sol["components"])
            all_grey_pairs.extend(sol["grey_pairs"])
            total_comp += sol["objective_num_components"]
            total_mixed += sol["objective_num_mixed_haplotype_edges"]
            total_bad += sol["objective_bad_nodes"]
        label = "phasing_" + "_".join(str(j) for j in combo)
        phasing = ilp.PhasingResult(
            components=all_components,
            grey_pairs=all_grey_pairs,
            subgraph_results={},
            num_components=total_comp,
            num_mixed_haplotype_edges=total_mixed,
            status="OPTIMAL",
        )
        options.append((label, phasing))

    return options, meta


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
            return _CROSS_CHR_PENALTY + abs(pos_u - pos_v)
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


def enumerate_full_options(graph, reference, L: int = 2, cpu_n: int = 1, n_workers: int = 1):
    """
    Returns (full, meta).

    full: list of dicts {option_id, phasing_label, loose_label, phasing,
    extra_pairs} -- one per (tied phasing) x (tied loose pairing) combo,
    for every tied-optimal phasing found (no deduplication: phasings that
    look alike up to haplotype labels can still differ in the scenario
    objective, since the relative haplotypes of cycles decide which genome
    components a step affects).

    cpu_n / n_workers: see enumerate_sample_phasing_options.
    """
    phasing_options, meta = enumerate_sample_phasing_options(
        graph, reference, L=L, cpu_n=cpu_n, n_workers=n_workers,
    )
    meta["n_phasings"] = len(phasing_options)

    full = []
    for plabel, phasing in phasing_options:
        pairing_options = enumerate_loose_pairings(graph, phasing)
        for llabel, extra_pairs in pairing_options:
            full.append({
                "phasing_label": plabel,
                "loose_label": llabel,
                "phasing": phasing,
                "extra_pairs": extra_pairs,
            })
    for i, opt in enumerate(full):
        opt["option_id"] = f"option_{i + 1:03d}"
    meta["n_loose_pairing_variants"] = len({o["loose_label"] for o in full})
    meta["n_full_options"] = len(full)
    return full, meta
