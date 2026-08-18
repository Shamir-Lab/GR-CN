"""
Exact enumeration of every tied-optimal haplotype-labeled phasing option.

Optimality is defined by a strict lexicographic hierarchy, solved jointly
(not greedily) at each stage with CP-SAT:

  stage 1: minimize number of phased connected components   (existing ILP obj)
  stage 2: minimize total cross-haplotype ("mixed") edges    (existing ILP obj)
  stage 3: minimize the COUNT of hap1 nodes that touch any mixed edge
           (the hap0-priority rule, solved exactly -- see
           analysis/phasing_tiebreak/joint_minimize_unresolved.py for the
           verification that this must be solved jointly, not greedily:
           greedy per-node locking in iid order was shown to leave 852 with
           22 "unresolved" nodes when the true joint minimum is only 6)

Every subgraph solution tied at (opt_comp, opt_mixed, opt_bad) is enumerated
via no-good cuts, deduplicated by its grey_pairs signature. Subgraphs combine
by Cartesian product (bounded) and each combined phasing is then crossed with
every tied-minimal loose-end pairing (gr_cn.phasing.loose logic).
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from ortools.sat.python import cp_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEGEN_DIR = REPO_ROOT / "analysis" / "phasing_degeneracy"
for p in (REPO_ROOT, DEGEN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gr_cn.phasing import ilp, loose
from degeneracy_experiment import build_subgraphs, _grey_pair_signature, enumerate_loose_pairings  # noqa: E402

SOLVE_TIME_LIMIT_S = 120.0
ENUM_SOLVE_TIME_LIMIT_S = 15.0
# Ties at the exact 3-stage optimum can genuinely number in the dozens+ for
# larger subgraphs (see analysis/phasing_tiebreak/ -- 852's subgraph alone
# has 182 hap1 candidate nodes). Enumerating literally all of them is not
# tractable in report-generation time, so this is a reporting cap, not a
# claim of exhaustiveness -- meta["truncated"] records when it was hit.
MAX_SOLUTIONS_PER_SUBGRAPH = 128
MAX_ATTEMPTS_PER_SUBGRAPH = 1500
MAX_TOTAL_PHASING_COMBOS = 128
VERBOSE = True


def _solver(time_limit: float = SOLVE_TIME_LIMIT_S, workers: int = 1) -> cp_model.CpSolver:
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    s.parameters.random_seed = 0
    return s


def _solve_or_die(solver: cp_model.CpSolver, model, label: str) -> int:
    """Solve and return the objective value, raising loudly instead of
    silently reading ObjectiveValue() off a timed-out/infeasible solve
    (CP-SAT returns 0 for that, which is indistinguishable from a real
    answer of 0 -- this caused a real 852 bug: a timed-out stage-1 solve
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


def exact_stage_values(G, alt_adj_weights, _retries: int = 3) -> Tuple[int, int, int]:
    """Return (opt_comp, opt_mixed, opt_bad) -- the exact joint optimum.

    Uses multiple search workers here (unlike the enumeration loop, which
    stays single-threaded for reproducible no-good-cut ordering) because
    these three solves just need one fast, certifiably-optimal answer.

    CP-SAT with a large worker count on a large model (e.g. 852's 352-node
    subgraph) has been observed -- rarely, non-reproducibly, with no
    associated exception or bad status -- to report OPTIMAL with opt_comp=0
    for a non-empty graph, which is structurally impossible (you cannot
    cover >0 nodes with 0 components). Retrying with a fresh model/solver
    has always recovered the correct answer, so guard against this rather
    than trust a single solve blindly.
    """
    n_nodes = len(G.nodes())
    for attempt in range(_retries):
        n_workers = min(8, max(1, __import__("os").cpu_count() or 1)) if attempt == 0 else 1

        d1 = ilp._build_model(G)
        ilp._add_alt_usage_constraints(d1, alt_adj_weights)
        d1["model"].Minimize(d1["num_comp_expr"])
        s1 = _solver(SOLVE_TIME_LIMIT_S, workers=n_workers)
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


def enumerate_subgraph_options(G, alt_adj_weights) -> List[Dict]:
    """All distinct-signature solutions tied at the exact 3-stage optimum."""
    if len(G.nodes()) == 0:
        return [{"components": [], "grey_pairs": [],
                  "objective_num_components": 0,
                  "objective_num_mixed_haplotype_edges": 0,
                  "objective_bad_nodes": 0}]

    opt_comp, opt_mixed, opt_bad = exact_stage_values(G, alt_adj_weights)

    # The locked model (opt_comp/opt_mixed/opt_bad all fixed as equalities)
    # is EXACTLY the constraint set exact_stage_values already proved
    # feasible inside its own stage-3 solve -- so a first-attempt INFEASIBLE
    # here is just as structurally impossible as exact_stage_values'
    # opt_comp=0 case, and has been observed to happen just as rarely/
    # non-reproducibly. Rebuild-and-retry rather than trust it silently.
    first_status = None
    for build_attempt in range(3):
        d4 = _build_locked_model(G, alt_adj_weights, opt_comp, opt_mixed, opt_bad)
        n_workers = min(4, max(1, __import__("os").cpu_count() or 1)) if build_attempt == 0 else 1
        solver = _solver(ENUM_SOLVE_TIME_LIMIT_S, workers=n_workers)
        first_status = solver.Solve(d4["model"])
        if first_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        if VERBOSE:
            print(f"    ... locked model impossibly returned "
                  f"{solver.StatusName(first_status)} on first solve "
                  f"(build attempt {build_attempt + 1}/3, workers={n_workers}); rebuilding", flush=True)
    else:
        raise RuntimeError(
            "enumerate_subgraph_options: locked model (opt_comp/opt_mixed/opt_bad fixed to "
            f"values exact_stage_values already proved feasible: {opt_comp}/{opt_mixed}/{opt_bad}) "
            "kept reporting infeasible after 3 rebuild attempts -- not a transient fluke."
        )

    model, x, A = d4["model"], d4["x"], d4["A"]
    solutions: List[Dict] = []
    seen_sigs = set()

    for attempt in range(MAX_ATTEMPTS_PER_SUBGRAPH):
        if len(solutions) >= MAX_SOLUTIONS_PER_SUBGRAPH:
            break
        status = first_status if attempt == 0 else solver.Solve(model)
        if VERBOSE and attempt % 10 == 0:
            print(f"    ... attempt {attempt}: found={len(solutions)} status={solver.StatusName(status)}",
                  flush=True)
        if status == cp_model.UNKNOWN:
            # Time limit hit without proving optimal/infeasible -- treat the
            # search as exhausted rather than looping until MAX_ATTEMPTS.
            if VERBOSE:
                print(f"    ... stopping: solver hit time limit without a verdict "
                      f"(found={len(solutions)})", flush=True)
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

    return solutions


def enumerate_sample_phasing_options(graph, reference, L: int = 2):
    """
    Returns (phasing_options, meta) where phasing_options is a list of
    (label, ilp.PhasingResult), each a distinct tied-optimal joint phasing,
    and meta carries per-subgraph diagnostics.
    """
    subgraphs, alt_adj_weights = build_subgraphs(graph, reference, L=L)
    per_subgraph_solutions = [enumerate_subgraph_options(sg, alt_adj_weights) for sg in subgraphs]

    meta = {
        "n_subgraphs": len(subgraphs),
        "per_subgraph_n_options": [len(s) for s in per_subgraph_solutions],
        "per_subgraph_stage_values": [
            (s[0]["objective_num_components"], s[0]["objective_num_mixed_haplotype_edges"],
             s[0]["objective_bad_nodes"]) if s else (0, 0, 0)
            for s in per_subgraph_solutions
        ],
        "per_subgraph_hit_solution_cap": [len(s) >= MAX_SOLUTIONS_PER_SUBGRAPH for s in per_subgraph_solutions],
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


def canonical_topology_signature(graph, reference, phasing, L: int = 2) -> tuple:
    """
    Haplotype-label-independent fingerprint of the downstream EBG cycle
    structure a phasing produces.

    Two tied-optimal phasings can differ at the raw arc level yet be
    identical up to a pure relabeling that carries no biological meaning:
      - swapping which physical strand is CALLED haplotype 0 vs 1 for any
        rearrangement region untouched by a deletion/gain (nothing in the
        ILP objective or the hap0-priority tiebreak distinguishes the two
        labels there -- see analysis/phasing_tiebreak/ for the empirical
        trace that first surfaced this),
      - swapping which extra gained copy is CALLED copy_index 1 vs 2 for a
        gained interval (both sit on haplotype 0 and are physically
        indistinguishable pieces of the same parental strand).
    Grouping candidate phasings by this signature (ebg.cycles._cycle_topology
    per cycle, the same haplotype-agnostic key merge_partner_cycles already
    uses to detect homozygous partner cycles) collapses exactly those
    label-only differences while keeping genuinely different downstream
    event structures apart -- this is what PM1284's "40+ tied phasings"
    inflation turned out to be.
    """
    from gr_cn.ebg import builder, cycles as ebg_cycles

    ebg_nodes = builder.build(graph, reference, phasing, L=L)
    ebg_cycles.remove_shared_adjacencies(ebg_nodes)
    raw_cycles = ebg_cycles.extract_cycles(ebg_nodes)
    per_cycle = sorted(tuple(sorted(ebg_cycles._cycle_topology(cyc))) for cyc in raw_cycles)
    return tuple(per_cycle)


def dedupe_phasing_options_by_topology(graph, reference, options, L: int = 2):
    """
    Groups (label, PhasingResult) options by canonical_topology_signature.
    Returns (representatives, equivalence_classes) where representatives is
    one (label, phasing) per distinct downstream structure (first one found
    in each group), and equivalence_classes maps representative label ->
    list of ALL raw labels (including itself) that collapsed into it.
    """
    groups: Dict[tuple, list] = {}
    for label, phasing in options:
        sig = canonical_topology_signature(graph, reference, phasing, L=L)
        groups.setdefault(sig, []).append((label, phasing))

    representatives = []
    equivalence_classes = {}
    for members in groups.values():
        rep_label, rep_phasing = members[0]
        representatives.append((rep_label, rep_phasing))
        equivalence_classes[rep_label] = [lbl for lbl, _ in members]

    return representatives, equivalence_classes


def enumerate_full_options(graph, reference, L: int = 2, dedupe: bool = True):
    """
    Returns (full, meta, equivalence_classes).

    full: list of dicts {option_id, phasing_label, loose_label, phasing,
    extra_pairs} -- one per (tied phasing) x (tied loose pairing) combo.

    dedupe=True (default): phasings that collapse to the same downstream EBG
    cycle structure under canonical_topology_signature (pure haplotype-label
    or gained-copy-label relabeling, see dedupe_phasing_options_by_topology)
    are deduplicated to a single representative BEFORE crossing with
    loose-end pairings.
    dedupe=False: every raw arc-level tied solution found gets its own
    option/folder, undeduplicated -- including near-duplicates that are the
    same event with an arbitrary haplotype/copy relabeling (see the
    conversation for concrete examples of what that looks like).

    equivalence_classes: representative phasing_label -> list of ALL raw
    phasing labels (arc-level distinct, pre-dedup) that collapsed into it.
    When dedupe=False this is the trivial 1:1 mapping (kept for schema
    consistency, not because any collapsing happened).
    """
    raw_phasing_options, meta = enumerate_sample_phasing_options(graph, reference, L=L)
    meta["n_raw_phasing_options"] = len(raw_phasing_options)

    if dedupe:
        phasing_options, equivalence_classes = dedupe_phasing_options_by_topology(
            graph, reference, raw_phasing_options, L=L,
        )
    else:
        phasing_options = raw_phasing_options
        equivalence_classes = {lbl: [lbl] for lbl, _ in raw_phasing_options}
    meta["n_canonical_phasing_options"] = len(phasing_options)
    meta["dedupe"] = dedupe

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
    return full, meta, equivalence_classes
