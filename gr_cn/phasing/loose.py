"""
Post-phasing loose-end resolver.

A JABBA LOOSE adjacency marks one known endpoint (source_iid/source_side)
whose real partner is unknown. Telomeric loose ends (chromosome starts/ends)
are already free path endpoints in the EBG and need no pairing.

Genuine (non-telomeric) loose ends are intentionally left unconnected going
into the phasing ILP (gr_cn.phasing.ilp.phase never adds an edge for a LOOSE
adjacency) so the solver produces its normal minimum-component phasing with
these positions as ordinary open path endpoints — exactly like the archived
implementation, which recorded them as a "LOOSE" boundary marker rather than
guessing a connection ahead of time.

This module runs *after* phasing: it looks at which haplotype copy of each
loose (iid, side) the solver actually left open (a path/singleton endpoint),
and pairs those specific copies up:

Rules
-----
n_open == 0:  nothing to do.
n_open == 2:  connect the two open ends directly (only one possibility).
n_open == 4:  try all 3 pairings, pick the one that minimises total genomic
              distance between paired breakpoints.  Genomic distance is the
              absolute difference in reference position; cross-chromosome
              pairs are penalised with a large constant.
n_open  > 4:  raise TooManyLooseEnds.

Resolving after phasing (rather than before) means we connect the exact
haplotype copy the ILP left dangling, instead of guessing one in advance —
avoiding any chance of colliding with a haplotype the solver already used
for something else.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple

from gr_cn.io.jabba import JabbaGraph

if TYPE_CHECKING:
    from gr_cn.phasing.ilp import PhaseNode, PhasingResult

_CROSS_CHR_PENALTY = 10_000_000_000  # larger than any real genomic distance


class TooManyLooseEnds(ValueError):
    """Raised when the sample has more open loose ends than the heuristic can handle."""


def _non_telomeric_loose_sides(graph: JabbaGraph) -> Set[Tuple[int, str]]:
    """(iid, side) pairs with a genuine LOOSE adjacency (telomeres excluded)."""
    sides = set()
    for adj in graph.loose_adjacencies:
        iv = graph.interval(adj.source_abs)
        if iv is None:
            continue
        if adj.source_side == "t" and iv.is_telomere_tail:
            continue
        if adj.source_side == "h" and iv.is_telomere_head:
            continue
        sides.add((adj.source_abs, adj.source_side))
    return sides


def _position(graph: JabbaGraph, node: "PhaseNode") -> Tuple[int, int]:
    iid, side, _hap, _copy = node
    iv = graph.interval(iid)
    chrom = iv.chromosome if iv else -1
    pos = (iv.start if side == "t" else iv.end) if iv else 0
    return chrom, pos


def resolve_post_phasing(
    graph: JabbaGraph,
    phasing: "PhasingResult",
) -> List[Tuple["PhaseNode", "PhaseNode"]]:
    """
    Find the phased path/singleton endpoints that sit on a genuine loose
    breakpoint and pair them up.

    Returns a list of (PhaseNode, PhaseNode) pairs to be added to
    phasing.grey_pairs before building the EBG.

    Raises
    ------
    TooManyLooseEnds
        If more than 4 open loose ends remain after phasing.
    """
    loose_sides = _non_telomeric_loose_sides(graph)
    if not loose_sides:
        return []

    open_ends: List["PhaseNode"] = []
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
        return []

    if n % 2 != 0:
        raise TooManyLooseEnds(
            f"Odd number of open loose path-ends ({n}) after phasing; cannot pair them."
        )

    if n > 4:
        raise TooManyLooseEnds(
            f"{n} open loose path-ends remain after phasing; heuristic only handles <= 4."
        )

    if n == 2:
        return [(open_ends[0], open_ends[1])]

    # n == 4: try all 3 perfect matchings, pick minimum total distance
    def pair_distance(a: "PhaseNode", b: "PhaseNode") -> int:
        chrom_a, pos_a = _position(graph, a)
        chrom_b, pos_b = _position(graph, b)
        if chrom_a != chrom_b:
            return _CROSS_CHR_PENALTY + abs(pos_a - pos_b)
        return abs(pos_a - pos_b)

    def total_distance(pairing: List[Tuple["PhaseNode", "PhaseNode"]]) -> int:
        return sum(pair_distance(a, b) for a, b in pairing)

    a, b, c, d = open_ends
    candidates = [
        [(a, b), (c, d)],
        [(a, c), (b, d)],
        [(a, d), (b, c)],
    ]
    return min(candidates, key=total_distance)
