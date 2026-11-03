"""
Loose-end helpers used by gr_cn.phasing.enumerate.enumerate_loose_pairings.

A JABBA LOOSE adjacency marks one known endpoint whose real partner is
unknown. Telomeric loose ends are already free path endpoints and need no
pairing; genuine (non-telomeric) loose ends are left open by the phasing ILP
and paired afterwards. Genomic distance between two ends is the absolute
difference in reference position, with cross-chromosome pairs penalised by a
large constant.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Set, Tuple

from gr_cn.io.jabba import JabbaGraph

if TYPE_CHECKING:
    from gr_cn.phasing.ilp import PhaseNode

_CROSS_CHR_PENALTY = 10_000_000_000  # larger than any real genomic distance


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
