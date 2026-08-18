"""
Extended Breakpoint Graph (EBG) builder.

Implements Definition 4 from the paper:

    Given Genome A (diploid reference) and Genome B (cancer, phased),
    build EBG(A, B) with:

    Nodes
    -----
    - INTERNAL (synteny) nodes: one (TAIL, HEAD) pair per interval × haplotype × copy.
    - TELOMERE nodes: chromosome ends.
    - DC nodes: one (T_DC, H_DC) pair per deleted copy
                (diploid_cn > cancer_cn for that interval).
    - AH nodes: one (T_AH, H_AH) pair per gained copy
                (cancer_cn > diploid_cn for that interval).

    Black edges (A-adjacencies absent from B)
    ------------------------------------------
    - REF adjacencies between INTERNAL nodes (normal chromosome structure).
    - {T_DC, H_DC} for each deleted interval (pre-deletion adjacency in A).
    - {T_AH, t_b+} and {h_b+, H_AH} for each gained copy (attachment in A+).

    Grey edges (B-adjacencies absent from A)
    -----------------------------------------
    - ALT adjacencies between INTERNAL nodes (somatic rearrangements in B).
    - {x, T_DC} and {y, H_DC} for each deletion (linking deleted extremities).
    - {T_AH, H_AH} for each gained copy (AH pair reconnects after insertion).
    - Insertion edges from phasing output (connecting gained copies into host).

    Shared adjacencies (in both A and B) are represented as BOTH black AND grey
    on the same node pair; cycles.remove_shared_adjacencies() strips them so
    that isolated vertices remain.

Entry point: build(graph, reference, phasing_result, L=2) -> List[EBGNode]
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from gr_cn.io.jabba import JabbaGraph
from gr_cn.io.reference import ReferenceGenome
from gr_cn.models.node import EBGNode, NodeType, End
from gr_cn.phasing.ilp import PhasingResult

# Node key type used as dict keys
NodeKey = Tuple  # (type_str, iid, side_str, haplotype, copy_index)


def _key(node: EBGNode) -> NodeKey:
    return (node.node_type.value, node.iid, node.end.value, node.haplotype, node.copy_index)


def build(
    graph: JabbaGraph,
    reference: ReferenceGenome,
    phasing: PhasingResult,
    L: int = 2,
) -> List[EBGNode]:
    """
    Build the Extended Breakpoint Graph nodes with black/grey neighbours set.

    Parameters
    ----------
    graph:
        Parsed JABBA graph (Genome B).
    reference:
        Reference genome (Genome A), from io.reference.build().
    phasing:
        Phasing result from phasing.ilp.phase().
    L:
        Number of haplotypes (default 2).

    Returns
    -------
    List of EBGNode with black_neighbor and grey_neighbor populated.
    Shared adjacencies are marked as both black and grey on the same pair;
    call ebg.cycles.remove_shared_adjacencies() before cycle extraction.
    """
    nodes: Dict[NodeKey, EBGNode] = {}

    # ------------------------------------------------------------------
    # 1.  Create INTERNAL (synteny) nodes
    # ------------------------------------------------------------------
    for iv in graph.intervals:
        cn_info = reference.get(iv.iid)
        cancer_cn = cn_info.cancer_cn if cn_info else L
        diploid_cn = cn_info.diploid_cn if cn_info else L

        # Baseline haplotype copies (hap 0..L-1, copy_index 0)
        for hap in range(L):
            for end in (End.TAIL, End.HEAD):
                pos = iv.start if end is End.TAIL else iv.end
                node = EBGNode.internal(
                    iid=iv.iid,
                    end=end,
                    haplotype=hap,
                    copy_index=0,
                    ref_chromosome=iv.chromosome,
                    ref_position=pos,
                    is_telomere=(iv.is_telomere_tail if end is End.TAIL else iv.is_telomere_head),
                    is_centromere_included=iv.is_centromere_included,
                )
                nodes[_key(node)] = node

        # Extra gained copies (copy_index > 0, haplotype 0 by convention)
        n_gained = max(0, cancer_cn - diploid_cn)
        for copy_i in range(1, n_gained + 1):
            for end in (End.TAIL, End.HEAD):
                pos = iv.start if end is End.TAIL else iv.end
                node = EBGNode.internal(
                    iid=iv.iid,
                    end=end,
                    haplotype=0,
                    copy_index=copy_i,
                    ref_chromosome=iv.chromosome,
                    ref_position=pos,
                    is_telomere=False,
                    is_centromere_included=iv.is_centromere_included,
                )
                nodes[_key(node)] = node

    # ------------------------------------------------------------------
    # 2.  Create DC nodes (one pair per deleted haplotype copy)
    # ------------------------------------------------------------------
    for info in reference.deleted_intervals:
        iv = info.interval
        for hap in range(info.n_deleted):
            for end in (End.TAIL, End.HEAD):
                dc = EBGNode.dc(iid=iv.iid, end=end, haplotype=hap, copy_index=0)
                nodes[_key(dc)] = dc

    # ------------------------------------------------------------------
    # 3.  Create AH nodes (one pair per gained copy)
    # ------------------------------------------------------------------
    for info in reference.gained_intervals:
        iv = info.interval
        for copy_i in range(1, info.n_gained + 1):
            for end in (End.TAIL, End.HEAD):
                ah = EBGNode.ah(iid=iv.iid, end=end, copy_index=copy_i)
                nodes[_key(ah)] = ah

    # ------------------------------------------------------------------
    # 4.  Convenience lookup helpers
    # ------------------------------------------------------------------

    def get(type_str: str, iid: int, side: str, hap: int, copy_i: int) -> Optional[EBGNode]:
        return nodes.get((type_str, iid, side, hap, copy_i))

    def synteny(iid: int, side: str, hap: int, copy_i: int = 0) -> Optional[EBGNode]:
        return get(NodeType.INTERNAL.value, iid, side, hap, copy_i)

    def dc_node(iid: int, side: str, hap: int) -> Optional[EBGNode]:
        return get(NodeType.DC.value, iid, side, hap, 0)

    def ah_node(iid: int, side: str, copy_i: int) -> Optional[EBGNode]:
        return get(NodeType.AH.value, iid, side, 0, copy_i)

    def set_black(a: Optional[EBGNode], b: Optional[EBGNode]) -> None:
        if a is not None and b is not None:
            a.black_neighbor = b
            b.black_neighbor = a

    def set_grey(a: Optional[EBGNode], b: Optional[EBGNode]) -> None:
        if a is not None and b is not None:
            a.grey_neighbor = b
            b.grey_neighbor = a

    # ------------------------------------------------------------------
    # 5.  Black edges from REF adjacencies (Genome A structure)
    #     These connect INTERNAL nodes that are adjacent in the normal genome.
    #     Deleted intervals still exist in A, so their black edges are kept.
    # ------------------------------------------------------------------
    for adj in reference.ref_adjacencies:
        if adj.sink_iid is None:
            continue
        src_iid, src_side = adj.source_abs, adj.source_side
        snk_iid, snk_side = adj.sink_abs, adj.sink_side

        for hap in range(L):
            set_black(
                synteny(src_iid, src_side, hap),
                synteny(snk_iid, snk_side, hap),
            )

    # ------------------------------------------------------------------
    # 6.  Black edges: DC pair internal adjacency {T_DC, H_DC}
    #     Present in A before deletion; absent from B.
    # ------------------------------------------------------------------
    for info in reference.deleted_intervals:
        iv = info.interval
        for hap in range(info.n_deleted):
            set_black(dc_node(iv.iid, "t", hap), dc_node(iv.iid, "h", hap))

    # ------------------------------------------------------------------
    # 7.  Black edges: AH attachment {T_AH, t_b+} and {h_b+, H_AH}
    #     Present in A+; absent from B (broken when copy is inserted).
    # ------------------------------------------------------------------
    for info in reference.gained_intervals:
        iv = info.interval
        for copy_i in range(1, info.n_gained + 1):
            set_black(ah_node(iv.iid, "t", copy_i), synteny(iv.iid, "t", 0, copy_i))
            set_black(ah_node(iv.iid, "h", copy_i), synteny(iv.iid, "h", 0, copy_i))

    # ------------------------------------------------------------------
    # 8.  Grey edges: phased ALT + cross-haplotype REF connections
    #     (Genome B's realized structure). The phasing result gives us
    #     paths/cycles over (iid, side, hap, copy) nodes; phasing.grey_pairs
    #     holds every used connection that needs an explicit grey edge here
    #     (see PhasingResult.grey_pairs docstring for why same-haplotype REF
    #     connections are excluded from that list).
    # ------------------------------------------------------------------
    # Build a lookup from phasing node tuple → EBGNode
    # PhaseNode = (iid, side, haplotype, copy_index); side ∈ {"t", "h"}
    phase_to_ebg: Dict[tuple, Optional[EBGNode]] = {}
    for node in nodes.values():
        if node.node_type is NodeType.INTERNAL:
            pk = (node.iid, node.end.value, node.haplotype, node.copy_index)
            phase_to_ebg[pk] = node

    for u_pk, v_pk in phasing.grey_pairs:
        set_grey(phase_to_ebg.get(u_pk), phase_to_ebg.get(v_pk))

    # ------------------------------------------------------------------
    # 9.  Grey edges for DC: {synteny_del, DC} pairs.
    #     Deleted haplotype copies use hap indices 0..n_deleted-1 (lowest first).
    #     Each deleted synteny node connects via grey to its DC counterpart.
    # ------------------------------------------------------------------
    for info in reference.deleted_intervals:
        iv = info.interval
        for del_hap in range(info.n_deleted):
            set_grey(synteny(iv.iid, "t", del_hap), dc_node(iv.iid, "t", del_hap))
            set_grey(synteny(iv.iid, "h", del_hap), dc_node(iv.iid, "h", del_hap))

    # ------------------------------------------------------------------
    # 10. Grey edges for AH: {T_AH, H_AH}
    #     After the gained copy is inserted, the AH pair reconnects.
    # ------------------------------------------------------------------
    for info in reference.gained_intervals:
        iv = info.interval
        for copy_i in range(1, info.n_gained + 1):
            set_grey(ah_node(iv.iid, "t", copy_i), ah_node(iv.iid, "h", copy_i))

    return list(nodes.values())
