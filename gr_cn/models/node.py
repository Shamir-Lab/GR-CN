from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(Enum):
    INTERNAL = "synteny"
    TELOMERE = "telomere"
    DC = "DC"
    AH = "AH"


class End(Enum):
    TAIL = "t"
    HEAD = "h"

    @classmethod
    def from_str(cls, s: str) -> "End":
        s = s.strip().lower()
        if s in ("t", "tail"):
            return cls.TAIL
        if s in ("h", "head"):
            return cls.HEAD
        raise ValueError(f"Unknown end: {s!r}")


@dataclass
class EBGNode:
    """
    A node in the Extended Breakpoint Graph.

    Each synteny block copy contributes two nodes (TAIL and HEAD).
    Deleted copies get DC (deletion-collector) telomere pairs.
    Gained copies get AH (addition-holder) telomere pairs.
    Chromosome ends get TELOMERE nodes.

    black_neighbor / grey_neighbor are set by ebg.builder and represent
    A-adjacencies and B-adjacencies respectively (absent from the other genome).
    Shared adjacencies are not stored here (removed by cycles.remove_shared_adjacencies).
    """
    node_type: NodeType
    iid: int                          # interval id; -1 for pure artificial nodes
    end: End
    haplotype: int                    # 0 or 1
    copy_index: int                   # 0 for baseline copies; >0 for extra gained copies
    ref_chromosome: int               # -1 for artificial (DC/AH with no reference locus)
    ref_position: int                 # genomic coordinate; -1 for artificial
    is_telomere: bool = False         # True if this node is at a chromosome end
    has_loose: bool = False           # True if the original JABBA adjacency was LOOSE
    is_centromere_included: bool = False

    black_neighbor: Optional["EBGNode"] = field(default=None, repr=False)
    grey_neighbor: Optional["EBGNode"] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def internal(
        cls,
        iid: int,
        end: End | str,
        haplotype: int,
        copy_index: int,
        ref_chromosome: int,
        ref_position: int,
        is_telomere: bool = False,
        has_loose: bool = False,
        is_centromere_included: bool = False,
    ) -> "EBGNode":
        return cls(
            node_type=NodeType.INTERNAL,
            iid=iid,
            end=End.from_str(end) if isinstance(end, str) else end,
            haplotype=haplotype,
            copy_index=copy_index,
            ref_chromosome=ref_chromosome,
            ref_position=ref_position,
            is_telomere=is_telomere,
            has_loose=has_loose,
            is_centromere_included=is_centromere_included,
        )

    @classmethod
    def dc(cls, iid: int, end: End | str, haplotype: int, copy_index: int) -> "EBGNode":
        return cls(
            node_type=NodeType.DC,
            iid=iid,
            end=End.from_str(end) if isinstance(end, str) else end,
            haplotype=haplotype,
            copy_index=copy_index,
            ref_chromosome=-1,
            ref_position=-1,
        )

    @classmethod
    def ah(cls, iid: int, end: End | str, copy_index: int) -> "EBGNode":
        return cls(
            node_type=NodeType.AH,
            iid=iid,
            end=End.from_str(end) if isinstance(end, str) else end,
            haplotype=0,
            copy_index=copy_index,
            ref_chromosome=-1,
            ref_position=-1,
        )

    # ------------------------------------------------------------------
    # Accessors used by dynamic_scenario_enumeration
    # ------------------------------------------------------------------

    @property
    def copy(self) -> int:
        return self.copy_index

    def is_real_synteny(self) -> bool:
        """True for INTERNAL nodes with a real reference locus (not artificial DC/AH copies)."""
        return self.node_type is NodeType.INTERNAL and self.ref_chromosome != -1

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def key(self) -> tuple:
        return (self.node_type.value, self.iid, self.end.value, self.haplotype, self.copy_index)

    def __repr__(self) -> str:
        return f"{self.node_type.value}_{self.iid}_{self.end.value}_{self.haplotype}_{self.copy_index}"

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other
