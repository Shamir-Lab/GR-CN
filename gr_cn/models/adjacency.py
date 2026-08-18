from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class AdjacencyKind(Enum):
    REF = "REF"      # reference (normal) adjacency — present in both A and B
    ALT = "ALT"      # somatic rearrangement — present in B only
    LOOSE = "LOOSE"  # unresolved end — partner unknown


@dataclass
class Adjacency:
    """
    A connection between two interval extremities in the JABBA graph.

    source_iid / sink_iid are signed: positive → head side, negative → tail side.
    LOOSE adjacencies have sink_iid = None (one-ended).
    """
    cid: int
    source_iid: int
    sink_iid: Optional[int]   # None for LOOSE
    kind: AdjacencyKind
    weight: Optional[float] = None

    @property
    def source_side(self) -> str:
        return "h" if self.source_iid > 0 else "t"

    @property
    def sink_side(self) -> Optional[str]:
        if self.sink_iid is None:
            return None
        return "h" if self.sink_iid > 0 else "t"

    @property
    def source_abs(self) -> int:
        return abs(self.source_iid)

    @property
    def sink_abs(self) -> Optional[int]:
        return abs(self.sink_iid) if self.sink_iid is not None else None

    @classmethod
    def from_jabba_dict(cls, d: Dict[str, Any]) -> "Adjacency":
        raw_kind = d.get("type") or d.get("kind", "")
        try:
            kind = AdjacencyKind(raw_kind.upper())
        except ValueError:
            kind = AdjacencyKind.ALT  # treat unknown as somatic

        source = d.get("source") or d.get("from") or d.get("s")
        sink = d.get("sink") or d.get("target") or d.get("to") or d.get("t")
        cid = d.get("cid") or d.get("id", 0)

        return cls(
            cid=int(cid),
            source_iid=int(source),
            sink_iid=int(sink) if sink not in (None, "NA", "") and kind is not AdjacencyKind.LOOSE else None,
            kind=kind,
            weight=float(d["weight"]) if d.get("weight") is not None else None,
        )
