from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

CHR_TO_INT: Dict[str, int] = {
    **{str(i): i for i in range(1, 23)},
    "X": 23,
    "Y": 24,
}
INT_TO_CHR: Dict[int, str] = {v: k for k, v in CHR_TO_INT.items()}


@dataclass
class Interval:
    """A single synteny block (genomic segment)."""
    iid: int
    chromosome: int        # integer (1-24; 23=X, 24=Y)
    start: int             # 1-based inclusive
    end: int               # 1-based inclusive
    cn: Optional[float] = None
    is_telomere_tail: bool = False    # leftmost interval on this chromosome
    is_telomere_head: bool = False    # rightmost interval on this chromosome
    is_centromere_included: bool = False

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)

    @property
    def chrom_str(self) -> str:
        return INT_TO_CHR.get(self.chromosome, str(self.chromosome))

    @classmethod
    def from_jabba_dict(cls, d: Dict[str, Any]) -> "Interval":
        raw_chr = str(d.get("chromosome", ""))
        chrom = CHR_TO_INT.get(raw_chr)
        if chrom is None:
            raise ValueError(f"Unknown chromosome: {raw_chr!r}")
        return cls(
            iid=int(d["iid"]),
            chromosome=chrom,
            start=int(d["startPoint"]),
            end=int(d["endPoint"]),
            cn=float(d["y"]) if d.get("y") is not None else None,
        )
