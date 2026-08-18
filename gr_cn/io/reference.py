"""
Reconstruct Genome A (diploid reference) from a parsed JABBA graph (Genome B).

Logic:
- REF adjacencies are shared by A and B; they define the normal chromosome
  adjacency structure that we can infer from B.
- A segment with CN=0 in B still exists in A with CN=diploid_cn.
- A segment with CN > diploid_cn in B has extra copies in A+ (as AH chromosomes).
- Diploid CN per chromosome is estimated from the arm-level weighted-mean CN
  of the JABBA graph (same logic as calculate_arm_cn in old preprocessing).

The returned ReferenceGenome carries per-interval diploid copy numbers and
flags which intervals are deleted (CN_B=0) or gained (CN_B > diploid) in B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gr_cn.models.interval import Interval
from gr_cn.models.adjacency import Adjacency, AdjacencyKind
from gr_cn.io.jabba import JabbaGraph, _centromere_dict, _CHR_ARM_ORDER

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHR_ANNOTATIONS_PATH = DATA_DIR / "chr_arm_annotations_modified_xy_hg19.xlsx"


def _arm_cn(graph: JabbaGraph) -> Dict[str, int]:
    """
    Estimate the baseline (diploid) copy number per chromosome arm from the
    weighted-mean CN of intervals overlapping that arm, rounded to nearest int.
    Mirrors calculate_arm_cn() from the old preprocessing code.
    """
    df = pd.read_excel(CHR_ANNOTATIONS_PATH)
    borders = dict(zip(_CHR_ARM_ORDER, zip(df["Start"].tolist(), df["End"].tolist())))

    arm_cn: Dict[str, int] = {}
    for arm in _CHR_ARM_ORDER:
        chrom = int(arm[:-1])
        arm_start, arm_end = borders[arm]
        ivs = [
            iv for iv in graph.intervals_on(chrom)
            if iv.start < arm_end and iv.end > arm_start and iv.cn is not None
        ]
        if not ivs:
            continue
        total_len = sum(iv.length for iv in ivs)
        if total_len == 0:
            continue
        weighted = sum(iv.cn * iv.length for iv in ivs) / total_len
        arm_cn[arm] = int(round(weighted))
    return arm_cn


def _diploid_cn_per_chrom(arm_cn: Dict[str, int]) -> Dict[int, int]:
    """
    Return the baseline ploidy per chromosome as max(p_arm_cn, q_arm_cn).
    Falls back to 2 if an arm is missing.
    """
    result: Dict[int, int] = {}
    for chrom in range(1, 25):
        p = arm_cn.get(f"{chrom}p", 2)
        q = arm_cn.get(f"{chrom}q", 2)
        result[chrom] = max(p, q)
    return result


@dataclass
class IntervalCNInfo:
    """Copy-number bookkeeping for one interval relative to A vs B."""
    interval: Interval
    diploid_cn: int       # baseline copies in A
    cancer_cn: int        # copies in B (rounded from JABBA float)
    n_deleted: int        # copies present in A but lost in B  (diploid_cn - cancer_cn if > 0)
    n_gained: int         # extra copies in B beyond diploid    (cancer_cn - diploid_cn if > 0)


@dataclass
class ReferenceGenome:
    """
    Genome A: the diploid reference inferred from a JABBA graph (Genome B).

    Attributes
    ----------
    cn_info:
        Per-interval copy-number comparison between A and B.
    ref_adjacencies:
        REF adjacencies from JABBA — the normal adjacency structure shared by A and B.
    diploid_cn_per_chrom:
        Baseline ploidy per chromosome (integer, typically 2).
    """
    cn_info: List[IntervalCNInfo] = field(default_factory=list)
    ref_adjacencies: List[Adjacency] = field(default_factory=list)
    diploid_cn_per_chrom: Dict[int, int] = field(default_factory=dict)

    _by_iid: Dict[int, IntervalCNInfo] = field(default_factory=dict, repr=False)

    def _build_index(self) -> None:
        self._by_iid = {info.interval.iid: info for info in self.cn_info}

    def get(self, iid: int) -> Optional[IntervalCNInfo]:
        return self._by_iid.get(iid)

    @property
    def deleted_intervals(self) -> List[IntervalCNInfo]:
        """Intervals that are fully or partially lost in B (need DC nodes)."""
        return [info for info in self.cn_info if info.n_deleted > 0]

    @property
    def gained_intervals(self) -> List[IntervalCNInfo]:
        """Intervals with extra copies in B (need AH nodes)."""
        return [info for info in self.cn_info if info.n_gained > 0]


def build(graph: JabbaGraph) -> ReferenceGenome:
    """
    Build Genome A from a parsed JABBA graph (Genome B).

    Parameters
    ----------
    graph:
        The parsed JABBA graph representing cancer Genome B.

    Returns
    -------
    ReferenceGenome
        Genome A with per-interval CN comparison and REF adjacency structure.
    """
    arm_cn = _arm_cn(graph)
    diploid_per_chrom = _diploid_cn_per_chrom(arm_cn)

    cn_info: List[IntervalCNInfo] = []
    for iv in graph.intervals:
        diploid = diploid_per_chrom.get(iv.chromosome, 2)
        cancer = int(round(iv.cn)) if iv.cn is not None else diploid
        n_deleted = max(0, diploid - cancer)
        n_gained = max(0, cancer - diploid)
        cn_info.append(IntervalCNInfo(
            interval=iv,
            diploid_cn=diploid,
            cancer_cn=cancer,
            n_deleted=n_deleted,
            n_gained=n_gained,
        ))

    ref = ReferenceGenome(
        cn_info=cn_info,
        ref_adjacencies=graph.ref_adjacencies,
        diploid_cn_per_chrom=diploid_per_chrom,
    )
    ref._build_index()
    return ref
