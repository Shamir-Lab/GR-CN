"""
Parse JABBA JSON graphs into Interval + Adjacency collections.

JABBA JSON structure:
  {
    "intervals": [ {"iid": ..., "chromosome": ..., "startPoint": ..., "endPoint": ..., "y": ...}, ... ],
    "connections": [ {"cid": ..., "type": "REF"|"ALT"|"LOOSE", "source": ..., "sink": ..., "weight": ...}, ... ]
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from gr_cn.models.interval import Interval
from gr_cn.models.adjacency import Adjacency, AdjacencyKind

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHR_ANNOTATIONS_PATH = DATA_DIR / "chr_arm_annotations_modified_xy_hg19.xlsx"

_CHR_ARM_ORDER = (
    [f"{i}p" for i in range(1, 23)] + [f"{i}q" for i in range(1, 23)] + ["23p", "23q", "24p", "24q"]
)


def _load_centromere_dict() -> Dict[int, tuple]:
    """Return {chromosome_int: (p_arm_end, q_arm_start)} from the reference XLSX."""
    df = pd.read_excel(CHR_ANNOTATIONS_PATH)
    borders = dict(zip(_CHR_ARM_ORDER, zip(df["Start"].tolist(), df["End"].tolist())))
    return {
        chrom: (borders[f"{chrom}p"][1], borders[f"{chrom}q"][0])
        for chrom in range(1, 25)
    }


_CENTROMERE_DICT: Optional[Dict[int, tuple]] = None


def _centromere_dict() -> Dict[int, tuple]:
    global _CENTROMERE_DICT
    if _CENTROMERE_DICT is None:
        _CENTROMERE_DICT = _load_centromere_dict()
    return _CENTROMERE_DICT


def _mark_telomeres(intervals: List[Interval]) -> None:
    """Set is_telomere_tail / is_telomere_head on each interval in-place."""
    chrom_min: Dict[int, int] = {}
    chrom_max: Dict[int, int] = {}
    for iv in intervals:
        chrom_min[iv.chromosome] = min(iv.start, chrom_min.get(iv.chromosome, iv.start))
        chrom_max[iv.chromosome] = max(iv.end, chrom_max.get(iv.chromosome, iv.end))
    for iv in intervals:
        iv.is_telomere_tail = iv.start <= chrom_min[iv.chromosome] <= iv.end
        iv.is_telomere_head = iv.start <= chrom_max[iv.chromosome] <= iv.end


def _mark_centromeres(intervals: List[Interval]) -> None:
    """Set is_centromere_included on each interval in-place."""
    cent = _centromere_dict()
    for iv in intervals:
        c_start, c_end = cent.get(iv.chromosome, (-1, -1))
        iv.is_centromere_included = iv.start <= c_start and c_end <= iv.end


@dataclass
class JabbaGraph:
    """Raw parsed JABBA graph for one sample."""
    sample_name: str
    intervals: List[Interval] = field(default_factory=list)
    adjacencies: List[Adjacency] = field(default_factory=list)

    # Fast lookups
    _by_iid: Dict[int, Interval] = field(default_factory=dict, repr=False)
    _by_chrom: Dict[int, List[Interval]] = field(default_factory=dict, repr=False)

    def _build_indices(self) -> None:
        self._by_iid = {iv.iid: iv for iv in self.intervals}
        self._by_chrom = {}
        for iv in self.intervals:
            self._by_chrom.setdefault(iv.chromosome, []).append(iv)
        for lst in self._by_chrom.values():
            lst.sort(key=lambda iv: iv.start)

    def interval(self, iid: int) -> Optional[Interval]:
        return self._by_iid.get(iid)

    def intervals_on(self, chromosome: int) -> List[Interval]:
        return self._by_chrom.get(chromosome, [])

    @property
    def ref_adjacencies(self) -> List[Adjacency]:
        return [a for a in self.adjacencies if a.kind is AdjacencyKind.REF]

    @property
    def alt_adjacencies(self) -> List[Adjacency]:
        return [a for a in self.adjacencies if a.kind is AdjacencyKind.ALT]

    @property
    def loose_adjacencies(self) -> List[Adjacency]:
        return [a for a in self.adjacencies if a.kind is AdjacencyKind.LOOSE]

    def aberrant_chromosomes(self) -> List[int]:
        """Chromosomes with >1 segment (i.e. rearranged)."""
        return [c for c, ivs in self._by_chrom.items() if len(ivs) > 1]


def load(path: str | Path, only_aberrant: bool = True, chromosomes=None) -> JabbaGraph:
    """
    Parse a JABBA JSON file and return a JabbaGraph.

    Parameters
    ----------
    path:
        Path to the JABBA JSON file.
    only_aberrant:
        If True (default), keep only intervals on chromosomes with >1 segment.
        Set to False to retain all chromosomes.
    chromosomes:
        If provided, keep only intervals on these chromosome numbers (e.g. [1, 5, 7, 22]).
        Applied after only_aberrant filtering.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    intervals = [Interval.from_jabba_dict(d) for d in raw.get("intervals", [])]
    adjacencies = [
        Adjacency.from_jabba_dict(d)
        for d in raw.get("connections", [])
        if d.get("type") in ("REF", "ALT", "LOOSE")
    ]

    graph = JabbaGraph(sample_name=path.stem, intervals=intervals, adjacencies=adjacencies)
    graph._build_indices()

    if only_aberrant:
        aberrant = set(graph.aberrant_chromosomes())
        graph.intervals = [iv for iv in intervals if iv.chromosome in aberrant]
        graph._build_indices()

    if chromosomes is not None:
        chr_set = set(chromosomes)
        graph.intervals = [iv for iv in graph.intervals if iv.chromosome in chr_set]
        valid_iids = {iv.iid for iv in graph.intervals}
        graph.adjacencies = [
            a for a in graph.adjacencies
            if a.source_abs in valid_iids
            and (a.sink_abs is None or a.sink_abs in valid_iids)
        ]
        graph._build_indices()

    _mark_telomeres(graph.intervals)
    _mark_centromeres(graph.intervals)

    return graph
