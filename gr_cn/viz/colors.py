"""
Color assignment for genome/EBG visualizations.

Chromosome identity is encoded as hue (categorical), using the validated
8-slot palette below in its fixed, CVD-safe order (see the data-viz skill's
references/palette.md). Haplotype is a secondary channel encoded by marker
style, not by a second hue, so it never collides with the chromosome colors.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# Fixed categorical order -- do not reorder or cycle arbitrarily.
CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
DC_FILL = "#e1e0d9"
DC_EDGE = "#898781"


def chromosome_colors(chromosomes: Sequence[int]) -> Dict[int, str]:
    """Assign each chromosome (in the order given) a fixed-order categorical hue."""
    ordered = list(dict.fromkeys(chromosomes))  # stable de-dup, preserve order
    if len(ordered) > len(CATEGORICAL_LIGHT):
        raise ValueError(
            f"{len(ordered)} chromosomes requested but the validated palette "
            f"only carries {len(CATEGORICAL_LIGHT)} categorical slots; "
            "facet into multiple figures instead of adding hues."
        )
    return {chrom: CATEGORICAL_LIGHT[i] for i, chrom in enumerate(ordered)}
