"""
Matplotlib rendering of one VizStep + its StepLayout onto an Axes.
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.path import Path

from gr_cn.models.node import EBGNode, NodeType
from gr_cn.viz import colors as pal
from gr_cn.viz.layout import StepLayout, edge_curve_offset
from gr_cn.viz.model import VizStep

NODE_R = 0.16
FONT_NODE = 6.5
FONT_CAP = 7.5
FONT_TAG = 7.5


def _node_label(node: EBGNode) -> str:
    side = node.end.value
    if node.node_type is NodeType.DC:
        return f"{side.upper()}_DC"
    if node.node_type is NodeType.AH:
        return f"{side.upper()}_AH"
    suffix = f".{node.copy_index}" if node.copy_index else ""
    return f"{side}{node.iid}{suffix}"


def _node_facecolor(node: EBGNode, chrom_colors: Dict[int, str]) -> str:
    if node.node_type in (NodeType.DC, NodeType.AH):
        return pal.DC_FILL
    return chrom_colors.get(node.ref_chromosome, pal.INK_MUTED)


def _draw_node(ax: Axes, node: EBGNode, xy, chrom_colors: Dict[int, str]) -> None:
    x, y = xy
    face = _node_facecolor(node, chrom_colors)
    is_satellite = node.node_type in (NodeType.DC, NodeType.AH)
    lw = 2.0 if node.haplotype else 1.1
    if is_satellite:
        w, h = 0.62, 0.30
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=face, edgecolor=pal.DC_EDGE, linewidth=1.0, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, _node_label(node), ha="center", va="center",
                 fontsize=FONT_NODE, color=pal.INK_SECONDARY, zorder=4)
        return

    circ = mpatches.Circle((x, y), NODE_R, facecolor=face, edgecolor=pal.INK_PRIMARY,
                            linewidth=lw, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, _node_label(node), ha="center", va="center",
             fontsize=FONT_NODE, color="white", fontweight="bold", zorder=4)


def _draw_cap(ax: Axes, node: EBGNode, xy, direction: float, chrom_colors: Dict[int, str]) -> None:
    x, y = xy
    cy = y + direction * 0.42
    face = _node_facecolor(node, chrom_colors)
    s = 0.20
    rect = mpatches.Rectangle((x - s / 2, cy - s / 2), s, s,
                               facecolor=face, edgecolor=pal.INK_PRIMARY, linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    ax.text(x, cy, node.end.value.upper(), ha="center", va="center",
             fontsize=FONT_CAP, color="white", fontweight="bold", zorder=4)
    ax.plot([x, x], [y + direction * NODE_R, cy - direction * s / 2],
            color=pal.INK_PRIMARY, linewidth=1.1, zorder=2)


def _edge_style(kind: str):
    """
    Genome-provenance encoding, reusing the existing ink tokens only (no new
    hue -- chromosome identity already owns hue, per gr_cn.viz.colors):
      base / shared    -- true in both genomes: INK_PRIMARY, solid.
      black            -- genome-A-only: INK_SECONDARY, solid.
      grey-resolved    -- genome-B-only, resolved: INK_MUTED, solid, heavier.
      grey-pending:*   -- genome-B-only, not yet resolved: INK_MUTED, dashed.
    """
    base_kind = kind.split(":", 1)[0]
    if base_kind == "grey-pending":
        return dict(color=pal.INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    if base_kind == "grey-resolved":
        return dict(color=pal.INK_MUTED, linewidth=1.6, linestyle="solid")
    if base_kind == "black":
        return dict(color=pal.INK_SECONDARY, linewidth=1.1, linestyle="solid")
    return dict(color=pal.INK_PRIMARY, linewidth=1.1, linestyle="solid")


def _curve_midpoint(x0: float, y0: float, x1: float, y1: float, offset: float) -> Tuple[float, float]:
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if not offset:
        return mx, my
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return mx, my
    px, py = -dy / length, dx / length
    return mx + px * offset, my + py * offset


def _draw_edge(ax: Axes, x0: float, y0: float, x1: float, y1: float, offset: float, style: dict) -> None:
    if not offset:
        ax.plot([x0, x1], [y0, y1], zorder=1, **style)
        return
    cx, cy = _curve_midpoint(x0, y0, x1, y1, offset)
    path = Path([(x0, y0), (cx, cy), (x1, y1)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(mpatches.PathPatch(
        path, facecolor="none", edgecolor=style["color"],
        linewidth=style["linewidth"], linestyle=style["linestyle"], zorder=1,
    ))


def draw_step(
    ax: Axes,
    step: VizStep,
    layout: StepLayout,
    chrom_colors: Dict[int, str],
    title: str = "",
) -> None:
    ax.set_title(title, fontsize=9, color=pal.INK_PRIMARY, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")

    node_pos = layout.node_pos

    non_base_connected: Set[int] = {
        id(u) for u, v, kind in step.active_edges if kind != "base"
    } | {
        id(v) for u, v, kind in step.active_edges if kind != "base"
    }
    satellite_anchor_ids: Set[int] = {id(a) for sat in layout.satellites for a in sat.anchors}

    # Active (real) edges: base + black + grey-resolved, drawn solid.
    # Whether a given "base" edge (an interval's own tail-head pairing) is
    # present at all is decided upstream, per step, by build_viz_steps /
    # genomes.full_genome_steps -- by the time it reaches here it is real
    # structure for this step and should be drawn.
    for u, v, kind in step.active_edges:
        if id(u) not in node_pos or id(v) not in node_pos:
            continue
        style = _edge_style(kind)
        (x0, y0), (x1, y1) = node_pos[id(u)], node_pos[id(v)]
        _draw_edge(ax, x0, y0, x1, y1, edge_curve_offset(layout, u, v), style)

    # Pending grey preview edges: dashed, labelled with the owning cycle tag.
    for u, v, kind in step.pending_edges:
        if id(u) not in node_pos or id(v) not in node_pos:
            continue
        style = _edge_style(kind)
        (x0, y0), (x1, y1) = node_pos[id(u)], node_pos[id(v)]
        offset = edge_curve_offset(layout, u, v)
        _draw_edge(ax, x0, y0, x1, y1, offset, style)
        tag = kind.split(":", 1)[1] if ":" in kind else ""
        if tag:
            tx, ty = _curve_midpoint(x0, y0, x1, y1, offset)
            ax.text(tx, ty, tag, fontsize=FONT_TAG,
                     color=pal.INK_SECONDARY, ha="center", va="center", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.12", facecolor=pal.SURFACE,
                                edgecolor="none", alpha=0.85))

    for chain in layout.chains:
        for nl in chain.node_layouts:
            _draw_node(ax, nl.node, node_pos[id(nl.node)], chrom_colors)

        first, last = chain.node_layouts[0], chain.node_layouts[-1]
        if not chain.is_circular:
            if id(first.node) not in non_base_connected and id(first.node) not in satellite_anchor_ids:
                _draw_cap(ax, first.node, node_pos[id(first.node)], -1.0, chrom_colors)
            if len(chain.node_layouts) > 1 and (
                id(last.node) not in non_base_connected and id(last.node) not in satellite_anchor_ids
            ):
                _draw_cap(ax, last.node, node_pos[id(last.node)], 1.0, chrom_colors)

    for sat in layout.satellites:
        _draw_node(ax, sat.tail_node, node_pos[id(sat.tail_node)], chrom_colors)
        _draw_node(ax, sat.head_node, node_pos[id(sat.head_node)], chrom_colors)

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    if xs and ys:
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(max(ys) + 0.6, min(ys) - 0.6)
