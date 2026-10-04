"""
Top-level entry points: run the pipeline and render the paper-style figures.

  plot_genome_a(...)  -- single panel, reference (black-only) structure
  plot_genome_b(...)  -- single panel, phased/realized (grey-if-present) structure
  plot_scenario(...)  -- one panel per scenario step, with {Ti,...} arrows
                         between them, exactly like Fig. 6 of the paper
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from gr_cn.viz import colors as pal
from gr_cn.viz.draw import draw_step
from gr_cn.viz.genomes import full_genome_steps
from gr_cn.viz.layout import compute_layout
from gr_cn.viz.model import build_cycle_infos, build_viz_steps, step_label
from gr_cn.viz.pipeline import PipelineResult, run_pipeline

PANEL_W_PER_COL = 1.15
PANEL_H_PER_ROW = 0.62
PANEL_MARGIN = 1.6
ARROW_GAP_IN = 1.05


def _chrom_colors(chromosomes: Optional[List[int]], pr: PipelineResult) -> dict:
    if chromosomes is None:
        chromosomes = sorted({iv.chromosome for iv in pr.graph.intervals})
    return pal.chromosome_colors(sorted(chromosomes))


def _draw_legend(fig, chrom_colors: dict, y: float = 0.01) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                    markerfacecolor=color, markeredgecolor=pal.INK_PRIMARY,
                    label=f"chr {chrom}")
        for chrom, color in chrom_colors.items()
    ]
    handles.append(plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                                markerfacecolor=pal.DC_FILL, markeredgecolor=pal.DC_EDGE,
                                label="DC / AH (deleted / gained copy)"))
    handles.append(plt.Line2D([0], [0], color=pal.INK_PRIMARY, linewidth=1.1,
                                label="shared (both genomes)"))
    handles.append(plt.Line2D([0], [0], color=pal.INK_SECONDARY, linewidth=1.1,
                                label="genome A only"))
    handles.append(plt.Line2D([0], [0], color=pal.INK_MUTED, linewidth=1.6,
                                label="genome B only (resolved)"))
    handles.append(plt.Line2D([0], [0], color=pal.INK_MUTED, linewidth=1.1,
                                linestyle=(0, (4, 3)), label="genome B only (pending)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, y))


def _single_panel_figure(layout_step, draw_step_obj, chrom_colors: dict, title: str):
    layout = compute_layout(layout_step)
    w = max(4.0, layout.n_cols * PANEL_W_PER_COL + PANEL_MARGIN)
    h = max(3.0, layout.n_rows * PANEL_H_PER_ROW + PANEL_MARGIN)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(pal.SURFACE)
    draw_step(ax, draw_step_obj, layout, chrom_colors, title=title)
    _draw_legend(fig, chrom_colors, y=0.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_genome_a(jabba_path: str, chromosomes: Optional[List[int]] = None,
                   pr: Optional[PipelineResult] = None, **pipeline_kwargs):
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    layout_step, draw_step_obj = full_genome_steps(pr.raw_nodes, use_grey=False)
    title = f"Genome A (diploid reference) -- {pr.graph.sample_name}"
    fig = _single_panel_figure(layout_step, draw_step_obj, chrom_colors, title)
    return fig, pr


def plot_genome_b(jabba_path: str, chromosomes: Optional[List[int]] = None,
                   pr: Optional[PipelineResult] = None, **pipeline_kwargs):
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    layout_step, draw_step_obj = full_genome_steps(pr.raw_nodes, use_grey=True)
    title = f"Genome B (phased tumor) -- {pr.graph.sample_name}"
    fig = _single_panel_figure(layout_step, draw_step_obj, chrom_colors, title)
    return fig, pr


def plot_ebg(jabba_path: str, chromosomes: Optional[List[int]] = None,
              pr: Optional[PipelineResult] = None, cycles_only: bool = False,
              **pipeline_kwargs):
    """Single panel: the EBG on the initial genome (first scenario panel).
    cycles_only draws just the nodes of the cycles, without the unchanged
    chromosome backbone -- needed for large graphs."""
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    step = build_viz_steps(pr.all_cycles, pr.scenario.partition,
                           all_nodes=None if cycles_only else pr.raw_nodes)[0]
    title = f"EBG -- {pr.graph.sample_name} ({len(pr.all_cycles)} cycles)"
    fig = _single_panel_figure(step, step, chrom_colors, title)
    return fig, pr


def plot_scenario(jabba_path: str, chromosomes: Optional[List[int]] = None,
                   pr: Optional[PipelineResult] = None, **pipeline_kwargs):
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    cycle_infos = build_cycle_infos(pr.all_cycles)

    if not pr.all_cycles:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No rearrangement cycles in this selection.",
                 ha="center", va="center", fontsize=10)
        return fig, pr

    steps = build_viz_steps(pr.all_cycles, pr.scenario.partition, all_nodes=pr.raw_nodes)
    layouts = [compute_layout(s) for s in steps]

    widths = [max(2.4, lay.n_cols * PANEL_W_PER_COL + PANEL_MARGIN) for lay in layouts]
    height = max(3.0, max(lay.n_rows for lay in layouts) * PANEL_H_PER_ROW + PANEL_MARGIN)
    total_w = sum(widths) + ARROW_GAP_IN * (len(steps) - 1) + 0.6

    fig = plt.figure(figsize=(total_w, height + 0.8))
    fig.patch.set_facecolor(pal.SURFACE)

    x_cursor = 0.0
    axes = []
    for i, (step, layout, w) in enumerate(zip(steps, layouts, widths)):
        left = x_cursor / total_w
        width_frac = w / total_w
        ax = fig.add_axes((left, 0.14, width_frac, 0.78))
        title = "Initial (EBG)" if i == 0 else f"After step {i}"
        draw_step(ax, step, layout, chrom_colors, title=title)
        axes.append(ax)
        x_cursor += w
        if i < len(steps) - 1:
            x_cursor += ARROW_GAP_IN

    for i in range(len(steps) - 1):
        left_ax, right_ax = axes[i], axes[i + 1]
        lb, rb = left_ax.get_position(), right_ax.get_position()
        y_mid = (lb.y0 + lb.y1) / 2
        arrow = FancyArrowPatch(
            (lb.x1 + 0.004, y_mid), (rb.x0 - 0.004, y_mid),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
            linewidth=2.2, color=pal.INK_PRIMARY,
        )
        fig.add_artist(arrow)
        label = step_label(pr.scenario.partition[i], cycle_infos)
        fig.text((lb.x1 + rb.x0) / 2, y_mid + 0.035, label,
                  ha="center", va="bottom", fontsize=10, fontweight="bold",
                  color=pal.INK_PRIMARY)

    fig.suptitle(
        f"{pr.graph.sample_name} -- scenario ({pr.scenario.method}, "
        f"max chromosomes touched = {pr.scenario.objective_value})",
        fontsize=11, color=pal.INK_PRIMARY, y=0.99,
    )
    _draw_legend(fig, chrom_colors, y=0.005)
    return fig, pr
