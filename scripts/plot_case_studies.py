"""
Figures for the case studies, drawn from analysis/case_studies/<sample>/case_study.json
(written by scripts/run_case_studies.py).

Writes to paper_outputs/figures/case_studies/ (PDF + PNG each):
  representative_scenarios   per sample, components affected at each step of
                             representative scenarios with different C_max
  effort_scenarios           complete scenarios generated: full enumeration vs B&B
  effort_nodes               search-tree nodes visited:    full enumeration vs B&B
  effort_time                running time:                 full enumeration vs B&B

Usage:
    python scripts/plot_case_studies.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "case_studies"
OUT = ROOT / "paper_outputs" / "figures" / "case_studies"
SAMPLES = ["CA-0098", "PM1284", "852_T1_33_23905"]

# Palette of the group's meeting slides (GR-CN 13-09-26): steel blue / light
# blue for the two methods, with a slate gray for the all-at-once scenario.
BLUE, BLUE_MID, BLUE_LIGHT, SLATE, TINT = "#4f78a8", "#7898ba", "#a9bed6", "#717a86", "#d8e3ed"
METHOD_COLORS = {"enumeration": BLUE, "branch_and_bound": BLUE_LIGHT}
METHOD_LABELS = {"enumeration": "Full enumeration", "branch_and_bound": "Branch-and-bound"}
SCENARIO_STYLE = {  # color, marker
    "All at once": (SLATE, "o"),
    "Suboptimal": (BLUE_LIGHT, "s"),
    "Singletons": (BLUE_MID, "D"),
    "Singletons (optimal)": (BLUE, "D"),
    "Optimal": (BLUE, "^"),
}
INK, INK_2, GRID, SURFACE = "#444444", "#595959", "#e6e6e6", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _load():
    return [json.loads((DATA / s / "case_study.json").read_text(encoding="utf-8")) for s in SAMPLES]


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{name} -> {OUT}")


def _partition_label(partition) -> str:
    steps = ["{" + ",".join(f"T$_{{{i + 1}}}$" for i in step) + "}" for step in partition]
    return r" $\rightarrow$ ".join(steps)


def _wrap_steps(label: str, per_line: int = 3) -> str:
    parts = label.split(r" $\rightarrow$ ")
    lines = [r" $\rightarrow$ ".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line)]
    return ("\n" + r"$\rightarrow$ ").join(lines)


def plot_representative(studies) -> None:
    fig, axes = plt.subplots(1, len(studies), figsize=(5.2 * len(studies), 4.6))
    for ax, st in zip(axes, studies):
        reps = st["representative_scenarios"]  # all at once, singletons, optimal, suboptimal
        handles, labeled = [], Counter()
        for r in reps:
            color, marker = SCENARIO_STYLE[r["name"]]
            x = list(range(1, len(r["step_costs"]) + 1))
            (h,) = ax.plot(x, r["step_costs"], color=color, marker=marker, markersize=8,
                           linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                           solid_capstyle="round", label=f"{r['name']}, $C_{{\\max}}$ = {r['cmax']}: "
                           + _wrap_steps(_partition_label(r["partition"])))
            handles.append(h)
            # label the last step that attains C_max (the right end of a flat line)
            peak = len(r["step_costs"]) - 1 - r["step_costs"][::-1].index(r["cmax"])
            spot = (x[peak], r["cmax"])  # stack labels that would land on the same point
            ax.annotate(f"$C_{{\\max}}$ = {r['cmax']}", spot,
                        xytext=(7, 7 + 12 * labeled[spot]), textcoords="offset points", va="bottom",
                        fontsize=8.5, color=INK_2, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9))
            labeled[spot] += 1
        n_steps = max(len(r["step_costs"]) for r in reps)
        ax.set_xlim(0.6, n_steps + 0.9)
        ax.set_ylim(0, max(r["cmax"] for r in reps) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Step")
        ax.set_title(f"{st['sample']} ({st['selected']['n_cycles']} cycles)", fontsize=11,
                     color=INK, fontweight="bold")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  frameon=False, fontsize=8, handlelength=2.2)
    axes[0].set_ylabel(r"Components affected, $\chi_t$")
    _save(fig, "representative_scenarios")


def _effort_bars(studies, key, ylabel, name, scale=1.0, unit="", fmt="{:,.0f}") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width, gap = 0.26, 0.03
    for i, st in enumerate(studies):
        e = st["effort"]
        vals = {m: e[m][key] * scale for m in ("enumeration", "branch_and_bound")}
        for j, m in enumerate(("enumeration", "branch_and_bound")):
            x = i + (j - 0.5) * (width + gap)
            ax.bar(x, vals[m], width, color=METHOD_COLORS[m], label=METHOD_LABELS[m] if i == 0 else None)
            ax.annotate(fmt.format(vals[m]) + unit, (x, vals[m]), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
        fold = vals["enumeration"] / vals["branch_and_bound"]
        top = max(vals.values())
        ax.annotate(f"{fold:,.0f}×" if fold >= 10 else f"{fold:.1f}×", (i, top), xytext=(0, 19),
                    textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=TINT, edgecolor="none"))
    ax.set_yscale("log")
    lo = min(st["effort"][m][key] * scale for st in studies for m in ("enumeration", "branch_and_bound"))
    hi = max(st["effort"][m][key] * scale for st in studies for m in ("enumeration", "branch_and_bound"))
    ax.set_ylim(lo / 3, hi * 12)
    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels([f"{st['sample']}\n({st['selected']['n_cycles']} cycles)" for st in studies])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    _save(fig, name)


def main() -> None:
    studies = _load()
    plot_representative(studies)
    _effort_bars(studies, "complete_scenarios", "Complete scenarios generated (log scale)",
                 "effort_scenarios")
    _effort_bars(studies, "tree_nodes", "Search-tree nodes visited (log scale)", "effort_nodes")
    _effort_bars(studies, "seconds", "Running time, ms (log scale)", "effort_time",
                 scale=1e3, unit=" ms", fmt="{:,.2f}")


if __name__ == "__main__":
    main()
