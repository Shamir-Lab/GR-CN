"""
Draw Genome A, Genome B and the optimal scenario for the paper's samples.

For each sample, writes to paper_outputs/figures/<sample>[_<option>]/:
  genome_a.{pdf,png}, genome_b.{pdf,png}, scenario.{pdf,png}   (matplotlib)
  scenario.html                                                (interactive SVG)

By default the phasing returned by ilp.phase() is drawn. Pass --option to
draw one specific tied phasing, numbered as in the canonical (deduplicated)
folders of analysis/haplotype_options_report/.

Usage:
    python scripts/make_figures.py                       # all samples
    python scripts/make_figures.py --sample PM1284 --option option_025
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from gr_cn.io import jabba, reference  # noqa: E402
from gr_cn.phasing import enumerate as penum  # noqa: E402
from gr_cn.viz import scenario_plot, svg_export  # noqa: E402
from gr_cn.viz.pipeline import run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_outputs" / "figures"

SAMPLES = {
    "CA-0098": ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23]),
    "PM1284": ("jabba_data/graphs_all/PM1284.json", [3, 11, 12]),
    "852_T1_33_23905": ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14]),
}


def _option_phasing(path: str, chroms, option_id: str):
    graph = jabba.load(path, chromosomes=chroms)
    ref = reference.build(graph)
    options, _meta, _eq = penum.enumerate_full_options(graph, ref, L=2, dedupe=True, cpu_n=1)
    opt = next((o for o in options if o["option_id"] == option_id), None)
    if opt is None:
        raise SystemExit(f"{option_id} not found ({len(options)} canonical options)")
    ph = opt["phasing"]
    return replace(ph, grey_pairs=list(ph.grey_pairs) + list(opt["extra_pairs"]))


def draw(sample: str, option_id: str | None) -> None:
    rel, chroms = SAMPLES[sample]
    path = str(ROOT / rel)
    phasing = _option_phasing(path, chroms, option_id) if option_id else None
    pr = run_pipeline(path, chromosomes=chroms, phasing_result=phasing)

    out = OUT / (f"{sample}_{option_id}" if option_id else sample)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("genome_a", scenario_plot.plot_genome_a),
                     ("genome_b", scenario_plot.plot_genome_b),
                     ("scenario", scenario_plot.plot_scenario)):
        fig, _ = fn(path, chromosomes=chroms, pr=pr)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    svg_export.export_scenario_html(path, chromosomes=chroms, pr=pr,
                                    out_path=str(out / "scenario.html"))
    print(f"{sample}{' ' + option_id if option_id else ''}: {len(pr.all_cycles)} cycles, "
          f"C*_max = {pr.scenario.objective_value} -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample", choices=list(SAMPLES), nargs="+", default=list(SAMPLES))
    p.add_argument("--option", default=None, help="canonical option id, e.g. option_025")
    a = p.parse_args()
    for s in a.sample:
        draw(s, a.option)


if __name__ == "__main__":
    main()
