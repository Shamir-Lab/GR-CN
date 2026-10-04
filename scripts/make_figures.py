"""
Draw Genome A, Genome B, the EBG and the optimal scenario for the paper's samples.

For each sample, writes to paper_outputs/figures/<sample>[_<phasing>]/:
  genome_a, genome_b, ebg, ebg_cycles, scenario  .{pdf,png}   (matplotlib;
  ebg_cycles draws only the cycle nodes, for large graphs)
  scenario.html                                  (interactive SVG)

By default the phasing returned by ilp.phase() is drawn. To draw one specific
tied phasing, pass either --option (numbered as in the canonical folders of
analysis/haplotype_options_report/) or --phasing/--loose (the stable raw
phasing_label / loose_pairing_label columns of the summary.csv files).

Usage:
    python scripts/make_figures.py                       # all samples
    python scripts/make_figures.py --sample PM1284 --option option_025
    python scripts/make_figures.py --sample 852_T1_33_23905 --phasing phasing_37 --loose ab_cd
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr_cn.io import jabba, reference  # noqa: E402
from gr_cn.phasing import enumerate as penum  # noqa: E402
from gr_cn.viz import scenario_plot, svg_export  # noqa: E402
from gr_cn.viz.pipeline import run_pipeline  # noqa: E402

OUT = ROOT / "paper_outputs" / "figures"

SAMPLES = {
    "CA-0098": ("jabba_data/graphs_all/CA-0098.json", [5, 18, 22, 23]),
    "PM1284": ("jabba_data/graphs_all/PM1284.json", [3, 11, 12]),
    "852_T1_33_23905": ("jabba_data/graphs_all/852_T1_33_23905.json", [6, 9, 12, 14]),
}


def _select_phasing(path: str, chroms, option_id: Optional[str],
                    phasing_label: Optional[str], loose_label: Optional[str]):
    graph = jabba.load(path, chromosomes=chroms)
    ref = reference.build(graph)
    # Canonical option ids index the deduplicated list; raw labels the full one.
    options, _meta, _eq = penum.enumerate_full_options(
        graph, ref, L=2, dedupe=option_id is not None, cpu_n=1)
    if option_id:
        matches = [o for o in options if o["option_id"] == option_id]
    else:
        matches = [o for o in options if o["phasing_label"] == phasing_label
                   and (loose_label is None or o["loose_label"] == loose_label)]
    if len(matches) != 1:
        raise SystemExit(f"{len(matches)} options match {option_id or (phasing_label, loose_label)} "
                         f"(of {len(options)}); pass --loose to disambiguate")
    opt = matches[0]
    ph = opt["phasing"]
    return replace(ph, grey_pairs=list(ph.grey_pairs) + list(opt["extra_pairs"]))


def draw(sample: str, option_id: Optional[str], phasing_label: Optional[str],
         loose_label: Optional[str]) -> None:
    rel, chroms = SAMPLES[sample]
    path = str(ROOT / rel)
    phasing = None
    if option_id or phasing_label:
        phasing = _select_phasing(path, chroms, option_id, phasing_label, loose_label)
    pr = run_pipeline(path, chromosomes=chroms, phasing_result=phasing)

    tag = option_id or "_".join(x for x in (phasing_label, loose_label) if x)
    out = OUT / (f"{sample}_{tag}" if tag else sample)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("genome_a", scenario_plot.plot_genome_a),
                     ("genome_b", scenario_plot.plot_genome_b),
                     ("ebg", scenario_plot.plot_ebg),
                     ("scenario", scenario_plot.plot_scenario)):
        fig, _ = fn(path, chromosomes=chroms, pr=pr)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    fig, _ = scenario_plot.plot_ebg(path, chromosomes=chroms, pr=pr, cycles_only=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ebg_cycles.{ext}", dpi=200, bbox_inches="tight")
    svg_export.export_scenario_html(path, chromosomes=chroms, pr=pr,
                                    out_path=str(out / "scenario.html"))
    print(f"{sample}{' ' + tag if tag else ''}: {len(pr.all_cycles)} cycles, "
          f"C*_max = {pr.scenario.objective_value} -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample", choices=list(SAMPLES), nargs="+", default=list(SAMPLES))
    p.add_argument("--option", default=None, help="canonical option id, e.g. option_025")
    p.add_argument("--phasing", default=None, help="raw phasing label, e.g. phasing_37")
    p.add_argument("--loose", default=None, help="loose-end pairing label, e.g. ab_cd")
    a = p.parse_args()
    for s in a.sample:
        draw(s, a.option, a.phasing, a.loose)


if __name__ == "__main__":
    main()
