"""
Build the paper's supplementary tables (Appendix C) from the analysis outputs.

Inputs (produced by scripts/reproduce_all.sh):
  analysis/haplotype_options_report/<sample>[_raw|_canonical]/summary.csv
  analysis/haplotype_options_report/<sample>[_raw|_canonical]/option_*/scenarios_summary.csv
  benchmark/results.csv

Outputs (CSV + LaTeX tabular for each table) in paper_outputs/tables/:
  S1_phasing_ambiguity   per sample: tied phasings (raw/canonical), EBG cycle
                         counts and the distribution of C*_max over raw phasings
  S2_cmax_distribution   per sample/option: number of scenarios at each C_max
  S3_search_effort       benchmark summarized by sample and number of cycles

Usage:
    python scripts/make_tables.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "analysis" / "haplotype_options_report"
BENCH = ROOT / "benchmark" / "results.csv"
OUT = ROOT / "paper_outputs" / "tables"


def _counts(series: pd.Series) -> str:
    """'3:16; 4:64; 5:48' -- value:count pairs in value order."""
    c = Counter(series)
    return "; ".join(f"{k}:{c[k]}" for k in sorted(c))


def _sample_dirs():
    """(sample, raw_dir, canonical_dir) -- unambiguous samples have one folder."""
    out = {}
    for d in sorted(p for p in REPORT.iterdir() if (p / "summary.csv").exists()):
        name = d.name
        if name.endswith("_raw"):
            out.setdefault(name[:-4], {})["raw"] = d
        elif name.endswith("_canonical"):
            out.setdefault(name[:-10], {})["canonical"] = d
        else:
            out.setdefault(name, {})["raw"] = d
            out[name]["canonical"] = d
    return out


def table_s1() -> pd.DataFrame:
    rows = []
    for sample, dirs in _sample_dirs().items():
        raw = pd.read_csv(dirs["raw"] / "summary.csv")
        can = pd.read_csv(dirs["canonical"] / "summary.csv")
        rows.append({
            "sample": sample,
            "raw_tied_phasings": len(raw),
            "canonical_phasings": len(can),
            "ebg_cycles (raw: n_cycles:count)": _counts(raw["n_ebg_cycles"]),
            "Cmax* (raw: value:count)": _counts(raw["optimal_max_chromosomes"]),
            "Cmax* (canonical: value:count)": _counts(can["optimal_max_chromosomes"]),
        })
    return pd.DataFrame(rows)


def table_s2() -> pd.DataFrame:
    rows = []
    for sample, dirs in _sample_dirs().items():
        for opt in sorted(dirs["canonical"].glob("option_*")):
            f = opt / "scenarios_summary.csv"
            if not f.exists():
                continue
            scen = pd.read_csv(f, usecols=["objective_max_chromosomes"])
            dist = Counter(scen["objective_max_chromosomes"])
            rows.append({
                "sample": sample, "option_id": opt.name, "n_scenarios": len(scen),
                "Cmax*": min(dist),
                **{f"Cmax={k}": dist[k] for k in sorted(dist)},
            })
    df = pd.DataFrame(rows)
    cmax_cols = sorted((c for c in df.columns if c.startswith("Cmax=")), key=lambda c: int(c[5:]))
    return df[["sample", "option_id", "n_scenarios", "Cmax*"] + cmax_cols].fillna(0).astype(
        {c: int for c in cmax_cols})


def table_s3() -> pd.DataFrame:
    b = pd.read_csv(BENCH)
    assert b["all_methods_agree"].all(), "benchmark methods disagree on some optimum"
    g = b.groupby(["sample", "n_cycles"])
    return g.agg(
        n_phasings=("option_id", "size"),
        n_scenarios=("n_scenarios", "first"),
        brute_evals=("brute_cost_evals", "median"),
        dp_evals=("dp_cost_evals", "median"),
        bnb_evals=("bnb_cost_evals", "median"),
        enum_s=("enum_seconds", "median"),
        brute_s=("brute_seconds", "median"),
        dp_s=("dp_seconds", "median"),
        bnb_s=("bnb_seconds", "median"),
    ).reset_index()


def _write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, float_format="%.4g"), encoding="utf-8")
    print(f"{name}: {len(df)} rows -> {OUT / name}.csv/.tex")


def main() -> None:
    _write(table_s1(), "S1_phasing_ambiguity")
    _write(table_s2(), "S2_cmax_distribution")
    if BENCH.exists():
        _write(table_s3(), "S3_search_effort")
    else:
        print(f"skipping S3: {BENCH} not found (run benchmark/run_benchmark.py)")


if __name__ == "__main__":
    main()
