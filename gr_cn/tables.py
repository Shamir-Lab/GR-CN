"""
Supplementary tables, built from the per-phasing rows and per-sample
summaries of run_pipeline.py (CSV + LaTeX tabular each):

  S1_phasings      per sample: tied optimal phasings, EBG cycle counts and
                   the distribution of C*_max over them
  S2_cmax          per sample, for the selected phasing: number of scenarios
                   at each value of C_max
  S3_effort        over all tied phasings, by sample and number of cycles:
                   complete scenarios, search-tree nodes and running time of
                   full enumeration vs branch-and-bound (medians)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _counts(counter: Counter) -> str:
    """'3:16, 4:64' -- value:count pairs in value order."""
    return ", ".join(f"{k}:{counter[k]}" for k in sorted(counter))


def table_s1(phasings: Dict[str, List[Dict]], summaries: List[Dict]) -> pd.DataFrame:
    rows = []
    for st in summaries:
        ph = phasings[st["sample"]]  # one row per tied phasing x tied loose-end pairing
        rows.append({
            "Sample": st["sample"], "Tied phasings": st["n_tied_phasings"], "Options": len(ph),
            "Search capped": st["phasing_search_capped"],
            "EBG cycles": _counts(Counter(r["n_cycles"] for r in ph)),
            "C*max": _counts(Counter(r["optimal_cmax"] for r in ph)),
        })
    return pd.DataFrame(rows)


def table_s2(summaries: List[Dict]) -> pd.DataFrame:
    values = sorted({int(v) for st in summaries for v in st["cmax_distribution"]})
    rows = []
    for st in summaries:
        d = {int(k): v for k, v in st["cmax_distribution"].items()}
        rows.append({"Sample": st["sample"], "Phasing": st["selected"]["phasing_label"],
                     "m": st["selected"]["n_cycles"], "Scenarios": sum(d.values()),
                     **{f"Cmax={v}": d.get(v, 0) for v in values}})
    return pd.DataFrame(rows)


def table_s3(phasings: Dict[str, List[Dict]]) -> pd.DataFrame:
    df = pd.DataFrame([{"Sample": s, **r} for s, ph in phasings.items() for r in ph])
    g = df.groupby(["Sample", "n_cycles"], sort=False)
    out = g.agg(
        Phasings=("option_id", "size"),
        Scenarios=("enum_complete_scenarios", "first"),
        BnB_complete=("bnb_complete_scenarios", "median"),
        Enum_nodes=("enum_tree_nodes", "first"),
        BnB_nodes=("bnb_tree_nodes", "median"),
        Enum_ms=("enum_seconds", lambda s: 1e3 * s.median()),
        BnB_ms=("bnb_seconds", lambda s: 1e3 * s.median()),
    ).reset_index().rename(columns={"n_cycles": "m"})
    return out.sort_values(["Sample", "m"], key=lambda c: c.map(
        {s: i for i, s in enumerate(phasings)}) if c.name == "Sample" else c)


def _write(df: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, float_format="%.4g"), encoding="utf-8")


def write_all(phasings: Dict[str, List[Dict]], summaries: List[Dict], out: Path) -> None:
    _write(table_s1(phasings, summaries), out, "S1_phasings")
    _write(table_s2(summaries), out, "S2_cmax")
    _write(table_s3(phasings), out, "S3_effort")
