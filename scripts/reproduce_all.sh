#!/usr/bin/env bash
# Reproduce every result in the paper from the three JaBbA inputs.
#
#   1. analysis/haplotype_options_report/  -- all tied phasings, EBGs, scenarios (Tables S1, S2)
#   2. benchmark/results.csv               -- search-effort benchmark (Table S3)
#   3. paper_outputs/tables, figures       -- tables and figures built from 1-2
#
# Each sample runs as its own process; the slowest (852_T1_33_23905) takes
# about 1.5 h on one core. Usage:  bash scripts/reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=logs
mkdir -p "$LOG" benchmark/results

for s in CA-0098_5_18_22_X PM1284_3_11_12 852_T1_33_23905_6_9_12_14; do
  "$PY" analysis/haplotype_options_report/run.py --sample "$s" > "$LOG/analysis_$s.log" 2>&1 &
done
for s in CA-0098 PM1284 852_T1_33_23905; do
  "$PY" benchmark/run_benchmark.py --only "$s" --out "benchmark/results/results_$s.csv" \
    > "$LOG/bench_$s.log" 2>&1 &
done
wait

"$PY" analysis/haplotype_options_report/run.py --overview-only
"$PY" benchmark/merge_results.py
"$PY" scripts/make_tables.py
"$PY" scripts/make_figures.py
echo "Done. See paper_outputs/."
