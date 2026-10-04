#!/usr/bin/env bash
# Reproduce every result in the paper from the three JaBbA inputs.
#
# Main text -- case studies (one phasing per sample):
#   1. analysis/case_studies/              -- phasing, selection of the phasing whose
#                                             scenarios vary most in C_max, its EBG,
#                                             all scenarios, optimum, search effort
#   2. paper_outputs/figures/case_studies/ -- representative scenarios; complete
#                                             scenarios, search-tree nodes and running
#                                             time of full enumeration vs branch-and-bound
# Appendix -- all tied phasings:
#   3. analysis/haplotype_options_report/  -- every distinct phasing (Tables S1, S2)
#   4. benchmark/results.csv               -- search effort over all phasings (Table S3)
#   5. paper_outputs/tables, figures       -- tables and figures built from 3-4
#
# Each sample runs as its own process; the slowest (852_T1_33_23905) takes
# about 1.5 h on one core. Usage:  bash scripts/reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=logs
mkdir -p "$LOG" benchmark/results

for s in CA-0098 PM1284 852_T1_33_23905; do
  "$PY" scripts/run_case_studies.py --sample "$s" > "$LOG/case_study_$s.log" 2>&1 &
done
for s in CA-0098_5_18_22_X PM1284_3_11_12 852_T1_33_23905_6_9_12_14; do
  "$PY" analysis/haplotype_options_report/run.py --sample "$s" > "$LOG/analysis_$s.log" 2>&1 &
done
for s in CA-0098 PM1284 852_T1_33_23905; do
  "$PY" benchmark/run_benchmark.py --only "$s" --out "benchmark/results/results_$s.csv" \
    > "$LOG/bench_$s.log" 2>&1 &
done
wait

"$PY" scripts/plot_case_studies.py
"$PY" analysis/haplotype_options_report/run.py --overview-only
"$PY" benchmark/merge_results.py
"$PY" scripts/make_tables.py
"$PY" scripts/make_figures.py
echo "Done. See paper_outputs/."
