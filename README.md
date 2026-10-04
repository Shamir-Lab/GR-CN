# GR_CN

A maximum-parsimony method for reconstructing multi-chromosomal rearrangement
histories with copy-number changes from JaBbA genome graphs.

Given a JaBbA graph for a cancer sample (the observed, rearranged genome —
"Genome B"), this pipeline:

1. reconstructs the diploid reference genome it descended from ("Genome A"),
2. phases Genome B's segments into haplotypes,
3. builds the **Extended Breakpoint Graph (EBG)** relating A and B,
4. and searches for the most parsimonious ordered sequence of rearrangement
   events ("scenario") that transforms A into B, minimizing the number of
   chromosomes disturbed at any single step.

The whole pipeline has been run end to end on 3 real cancer samples (see
[Samples](#samples)); this README documents both how to reproduce that and
exactly what every output file means.

## Quick start

```bash
git clone https://github.com/Talbenyishay/GR-CN.git && cd GR-CN
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # exact versions used for the paper
pip install -e .
pytest                               # checks the paper's CA-0098 numbers (seconds)

# one sample, one phasing, best scenario
python main.py jabba_data/graphs_all/CA-0098.json --chromosomes 5 18 22 23 --quick
```

## Reproducing the paper

```bash
bash scripts/reproduce_all.sh
```

| Paper item | Produced by | Output |
|---|---|---|
| Case-study results, Table S1 (phasing ambiguity), Table S2 (C_max distributions) | `analysis/haplotype_options_report/run.py` | `analysis/haplotype_options_report/` |
| Table S3 (search effort: brute force vs DP vs branch-and-bound) | `benchmark/run_benchmark.py` | `benchmark/results.csv` |
| Tables S1–S3 as CSV + LaTeX | `scripts/make_tables.py` | `paper_outputs/tables/` |
| Genome A / Genome B / scenario figures | `scripts/make_figures.py` | `paper_outputs/figures/` |

Each sample runs as its own process; the slowest (852_T1_33_23905) takes
about 1.5 h on one core, the others minutes. CP-SAT runs with one worker
(`--cpu 1`) so phasing enumeration is deterministic.

## Pipeline

```
JaBbA JSON
    │
    ▼
1. Preprocessing            gr_cn/io/jabba.py, gr_cn/io/reference.py
   parse intervals & adjacencies, infer diploid Genome A from Genome B's copy numbers
    │
    ▼
2. Phasing                  gr_cn/phasing/ilp.py, gr_cn/phasing/enumerate.py, gr_cn/phasing/loose.py
   assign every segment copy to a haplotype via a 3-stage lexicographic
   CP-SAT solve (min components, then min cross-haplotype edges, then a
   hap0-priority tie-break); where this optimum is tied, enumerate every
   tied phasing exactly via no-good cuts; pair up any unresolved loose
   breakpoints
    │
    ▼
3. Extended Breakpoint Graph        gr_cn/ebg/builder.py, gr_cn/ebg/cycles.py
   build the EBG (Definition 4), strip shared A∩B adjacencies, extract
   alternating cycles (Lemma 1)
    │
    ▼
4. Scenario search                  gr_cn/scenario/enumeration.py, clustering.py
   find the cycle ordering minimizing the max chromosomes touched per step —
   cycles are first split into independent chromosome clusters, then each
   cluster is solved exactly (subset-DP or branch-and-bound), cross-checked
   against true brute-force enumeration wherever tractable
    │
    ▼
5. Reporting                        gr_cn/report.py
   CSV tables: Genome A, Genome B, EBG structure, cycle summary, and every
   enumerated scenario, per tied-phasing option, per sample
```

Two entry points share this machinery for different purposes:

- **`main.py`** — a single run: load one sample, phase it once (`--quick`),
  or exhaustively enumerate every tied-optimal phasing option (default), and
  find the best scenario for each. This is the library's CLI entry point.
- **`analysis/haplotype_options_report/run.py`** — the thin driver that runs
  `main.py`'s exhaustive path across all 3 named samples and writes the
  full CSV deliverable tree documented below. This is what actually
  produced everything under `analysis/haplotype_options_report/`.

## Repository layout

```
gr_cn/                          core library package
  io/            jabba.py, reference.py       — parsing + Genome A reconstruction
  models/        interval.py, adjacency.py, node.py  — data model
  phasing/       ilp.py                        — phasing CP-SAT model + one-optimal-solution phase()
                 enumerate.py                  — exhaustive tied-optimal phasing enumeration (no-good cuts)
                 loose.py                      — loose-end (unresolved breakpoint) matching
  ebg/           builder.py, cycles.py         — Extended Breakpoint Graph construction + cycle extraction
  scenario/      enumeration.py                — scenario search (DP / branch-and-bound, exact)
                 clustering.py                 — independent chromosome-cluster decomposition (union-find)
                 objective.py                  — alternative scenario objectives (max, sum, max-then-sum)
  report.py                                    — CSV writers for one tied-phasing option's full deliverable
  viz/                                         — Genome A/B and scenario drawings (matplotlib + interactive SVG)

main.py                         CLI entry point (run_quick / run_exhaustive)

analysis/
  haplotype_options_report/     the 3-sample results deliverable (see below)
    run.py                        drives all 3 samples through main.run_exhaustive()
benchmark/
  run_benchmark.py              search-effort benchmark (Table S3)
  merge_results.py              combines per-sample benchmark runs
  results.csv                   benchmark output, one row per canonical phasing option
scripts/
  reproduce_all.sh              runs everything below in order
  make_tables.py                Tables S1–S3 (CSV + LaTeX) from the outputs above
  make_figures.py               Genome A/B/scenario figures for any sample / phasing option
paper_outputs/                  generated tables and figures

data/                            reference annotation (chromosome arm boundaries)
jabba_data/graphs_all/           the 3 JaBbA samples used by this repo
tests/                           regression tests pinning the paper's CA-0098 results
requirements.txt                 exact dependency versions used for the paper
CITATION.cff                     citation metadata
LICENSE                          MIT

archive/                         historical/exploratory material, gitignored -- not part of the live pipeline
```

## Samples

| Sample | Chromosomes | JaBbA file |
|---|---|---|
| `CA-0098` | 5, 18, 22, X | `jabba_data/graphs_all/CA-0098.json` |
| `PM1284` | 3, 11, 12 | `jabba_data/graphs_all/PM1284.json` |
| `852_T1_33_23905` | 6, 9, 12, 14 | `jabba_data/graphs_all/852_T1_33_23905.json` |

## Usage

### Single run (`main.py`)

```bash
# Exhaustive (default): every tied-optimal phasing option, full CSV deliverable
python main.py jabba_data/graphs_all/PM1284.json --chromosomes 3 11 12

# Quick: one arbitrary-but-optimal phasing, just print the best scenario
python main.py jabba_data/graphs_all/PM1284.json --chromosomes 3 11 12 --quick
```

Key options:

| Flag | Default | Meaning |
|---|---|---|
| `--L` | 2 | Number of haplotypes |
| `--chromosomes` | all | Restrict to specific chromosome numbers |
| `--quick` | off | Single optimal phasing instead of exhaustive enumeration |
| `--method` | dp | Scenario search: `dp` \| `branch_and_bound` (both exact) |
| `--timing` | before | Cost timing: `before`\|`after`\|`both` |
| `--cpu` | 1 | CP-SAT search workers *within* one solve |
| `--workers` | 1 | Parallel processes *across* independent subgraphs/options |
| `--dedupe` | off | (exhaustive mode) collapse pure haplotype/copy-label relabelings — see below |
| `--out-dir` | — | (exhaustive mode) where to write the CSV deliverable |
| `--merge-haplotypes` | off | (quick mode) merge same-topology cycles across haplotypes into one event |
| `--out` | — | (quick mode) write the scenario result to a JSON file |

### Reproducing the full 3-sample results deliverable

```bash
python analysis/haplotype_options_report/run.py                          # all samples, sequentially
python analysis/haplotype_options_report/run.py --sample PM1284_3_11_12  # one sample
python analysis/haplotype_options_report/run.py --overview-only          # combine into samples_overview.csv
```

Runs all three samples end to end. **For each sample, the raw (undeduplicated)
tied-phasing enumeration always runs first.** If a sample's phasing is
completely unambiguous (only CA-0098, currently), that single result is
written with no suffix. Otherwise, a second, topology-deduplicated
("canonical") pass also runs, and *both* are written side by side — see
[Raw vs. canonical phasing options](#raw-vs-canonical-phasing-options) for
why both are kept.

## Output structure

```
analysis/haplotype_options_report/
  CA-0098_5_18_22_X/                       unambiguous sample: single result
    option_001/*.csv
    summary.csv
  PM1284_3_11_12_raw/                      ambiguous sample: every raw tied phasing
    option_001/*.csv ... option_128/*.csv
    summary.csv
  PM1284_3_11_12_canonical/                same sample: topology-deduplicated
    option_001/*.csv ... option_037/*.csv
    summary.csv
    phasing_equivalence_classes.csv
  852_T1_33_23905_6_9_12_14_raw/
  852_T1_33_23905_6_9_12_14_canonical/
  samples_overview.csv                     one row per sample, cross-sample comparison
```

**Only `summary.csv`, `samples_overview.csv`, and
`phasing_equivalence_classes.csv` are checked into git.** The per-option
`option_NNN/` folders (which can number in the hundreds across all options
of a sample) are `.gitignore`d and are exactly, deterministically
reproduced by rerunning `analysis/haplotype_options_report/run.py` — see
`.gitignore` for the pattern.

### Raw vs. canonical phasing options

JaBbA graphs report unphased somatic copy number, so before the EBG can be
built, every segment copy must be assigned to a haplotype (`gr_cn.phasing`).
This phasing optimum is found via a 3-stage lexicographic CP-SAT solve
(minimize components, then cross-haplotype edges, then a hap0-priority
tie-break — see [Method](#method)), and it is frequently **tied**: several
different concrete haplotype assignments achieve the exact same 3-stage
optimum.

Not all ties are biologically meaningful. A tied phasing can differ from
another purely by:
- **haplotype relabeling** — which physical strand is *called* haplotype 0
  vs. 1 is arbitrary wherever nothing else pins it down, or
- **gained-copy relabeling** — which of several identical extra
  (non-reference) copies of a segment is *called* copy_index 1 vs. 2.

Both are pure bookkeeping with no biological content. `gr_cn.phasing.enumerate`
enumerates every **raw** tied phasing (via CP-SAT no-good cuts), and
`--dedupe` additionally canonicalizes them by downstream EBG cycle topology
(stripping the haplotype label from every cycle's node signature) to collapse
relabeling duplicates down to genuinely distinct **canonical** reconstructions.

This matters because the two numbers tell you different things, and neither
alone is the full picture:

| Sample | Raw tied phasings | Canonical (distinct) phasings | Optimal score across options |
|---|---|---|---|
| CA-0098 | 1 | 1 | 2 (fixed — zero ambiguity) |
| PM1284 | 128 (search plateaued — likely at/near the true count) | 37 | **3 / 4 / 5 — genuinely varies** |
| 852_T1_33_23905 | 128 (hit a hard cap, no plateau — a confirmed floor, not the true count) | 80 | **3 for all 80 — invariant** |

Two importantly different situations, both real:
- **PM1284**: most of its 128 raw ties (91 of them) really are relabeling
  noise, but the 37 that remain are *genuinely structurally different*
  reconstructions, and they disagree on the final answer. The choice of
  phasing is a real, consequential ambiguity here.
- **852**: only a minority of its 128 raw ties (48) are relabeling noise —
  80 are genuinely distinct reconstructions — yet every single one of those
  80 agrees on the final score. This is a *stronger* robustness result than
  PM1284's would be even if it had collapsed further: real structural
  ambiguity that provably doesn't change the reconstructed parsimony
  objective, not just cosmetic ambiguity that trivially doesn't matter.

**Caveat:** for PM1284 and 852, the raw search is capped at
`MAX_SOLUTIONS_PER_SUBGRAPH = 128` per phasing subgraph
(`gr_cn/phasing/enumerate.py`) — a reporting cap, not a claim of
exhaustiveness (`raw_enumeration_capped` in `summary.csv`/
`samples_overview.csv` records whether it was hit). For PM1284 the search's
own find-curve plateaued well before the cap (no new solution found in the
last 30 attempts), so 128 is plausibly the true count. For 852 the find-curve
never plateaued (a new solution on almost every attempt, right up to the
cap), so both its raw and canonical counts are floors, not proven totals —
though the score itself (3, stable across all 80 canonical options found so
far) is unlikely to be affected by finding more.

## CSV file reference

### `genome_a.csv` — the reconstructed diploid reference

One row per reference interval.

| Column | Meaning |
|---|---|
| `iid` | Interval id |
| `chromosome`, `start`, `end`, `length` | Reference coordinates |
| `diploid_cn` | Expected diploid copy number (normally 2) |
| `cancer_cn_in_sample` | Observed copy number in Genome B |
| `n_deleted_copies` | How many of the diploid copies are absent in Genome B |
| `n_gained_copies` | How many extra (non-reference) copies exist in Genome B |
| `is_telomere_tail`, `is_telomere_head` | Chromosome-end flags |
| `is_centromere_included` | Whether this interval spans a centromere |

### `genome_b.csv` — the observed rearranged genome's connectivity

One row per haplotype-and-copy-specific node (`node_type, iid, side, haplotype,
copy_index`), with its actual Genome B partner:

| Column | Meaning |
|---|---|
| `node_type` | `synteny` (real segment) or `DC`/`AH` (synthetic deletion-collector / addition-holder) |
| `iid`, `side`, `haplotype`, `copy_index` | This node's identity |
| `chromosome`, `position` | Reference location (`-1` for synthetic nodes) |
| `is_telomere` | Whether this side is a chromosome end |
| `partner_type`, `partner_iid`, `partner_side`, `partner_haplotype`, `partner_copy_index` | The node it's actually adjacent to in Genome B |
| `edge_type` | `REF_shared` (unchanged reference adjacency), `ALT_rearranged` (a structural junction), or `UNRESOLVED` (unpaired loose end) |

### `ebg_structure.csv` — the Extended Breakpoint Graph's edges

One row per edge (`cycle_id` groups edges into the alternating cycle they
belong to):

| Column | Meaning |
|---|---|
| `cycle_id` | Which extracted cycle this edge belongs to |
| `color` | `black` (Genome A / reference adjacency) or `grey` (Genome B / this phasing option's adjacency) |
| `u_*`, `v_*` | The two endpoint nodes' full identity (`node_type, iid, side, haplotype, copy_index, chromosome, position`) |

### `ebg_components_summary.csv` — one row per extracted cycle

| Column | Meaning |
|---|---|
| `cycle_id` | Matches `ebg_structure.csv` |
| `n_nodes`, `n_black_edges`, `n_grey_edges` | Cycle size |
| `chromosomes` | `;`-separated reference chromosomes this cycle touches |
| `haplotypes_involved` | Which haplotype(s) this cycle runs on |
| `node_types` | `;`-separated distinct node types in the cycle |
| `ordered_nodes` | The cycle's nodes in traversal order, `type:iid+side:haplotype:copy` |

### `scenarios_summary.csv` — every possible ordered scenario for one phasing option

One row per ordered partition of that option's cycles into simultaneous
steps — i.e. literal brute-force enumeration of the scenario search space:

| Column | Meaning |
|---|---|
| `scenario_id` | Row identity |
| `n_steps` | How many steps this ordering uses |
| `objective_max_chromosomes` | This ordering's cost: the max chromosomes disturbed in any one step |
| `is_global_optimal` | Whether this ordering achieves the option's true minimum cost |
| `steps` | `\|`-separated steps, each a `,`-separated list of `cycle_id`s executed together |
| `step_costs` | `\|`-separated per-step chromosome-disturbance count |
| `step_chromosomes_before` | `\|`-separated chromosome count present just before each step |

### `summary.csv` — one row per tied-phasing option, per sample folder

| Column | Meaning |
|---|---|
| `option_id` | `option_NNN`, matches the subfolder |
| `phasing_label`, `loose_pairing_label` | Which tied phasing / loose-end pairing this option is |
| `n_components`, `n_mixed_haplotype_edges` | This phasing's 3-stage CP-SAT objective values (stages 1–2) |
| `n_ebg_cycles`, `n_ebg_nodes` | EBG size for this option |
| `n_possible_scenarios` | Brute-force count of all possible orderings (`scenarios_summary.csv` row count) |
| `scenarios_enumeration_truncated` | Whether brute-force enumeration itself was capped (separate from the phasing cap) |
| `optimal_max_chromosomes` | This option's parsimony score |
| `n_optimal_scenarios` | How many of the possible orderings achieve that score |
| `n_full_scenarios_bruteforce` | Same as `n_possible_scenarios` (kept for clarity alongside the B&B columns) |
| `bnb_expanded_states` | How many states the exact branch-and-bound search actually visited |
| `bnb_matches_full_enum` | Whether B&B's optimum agrees with the brute-force ground truth (always checked) |
| `bnb_speedup_factor` | `n_full_scenarios_bruteforce / bnb_expanded_states` |
| `n_real_chromatids_genome_a`, `n_real_chromatids_genome_b` | Real (non-synthetic) chromatid counts, sanity-checked equal unless the sample has a genuine net chromatid gain/loss |

### `phasing_equivalence_classes.csv` — canonical folders only

| Column | Meaning |
|---|---|
| `canonical_phasing_label` | The representative raw phasing kept for this canonical option |
| `n_raw_ties_collapsed` | How many raw ties share this canonical option's downstream topology |
| `raw_phasing_labels` | `;`-separated list of all those raw phasing labels |

### `samples_overview.csv` — one row per sample, cross-sample comparison

| Column | Meaning |
|---|---|
| `sample` | Sample label |
| `n_subgraphs` | Independent weakly-connected phasing subgraphs |
| `n_raw_tied_phasings`, `n_canonical_tied_phasings` | See [Raw vs. canonical](#raw-vs-canonical-phasing-options) |
| `raw_enumeration_capped` | Whether the raw search hit its solution cap (non-exhaustive) rather than proving completeness |
| `n_loose_pairing_variants` | Distinct tied loose-end pairings found |
| `distinct_optimal_max_chromosomes` | `;`-separated set of parsimony scores seen across all options |
| `objective_is_stable_across_options` | Whether every option agrees on the score |

## Method

The reconstruction follows the paper's formal framework:

- **Definition 4 (Extended Breakpoint Graph):** nodes for every synteny
  block × haplotype × copy, plus synthetic DC (deletion-collector) and AH
  (addition-holder) node pairs for deleted/gained segments; black edges are
  Genome A's adjacencies, grey edges are Genome B's.
- **Lemma 1:** after removing adjacencies shared by both genomes, the EBG
  decomposes into alternating black/grey cycles.
- **Scenario objective:** an ordered partition of these cycles into
  simultaneous "steps"; the cost of a scenario is the maximum number of
  chromosome components touched by any single step, minimized over all
  valid orderings. Cycles are first partitioned into independent
  chromosome clusters (union-find over shared chromosomes,
  `gr_cn/scenario/clustering.py`) since disjoint clusters can always be
  scheduled independently without loss of optimality; each cluster is then
  solved exactly by one of two independently-implemented algorithms
  (`gr_cn/scenario/enumeration.py`):
  - `dp`: subset dynamic programming over all `2^n_cycles` states.
  - `branch_and_bound`: depth-first search with memoization and pruning,
    typically visiting far fewer states in practice (see
    `bnb_expanded_states`/`bnb_speedup_factor` in `summary.csv`).
  Per-scenario-state connectivity (how many distinct chromosome components
  exist at a given point in the schedule) is evaluated via union-find with
  path compression and union-by-rank.
  Both are cross-checked against true brute-force enumeration of every
  possible ordering (the Fubini/ordered-Bell number of orderings, capped
  only when that count itself becomes intractable — `n_full_scenarios_bruteforce`
  in `summary.csv`), so exactness is verified three independent ways, not
  just asserted.
- **Phasing:** a 3-stage lexicographic CP-SAT objective (minimize phased
  components, then cross-haplotype edges, then a hap0-priority tie-break
  solved exactly rather than greedily — a greedy per-node tie-break was
  found to leave a real sample with 22 "unresolved" nodes where the true
  joint minimum was only 6). Where this optimum is tied, every tied
  solution is enumerated exactly via CP-SAT no-good cuts
  (`gr_cn/phasing/enumerate.py`), then optionally canonicalized by
  downstream EBG topology to separate genuine structural ambiguity from
  haplotype/copy-label relabeling artifacts (see
  [Raw vs. canonical](#raw-vs-canonical-phasing-options)).

## Installation

See [Quick start](#quick-start). Requires Python ≥3.10 (results produced
with 3.13). Core dependencies: `ortools` (CP-SAT), `networkx`, `pandas`,
`openpyxl`, `matplotlib`; `requirements.txt` pins the exact versions.

## License

MIT -- see [LICENSE](LICENSE).
