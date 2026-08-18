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

## Pipeline

```
JaBbA JSON
    │
    ▼
1. Preprocessing            gr_cn/io/jabba.py, gr_cn/io/reference.py
   parse intervals & adjacencies, infer diploid Genome A from Genome B's copy numbers
    │
    ▼
2. Phasing + loose-end matching     gr_cn/phasing/ilp.py, gr_cn/phasing/loose.py
   assign every segment copy to a haplotype via CP-SAT (min components, then
   min cross-haplotype edges); pair up any unresolved loose breakpoints
    │
    ▼
3. Extended Breakpoint Graph        gr_cn/ebg/builder.py, gr_cn/ebg/cycles.py
   build the EBG (Definition 4), strip shared A∩B adjacencies, extract
   alternating cycles (Lemma 1)
    │
    ▼
4. Scenario search                  gr_cn/scenario/enumeration.py, clustering.py
   find the cycle ordering minimizing the max chromosomes touched per step —
   exact branch-and-bound / DP / beam search / brute-force, by instance size
    │
    ▼
5. Summarization                    analysis/haplotype_options_report/
   CSV tables: Genome A, Genome B, EBG structure, cycle summary, and every
   enumerated scenario, per sample
```

Two entry points share this machinery for different purposes:

- **`main.py`** — a single run: load one sample, phase it once, find one best
  scenario. This is the fast, direct path.
- **`analysis/haplotype_options_report/`** — the phasing ILP is frequently
  *tied* at its optimum (multiple equally-valid haplotype assignments and
  loose-end pairings). This path enumerates **every** tied-optimal
  combination exhaustively (via CP-SAT no-good cuts, proven exhaustive rather
  than capped) and writes a full CSV deliverable comparing them all.

## Repository layout

```
gr_cn/                          core library package
  io/            jabba.py, reference.py       — parsing + Genome A reconstruction
  models/        interval.py, adjacency.py, node.py  — data model
  phasing/       ilp.py, loose.py             — phasing CP-SAT + loose-end matching
  ebg/           builder.py, cycles.py        — Extended Breakpoint Graph
  scenario/      enumeration.py, clustering.py — scenario search (exact/heuristic)

main.py                         single-run CLI entry point

analysis/
  haplotype_options_report/     exhaustive tied-option enumeration + CSV summarization
    enumerate_options.py          3-stage lexicographic CP-SAT + exhaustive tie enumeration
    write_csvs.py                 per-option CSV writers
    run.py                        drives all samples, writes summary.csv / samples_overview.csv
    <sample>/option_NNN/*.csv     generated deliverable (checked in)
  phasing_degeneracy/
    degeneracy_experiment.py      subgraph builder + loose-pairing enumerator
                                   (shared by enumerate_options.py)

data/                            reference annotation (chromosome arm boundaries)
jabba_data/graphs_all/           the 3 JaBbA samples used by this repo
tests/                           test package (scaffold)
GR_CN.pdf                        the paper
```

Historical, superseded, and exploratory material (an old visualization
subpackage, unused sample data, earlier standalone experiment scripts, other
tie-break investigations) is intentionally excluded from git — see
`.gitignore`.

## Samples

| Sample | Chromosomes |
|---|---|
| `CA-0098` | 5, 18, 22, X |
| `PM1284` | 3, 11, 12 |
| `852_T1_33_23905` | 6, 9, 12, 14 |

## Usage

### Single run (`main.py`)

```bash
python main.py jabba_data/graphs_all/PM1284.json --chromosomes 3 11 12
```

Key options:

| Flag | Default | Meaning |
|---|---|---|
| `--L` | 2 | Number of haplotypes |
| `--time-limit` | 200 | Per-subgraph phasing ILP time limit (s) |
| `--cpu` | 1 | CP-SAT worker threads |
| `--method` | auto | Scenario search: `auto`\|`exact`\|`ilp`\|`beam`\|`greedy` |
| `--exact-max` | 16 | Max cycles for exact branch-and-bound in `auto` mode |
| `--ilp-max` | 18 | Max cycles for the DP-exact solver in `ilp` mode |
| `--timing` | before | Cost timing: `before`\|`after`\|`both` |
| `--chromosomes` | all | Restrict to specific chromosome numbers |
| `--no-cluster` | off | Disable independent chromosome-cluster optimization |
| `--merge-haplotypes` | off | Merge same-topology cycles across haplotypes into one event |
| `--out` | — | Write the scenario result to a JSON file |

### Exhaustive tied-option enumeration + CSV report

```bash
python analysis/haplotype_options_report/run.py
```

Runs all three samples end to end and writes, per sample, one folder per
tied-optimal (phasing × loose-pairing) option:

```
analysis/haplotype_options_report/<sample>/option_NNN/
  genome_a.csv                  Genome A (identical across every option)
  genome_b.csv                  Genome B connectivity for this option
  ebg_structure.csv             every black/grey EBG edge
  ebg_components_summary.csv    one row per extracted EBG cycle
  scenarios_summary.csv         one row per scenario ordering, plus an exact
                                 branch-and-bound vs. brute-force comparison

analysis/haplotype_options_report/<sample>/summary.csv     one row per option
analysis/haplotype_options_report/samples_overview.csv     one row per sample
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e .
```

Requires Python ≥3.10. Core dependencies: `ortools` (CP-SAT), `networkx`,
`pandas`, `openpyxl`.

## Method

The reconstruction follows the paper's formal framework (`GR_CN.pdf`):

- **Definition 4 (Extended Breakpoint Graph):** nodes for every synteny
  block × haplotype × copy, plus synthetic DC (deletion-collector) and AH
  (addition-holder) node pairs for deleted/gained segments; black edges are
  Genome A's adjacencies, grey edges are Genome B's.
- **Lemma 1:** after removing adjacencies shared by both genomes, the EBG
  decomposes into alternating black/grey cycles.
- **Scenario objective:** an ordered partition of these cycles into
  simultaneous "steps"; the cost of a scenario is the maximum number of
  chromosome components touched by any single step, minimized over all
  valid orderings. Solved exactly via branch-and-bound (or DP for larger
  instances up to a tractable cycle count), cross-checked against true
  brute-force enumeration (the Fubini/ordered-Bell number of orderings)
  wherever that's still tractable.
