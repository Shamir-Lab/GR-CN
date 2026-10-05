# GR-CN

Reconstructing the evolutionary history of complex genome rearrangements
with copy-number changes, from JaBbA genome graphs.

Given a JaBbA graph of a cancer genome (the observed, rearranged genome,
"Genome B"), GR-CN reconstructs the diploid genome it descended from
("Genome A"), builds the **Extended Breakpoint Graph (EBG)** of A and B, and
searches the rearrangement scenarios that transform A into B — ordered
partitions of the EBG's alternating cycles into simultaneous steps — for one
that minimizes the maximum number of genome components affected by any
single step (C_max).

## Quick start

```bash
git clone https://github.com/Talbenyishay/GR-CN.git && cd GR-CN
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # exact versions used for the paper
pip install -e .
pytest                               # checks the paper's CA-0098 numbers (seconds)
python run_pipeline.py               # everything in the paper (~1.5 h)
```

## The pipeline

`run_pipeline.py` is the single entry point. For each of the three samples it
runs, in order:

| Stage | What | Code |
|---|---|---|
| 1 | Parse the JaBbA graph (Genome B) and reconstruct the diploid Genome A | `gr_cn/io/` |
| 2 | Phase Genome B: enumerate every tied-optimal phasing | `gr_cn/phasing/` |
| 3 | For every phasing: build the EBG, extract its cycles, enumerate all scenarios (C_max distribution), and measure full enumeration vs branch-and-bound | `gr_cn/ebg/`, `gr_cn/scenario/`, `gr_cn/analysis.py` |
| 4 | Select the phasing whose scenarios vary most in C_max | `gr_cn/analysis.py` |
| 5 | Write the EBG cycle data of the selected phasing, its optimum and representative scenarios | `gr_cn/report.py`, `gr_cn/analysis.py` |

and then, over all samples:

| Stage | What | Code |
|---|---|---|
| 6 | Figures: representative scenarios; complete scenarios, search-tree nodes and running time of full enumeration vs branch-and-bound | `gr_cn/figures.py` |
| 7 | Supplementary tables S1–S3 | `gr_cn/tables.py` |

Phasing (stage 2) dominates the running time — about 25 min for PM1284 and
45 min for 852_T1_33_23905 on one core. It is cached, so
`python run_pipeline.py --reuse-phasing` reruns everything else in minutes.
CP-SAT runs with one worker (`--cpu 1`), but its solves use wall-clock time
limits (a timed-out solve is retried with a longer limit). For
852_T1_33_23905, whose tied-phasing search stops at the 128-phasing cap, the
set of phasings found can therefore differ between runs on a loaded machine;
its optimum (C*_max = 4 for every phasing) does not.

### Samples

| Sample | Chromosomes of the event | JaBbA graph |
|---|---|---|
| CA-0098 | 5, 18, 22, X | `jabba_data/graphs_all/CA-0098.json` |
| PM1284 | 3, 11, 12 | `jabba_data/graphs_all/PM1284.json` |
| 852_T1_33_23905 | 6, 9, 12, 14 | `jabba_data/graphs_all/852_T1_33_23905.json` |

### Choices made by the pipeline

- **Tied phasings.** The phasing optimum (a 3-stage lexicographic CP-SAT
  objective: fewest phased components, then fewest cross-haplotype edges,
  then a haplotype-0-priority tie-break) is often tied. All tied phasings
  are enumerated with no-good cuts, capped at 128 per phasing subgraph
  (`MAX_SOLUTIONS_PER_SUBGRAPH`), and each is crossed with every tied
  pairing of unresolved loose ends; every resulting option is analyzed.
  Phasings are not deduplicated: two phasings with the same cycles up to
  haplotype labels can still differ in C*_max, because the relative
  haplotypes of the cycles decide which genome components a step affects.
- **Selected phasing.** Per sample, the phasing with the most distinct C_max
  values over its scenarios, then the most cycles, then the highest entropy
  of its C_max distribution (`analysis.select_phasing`). The criterion looks
  only at how much the objective varies, not at search effort.
- **Representative scenarios** (`analysis.representative_scenarios`): all
  cycles at once; all cycles as singletons (the worst-scoring order); the
  optimum, unless one of these attains it; and a suboptimal scenario that is
  neither, preferably with three steps.
- **Search effort.** For each phasing, full enumeration (every ordered
  partition, with its step profile) and branch-and-bound each run on a fresh
  cost model; reported are complete scenarios generated, search-tree nodes
  visited and the median running time over 5 runs. Both, and the subset
  dynamic program, must reach the same optimum. Committed times were measured
  on a shared Linux server; absolute times vary by machine.

## Outputs

```
results/
  <sample>/
    phasings.csv          every tied phasing (x loose-end pairing): cycles,
                          C_max distribution, search effort; `selected` marks the chosen one
    selected_phasing/     EBG cycle data of the selected phasing (below)
    summary.json          selection, C_max distribution, optimum, representative
                          scenarios, search effort
  figures/                representative_scenarios, effort_scenarios, effort_nodes,
                          effort_time (PDF + PNG)
  tables/                 S1_phasings, S2_cmax, S3_effort (CSV + LaTeX)
```

### EBG cycle data (`results/<sample>/selected_phasing/`)

**`genome_a.csv`** — the reconstructed diploid Genome A, one row per interval.

| Column | Meaning |
|---|---|
| `iid` | Interval id |
| `chromosome`, `start`, `end`, `length` | Reference coordinates |
| `diploid_cn` | Expected diploid copy number (normally 2) |
| `cancer_cn_in_sample` | Observed copy number in Genome B |
| `n_deleted_copies`, `n_gained_copies` | Diploid copies absent from / extra copies present in Genome B |
| `is_telomere_tail`, `is_telomere_head` | Chromosome-end flags |
| `is_centromere_included` | Whether the interval spans a centromere |

**`genome_b.csv`** — Genome B connectivity, one row per haplotype- and
copy-specific segment extremity.

| Column | Meaning |
|---|---|
| `node_type` | Always `synteny` (real segment extremities) |
| `iid`, `side`, `haplotype`, `copy_index` | The extremity's identity |
| `chromosome`, `position` | Reference location |
| `is_telomere` | Whether this side is a chromosome end |
| `partner_*` | The node it is adjacent to in Genome B; `partner_type` can also be `DC`/`AH` (deletion-collector / addition-holder) or `telomere` |
| `edge_type` | `REF_shared` (unchanged adjacency), `ALT_rearranged` (structural junction) or `UNRESOLVED` (unpaired loose end) |

**`ebg_structure.csv`** — the EBG's edges (adjacencies shared by A and B
removed), one row per edge.

| Column | Meaning |
|---|---|
| `cycle_id` | The alternating cycle the edge belongs to |
| `color` | `black` (Genome A adjacency) or `grey` (Genome B adjacency) |
| `u_*`, `v_*` | The two endpoints (`node_type, iid, side, haplotype, copy_index, chromosome, position`) |

**`ebg_components_summary.csv`** — one row per cycle.

| Column | Meaning |
|---|---|
| `cycle_id` | Matches `ebg_structure.csv` |
| `n_nodes`, `n_black_edges`, `n_grey_edges` | Cycle size |
| `chromosomes` | `;`-separated reference chromosomes the cycle touches |
| `haplotypes_involved` | Haplotype(s) the cycle runs on |
| `node_types` | `;`-separated node types in the cycle |
| `ordered_nodes` | The cycle's nodes in traversal order, `type:<iid><side>:h<haplotype>:c<copy>` |

**`scenarios_summary.csv`** — every scenario (ordered partition of the
cycles). Not tracked in git (47,293 rows for 7 cycles); regenerated by the
pipeline.

| Column | Meaning |
|---|---|
| `scenario_id` | Row id |
| `n_steps` | Number of steps |
| `objective_max_chromosomes` | C_max of the scenario |
| `is_global_optimal` | Whether it attains the optimum |
| `steps` | `\|`-separated steps, each a `,`-separated list of `cycle_id`s |
| `step_costs` | `\|`-separated components affected by each step |
| `step_chromosomes_before` | `\|`-separated number of components before each step |

## Method notes

- **EBG.** Nodes for every synteny block extremity × haplotype × copy, plus
  DC node pairs for deleted copies, AH node pairs for gained copies, and
  telomere vertices that close the paths ending at chromosome ends that
  differ between A and B; black edges are Genome A's adjacencies, grey edges
  Genome B's. With shared adjacencies removed, the EBG is a union of
  alternating cycles.
- **Objective.** Each step of a scenario executes the union of its cycles as
  one (k,d,g)-break; its cost is the number of current genome components it
  affects, computed on the intermediate genome by union-find. C_max is the
  maximum over the steps.
- **Exact search** (`gr_cn/scenario/enumeration.py`). Branch-and-bound over
  sets of executed cycles: prunes children whose running maximum reaches the
  incumbent, states already reached with a no-larger running maximum, and
  steps whose costliest single cycle cannot beat the incumbent; children are
  explored cheapest first. A subset dynamic program over all 2^m states and
  full enumeration of the F_m ordered partitions (F_m the ordered Bell
  number) serve as independent checks.

## Repository layout

```
run_pipeline.py        the pipeline (stages 1-7)
gr_cn/                 library
  io/                  JaBbA parsing, Genome A reconstruction
  models/              intervals, adjacencies, EBG nodes
  phasing/             phasing CP-SAT model, tied-phasing enumeration, loose-end pairing
  ebg/                 EBG construction, cycle extraction
  scenario/            scenario cost model, branch-and-bound, DP, enumeration
  analysis.py          per-phasing analysis, selection, representative scenarios, search effort
  report.py            CSV writers for the EBG cycle data
  figures.py           figures
  tables.py            supplementary tables
results/               pipeline outputs
jabba_data/graphs_all/ the three JaBbA graphs
data/                  chromosome-arm annotation (hg19)
tests/                 regression tests on the paper's CA-0098 results
```

## License

MIT — see [LICENSE](LICENSE).
