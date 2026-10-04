"""
Re-runs the gr_cn pipeline (same steps as main.run_quick) but keeps every
intermediate object the visualizers need: the reference genome, the phasing
result, both the raw and stripped EBG node lists, the extracted cycles, and
the scenario result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from gr_cn.ebg import builder, cycles as ebg_cycles
from gr_cn.io import jabba, reference
from gr_cn.models.node import EBGNode
from gr_cn.phasing import ilp, loose
from gr_cn.scenario import enumeration


@dataclass
class PipelineResult:
    graph: "jabba.JabbaGraph"
    ref: "reference.ReferenceGenome"
    phasing_result: "ilp.PhasingResult"
    raw_nodes: List[EBGNode]     # unstripped -- for full Genome A/B panels
    ebg_nodes: List[EBGNode]     # stripped -- for cycle extraction
    all_cycles: List[List[EBGNode]]
    scenario: enumeration.ScenarioResult


def run_pipeline(
    jabba_path: str,
    L: int = 2,
    time_limit_s: float = 200.0,
    cpu_n: int = 1,
    method: str = "dp",
    timing: str = "before",
    chromosomes: Optional[List[int]] = None,
    merge_haplotypes: bool = False,
    phasing_result: Optional["ilp.PhasingResult"] = None,
) -> PipelineResult:
    """
    phasing_result: draw a specific phasing (e.g. one tied option from
    gr_cn.phasing.enumerate) instead of the single phasing ilp.phase() returns.
    """
    graph = jabba.load(jabba_path, chromosomes=chromosomes)
    ref = reference.build(graph)
    if phasing_result is None:
        phasing_result = ilp.phase(graph, ref, L=L, time_limit_s=time_limit_s, cpu_n=cpu_n)

        if graph.loose_adjacencies:
            try:
                extra_pairs = loose.resolve_post_phasing(graph, phasing_result)
                phasing_result.grey_pairs.extend(extra_pairs)
            except loose.TooManyLooseEnds:
                pass

    raw_nodes = builder.build(graph, ref, phasing_result, L=L)
    # Cycle extraction needs its own (mutated) copy -- rebuild rather than
    # deep-copy so raw_nodes stays untouched for the full-genome panels.
    ebg_nodes = builder.build(graph, ref, phasing_result, L=L)
    ebg_cycles.remove_shared_adjacencies(ebg_nodes)
    raw_cycles = ebg_cycles.extract_cycles(ebg_nodes)
    all_cycles = ebg_cycles.merge_partner_cycles(raw_cycles) if merge_haplotypes else raw_cycles

    scenario = enumeration.find_best_scenario(
        cycles=ebg_cycles.remap_cycles_to_nodes(all_cycles, raw_nodes),
        all_nodes=raw_nodes, timing=timing, method=method,
    )

    return PipelineResult(
        graph=graph, ref=ref, phasing_result=phasing_result,
        raw_nodes=raw_nodes, ebg_nodes=ebg_nodes, all_cycles=all_cycles,
        scenario=scenario,
    )
