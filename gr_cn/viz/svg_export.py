"""
Self-contained, pannable/zoomable HTML export of the same panels
gr_cn.viz.scenario_plot renders with matplotlib.

This is a second rendering backend for the *same* data: it consumes the
exact StepLayout / VizStep objects gr_cn.viz.layout and gr_cn.viz.model
already produce (so node placement, satellite anchoring, and row alignment
are identical to the static PNGs) and emits SVG instead of matplotlib
artists. The page is a single static HTML file with inline CSS/JS (no
network dependency, nothing uploaded anywhere) -- open it directly in a
browser to pan (click-drag), zoom (scroll / pinch), and -- on the scenario
export -- hover any node/edge to highlight the whole cycle it belongs to.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from gr_cn.models.node import EBGNode, NodeType
from gr_cn.viz import colors as pal
from gr_cn.viz.genomes import full_genome_steps
from gr_cn.viz.layout import StepLayout, compute_layout, edge_curve_offset
from gr_cn.viz.model import VizStep, build_cycle_infos, build_viz_steps, remap_cycles_to_nodes, step_label
from gr_cn.viz.pipeline import PipelineResult, run_pipeline

UNIT = 92.0
NODE_R = 15.0
SAT_W, SAT_H = 60.0, 28.0
CAP_S = 20.0
CAP_GAP = 40.0
PANEL_GAP = 90.0
MARGIN = 60.0


def _node_label(node: EBGNode) -> str:
    side = node.end.value
    if node.node_type is NodeType.DC:
        return f"{side.upper()}_DC"
    if node.node_type is NodeType.AH:
        return f"{side.upper()}_AH"
    suffix = f".{node.copy_index}" if node.copy_index else ""
    return f"{side}{node.iid}{suffix}"


def _node_svg(node: EBGNode, x: float, y: float, chrom_colors: Dict[int, str], cyc: str = "") -> str:
    label = escape(_node_label(node))
    if node.node_type in (NodeType.DC, NodeType.AH):
        return (
            f'<g class="node"{cyc}><rect class="sat" x="{x - SAT_W / 2:.1f}" y="{y - SAT_H / 2:.1f}" '
            f'width="{SAT_W:.0f}" height="{SAT_H:.0f}" rx="5"/>'
            f'<text class="lbl-sat" x="{x:.1f}" y="{y + 4:.1f}">{label}</text></g>'
        )
    fill = chrom_colors.get(node.ref_chromosome, pal.INK_MUTED)
    lw = 2.6 if node.haplotype else 1.4
    return (
        f'<g class="node"{cyc}><circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_R:.0f}" '
        f'fill="{fill}" stroke-width="{lw}"/>'
        f'<text class="lbl" x="{x:.1f}" y="{y + 4:.1f}">{label}</text></g>'
    )


def _edge_css_class(kind: str) -> str:
    """Mirrors gr_cn.viz.draw._edge_style's genome-provenance encoding."""
    base_kind = kind.split(":", 1)[0]
    if base_kind == "grey-pending":
        return "edge pending"
    if base_kind == "grey-resolved":
        return "edge resolved"
    if base_kind == "black":
        return "edge black"
    return "edge"


def _curve_control_point(x0: float, y0: float, x1: float, y1: float, offset: float) -> Tuple[float, float]:
    """Perpendicular-offset control point at the edge's midpoint (mirrors gr_cn.viz.draw)."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if not offset:
        return mx, my
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return mx, my
    px, py = -dy / length, dx / length
    return mx + px * offset, my + py * offset


def _edge_svg(x0: float, y0: float, x1: float, y1: float, offset_px: float, cls: str, cyc: str) -> Tuple[str, float, float]:
    """Returns (svg_element, midpoint_x, midpoint_y) -- straight <line> when offset_px is 0, else a quadratic Bezier <path>."""
    if not offset_px:
        return f'<line class="{cls}"{cyc} x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>', (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy = _curve_control_point(x0, y0, x1, y1, offset_px)
    return f'<path class="{cls}"{cyc} d="M {x0:.1f},{y0:.1f} Q {cx:.1f},{cy:.1f} {x1:.1f},{y1:.1f}"/>', cx, cy


def _cap_svg(node: EBGNode, x: float, y: float, direction: float, chrom_colors: Dict[int, str], cyc: str = "") -> str:
    fill = chrom_colors.get(node.ref_chromosome, pal.INK_MUTED)
    cy = y + direction * CAP_GAP
    s = CAP_S
    return (
        f'<g class="cap"{cyc}>'
        f'<line x1="{x:.1f}" y1="{y + direction * NODE_R:.1f}" x2="{x:.1f}" y2="{cy - direction * s / 2:.1f}"/>'
        f'<rect x="{x - s / 2:.1f}" y="{cy - s / 2:.1f}" width="{s:.0f}" height="{s:.0f}" fill="{fill}"/>'
        f'<text class="lbl-cap" x="{x:.1f}" y="{cy + 4:.1f}">{node.end.value.upper()}</text></g>'
    )


def render_step_svg(
    step: VizStep,
    layout: StepLayout,
    chrom_colors: Dict[int, str],
    title: str = "",
    x_offset: float = 0.0,
    node_cycle: Optional[Dict[int, str]] = None,
) -> Tuple[str, float, float, float, float]:
    """
    Render one VizStep/StepLayout to an SVG fragment (a <g> element, meant to
    be embedded inside a parent <svg>), placed with its left edge at
    x_offset. Returns (svg_fragment, left, top, right, bottom) in px -- the
    fragment's own bounding box (including its title) -- so callers can lay
    panels out side by side/stacked and size the overall canvas.

    node_cycle, if given, maps id(EBGNode) -> owning raw-cycle sub-label
    (e.g. "T1a"); every element touching a node is tagged
    data-cyc="<labels>" so the page's hover script can highlight everything
    sharing a cycle.
    """
    node_pos = layout.node_pos
    if not node_pos:
        return '<g class="panel"></g>', x_offset, 0.0, x_offset + 200, 100.0

    node_cycle = node_cycle or {}

    def cyc_attr(*node_ids: int) -> str:
        cycs = sorted({node_cycle[nid] for nid in node_ids if nid in node_cycle})
        return f' data-cyc="{escape(" ".join(cycs))}"' if cycs else ""

    def P(nid: int) -> Tuple[float, float]:
        x, y = node_pos[nid]
        return x * UNIT + x_offset, y * UNIT

    non_base_connected = {
        id(u) for u, v, kind in step.active_edges if kind != "base"
    } | {
        id(v) for u, v, kind in step.active_edges if kind != "base"
    }
    satellite_anchor_ids = {id(a) for sat in layout.satellites for a in sat.anchors}

    edge_parts: List[str] = []
    for u, v, kind in step.active_edges:
        if id(u) not in node_pos or id(v) not in node_pos:
            continue
        x0, y0 = P(id(u))
        x1, y1 = P(id(v))
        # base/shared edges are the fixed backbone, not any one cycle's own
        # edge -- their two endpoints can legitimately belong to two
        # different cycles (e.g. a segment's tail and head broken into
        # separate rearrangements), so tagging them would make hovering
        # highlight every cycle they happen to touch instead of just one.
        cyc = "" if kind in ("base", "shared") else cyc_attr(id(u), id(v))
        offset_px = edge_curve_offset(layout, u, v) * UNIT
        svg, _mx, _my = _edge_svg(x0, y0, x1, y1, offset_px, _edge_css_class(kind), cyc)
        edge_parts.append(svg)

    tag_parts: List[str] = []
    for u, v, kind in step.pending_edges:
        if id(u) not in node_pos or id(v) not in node_pos:
            continue
        x0, y0 = P(id(u))
        x1, y1 = P(id(v))
        cyc = cyc_attr(id(u), id(v))
        offset_px = edge_curve_offset(layout, u, v) * UNIT
        svg, mx, my = _edge_svg(x0, y0, x1, y1, offset_px, _edge_css_class(kind), cyc)
        edge_parts.append(svg)
        tag = kind.split(":", 1)[1] if ":" in kind else ""
        if tag:
            tag_parts.append(
                f'<g class="tag"{cyc}><rect x="{mx - 13:.1f}" y="{my - 11:.1f}" width="26" height="22" rx="4"/>'
                f'<text x="{mx:.1f}" y="{my + 5:.1f}">{escape(tag)}</text></g>'
            )

    node_parts: List[str] = []
    for chain in layout.chains:
        for nl in chain.node_layouts:
            x, y = P(id(nl.node))
            node_parts.append(_node_svg(nl.node, x, y, chrom_colors, cyc_attr(id(nl.node))))
        first, last = chain.node_layouts[0], chain.node_layouts[-1]
        if not chain.is_circular:
            if id(first.node) not in non_base_connected and id(first.node) not in satellite_anchor_ids:
                x, y = P(id(first.node))
                node_parts.append(_cap_svg(first.node, x, y, -1.0, chrom_colors, cyc_attr(id(first.node))))
            if len(chain.node_layouts) > 1 and (
                id(last.node) not in non_base_connected and id(last.node) not in satellite_anchor_ids
            ):
                x, y = P(id(last.node))
                node_parts.append(_cap_svg(last.node, x, y, 1.0, chrom_colors, cyc_attr(id(last.node))))

    for sat in layout.satellites:
        cyc = cyc_attr(id(sat.tail_node), id(sat.head_node))
        x, y = P(id(sat.tail_node))
        node_parts.append(_node_svg(sat.tail_node, x, y, chrom_colors, cyc))
        x, y = P(id(sat.head_node))
        node_parts.append(_node_svg(sat.head_node, x, y, chrom_colors, cyc))

    xs = [node_pos[i][0] * UNIT + x_offset for i in node_pos]
    ys = [node_pos[i][1] * UNIT for i in node_pos]
    left, right = min(xs) - 70, max(xs) + 70
    node_top, bottom = min(ys) - 90, max(ys) + 70
    title_y = node_top - 30
    top = title_y - 20 if title else node_top

    title_svg = ""
    if title:
        title_svg = f'<text class="panel-title" x="{(left + right) / 2:.1f}" y="{title_y:.1f}">{escape(title)}</text>'

    fragment = (
        f'<g class="panel">{title_svg}'
        + "".join(edge_parts) + "".join(node_parts) + "".join(tag_parts) +
        "</g>"
    )
    return fragment, left, top, right, bottom


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --ink: {ink_primary}; --ink2: {ink_secondary}; --muted: {ink_muted};
    --surface: {surface}; --dc-fill: {dc_fill}; --dc-edge: {dc_edge};
    --grid: {gridline};
  }}
  html, body {{ margin:0; padding:0; height:100%; background:var(--surface);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; overflow:hidden; }}
  #hud {{ position:fixed; top:10px; left:12px; z-index:10; font-size:13px; color:var(--ink2);
    background:rgba(255,255,255,0.82); padding:6px 10px; border-radius:6px; pointer-events:none; }}
  #legend {{ position:fixed; bottom:10px; left:12px; z-index:10; font-size:13px; color:var(--ink);
    background:rgba(255,255,255,0.9); padding:8px 12px; border-radius:6px; display:flex; gap:14px; flex-wrap:wrap; max-width:90vw; }}
  #legend .item {{ display:flex; align-items:center; gap:6px; }}
  #legend .swatch {{ width:12px; height:12px; border-radius:50%; border:1px solid var(--ink); display:inline-block; }}
  #legend .line-swatch {{ width:20px; height:0; border-top-width:2px; border-top-style:solid; display:inline-block; }}
  #viewport {{ width:100vw; height:100vh; cursor:grab; touch-action:none; }}
  #viewport.grabbing {{ cursor:grabbing; }}
  svg {{ width:100%; height:100%; display:block; }}
  .node circle {{ stroke: var(--ink); }}
  .node .lbl {{ fill:#fff; font-size:12px; font-weight:600; text-anchor:middle; pointer-events:none; }}
  .node .sat {{ fill: var(--dc-fill); stroke: var(--dc-edge); stroke-width:1.2; }}
  .node .lbl-sat {{ fill: var(--ink2); font-size:11px; text-anchor:middle; pointer-events:none; }}
  .edge {{ stroke: var(--ink); stroke-width:1.4; fill: none; }}
  .edge.black {{ stroke: var(--ink2); }}
  .edge.resolved {{ stroke: var(--muted); stroke-width:1.9; }}
  .edge.pending {{ stroke: var(--muted); stroke-dasharray:7,5; }}
  .cap line {{ stroke: var(--ink); stroke-width:1.4; }}
  .cap rect {{ stroke: var(--ink); stroke-width:1.4; }}
  .cap .lbl-cap {{ fill:#fff; font-size:11px; font-weight:700; text-anchor:middle; pointer-events:none; }}
  .tag rect {{ fill: var(--surface); opacity:0.9; }}
  .tag text {{ fill: var(--ink2); font-size:12px; text-anchor:middle; pointer-events:none; font-weight:600; }}
  .panel-title {{ fill: var(--ink); font-size:18px; text-anchor:middle; }}
  .arrow {{ stroke: var(--ink); stroke-width:2.4; fill:none; marker-end:url(#arrowhead); }}
  .arrow-label {{ fill: var(--ink); font-size:15px; font-weight:700; text-anchor:middle; }}
  .suptitle {{ fill: var(--ink); font-size:20px; text-anchor:middle; }}
  [data-cyc] {{ transition: opacity 0.08s linear; cursor: pointer; }}
  #world.hovering [data-cyc] {{ opacity: 0.18; }}
  #world.hovering [data-cyc].hi {{ opacity: 1; }}
  #world.hovering .hi.edge {{ stroke-width: 3; }}
  #world.hovering .hi.node circle {{ stroke-width: 3.4; }}
</style>
</head>
<body>
<div id="hud">scroll / pinch to zoom &middot; drag to pan &middot; double-click to reset{hover_hint}</div>
<div id="viewport">
  <svg id="canvas" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 Z" fill="{ink_primary}"/>
      </marker>
    </defs>
    <g id="world">
{body}
    </g>
  </svg>
</div>
{legend}
<script>
(function() {{
  var world = document.getElementById('world');
  var viewport = document.getElementById('viewport');
  var contentW = {content_w}, contentH = {content_h};
  var scale = 1, tx = 0, ty = 0;

  function fit() {{
    var vw = viewport.clientWidth, vh = viewport.clientHeight;
    scale = Math.min(vw / contentW, vh / contentH) * 0.94;
    tx = (vw - contentW * scale) / 2;
    ty = (vh - contentH * scale) / 2;
    apply();
  }}
  function apply() {{
    world.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')');
  }}
  viewport.addEventListener('wheel', function(e) {{
    e.preventDefault();
    var rect = viewport.getBoundingClientRect();
    var mx = e.clientX - rect.left, my = e.clientY - rect.top;
    var factor = Math.exp(-e.deltaY * 0.0015);
    var newScale = Math.min(Math.max(scale * factor, 0.02), 40);
    var wx = (mx - tx) / scale, wy = (my - ty) / scale;
    tx = mx - wx * newScale;
    ty = my - wy * newScale;
    scale = newScale;
    apply();
  }}, {{ passive: false }});
  var dragging = false, moved = false, lastX = 0, lastY = 0;
  viewport.addEventListener('mousedown', function(e) {{
    dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
    viewport.classList.add('grabbing');
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    tx += e.clientX - lastX; ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    moved = true;
    apply();
  }});
  window.addEventListener('mouseup', function() {{
    dragging = false; viewport.classList.remove('grabbing');
  }});
  viewport.addEventListener('dblclick', fit);
  window.addEventListener('resize', fit);

  // Hover any node/edge/tag to highlight every element sharing a cycle
  // index with it (data-cyc="i j ..."), dimming everything else.
  var cycIndex = {{}};
  Array.prototype.forEach.call(document.querySelectorAll('[data-cyc]'), function(el) {{
    el.getAttribute('data-cyc').split(' ').forEach(function(c) {{
      (cycIndex[c] = cycIndex[c] || []).push(el);
    }});
  }});
  var hasCycles = Object.keys(cycIndex).length > 0;
  if (hasCycles) {{
    world.addEventListener('mouseover', function(e) {{
      var el = e.target.closest('[data-cyc]');
      if (!el) return;
      world.classList.add('hovering');
      var cycs = el.getAttribute('data-cyc').split(' ');
      var hit = new Set();
      cycs.forEach(function(c) {{ (cycIndex[c] || []).forEach(function(x) {{ hit.add(x); }}); }});
      hit.forEach(function(x) {{ x.classList.add('hi'); }});
    }});
    world.addEventListener('mouseout', function(e) {{
      var el = e.target.closest('[data-cyc]');
      if (!el) return;
      world.classList.remove('hovering');
      Array.prototype.forEach.call(document.querySelectorAll('.hi'), function(x) {{ x.classList.remove('hi'); }});
    }});
  }}

  fit();
}})();
</script>
</body>
</html>
"""


def _legend_html(chrom_colors: Dict[int, str]) -> str:
    items = "".join(
        f'<span class="item"><span class="swatch" style="background:{color}"></span>chr {chrom}</span>'
        for chrom, color in chrom_colors.items()
    )
    items += (
        f'<span class="item"><span class="swatch" style="background:{pal.DC_FILL};'
        f'border-radius:3px;"></span>DC / AH (deleted / gained copy)</span>'
    )
    for label, color, style in (
        ("shared (both genomes)", pal.INK_PRIMARY, "solid"),
        ("genome A only", pal.INK_SECONDARY, "solid"),
        ("genome B only (resolved)", pal.INK_MUTED, "solid"),
        ("genome B only (pending)", pal.INK_MUTED, "dashed"),
    ):
        items += (
            f'<span class="item"><span class="line-swatch" '
            f'style="border-top-color:{color};border-top-style:{style};"></span>{escape(label)}</span>'
        )
    return f'<div id="legend">{items}</div>'


def _write_page(title: str, body: str, content_w: float, content_h: float,
                 legend_html: str, out_path: Path, hover_hint: str = "") -> Path:
    html = _PAGE_TEMPLATE.format(
        title=escape(title), body=body, content_w=f"{content_w:.0f}", content_h=f"{content_h:.0f}",
        legend=legend_html, hover_hint=hover_hint,
        ink_primary=pal.INK_PRIMARY, ink_secondary=pal.INK_SECONDARY, ink_muted=pal.INK_MUTED,
        surface=pal.SURFACE, dc_fill=pal.DC_FILL, dc_edge=pal.DC_EDGE, gridline=pal.GRIDLINE,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _chrom_colors(chromosomes: Optional[Sequence[int]], pr: PipelineResult) -> Dict[int, str]:
    if chromosomes is None:
        chromosomes = sorted({iv.chromosome for iv in pr.graph.intervals})
    return pal.chromosome_colors(sorted(chromosomes))


def export_genome_html(
    jabba_path: str,
    use_grey: bool,
    chromosomes: Optional[Sequence[int]] = None,
    pr: Optional[PipelineResult] = None,
    out_path: Optional[str] = None,
    **pipeline_kwargs,
) -> str:
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    layout_step, draw_step_obj = full_genome_steps(pr.raw_nodes, use_grey=use_grey)
    layout = compute_layout(layout_step)
    label = "Genome B (phased tumor)" if use_grey else "Genome A (diploid reference)"
    title = f"{label} -- {pr.graph.sample_name}"
    fragment, left, top, right, bottom = render_step_svg(
        draw_step_obj, layout, chrom_colors, title=title,
    )
    content_w = right - left + 2 * MARGIN
    content_h = bottom - top + 2 * MARGIN
    body = f'<g transform="translate({MARGIN - left:.1f},{MARGIN - top:.1f})">{fragment}</g>'
    suffix = "genome_b" if use_grey else "genome_a"
    out = Path(out_path) if out_path else Path(f"{pr.graph.sample_name}_{suffix}.html")
    _write_page(title, body, content_w, content_h, _legend_html(chrom_colors), out)
    return str(out)


def export_genome_a_html(jabba_path: str, chromosomes: Optional[Sequence[int]] = None,
                          pr: Optional[PipelineResult] = None, out_path: Optional[str] = None,
                          **pipeline_kwargs) -> str:
    return export_genome_html(jabba_path, False, chromosomes, pr, out_path, **pipeline_kwargs)


def export_genome_b_html(jabba_path: str, chromosomes: Optional[Sequence[int]] = None,
                          pr: Optional[PipelineResult] = None, out_path: Optional[str] = None,
                          **pipeline_kwargs) -> str:
    return export_genome_html(jabba_path, True, chromosomes, pr, out_path, **pipeline_kwargs)


def export_scenario_html(
    jabba_path: str,
    chromosomes: Optional[Sequence[int]] = None,
    pr: Optional[PipelineResult] = None,
    out_path: Optional[str] = None,
    **pipeline_kwargs,
) -> str:
    pr = pr or run_pipeline(jabba_path, chromosomes=chromosomes, **pipeline_kwargs)
    chrom_colors = _chrom_colors(chromosomes, pr)
    cycle_infos = build_cycle_infos(pr.all_cycles)

    suptitle = (
        f"{pr.graph.sample_name} -- scenario ({pr.scenario.method}, "
        f"max chromosomes touched = {pr.scenario.objective_value})"
    )

    if not pr.all_cycles:
        body = '<text class="suptitle" x="200" y="60">No rearrangement cycles in this selection.</text>'
        out = Path(out_path) if out_path else Path(f"{pr.graph.sample_name}_scenario.html")
        _write_page(suptitle, body, 500, 150, "", out)
        return str(out)

    # render_step_svg draws pr.raw_nodes (via build_viz_steps' all_nodes=),
    # not the cycle-extraction node objects cycle_infos above was built
    # from -- rebuild cycle_infos on raw-node identity so hover-highlight
    # data-cyc attributes match the nodes actually on the page.
    cycles_on_raw = remap_cycles_to_nodes(pr.all_cycles, pr.raw_nodes)
    cycle_infos_raw = build_cycle_infos(cycles_on_raw)
    node_cycle: Dict[int, str] = {
        id(n): info.node_label[id(n)] for info in cycle_infos_raw for n in info.nodes
    }

    steps = build_viz_steps(pr.all_cycles, pr.scenario.partition, all_nodes=pr.raw_nodes)
    layouts = [compute_layout(s) for s in steps]

    fragments = []
    max_span = 0.0
    for i, (step, layout) in enumerate(zip(steps, layouts)):
        title = "Initial (EBG)" if i == 0 else f"After step {i}"
        fragment, left, top, right, bottom = render_step_svg(
            step, layout, chrom_colors, title=title, node_cycle=node_cycle,
        )
        fragments.append((fragment, left, top, right, bottom))
        max_span = max(max_span, bottom - top)

    # Each panel is top-aligned to y=0 in its own frame (matplotlib instead
    # gives each panel an independent auto-centered axes box; top-aligning
    # is the simplest SVG equivalent and keeps every panel's title level).
    parts: List[str] = []
    arrows: List[str] = []
    x_cursor = MARGIN
    panel_edges: List[Tuple[float, float]] = []
    for fragment, left, top, right, bottom in fragments:
        width = right - left
        shift_x = x_cursor - left
        shift_y = -top
        parts.append(f'<g transform="translate({shift_x:.1f},{shift_y:.1f})">{fragment}</g>')
        panel_edges.append((x_cursor, x_cursor + width))
        x_cursor += width + PANEL_GAP

    total_w = x_cursor - PANEL_GAP + MARGIN
    y_mid = max_span / 2

    for i in range(len(steps) - 1):
        x0 = panel_edges[i][1]
        x1 = panel_edges[i + 1][0]
        label = step_label(pr.scenario.partition[i], cycle_infos)
        arrows.append(
            f'<path class="arrow" d="M {x0 + 10:.1f},{y_mid:.1f} L {x1 - 14:.1f},{y_mid:.1f}"/>'
            f'<text class="arrow-label" x="{(x0 + x1) / 2:.1f}" y="{y_mid - 14:.1f}">{escape(label)}</text>'
        )

    suptitle_h = 60.0
    body = (
        f'<text class="suptitle" x="{total_w / 2:.1f}" y="30">{escape(suptitle)}</text>'
        f'<g transform="translate(0,{suptitle_h:.1f})">' + "".join(parts) + "".join(arrows) + "</g>"
    )
    total_h = max_span + suptitle_h + MARGIN

    out = Path(out_path) if out_path else Path(f"{pr.graph.sample_name}_scenario.html")
    _write_page(
        suptitle, body, total_w, total_h, _legend_html(chrom_colors), out,
        hover_hint=" &middot; hover a node/edge to highlight its cycle",
    )
    return str(out)
