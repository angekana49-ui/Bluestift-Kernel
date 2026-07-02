"""Generate a standalone interactive HTML visualization of the Kernel Graph.

Pulls all concept_nodes + concept_edges from Supabase and renders a force-directed
network with vis-network (loaded from CDN — the output is a single self-contained
HTML file). Nodes are coloured by subject and sized by how many concepts depend on
them (foundational concepts are bigger); cross-subject *bridge* edges are drawn in
red so they pop.

Usage:
    python scripts/visualize_graph.py            # -> graph_viz.html
    python scripts/visualize_graph.py out.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Subject -> node colour.
SUBJECT_COLORS = {
    "MATH": "#4C9BE8",
    "PHYSICS": "#E8954C",
    "ENGLISH": "#7BC86C",
    "CHEMISTRY": "#B07BE8",
    "HISTORY": "#E8C84C",
    "BIOLOGY": "#4CC8B0",
}
DEFAULT_COLOR = "#9aa5b1"
CROSS_EDGE_COLOR = "#E8454C"      # cross-subject bridges — red, bold
WITHIN_EDGE_COLOR = "#c7ced6"     # within-subject — light grey


def build_html(nodes: list[dict], edges: list[dict]) -> str:
    by_id = {n["id"]: n for n in nodes}

    # Out-degree (how many concepts list this node as a prerequisite) -> size.
    feeds: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        if e["prerequisite_id"] in feeds:
            feeds[e["prerequisite_id"]] += 1

    vis_nodes = []
    subjects = sorted({n["subject"] for n in nodes})
    for n in nodes:
        color = SUBJECT_COLORS.get(n["subject"], DEFAULT_COLOR)
        size = 10 + 3 * feeds.get(n["id"], 0)
        vis_nodes.append(
            {
                "id": n["id"],
                "label": n["label"],
                "group": n["subject"],
                "color": color,
                "size": size,
                "title": f"{n['label']}  ·  {n['subject']}  ·  {n.get('level', '')}",
            }
        )

    vis_edges = []
    cross_count = 0
    for e in edges:
        src, dst = by_id.get(e["prerequisite_id"]), by_id.get(e["concept_id"])
        if not src or not dst:
            continue
        is_cross = src["subject"] != dst["subject"]
        cross_count += is_cross
        vis_edges.append(
            {
                "from": e["prerequisite_id"],
                "to": e["concept_id"],
                "color": CROSS_EDGE_COLOR if is_cross else WITHIN_EDGE_COLOR,
                "width": 3 if is_cross else 1,
            }
        )

    legend = " ".join(
        f'<span style="color:{SUBJECT_COLORS.get(s, DEFAULT_COLOR)}">&#9679;</span> {s}'
        for s in subjects
    )

    html = _TEMPLATE
    replacements = {
        "__NODES__": json.dumps(vis_nodes, ensure_ascii=False),
        "__EDGES__": json.dumps(vis_edges, ensure_ascii=False),
        "__LEGEND__": legend,
        "__NNODES__": str(len(vis_nodes)),
        "__NEDGES__": str(len(vis_edges)),
        "__NCROSS__": str(cross_count),
        "__CROSSCOLOR__": CROSS_EDGE_COLOR,
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Bluestift Kernel Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#0f141a;font-family:system-ui,Arial}
  #net{width:100%;height:100vh}
  #hud{position:fixed;top:12px;left:12px;color:#e6edf3;background:#1b232c;
       padding:12px 16px;border-radius:10px;font-size:13px;line-height:1.7;
       box-shadow:0 4px 20px rgba(0,0,0,.4)}
  #hud b{font-size:15px}
  .cross{color:__CROSSCOLOR__;font-weight:600}
</style></head>
<body>
<div id="hud">
  <b>Bluestift Cognitive Kernel &mdash; Knowledge Graph</b><br>
  __NNODES__ concepts &middot; __NEDGES__ prerequis &middot; <span class="cross">__NCROSS__ ponts inter-matieres</span><br>
  __LEGEND__ &nbsp;|&nbsp; <span class="cross">&#9644;</span> pont cross-subject<br>
  <small>taille = nb de concepts qui en dependent (fondamentaux = plus gros)</small>
</div>
<div id="net"></div>
<script>
  const nodes = new vis.DataSet(__NODES__);
  const edges = new vis.DataSet(__EDGES__);
  new vis.Network(document.getElementById('net'), {nodes, edges}, {
    nodes:{shape:'dot',font:{color:'#cfd8e3',size:12},borderWidth:0},
    edges:{arrows:{to:{enabled:true,scaleFactor:0.4}},smooth:{type:'continuous'}},
    physics:{solver:'forceAtlas2Based',
             forceAtlas2Based:{gravitationalConstant:-45,springLength:110,springConstant:0.05},
             stabilization:{iterations:250}},
    interaction:{hover:true,tooltipDelay:120}
  });
</script>
</body></html>
"""


def main() -> None:
    from services import db

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("graph_viz.html")
    client = db.get_client()
    nodes = db.load_concept_nodes(client)
    edges = db.load_concept_edges(client)
    html = build_html(nodes, edges)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} — {len(nodes)} nodes, {len(edges)} edges. Open it in a browser.")


if __name__ == "__main__":
    main()
