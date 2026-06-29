"""Kernel Graph — a NetworkX DiGraph over knowledge components (KCs).

Edge direction convention: prerequisite -> concept.
So `graph.predecessors(concept)` yields that concept's prerequisites, which is
exactly what the root-cause DFS walks.
"""
from __future__ import annotations

import networkx as nx


def build_graph(nodes: list[dict], edges: list[dict]) -> nx.DiGraph:
    """Build a DiGraph from concept_nodes + concept_edges rows.

    Nodes are keyed by their `label` (snake_case, unique per subject in practice).
    Each node keeps a copy of its DB row in attributes. Each edge points from the
    prerequisite to the dependent concept.

    Args:
        nodes: rows from kernel.concept_nodes.
        edges: rows from kernel.concept_edges, with concept_id/prerequisite_id.
    """
    graph = nx.DiGraph()

    # Map id -> label so edges (stored by id) can be wired by label.
    id_to_label: dict[str, str] = {}
    for node in nodes:
        label = node["label"]
        id_to_label[node["id"]] = label
        graph.add_node(label, **node)

    for edge in edges:
        prereq_label = id_to_label.get(edge["prerequisite_id"])
        concept_label = id_to_label.get(edge["concept_id"])
        if prereq_label is None or concept_label is None:
            # Dangling edge (node missing); skip rather than crash.
            continue
        graph.add_edge(prereq_label, concept_label, weight=edge.get("weight", 1.0))

    return graph


def node_id(graph: nx.DiGraph, label: str) -> str | None:
    """Return the DB id for a node label, or None if absent."""
    if label in graph.nodes:
        return graph.nodes[label].get("id")
    return None
