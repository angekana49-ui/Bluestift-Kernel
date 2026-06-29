"""Root-cause detection via DFS over the Kernel Graph.

Given a set of failing KCs, walk the prerequisite chain (predecessors) downward
to find the deepest prerequisite that is itself failing. That deepest failing
node is the *root gap*: the place where remediation should start.
"""
from __future__ import annotations

import networkx as nx

# A concept is considered "failing" when its effective mastery is below this.
FAILING_THRESHOLD = 0.5


def detect_root_cause(
    graph: nx.DiGraph,
    failing_kcs: list[str],
    concept_states: dict[str, float],
) -> dict:
    """Find the root gap among failing KCs.

    Args:
        graph: the Kernel Graph (prerequisite -> concept).
        failing_kcs: labels the student is currently failing.
        concept_states: label -> effective mastery in [0, 1].

    Returns:
        dict with root_gap, detection_path (surface -> root), and confidence.
    """
    root_gap = None
    detection_path: list[str] = []

    for kc in failing_kcs:
        if kc not in graph:
            continue
        path = _dfs_find_root(graph, kc, concept_states, visited=set())
        # Prefer the longest chain — it digs to the deepest underlying gap.
        if path and len(path) > len(detection_path):
            detection_path = path
            root_gap = path[-1]

    return {
        "root_gap": root_gap,
        "detection_path": detection_path,
        "confidence": _compute_confidence(detection_path, concept_states),
    }


def _dfs_find_root(
    graph: nx.DiGraph,
    node: str,
    states: dict[str, float],
    visited: set,
) -> list[str]:
    """Walk prerequisites; return the chain to the deepest failing prerequisite."""
    if node in visited:
        return []
    visited.add(node)

    prerequisites = list(graph.predecessors(node))
    if not prerequisites:
        return [node]

    # Descend into the weakest failing prerequisite chain.
    best_chain: list[str] = []
    for prereq in prerequisites:
        if states.get(prereq, 0.5) < FAILING_THRESHOLD:
            deeper = _dfs_find_root(graph, prereq, states, visited)
            if len(deeper) > len(best_chain):
                best_chain = deeper

    if best_chain:
        return [node] + best_chain
    return [node]


def _compute_confidence(detection_path: list[str], states: dict[str, float]) -> float:
    """Confidence that the root gap is real.

    Higher when (a) the chain is long (consistent prerequisite weakness) and
    (b) the root node's mastery is clearly low. Bounded to [0, 1].
    """
    if not detection_path:
        return 0.0

    root = detection_path[-1]
    root_mastery = states.get(root, 0.5)

    # Depth signal: each extra hop adds evidence, saturating around 3 hops.
    depth_signal = min(1.0, len(detection_path) / 3.0)
    # Severity signal: the lower the root mastery, the more confident.
    severity_signal = max(0.0, 1.0 - root_mastery)

    confidence = 0.5 * depth_signal + 0.5 * severity_signal
    return round(min(1.0, max(0.0, confidence)), 4)


def recommended_path(graph: nx.DiGraph, root_gap: str | None, max_len: int = 4) -> list[str]:
    """Suggest a remediation order: root gap first, then its dependents upward.

    A short forward walk from the root gap through the concepts that build on it,
    giving RAYA a concrete sequence to rebuild from the foundation.
    """
    if not root_gap or root_gap not in graph:
        return [] if not root_gap else [root_gap]

    path = [root_gap]
    current = root_gap
    visited = {root_gap}
    while len(path) < max_len:
        successors = [s for s in graph.successors(current) if s not in visited]
        if not successors:
            break
        # Pick the dependent with the most prerequisites satisfied (lowest fan-in
        # of unmet deps is hard to know here, so take the first deterministic one).
        nxt = sorted(successors)[0]
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path
