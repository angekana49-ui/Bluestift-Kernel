"""Root-cause detection via DFS over the Kernel Graph.

Given a set of failing KCs, walk the prerequisite chain (predecessors) downward
to find the deepest prerequisite that is itself failing. That deepest failing
node is the *root gap*: the place where remediation should start.
"""
from __future__ import annotations

import networkx as nx

# A concept is considered "failing" when its effective mastery is below this.
FAILING_THRESHOLD = 0.5


# How many consecutive *unknown* (never-practised) prerequisites the search may
# bridge before stopping. Lets it cross small evidence gaps to reach a deeper
# weak concept, without bottoming out at the most elementary leaf every time.
UNKNOWN_BUDGET = 2


def detect_root_cause(
    graph: nx.DiGraph,
    failing_kcs: list[str],
    concept_states: dict[str, float],
    known: set[str] | None = None,
) -> dict:
    """Find the root gap among failing KCs.

    The root is chosen by **convergence** first, not by chain length: a weak or
    unverified concept that many failing KCs depend on is the strongest
    explanation for the struggle. Ties break on evidence (a known-weak concept
    beats a merely-unknown one), then on depth (more foundational).

    Args:
        graph: the Kernel Graph (prerequisite -> concept).
        failing_kcs: labels the student is currently failing.
        concept_states: label -> effective mastery in [0, 1].
        known: labels with actual student evidence. Labels absent from this set
            are treated as *unknown* (never practised), so the search descends
            into them as suspects. When None, every label in concept_states is
            considered known.

    Returns:
        dict with root_gap, detection_path (surface -> root), and confidence.
    """
    if known is None:
        known = set(concept_states.keys())

    failing_in_graph = [kc for kc in failing_kcs if kc in graph]
    if not failing_in_graph:
        return {"root_gap": None, "detection_path": [], "confidence": 0.0}

    # Each failing KC -> its chain down to a deepest weak/unknown concept.
    chains: dict[str, list[str]] = {
        kc: _dfs_find_root(graph, kc, concept_states, known, set(), UNKNOWN_BUDGET)
        for kc in failing_in_graph
    }

    # Candidate roots: every concept appearing on any failing chain.
    candidates: set[str] = set()
    for chain in chains.values():
        candidates.update(chain)
    if not candidates:
        return {"root_gap": None, "detection_path": [], "confidence": 0.0}

    def convergence(r: str) -> int:
        # How many failing KCs depend (transitively) on r — the core signal.
        return sum(
            1 for kc in failing_in_graph if kc == r or nx.has_path(graph, r, kc)
        )

    def has_evidence(r: str) -> int:
        return 1 if (r in known and concept_states.get(r, 0.5) < FAILING_THRESHOLD) else 0

    def depth(r: str) -> int:
        return max((len(ch) for ch in chains.values() if ch and ch[-1] == r), default=1)

    root_gap = max(candidates, key=lambda r: (convergence(r), has_evidence(r), depth(r)))

    # detection_path: a real surface -> root prerequisite chain.
    detection_path = _path_to_root(graph, chains, root_gap, failing_in_graph)

    return {
        "root_gap": root_gap,
        "detection_path": detection_path,
        "confidence": _compute_confidence(
            detection_path, concept_states, convergence(root_gap), len(failing_in_graph)
        ),
    }


def _path_to_root(
    graph: nx.DiGraph,
    chains: dict[str, list[str]],
    root: str,
    failing: list[str],
) -> list[str]:
    """Build a surface -> root chain for the chosen root.

    Prefers a DFS chain that already ends at the root; otherwise reconstructs the
    prerequisite chain from the root up to the failing KC it best explains (the
    one giving the deepest chain), using the graph.
    """
    # A real chain has length > 1; the root's own length-1 self-chain doesn't count.
    ending = [ch for ch in chains.values() if len(ch) > 1 and ch[-1] == root]
    if ending:
        return max(ending, key=len)

    best = [root]
    for kc in failing:
        if kc != root and nx.has_path(graph, root, kc):
            # shortest_path gives root..kc (prereq -> concept); reverse to surface -> root.
            chain = list(reversed(nx.shortest_path(graph, root, kc)))
            if len(chain) > len(best):
                best = chain
    return best


def _dfs_find_root(
    graph: nx.DiGraph,
    node: str,
    states: dict[str, float],
    known: set[str],
    visited: set,
    budget: int,
) -> list[str]:
    """Walk prerequisites to the deepest weak-or-unverified concept.

    A prerequisite is descended into when it is either known-weak (mastery below
    the failing threshold) or unknown (no evidence). Known-mastered prerequisites
    act as a barrier — the gap is above them. The `budget` caps how many unknown
    prerequisites in a row may be crossed; it refills whenever the search lands on
    a known-weak concept (fresh evidence).
    """
    if node in visited:
        return []
    visited.add(node)

    prerequisites = list(graph.predecessors(node))
    if not prerequisites:
        return [node]

    best_chain: list[str] = []
    for prereq in prerequisites:
        is_known = prereq in known
        mastery = states.get(prereq, 0.5)

        if is_known and mastery >= FAILING_THRESHOLD:
            continue  # mastered prerequisite -> barrier, don't descend
        if is_known:
            # Known-weak: strong evidence, descend and refill the unknown budget.
            deeper = _dfs_find_root(graph, prereq, states, known, visited, UNKNOWN_BUDGET)
        else:
            # Unknown: a suspected gap, descend only while budget remains.
            if budget <= 0:
                continue
            deeper = _dfs_find_root(graph, prereq, states, known, visited, budget - 1)

        if len(deeper) > len(best_chain):
            best_chain = deeper

    if best_chain:
        return [node] + best_chain
    return [node]


def _compute_confidence(
    detection_path: list[str],
    states: dict[str, float],
    convergence: int = 1,
    failing_count: int = 1,
) -> float:
    """Confidence that the root gap is real.

    Blends three signals, each in [0, 1]:
      - depth: longer prerequisite chains are stronger (saturates ~3 hops),
      - severity: the lower the root's mastery, the more confident,
      - convergence: the share of failing KCs that trace to this root.
    """
    if not detection_path:
        return 0.0

    root = detection_path[-1]
    root_mastery = states.get(root, 0.5)

    depth_signal = min(1.0, len(detection_path) / 3.0)
    severity_signal = max(0.0, 1.0 - root_mastery)
    convergence_signal = min(1.0, convergence / max(1, failing_count))

    confidence = 0.4 * convergence_signal + 0.3 * depth_signal + 0.3 * severity_signal
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
