"""Root-cause detection via DFS over the Kernel Graph.

Given a set of failing KCs, walk the prerequisite chain (predecessors) downward
to find the deepest prerequisite that is itself failing. That deepest failing
node is the *root gap*: the place where remediation should start.
"""
from __future__ import annotations

import networkx as nx

# [design] The most a concept's mastery may be and still count as failing.
FAILING_THRESHOLD = 0.5


def failing_threshold(p_init: float) -> float:
    """Below what mastery a KC counts as FAILING: under its own prior, capped at 0.5.

    "Failing" has to mean the evidence points towards not knowing — a posterior
    below the prior — not merely "not yet shown to be known". With the
    literature p_init of 0.3, a flat 0.5 threshold made any weak positive
    evidence a failure: one assisted success on a mastered concept leaves K
    near 0.45, and that concept then became a root gap. It also keeps a
    concept that has only decayed (towards p_init) from ever reading as failed
    without fresh evidence. Once calibration raises p_init above 0.5, the cap
    applies: more likely unknown than known.
    """
    return min(FAILING_THRESHOLD, p_init)


# How many consecutive *unknown* (never-practised) prerequisites the search may
# bridge before stopping. Lets it cross small evidence gaps to reach a deeper
# weak concept, without bottoming out at the most elementary leaf every time.
UNKNOWN_BUDGET = 2


def detect_root_cause(
    graph: nx.DiGraph,
    failing_kcs: list[str],
    concept_states: dict[str, float],
    known: set[str] | None = None,
    weak_below: dict[str, float] | None = None,
) -> dict:
    """Find the root gap among failing KCs.

    The root is chosen by **convergence** first, not by chain length: a weak or
    unverified concept that many failing KCs depend on is the strongest
    explanation for the struggle. Ties break on evidence (a known-weak concept
    beats a merely-unknown one), then on depth (more foundational), then on
    severity (the weaker concept), and finally on the label — so two identical
    requests always get the same answer, whatever the process's hash seed.

    Args:
        graph: the Kernel Graph (prerequisite -> concept).
        failing_kcs: labels the student is currently failing.
        concept_states: label -> effective mastery in [0, 1].
        known: labels with actual student evidence. Labels absent from this set
            are treated as *unknown* (never practised), so the search descends
            into them as suspects. When None, every label in concept_states is
            considered known.
        weak_below: per-label failing threshold (see `failing_threshold`);
            FAILING_THRESHOLD for labels absent from it.

    Returns:
        dict with root_gap, detection_path (surface -> root), and confidence.
    """
    if known is None:
        known = set(concept_states.keys())
    limits = weak_below or {}

    def is_weak(label: str) -> bool:
        return concept_states.get(label, 0.5) < limits.get(label, FAILING_THRESHOLD)

    failing_in_graph = [kc for kc in failing_kcs if kc in graph]
    if not failing_in_graph:
        return {"root_gap": None, "detection_path": [], "confidence": 0.0}

    # Each failing KC -> its chain down to a deepest weak/unknown concept.
    chains: dict[str, list[str]] = {
        kc: _dfs_find_root(graph, kc, is_weak, known, set(), UNKNOWN_BUDGET)
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
        return 1 if (r in known and is_weak(r)) else 0

    def depth(r: str) -> int:
        # How far below a failing concept r sits, on any chain. It used to count
        # only chains ENDING at r, so a known-weak prerequisite in the middle of
        # a chain (the chain going on into unverified concepts below it) scored
        # depth 1, like the surface itself — and lost to the surface on
        # severity. The module's own rule is the deepest failing prerequisite.
        return max((ch.index(r) + 1 for ch in chains.values() if r in ch), default=1)

    # `candidates` is a set: iterate it sorted, because max() keeps the first
    # maximal element and set order changes between processes.
    root_gap = max(
        sorted(candidates),
        key=lambda r: (convergence(r), has_evidence(r), depth(r), -concept_states.get(r, 0.5)),
    )

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
    is_weak,
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

    # Sorted: the first-longest chain wins ties below, and edge order comes from
    # an unordered DB read.
    prerequisites = sorted(graph.predecessors(node))
    if not prerequisites:
        return [node]

    best_chain: list[str] = []
    for prereq in prerequisites:
        is_known = prereq in known

        if is_known and not is_weak(prereq):
            continue  # mastered prerequisite -> barrier, don't descend
        if is_known:
            # Known-weak: strong evidence, descend and refill the unknown budget.
            deeper = _dfs_find_root(graph, prereq, is_weak, known, visited, UNKNOWN_BUDGET)
        else:
            # Unknown: a suspected gap, descend only while budget remains.
            if budget <= 0:
                continue
            deeper = _dfs_find_root(graph, prereq, is_weak, known, visited, budget - 1)

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


def recommended_path(
    graph: nx.DiGraph,
    root_gap: str | None,
    max_len: int = 4,
    priorities: dict[str, float] | None = None,
    detection_path: list[str] | None = None,
) -> list[str]:
    """Suggest a remediation order: root gap first, then its dependents upward.

    When the diagnosis has a detection path (surface -> root), the remediation
    climbs it back (root -> surface): those are the concepts standing between
    the foundation and what the student is actually stuck on. A free walk from
    the root could wander off towards an unrelated dependent — it used to pick
    the alphabetically first one. Beyond the path, or without one, the walk
    continues forward through the concepts that build on the last step.

    `priorities` is the school's per-concept weighting (see core/curriculum.py).
    It only ever chooses *between* concepts that are already valid next steps —
    the walk still follows the prerequisite graph, so a school can say what to
    reach for first but cannot ask for a concept before its foundations.
    """
    if not root_gap or root_gap not in graph:
        return [] if not root_gap else [root_gap]

    weights = priorities or {}
    climb = list(reversed(detection_path or []))
    if not climb or climb[0] != root_gap:
        climb = [root_gap]
    path = climb[:max_len]
    current = path[-1]
    visited = set(path)
    while len(path) < max_len:
        successors = [s for s in graph.successors(current) if s not in visited]
        if not successors:
            break
        # Highest school priority first; alphabetical as the deterministic
        # tiebreak (and the whole rule when no school weighting applies).
        nxt = sorted(successors, key=lambda s: (-weights.get(s, 1.0), s))[0]
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path
