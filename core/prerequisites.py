"""Multi-hop prerequisite reasoning — the graph half of GraphRAG.

The question this answers is the one vector retrieval cannot: *which
prerequisites of this concept has THIS student not yet mastered?* Similarity
search can find passages that look like a concept; only the graph knows that
`derivation_fonction` rests on `notion_de_fonction`, which rests on
`notion_de_variable`, and only the student's state says which of those is
actually missing. Retrieval then has something worth retrieving for.

Three rules do the pedagogical work here, and each one is a decision:

1. **A mastered prerequisite is a wall, not a door.** The walk stops there and
   does not expand through it. If a student has `notion_de_fonction`, whatever
   that concept itself rests on is already carried — listing it would bury the
   real gap under foundations the student demonstrably holds. This is what
   turns an exhaustive ancestor dump into a short, actionable list.

2. **Unknown is not mastered.** A concept the student has never touched is
   expanded through, because we have no evidence either way, but it is reported
   as unknown rather than as a gap. Treating silence as mastery is how a
   tutoring system walks a child past the hole they are standing in.

3. **Order by foundation, not by distance.** A three-hop prerequisite often has
   to be taught before a one-hop one. The output is a topological order over
   the gaps found, so following it top to bottom never teaches something before
   what it rests on.
"""
from __future__ import annotations

from collections import deque

import networkx as nx

# How deep the walk goes by default. The graph averages ~2 prerequisites per
# concept, so four hops is already a few dozen candidates — past that a report
# stops being something a teacher or a tutor can act on in one sitting.
DEFAULT_MAX_HOPS = 4
MAX_HOPS_CEILING = 8


def prerequisite_frontier(
    graph: nx.DiGraph,
    target: str,
    status_by_label: dict[str, str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict[str, int]:
    """Walk prerequisites breadth-first, stopping at mastered concepts.

    Returns every concept reached, mapped to the fewest hops it took to get
    there. The target itself is excluded — the question is what it rests on.

    `status_by_label` carries one of "mastered" / "partial" / "gap" / "unknown"
    per concept. Anything absent is treated as unknown.
    """
    if target not in graph:
        return {}

    hops: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(target, 0)])
    seen = {target}

    while queue:
        label, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for prereq in graph.predecessors(label):
            if prereq not in hops or hops[prereq] > depth + 1:
                hops[prereq] = depth + 1
            if prereq in seen:
                continue
            seen.add(prereq)
            # Rule 1: a mastered prerequisite is recorded but never expanded
            # through. Rule 2: unknown is expanded, because absence of evidence
            # is not evidence of mastery.
            if status_by_label.get(prereq, "unknown") != "mastered":
                queue.append((prereq, depth + 1))

    return hops


def teaching_order(
    graph: nx.DiGraph, labels: list[str], hops: dict[str, int] | None = None
) -> list[str]:
    """Order concepts so nothing is taught before what it rests on.

    Topological over the subgraph induced by `labels`, which is the hard
    constraint. Between branches that do not depend on each other topology says
    nothing, so the tie is broken by depth, deepest first: that is the Kernel's
    whole thesis — work on the foundation the other gaps trace back to, not on
    whichever surface concept happened to come up first.

    The graph is built as a DAG, but a cycle introduced upstream must not take
    the route down with it: on a cycle this falls back to the caller's order
    rather than raising.
    """
    present = [lbl for lbl in labels if lbl in graph]
    if not present:
        return []
    sub = graph.subgraph(present)
    depth = hops or {}
    try:
        return list(
            nx.lexicographical_topological_sort(
                sub, key=lambda lbl: (-depth.get(lbl, 0), lbl)
            )
        )
    except nx.NetworkXUnfeasible:  # pragma: no cover - the builder enforces a DAG
        return present


def gap_report(
    graph: nx.DiGraph,
    target: str,
    status_by_label: dict[str, str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict:
    """The full answer: what this concept rests on, and what is missing.

    `gaps` are the concepts to actually work on, in teaching order. `frontier`
    is where the walk stopped because the student already holds the concept —
    worth returning, because "we stopped here" is the evidence that the short
    list above is short for a reason, not because the walk gave up.
    """
    hops = prerequisite_frontier(graph, target, status_by_label, max_hops)

    gaps: list[str] = []
    frontier: list[str] = []
    for label in hops:
        if status_by_label.get(label, "unknown") == "mastered":
            frontier.append(label)
        else:
            gaps.append(label)

    ordered = teaching_order(graph, gaps, hops)
    return {
        "target": target,
        "gaps": [{"label": lbl, "hops": hops[lbl]} for lbl in ordered],
        "frontier": [
            {"label": lbl, "hops": hops[lbl]}
            for lbl in sorted(frontier, key=lambda lbl: (hops[lbl], lbl))
        ],
        "max_hops": max_hops,
        # True only when the walk actually had somewhere left to go: a concept
        # sitting at the depth limit, still unmastered, that rests on something
        # we never looked at. A node that merely happens to be that deep is not
        # truncation — and a caller must never read a cut list as a complete
        # one, because on this screen "nothing else is missing" is a claim.
        "truncated": any(
            depth >= max_hops
            and status_by_label.get(label, "unknown") != "mastered"
            and next(graph.predecessors(label), None) is not None
            for label, depth in hops.items()
        ),
    }
