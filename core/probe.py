"""Active probing: which single question would best settle the diagnosis?

The root-gap search can only reason about what the student has shown. A
prerequisite that was never practised is a *suspect*, and the benchmark shows
the cost: the Kernel names a planted gap far more often when it was practised
than when it never was. Waiting for the conversation to wander onto the right
concept is luck. Asking about it is a decision.

The decision is Bayesian. The hypotheses are "the root gap is concept h", for
every concept within reach of what the student is failing, plus "no gap" (the
failures are noise). A gap propagates upwards: under h, h and every concept
built on it are most likely unlearned, and the rest most likely learned. Each
concept the student has practised is evidence for or against each hypothesis,
read off its mastery K:

    P(evidence | learned) / P(evidence | unlearned)  =  odds(K) / odds(p_init)

i.e. how far the student's own record moved that concept's belief away from its
prior — a concept never practised says nothing. That gives a posterior over
where the gap is. For every candidate question we then ask: if the student
answered it (P(correct) = 1 - slip if learned, guess if not), how much would
the entropy of that posterior fall, on average over the two answers? The
question with the largest expected fall is asked — the classic adaptive-testing
rule (maximum expected information), applied to the location of the gap rather
than to an ability score.

This only CHOOSES the question. The root itself is still named by
core/detector.py from the evidence, the probe's answer included once it lands.
"""
from __future__ import annotations

import math

import networkx as nx

from core import bkt, detector

# [design] How far below the failing concepts a hypothesis or a question may
# sit. Matches the default depth of /prerequisite_gaps: past four hops a
# concept explains the struggle too remotely to be worth a diagnostic question.
PROBE_MAX_HOPS = 4
# [design] Hypotheses (and so candidate questions) kept per request, nearest
# first. The cost is hypotheses x questions x evidence; this bounds it.
PROBE_MAX_HYPOTHESES = 60
# [design] How strictly a gap propagates. Under "the gap is h", a concept built
# on h is unlearned with this probability, and any other concept learned with
# it. Not 1: a student can have more than one weakness, and can have learned a
# concept despite a shaky foundation. Lower values make the posterior trust
# the graph less and the evidence on each concept more.
PROPAGATION = 0.85
# [design] Below this expected information, in bits, a question is not worth
# the student's time.
PROBE_MIN_GAIN = 0.05
# Odds are clipped so a K of exactly 0 or 1 cannot produce an infinite ratio.
_EPS = 1e-4


def _entropy(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _odds(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, p))
    return p / (1 - p)


def _region(graph: nx.DiGraph, failing: list[str]) -> list[tuple[int, str]]:
    """The failing concepts and their prerequisites within reach, nearest first."""
    hops: dict[str, int] = {kc: 0 for kc in failing if kc in graph}
    frontier = sorted(hops)
    for depth in range(1, PROBE_MAX_HOPS + 1):
        nxt = []
        for node in frontier:
            for prereq in sorted(graph.predecessors(node)):
                if prereq not in hops:
                    hops[prereq] = depth
                    nxt.append(prereq)
        frontier = nxt
    return sorted((h, label) for label, h in hops.items())[:PROBE_MAX_HYPOTHESES]


def _params(graph: nx.DiGraph, label: str) -> dict:
    return bkt.get_bkt_params(dict(graph.nodes[label]) if label in graph else None)


def root_posterior(
    graph: nx.DiGraph, failing: list[str], states: dict[str, float], known: set[str]
) -> dict[str | None, float]:
    """P(the root gap is h | everything the student has shown), h=None for "no gap"."""
    hypotheses = [label for _, label in _region(graph, failing)]
    if not hypotheses:
        return {}
    covered = {h: {h} | nx.descendants(graph, h) for h in hypotheses}
    # Log likelihood ratio learned:unlearned of each piece of evidence.
    llr = {}
    for label in known:
        if label in graph and label in states:
            p0 = _params(graph, label)["p_init"]
            llr[label] = math.log(_odds(states[label]) / _odds(p0))

    def log_lik(cover: set[str]) -> float:
        total = 0.0
        for label, r in llr.items():
            ratio = math.exp(r)  # P(e|learned) / P(e|unlearned)
            q_learned = (1 - PROPAGATION) if label in cover else PROPAGATION
            # Up to a factor common to every hypothesis, P(e|unlearned) = 1.
            total += math.log(q_learned * ratio + (1 - q_learned))
        return total

    logs = {h: log_lik(covered[h]) for h in hypotheses}
    logs[None] = log_lik(set())
    # Uniform prior: no concept is assumed a likelier gap than another, and
    # "no gap" weighs as much as any single concept.
    top = max(logs.values())
    weights = {h: math.exp(v - top) for h, v in logs.items()}
    z = sum(weights.values())
    return {h: w / z for h, w in weights.items()}


def choose_probe(
    graph: nx.DiGraph,
    failing: list[str],
    states: dict[str, float],
    known: set[str],
    weak_below: dict[str, float],
    current_root: str | None,
) -> dict | None:
    """The most informative diagnostic question, or None if none is worth asking.

    Args:
        graph: the Kernel Graph (prerequisite -> concept).
        failing / states / known / weak_below: exactly what the root-gap search
            was given, so the probe reasons on the same evidence.
        current_root: the root that search returned.

    Returns:
        {label, concept_id, expected_gain, p_correct, root_if_correct,
         root_if_wrong, confirms_root} for the best question.
    """
    if not failing:
        return None
    posterior = root_posterior(graph, failing, states, known)
    if not posterior:
        return None
    prior_entropy = _entropy(posterior.values())
    hops = {label: h for h, label in _region(graph, failing)}
    covered = {h: ({h} | nx.descendants(graph, h)) if h is not None else set() for h in posterior}

    scored = []
    for label in sorted(hops):
        if label in failing:
            continue  # already the topic of the conversation
        params = _params(graph, label)
        p_if_learned, p_if_unlearned = 1 - params["p_slip"], params["p_guess"]
        # P(correct | h): the concept is unlearned with prob PROPAGATION if h
        # covers it, learned with prob PROPAGATION otherwise.
        p_correct_h = {}
        for h, cover in covered.items():
            q_learned = (1 - PROPAGATION) if label in cover else PROPAGATION
            p_correct_h[h] = q_learned * p_if_learned + (1 - q_learned) * p_if_unlearned
        p_correct = sum(posterior[h] * p_correct_h[h] for h in posterior)
        expected = 0.0
        for p_answer, lik in (
            (p_correct, p_correct_h),
            (1 - p_correct, {h: 1 - p for h, p in p_correct_h.items()}),
        ):
            if p_answer <= 0:
                continue
            after = [posterior[h] * lik[h] / p_answer for h in posterior]
            expected += p_answer * _entropy(after)
        gain = prior_entropy - expected
        if gain >= PROBE_MIN_GAIN:
            # Ties: the nearer concept, then the label, so the choice is stable.
            scored.append(((round(gain, 6), -hops[label], _neg(label)), label, gain, p_correct))

    # Most informative first, and the first whose two answers lead the
    # detector to different roots. The posterior may learn from a question
    # whose answer the diagnosis would ignore — a concept below one the
    # student has mastered, say — but the student would see nothing come of it.
    for _, label, gain, p_correct in sorted(scored, reverse=True):
        if_correct = _root_after(graph, failing, states, known, weak_below, label, True)
        if_wrong = _root_after(graph, failing, states, known, weak_below, label, False)
        if if_correct == if_wrong:
            continue
        return {
            "label": label,
            "concept_id": graph.nodes[label].get("id"),
            "expected_gain": round(gain, 4),
            "p_correct": round(p_correct, 4),
            "root_if_correct": if_correct,
            "root_if_wrong": if_wrong,
            "confirms_root": label == current_root,
        }
    return None


def _root_after(graph, failing, states, known, weak_below, label: str, correct: bool) -> str | None:
    """The root the detector would name once this answer is in."""
    params = _params(graph, label)
    k = states.get(label) if label in known else params["p_init"]
    trial_states = dict(states)
    trial_states[label] = bkt.update_bkt(k, correct=correct, params=params)
    trial_weak = dict(weak_below)
    trial_weak.setdefault(label, detector.failing_threshold(params["p_init"]))
    return detector.detect_root_cause(
        graph, failing, trial_states, known=known | {label}, weak_below=trial_weak
    )["root_gap"]


def _neg(label: str) -> tuple:
    """Sort key making the alphabetically FIRST label win under max()."""
    # The trailing 0 sorts a prefix ("ab") above its extensions ("abc").
    return tuple(-ord(c) for c in label) + (0,)
