"""Mindset score M.

M = sigmoid(GAIN * (linear - 0.5)), where `linear` blends two readings:
  - the conversation (the corpus formula: abandon, persistence, time on task,
    interaction quality), with abandon-after-error MEASURED in the conversation
    whenever there was an error to abandon after;
  - the Kernel's own traces: learning speed (V), progression across sessions,
    recovery after an error, stability (P without M). See `measured_linear`.

A sigmoid blend of behavioural signals, clamped away from {0,1} because — per
Dweck — a fully fixed or fully growth mindset is never a settled fact.

The four weights are POSITIVE and sum to 1, over signals that all point the
same way (more is better), so the weighted sum is itself a score in [0,1] whose
midpoint is a genuinely neutral student. `abandon_rate` points the other way,
so it enters as its complement: not giving up is the signal.

That normalisation is the whole point of the shape, and it is what this module
used to be missing. A sigmoid applied to the RAW weighted sum is not a
rescaling, it is a compression towards 0.5: with the old signed weights the
reachable output was [0.413, 0.657], so no student could ever cross
`classify_mindset`'s growth threshold (0.66) or its fixed one (0.40). Every
learner read as "mixed" forever, `detect_fixed_mindset` could never fire, and
the clamps below were decorative. Centring on the neutral point and scaling by
GAIN is what makes the thresholds, the alert and the clamps mean anything.
"""
from __future__ import annotations

import math

# Weights (positive, sum = 1). Giving up weighs most, because it is the
# behaviour the construct is actually about.
W_ABANDON = 0.35       # applied to (1 - abandon_rate)
W_PERSISTENCE = 0.30
W_TIME = 0.20
W_QUALITY = 0.15

NEUTRAL_M = 0.5

# Steepness around the neutral midpoint. 6.0 maps the extremes of the weighted
# sum to ~0.047 / ~0.953 — just outside the clamps, so the clamps guard the
# arithmetic instead of doing the shaping, and the middle of the range stays
# roughly linear where most students actually sit.
GAIN = 6.0

M_FLOOR = 0.05
M_CEIL = 0.95

# How much of one conversation's reading is allowed to move the stored score.
# M is a trait estimate read through a single exchange, and the label it feeds
# has consequences: `fixed_mindset` tells the tutor to stop working the concept
# and address the student's self-belief, and routes the turn to the expensive
# model. One terse afternoon must not be enough to say that about a learner —
# ~7 conversations of consistent evidence get 80% of the way to a new level.
EMA_WEIGHT = 0.3


def _unit(value: float) -> float:
    """Clamp a signal to [0,1]. The prompt asks for that range; an LLM is not
    bound by it, and out-of-range values now move a label that matters."""
    return max(0.0, min(1.0, value))


def _squash(linear: float) -> float:
    """The centred sigmoid, clamped: a linear score in [0,1] -> M."""
    m_score = 1.0 / (1.0 + math.exp(-GAIN * (linear - NEUTRAL_M)))
    return max(M_FLOOR, min(M_CEIL, m_score))


def conversation_linear(
    abandon_rate: float,
    persistence_score: float,
    time_on_task: float,
    interaction_quality: float,
) -> float:
    """The behaviour read in one conversation, as a linear score in [0, 1]."""
    return (
        W_ABANDON * (1.0 - _unit(abandon_rate))
        + W_PERSISTENCE * _unit(persistence_score)
        + W_TIME * _unit(time_on_task)
        + W_QUALITY * _unit(interaction_quality)
    )


def compute_mindset_score(
    abandon_rate: float,
    persistence_score: float,
    time_on_task: float,
    interaction_quality: float,
) -> float:
    """Blend behavioural signals into a mindset score in [0.05, 0.95]."""
    return _squash(conversation_linear(abandon_rate, persistence_score, time_on_task, interaction_quality))


# --------------------------------------------------------------------------- #
# Measured M: the Kernel's own traces
# --------------------------------------------------------------------------- #
# The conversation signals above are an LLM's reading of prose. These are
# measured on what the student actually did, and they are about DYNAMICS —
# how the student reacts to difficulty and whether they improve — never about
# LEVEL. K (what the student knows) and forgetting (what their memory keeps)
# are deliberately left out: feeding them into M would label a weak or
# forgetful student "fixed mindset" for their level, the stereotype-threat
# mechanism the corpus warns about (§2.7). Dweck's mindset is a belief about
# ability, not ability.
#
# [design] Weights of the measured signals (renormalised over the ones
# available for this student).
W_MEASURED = {
    "learning_speed": 0.30,   # V: share of the mastery gap closed per attempt
    "progression": 0.30,      # does K rise across sessions?
    "recovery": 0.20,         # after a failure, does the next try succeed?
    "stability": 0.20,        # P without M: 1 - personal slip
}
# [design] How much measured evidence it takes to weigh as much as half its
# maximum share, and that maximum: the conversation always counts at least
# half, because the measured signals are indirect proxies of a belief.
MEASURED_HALF_WEIGHT_AT = 5
MAX_MEASURED_SHARE = 0.5


def session_behaviour(attempts: list[tuple[str, float]]) -> dict:
    """Abandon-after-error and recovery, measured on one conversation.

    Args:
        attempts: (kc_label, credit) in conversation order.

    A failure is ABANDONED when the student makes no further attempt on that KC
    in the conversation — the corpus' "abandon post-erreur", measured instead
    of guessed. RECOVERY is the share of those retries that succeed.
    """
    failures = retried = recovered = 0
    for i, (kc, credit) in enumerate(attempts):
        if credit >= 0.5:
            continue
        failures += 1
        nxt = next((c for k, c in attempts[i + 1:] if k == kc), None)
        if nxt is not None:
            retried += 1
            recovered += nxt >= 0.5
    return {
        "abandon_after_error": 1 - retried / failures if failures else None,
        "recovery": recovered / retried if retried else None,
        "failures": failures,
    }


def measured_linear(states: list[dict], trajectories: dict[str, list[float]], recovery: float | None):
    """The measured signals as (linear score in [0, 1], evidence count), or (None, 0).

    Args:
        states: the student's KC states BEFORE this session.
        trajectories: concept_id -> K series, oldest first.
        recovery: this session's recovery rate (session_behaviour), if any.
    """
    signals: dict[str, float] = {}
    n = 0
    practised = [s for s in states if (s.get("interactions_on_kc") or 0) >= 2]
    speeds = [float(s["v_score"]) for s in practised if s.get("v_score") is not None]
    if speeds:
        signals["learning_speed"] = sum(speeds) / len(speeds)
        n += len(speeds)
    slips = [float(s["p_slip_personal"]) for s in practised if s.get("p_slip_personal") is not None]
    if slips:
        signals["stability"] = 1.0 - sum(slips) / len(slips)
    rises = [series[-1] - series[0] for series in trajectories.values() if len(series) >= 2]
    if rises:
        # A rise of +0.5 across the window reads as fully progressing, a fall
        # of 0.5 as not at all.
        signals["progression"] = _unit(0.5 + sum(rises) / len(rises))
        n += len(rises)
    if recovery is not None:
        signals["recovery"] = recovery
        n += 1
    if not signals:
        return None, 0
    total = sum(W_MEASURED[k] for k in signals)
    return sum(W_MEASURED[k] * v for k, v in signals.items()) / total, n


def combine(conversation: float | None, measured: float | None, evidence: int) -> float | None:
    """M from the conversation reading and the measured signals (linear -> M).

    The measured share grows with the evidence behind it, up to
    MAX_MEASURED_SHARE. With only one of the two, that one stands alone.
    """
    if conversation is None and measured is None:
        return None
    if measured is None:
        return _squash(conversation)
    if conversation is None:
        return _squash(measured)
    share = MAX_MEASURED_SHARE * evidence / (evidence + MEASURED_HALF_WEIGHT_AT)
    return _squash((1 - share) * conversation + share * measured)


def blend_mindset(previous: float | None, observed: float) -> float:
    """Fold one conversation's reading into the running estimate.

    No history (a student's first analysed conversation) means there is nothing
    to smooth against, so the reading stands on its own — otherwise every new
    learner would start life pinned to whatever NEUTRAL_M is.
    """
    if previous is None:
        return observed
    return (1.0 - EMA_WEIGHT) * previous + EMA_WEIGHT * observed


def classify_mindset(m_score: float) -> str:
    """Map a mindset score to a coarse label."""
    if m_score >= 0.66:
        return "growth"
    if m_score <= 0.4:
        return "fixed"
    return "mixed"
