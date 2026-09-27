"""Mindset score M.

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


def compute_mindset_score(
    abandon_rate: float,
    persistence_score: float,
    time_on_task: float,
    interaction_quality: float,
) -> float:
    """Blend behavioural signals into a mindset score in [0.05, 0.95]."""
    linear = (
        W_ABANDON * (1.0 - _unit(abandon_rate))
        + W_PERSISTENCE * _unit(persistence_score)
        + W_TIME * _unit(time_on_task)
        + W_QUALITY * _unit(interaction_quality)
    )
    m_score = 1.0 / (1.0 + math.exp(-GAIN * (linear - NEUTRAL_M)))
    return max(M_FLOOR, min(M_CEIL, m_score))


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
