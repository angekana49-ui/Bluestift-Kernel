"""Bayesian Knowledge Tracing (BKT).

Posterior update of a learner's mastery probability K after an observation.

Parameters are *living*: they are never hardcoded into the update. The literature
priors (Corbett & Anderson 1995) are used only as a fallback when a concept node
has no empirically calibrated values yet. See `get_bkt_params`.

Partial-credit handling follows Ostrow & Heffernan (2015); the asymmetric update
(errors weigh more than successes) follows Hooshyar et al.
"""
from __future__ import annotations

# Literature priors — used only as a fallback prior, never as a hard constant.
BKT_PRIORS = {
    "p_init": 0.3,      # p(L0) initial mastery probability
    "p_transit": 0.1,   # p(T) probability of learning on a trial
    "p_slip": 0.1,      # p(S) probability of error despite mastery
    "p_guess": 0.2,     # p(G) probability of success by chance
}

# Recognized partial-credit bins (Ostrow & Heffernan 2015).
PARTIAL_CREDIT_BINS = (0.0, 0.3, 0.6, 0.7, 0.8, 1.0)


def get_bkt_params(concept_node: dict | None) -> dict:
    """Read BKT params from a concept node, falling back to literature priors.

    A node may carry empirically calibrated `p_init/p_transit/p_slip/p_guess`.
    Any missing or null value falls back to the prior.
    """
    node = concept_node or {}
    return {
        "p_init": node.get("p_init") or BKT_PRIORS["p_init"],
        "p_transit": node.get("p_transit") or BKT_PRIORS["p_transit"],
        "p_slip": node.get("p_slip") or BKT_PRIORS["p_slip"],
        "p_guess": node.get("p_guess") or BKT_PRIORS["p_guess"],
    }


def snap_partial_credit(value: float) -> float:
    """Snap a raw partial-credit score to the nearest recognized bin."""
    return min(PARTIAL_CREDIT_BINS, key=lambda b: abs(b - value))


def update_bkt(
    k_current: float,
    correct: bool,
    partial_credit: float | None = None,
    params: dict | None = None,
) -> float:
    """Bayesian update of K after one observation.

    Args:
        k_current: prior mastery probability in [0, 1].
        correct: whether the attempt succeeded (ignored if partial_credit given).
        partial_credit: graded outcome in [0, 1]; snapped to recognized bins.
        params: BKT params dict (p_init/p_transit/p_slip/p_guess). Falls back to
            literature priors when omitted.

    Returns:
        Posterior mastery probability in [0, 1].
    """
    p = params or BKT_PRIORS
    p_slip = p["p_slip"]
    p_guess = p["p_guess"]
    p_transit = p["p_transit"]

    if partial_credit is not None:
        correct_weight = snap_partial_credit(partial_credit)
    else:
        correct_weight = 1.0 if correct else 0.0

    # Asymmetric likelihoods (Hooshyar): failures pull mastery down harder.
    if correct_weight < 0.5:
        p_obs_given_learned = p_slip * (1 - correct_weight)
        p_obs_given_not_learned = (1 - p_guess) * (1 - correct_weight)
    else:
        p_obs_given_learned = (1 - p_slip) * correct_weight
        p_obs_given_not_learned = p_guess * correct_weight

    numerator = p_obs_given_learned * k_current
    denominator = numerator + p_obs_given_not_learned * (1 - k_current)
    k_posterior = numerator / denominator if denominator > 0 else k_current

    # Learning transition: chance to move from "not learned" to "learned".
    k_new = k_posterior + (1 - k_posterior) * p_transit
    return min(max(k_new, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# Selective-update gate
# --------------------------------------------------------------------------- #
def should_update_bkt(
    consecutive_interactions: int,
    partial_credit_now: float,
    partial_credit_prev: float | None,
    anomalous_pattern: bool = False,
) -> bool:
    """Decide whether a BKT update should fire (avoid noisy micro-updates).

    Fires when any of:
      - 3+ consecutive interactions on the same KC in the same session, OR
      - partial-credit change > 0.3 vs the last stored value, OR
      - an anomalous pattern is detected.
    """
    if anomalous_pattern:
        return True
    if consecutive_interactions >= 3:
        return True
    if partial_credit_prev is not None and abs(partial_credit_now - partial_credit_prev) > 0.3:
        return True
    return False


# --------------------------------------------------------------------------- #
# Mastery criterion (dual condition)
# --------------------------------------------------------------------------- #
MASTERY_THRESHOLD = 0.95
MAX_SLIP_FOR_MASTERY = 0.15
MIN_PARTIAL_CREDIT_AVG = 0.7


def is_mastered(k_effective: float, p_slip: float, partial_credit_avg: float) -> bool:
    """Dual mastery condition: high effective mastery AND low slip AND solid credit."""
    return (
        k_effective >= MASTERY_THRESHOLD
        and p_slip <= MAX_SLIP_FOR_MASTERY
        and partial_credit_avg >= MIN_PARTIAL_CREDIT_AVG
    )


# Status thresholds used across the API.
GAP_THRESHOLD = 0.4      # below -> "gap"
PARTIAL_THRESHOLD = 0.7  # below -> "partial", at/above -> "mastered"


def classify_status(
    k_effective: float,
    p_slip: float = BKT_PRIORS["p_slip"],
    partial_credit_avg: float = 0.5,
) -> str:
    """Map an effective mastery to a human-facing status label."""
    if is_mastered(k_effective, p_slip, partial_credit_avg):
        return "mastered"
    if k_effective < GAP_THRESHOLD:
        return "gap"
    if k_effective < PARTIAL_THRESHOLD:
        return "partial"
    return "mastered"
