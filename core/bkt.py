"""Bayesian Knowledge Tracing (BKT).

Posterior update of a learner's mastery probability K after an observation.

Parameters are *living*: they are never hardcoded into the update. The literature
priors (Corbett & Anderson 1995) are used only as a fallback when a concept node
has no empirically calibrated values yet. See `get_bkt_params`.

Partial-credit handling follows Ostrow & Heffernan (2015): a graded answer is
soft evidence, weighing the correct and incorrect likelihoods by its credit.
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


# Slip and guess must each stay below 0.5. At slip + guess >= 1 the model
# inverts — a correct answer becomes evidence of NOT knowing — and near that
# line it stops being identifiable (Baker, Corbett & Aleven 2008). Whatever a
# calibration run or a hand edit writes to a node, the update never sees that.
MAX_SLIP_OR_GUESS = 0.45


def _param(node: dict, name: str, lo: float, hi: float) -> float:
    """One param from the node, or the prior when missing/unreadable; bounded.

    `None` means "not calibrated"; 0.0 is a calibrated value and is kept.
    """
    value = node.get(name)
    try:
        value = float(value) if value is not None else BKT_PRIORS[name]
    except (TypeError, ValueError):
        value = BKT_PRIORS[name]
    return max(lo, min(hi, value))


def get_bkt_params(concept_node: dict | None) -> dict:
    """Read BKT params from a concept node, falling back to literature priors.

    A node may carry empirically calibrated `p_init/p_transit/p_slip/p_guess`.
    Any missing or null value falls back to the prior; every value is bounded
    to the identifiable region.
    """
    node = concept_node or {}
    return {
        "p_init": _param(node, "p_init", 0.0, 1.0),
        "p_transit": _param(node, "p_transit", 0.0, 1.0),
        "p_slip": _param(node, "p_slip", 0.0, MAX_SLIP_OR_GUESS),
        "p_guess": _param(node, "p_guess", 0.0, MAX_SLIP_OR_GUESS),
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

    # Soft evidence: a credit c is read as "correct with weight c, incorrect with
    # weight 1 - c", so the likelihood of the observation is the mixture of the
    # two binary BKT likelihoods. c = 1 and c = 0 are exactly classic BKT.
    #
    # The previous form scaled BOTH likelihoods by the same factor, which cancels
    # in the posterior ratio: every credit below 0.5 updated like a plain failure
    # and every credit at/above 0.5 like a plain success — partial credit was
    # binarised, whatever the bins said.
    p_obs_given_learned = correct_weight * (1 - p_slip) + (1 - correct_weight) * p_slip
    p_obs_given_not_learned = correct_weight * p_guess + (1 - correct_weight) * (1 - p_guess)

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
GAP_THRESHOLD = 0.4      # below -> "gap"; otherwise "partial" until mastered


def classify_status(
    k_effective: float,
    p_slip: float = BKT_PRIORS["p_slip"],
    partial_credit_avg: float = 0.5,
) -> str:
    """Map an effective mastery to a human-facing status label.

    "mastered" is reserved for the dual condition. It used to also be granted to
    any K >= 0.7, which made the dual condition dead code: a KC slipped on 40%
    of the time read "mastered" on the same response that raised false_mastery.
    """
    if is_mastered(k_effective, p_slip, partial_credit_avg):
        return "mastered"
    if k_effective < GAP_THRESHOLD:
        return "gap"
    return "partial"


def effective_state(concept_node: dict, student_state: dict | None) -> dict:
    """One KC's current standing for one student: decayed mastery and status.

    The same three lines used to sit in /load_profile and would have had to be
    copied into every route that asks "how is this student doing on this
    concept". Two copies of a mastery rule drift, and the drift shows up as two
    screens disagreeing about whether a child has understood something.

    A student with no row for the KC is `unknown`, never `gap`: never having
    been asked is not the same as having failed.
    """
    from core import forgetting  # local: forgetting is a leaf, this keeps it one

    if not student_state:
        return {"k_raw": None, "k_effective": None, "p_slip": None, "status": "unknown"}

    k_raw = student_state.get("mastery_score_raw") or 0.0
    last_at = student_state.get("last_strong_signal_at")
    k_effective = forgetting.compute_effective_mastery(
        k_raw,
        concept_node.get("type_kc", "conceptual"),
        last_at,
        lambda_override=forgetting.get_lambda(concept_node, student_state),
    )
    p_slip = student_state.get("p_slip_personal") or get_bkt_params(concept_node)["p_slip"]
    pc_avg = student_state.get("partial_credit_avg") or 0.5
    return {
        "k_raw": round(k_raw, 4),
        "k_effective": round(k_effective, 4),
        "p_slip": p_slip,
        "status": classify_status(k_effective, p_slip, pc_avg),
    }
