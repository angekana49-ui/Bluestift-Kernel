"""Bayesian Knowledge Tracing (BKT).

Posterior update of a learner's mastery probability K after an observation.

Parameters are *living*: they are never hardcoded into the update. The literature
priors (Corbett & Anderson 1995) are used only as a fallback when a concept node
has no empirically calibrated values yet. See `get_bkt_params`.

Partial-credit handling follows Ostrow & Heffernan (2015): a graded answer is
soft evidence, weighing the correct and incorrect likelihoods by its credit.
"""
from __future__ import annotations

# [prior] Literature priors — used only as a fallback, never as a hard constant.
# Typical of fitted Cognitive Tutor / ASSISTments values (Corbett & Anderson
# 1995; Baker, Corbett & Aleven 2008), not fitted on our students. They are
# replaced per KC by `calibration.fit_bkt_em` once enough evidence exists.
BKT_PRIORS = {
    "p_init": 0.3,      # p(L0) initial mastery probability
    "p_transit": 0.1,   # p(T) probability of learning on a trial
    "p_slip": 0.1,      # p(S) probability of error despite mastery
    "p_guess": 0.2,     # p(G) probability of success by chance
}

# [prior, to verify] The partial-credit grading scale (Ostrow & Heffernan 2015).
# A scale for GRADERS (the extraction LLM, the app) to report on. The update
# itself takes any credit in [0, 1]: under soft evidence, snapping to these
# bins only destroyed information — it even turned the uninformative 0.5 into
# a 0.6 that counts as evidence of mastery.
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


# [design] Guess rate of an assisted attempt: with help, a student who has not
# learned the concept succeeds about half the time. A deliberately conservative
# placeholder — calibrate it separately from p_guess once learning_events hold
# enough assisted attempts.
ASSISTED_GUESS = 0.5


def update_bkt(
    k_current: float,
    correct: bool,
    partial_credit: float | None = None,
    params: dict | None = None,
    assisted: bool = False,
) -> float:
    """Bayesian update of K after one observation.

    An assisted attempt is a different observation, not a discounted one: with
    help, a student who has NOT learned the concept succeeds far more often,
    so its guess rate is raised to ASSISTED_GUESS. A success then still counts,
    but as much weaker evidence (Bastani et al. 2024: assisted practice
    performance barely predicts unassisted exam performance). Capping the
    credit at 0.9, as before, left an assisted success ~70% as convincing as an
    autonomous one.

    Args:
        k_current: prior mastery probability in [0, 1].
        correct: whether the attempt succeeded (ignored if partial_credit given).
        partial_credit: graded outcome in [0, 1] (clamped).
        params: BKT params dict (p_init/p_transit/p_slip/p_guess). Falls back to
            literature priors when omitted.
        assisted: the attempt was made with the tutor's help.

    Returns:
        Posterior mastery probability in [0, 1].
    """
    p = params or BKT_PRIORS
    p_slip = p["p_slip"]
    p_guess = p["p_guess"]
    p_transit = p["p_transit"]
    if assisted:
        p_guess = max(p_guess, ASSISTED_GUESS)

    if partial_credit is not None:
        correct_weight = max(0.0, min(1.0, partial_credit))
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
# Evidence rules: blocage type
# --------------------------------------------------------------------------- #
def blocage_evidence(credit: float, blocage_type: str | None) -> float | None:
    """The credit an attempt counts for, given why the student was blocked.

    K measures the concept, not the language it was asked in, so a failure
    caused by a LINGUISTIC block is no evidence about K and is not counted
    (None). An AMBIGUOUS failure counts half: its credit is pulled halfway to
    0.5, which is exactly the uninformative credit (at c = 0.5 both likelihoods
    equal 0.5, so the posterior equals the prior). Successes and conceptual
    failures count in full. [design: the half weight for ambiguous]
    """
    if credit >= 0.5 or blocage_type in (None, "", "none", "conceptual"):
        return credit
    if blocage_type == "linguistic":
        return None
    if blocage_type == "ambiguous":
        return (credit + 0.5) / 2
    return credit


# --------------------------------------------------------------------------- #
# tau — the KC's rigour
# --------------------------------------------------------------------------- #
# tau in [0, 1] says how EXACT a KC must be to count as known: a definition to
# recall word for word or a procedure that must be carried out without error
# (high tau) versus a general idea (low tau). It comes with the KC (inferred at
# creation from its type and target level), 0.5 being neutral: at tau = 0.5
# every rule below reduces exactly to the behaviour without tau.
#
# tau acts on the rules WE own, never on a parameter fitted to data:
#   - how much a partial answer is worth (rigor_credit),
#   - the mastery threshold (mastery_threshold),
#   - the forgetting prior (forgetting.get_lambda),
# and it is exposed to RAYA, which chooses the EMT entry point with it.
TAU_NEUTRAL = 0.5


def kc_tau(concept_node: dict | None) -> float:
    """The KC's rigour, bounded to [0, 1]; neutral when missing or unreadable."""
    try:
        value = float((concept_node or {}).get("tau"))
    except (TypeError, ValueError):
        return TAU_NEUTRAL
    return max(0.0, min(1.0, value))


def rigor_credit(credit: float, tau: float) -> float:
    """What a partial answer is worth on a KC of rigour tau: credit ** (2 tau).

    Full success and full failure are unchanged; only partial answers move. On
    a rigorous KC (tau 0.8) a 0.7 answer counts as 0.56 — close is not enough
    when exactness is the point; on a loose one (tau 0.3), as 0.81. This is the
    "update speed" tau modulates: how fast partial answers build mastery,
    without touching p_transit, which is fitted to data. [design: the exponent]
    """
    if credit <= 0.0 or credit >= 1.0:
        return credit
    return credit ** (2.0 * tau)


def observation(credit: float, assisted: bool, blocage_type: str | None, tau: float = TAU_NEUTRAL) -> dict:
    """One graded attempt, as the update and the ledger use it.

    The single definition shared by /analyze and /update_concept_state, so the
    two doors into a student's state apply the same evidence rules: rigour
    first (what the answer is worth on this KC), then the blocage rule
    (whether a failure says anything about the concept at all).
    """
    worth = rigor_credit(max(0.0, min(1.0, credit)), tau)
    counted = blocage_evidence(worth, blocage_type)
    return {
        "credit": counted if counted is not None else worth,
        "raw_credit": credit,
        "assisted": bool(assisted),
        "blocage_type": blocage_type,
        "counted": counted is not None,
    }


# --------------------------------------------------------------------------- #
# Mastery criterion (dual condition)
# --------------------------------------------------------------------------- #
# [prior] Corbett & Anderson (1995) mastery criterion, at neutral rigour.
MASTERY_THRESHOLD = 0.95
# [design] How far rigour moves it: 0.93 at tau 0.3, 0.98 at tau 0.8.
MASTERY_TAU_SLOPE = 0.10


def mastery_threshold(tau: float = TAU_NEUTRAL) -> float:
    """The K a KC of rigour tau must reach to count as mastered."""
    return min(0.99, MASTERY_THRESHOLD + MASTERY_TAU_SLOPE * (tau - TAU_NEUTRAL))
# [design] What "low slip" means for mastery; no published value for K-12.
MAX_SLIP_FOR_MASTERY = 0.15
# [design] Solid credit on the recent autonomous attempts.
MIN_PARTIAL_CREDIT_AVG = 0.7
# How many recent autonomous attempts the credit condition reads.
RECENT_AUTONOMOUS_WINDOW = 3


def push_autonomous_credit(recent: list | None, credit: float) -> list[float]:
    """Append an unassisted attempt's credit, keeping the last WINDOW, oldest first."""
    return (list(recent or []) + [round(credit, 4)])[-RECENT_AUTONOMOUS_WINDOW:]


def mastery_credit(student_state: dict | None) -> float | None:
    """The credit figure the dual condition reads for a student's KC state.

    The mean of the last three AUTONOMOUS attempts, and None (so: not
    mastered) while fewer than three exist — mastery has to be shown without
    help. Rows written before migration 011 have no list at all (NULL); for
    those, fall back to the all-time average until new attempts fill the list.
    """
    if not student_state:
        return None
    recent = student_state.get("recent_autonomous_credits")
    if recent is None:
        return student_state.get("partial_credit_avg")
    if len(recent) < RECENT_AUTONOMOUS_WINDOW:
        return None
    return sum(recent) / len(recent)


def is_mastered(
    k_effective: float, p_slip: float, partial_credit_avg: float | None, tau: float = TAU_NEUTRAL
) -> bool:
    """Dual mastery condition: high effective mastery AND low slip AND solid credit."""
    return (
        k_effective >= mastery_threshold(tau)
        and p_slip <= MAX_SLIP_FOR_MASTERY
        and partial_credit_avg is not None
        and partial_credit_avg >= MIN_PARTIAL_CREDIT_AVG
    )


# [design] Status threshold used across the API.
GAP_THRESHOLD = 0.4      # below -> "gap"; otherwise "partial" until mastered


def classify_status(
    k_effective: float,
    p_slip: float = BKT_PRIORS["p_slip"],
    partial_credit_avg: float | None = 0.5,
    tau: float = TAU_NEUTRAL,
) -> str:
    """Map an effective mastery to a human-facing status label.

    "mastered" is reserved for the dual condition. It used to also be granted to
    any K >= 0.7, which made the dual condition dead code: a KC slipped on 40%
    of the time read "mastered" on the same response that raised false_mastery.
    """
    if is_mastered(k_effective, p_slip, partial_credit_avg, tau):
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
        return {"k_raw": None, "k_effective": None, "p_slip": None, "status": "unknown",
                "tau": kc_tau(concept_node)}

    k_raw = student_state.get("mastery_score_raw") or 0.0
    last_at = student_state.get("last_strong_signal_at")
    k_effective = forgetting.compute_effective_mastery(
        k_raw,
        concept_node.get("type_kc", "conceptual"),
        last_at,
        lambda_override=forgetting.get_lambda(concept_node, student_state),
        floor=get_bkt_params(concept_node)["p_init"],
    )
    p_slip = student_state.get("p_slip_personal") or get_bkt_params(concept_node)["p_slip"]
    pc_avg = mastery_credit(student_state)
    return {
        "k_raw": round(k_raw, 4),
        "k_effective": round(k_effective, 4),
        "p_slip": p_slip,
        "status": classify_status(k_effective, p_slip, pc_avg, kc_tau(concept_node)),
        "tau": kc_tau(concept_node),
    }
