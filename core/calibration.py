"""Self-calibration of living parameters.

The Kernel starts on literature priors and refines parameters from real data:
  - per-student personal lambda (observed forgetting),
  - per-KC empirical difficulty and lambda (aggregated over students),
  - per-KC BKT parameters, fitted by EM on the logged evidence (fit_bkt_em).

These run in the background and never block /analyze.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# Bounds for an observed lambda — values outside are treated as noise.
LAMBDA_MIN = 0.001
LAMBDA_MAX = 0.2

# [design] Step size of the online maximum-likelihood update of a personal
# lambda, in log-space. At 0.5, one failure a month after a solid session raises
# lambda by ~20%, one success lowers it by ~10%: it takes several consistent
# returns to move a student's forgetting rate far from the prior.
LAMBDA_STEP = 0.5

# [design] Calibration gates.
MIN_STUDENTS_FOR_KC_CALIBRATION = 10
MIN_INTERACTIONS_PER_STUDENT = 5

# How long a KC's calibration stays fresh. Recalibration reads every student's
# state for that KC, so doing it on each interaction would turn a popular KC
# into a full table scan per conversation, for an aggregate that barely moves
# when one more student answers one more question.
CALIBRATION_COOLDOWN_HOURS = 6


def is_calibration_due(node: dict, now: datetime | None = None) -> bool:
    """Whether this KC is worth recalibrating right now.

    Takes the KC node the caller already holds, so the check costs no query —
    it is meant to be called on the hot path before scheduling background work.
    A KC that has never been calibrated is always due.
    """
    last = node.get("last_calibration_at")
    if not last:
        return True
    if isinstance(last, str):
        try:
            last = datetime.fromisoformat(last.replace("Z", "+00:00"))
        except ValueError:
            return True  # unparseable timestamp: treat as never calibrated
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - last) >= timedelta(hours=CALIBRATION_COOLDOWN_HOURS)


def update_personal_lambda(
    current_lambda: float,
    k_stored: float,
    floor: float,
    delta_days: float,
    credit: float,
    p_slip: float,
    p_guess: float,
) -> float:
    """One online maximum-likelihood step on a student's forgetting rate.

    After `delta_days` without practice the decay model predicts
        K(lambda)  = floor + (k_stored - floor) * exp(-lambda * delta_days)
        P(correct) = p_guess + (1 - p_slip - p_guess) * K(lambda)
    and the first attempt back has `credit`. Move lambda along the gradient of
    that attempt's log-likelihood: an unexpected failure means the student
    forgets faster than assumed, an unexpected success means slower.

    The previous estimator solved `k_after = k_before * exp(-lambda * dt)` with
    the attempt's CREDIT standing in for k_after — a single 0/1 grade treated
    as a measurement of mastery. A success gave a negative lambda (clamped to
    the minimum), a failure no estimate at all.
    """
    gap = k_stored - floor
    if delta_days < 1 or gap <= 0 or current_lambda <= 0:
        return current_lambda  # no forgetting to learn from
    decay = math.exp(-current_lambda * delta_days)
    discrimination = 1.0 - p_slip - p_guess
    p_correct = p_guess + discrimination * (floor + gap * decay)
    p_correct = min(1 - 1e-6, max(1e-6, p_correct))
    dloglik_dp = credit / p_correct - (1.0 - credit) / (1.0 - p_correct)
    dp_dlambda = discrimination * (-delta_days * gap * decay)
    grad = dloglik_dp * dp_dlambda
    # Log-space step: multiplicative, so lambda stays positive and moves by
    # the same proportion whatever its scale.
    new = current_lambda * math.exp(LAMBDA_STEP * current_lambda * grad)
    return max(LAMBDA_MIN, min(LAMBDA_MAX, new))


# --------------------------------------------------------------------------- #
# BKT parameter fitting (EM / Baum-Welch on a two-state HMM)
# --------------------------------------------------------------------------- #
# [design] Minimum evidence before a KC's fitted parameters replace the priors.
MIN_SEQUENCES_FOR_BKT_FIT = 30
MIN_EVENTS_FOR_BKT_FIT = 150
# [design] Strength of the literature prior, in pseudo-observations. MAP rather
# than maximum likelihood: with little data the fit stays near the prior
# instead of jumping to whatever a few students happened to do, and the prior's
# pull fades as evidence accumulates.
BKT_PRIOR_STRENGTH = 10.0
EM_MAX_ITERATIONS = 100
EM_TOLERANCE = 1e-6


def _emissions(c: float, slip: float, guess: float) -> tuple[float, float]:
    """P(observation | learned), P(observation | not learned) for credit c.

    The same soft-evidence mixture as bkt.update_bkt, so the fitted parameters
    mean exactly what the update uses them for.
    """
    return c * (1 - slip) + (1 - c) * slip, c * guess + (1 - c) * (1 - guess)


def fit_bkt_em(sequences: list[list[float]], priors: dict | None = None) -> dict | None:
    """Fit p_init, p_transit, p_slip, p_guess to per-student credit sequences.

    Args:
        sequences: one chronological list of credits in [0, 1] per student,
            autonomous attempts only (assisted ones follow another guess rate).
        priors: the values to shrink towards; the literature priors by default.

    Returns:
        The fitted parameters, bounded to the identifiable region, or None when
        there is not enough evidence to replace the priors.
    """
    from core.bkt import BKT_PRIORS, MAX_SLIP_OR_GUESS

    sequences = [seq for seq in sequences if seq]
    if (
        len(sequences) < MIN_SEQUENCES_FOR_BKT_FIT
        or sum(len(seq) for seq in sequences) < MIN_EVENTS_FOR_BKT_FIT
    ):
        return None
    prior = dict(priors or BKT_PRIORS)
    p0, pt, ps, pg = prior["p_init"], prior["p_transit"], prior["p_slip"], prior["p_guess"]
    a = BKT_PRIOR_STRENGTH
    prev_ll = -math.inf

    for _ in range(EM_MAX_ITERATIONS):
        init_num = trans_num = trans_den = 0.0
        slip_num = slip_den = guess_num = guess_den = 0.0
        ll = 0.0
        for seq in sequences:
            n = len(seq)
            em = [_emissions(c, ps, pg) for c in seq]
            # Forward pass (L = learned, U = not yet), normalised per step.
            alpha, scales = [], []
            for t in range(n):
                if t == 0:
                    f_l, f_u = p0 * em[0][0], (1 - p0) * em[0][1]
                else:
                    pl, pu = alpha[-1]
                    f_l = (pl + pu * pt) * em[t][0]
                    f_u = pu * (1 - pt) * em[t][1]
                z = (f_l + f_u) or 1e-300
                scales.append(z)
                alpha.append((f_l / z, f_u / z))
            ll += sum(math.log(z) for z in scales)
            # Backward pass with the same scales. No forgetting: L -> L always.
            beta = [(1.0, 1.0)] * n
            for t in range(n - 2, -1, -1):
                bl, bu = beta[t + 1]
                el, eu = em[t + 1]
                z = scales[t + 1]
                beta[t] = (el * bl / z, (pt * el * bl + (1 - pt) * eu * bu) / z)
            gammas = []
            for t in range(n):
                gl = alpha[t][0] * beta[t][0]
                gu = alpha[t][1] * beta[t][1]
                zz = (gl + gu) or 1e-300
                gammas.append((gl / zz, gu / zz))
            init_num += gammas[0][0]
            for t in range(n - 1):
                # Expected U -> L transitions between t and t+1.
                el = em[t + 1][0]
                trans_num += alpha[t][1] * pt * el * beta[t + 1][0] / scales[t + 1]
                trans_den += gammas[t][1]
            for (gl, gu), c, (el, eu) in zip(gammas, seq, em):
                # Soft evidence: the expected share of the observation that was
                # an incorrect answer given L (a slip), a correct one given U (a
                # guess). For c in {0, 1} this is the textbook count.
                slip_num += gl * ((1 - c) * ps / el if el > 0 else 0.0)
                slip_den += gl
                guess_num += gu * (c * pg / eu if eu > 0 else 0.0)
                guess_den += gu

        # M-step: MAP with Beta pseudo-counts centred on the prior.
        p0 = (init_num + a * prior["p_init"]) / (len(sequences) + a)
        pt = (trans_num + a * prior["p_transit"]) / (trans_den + a)
        ps = min((slip_num + a * prior["p_slip"]) / (slip_den + a), MAX_SLIP_OR_GUESS)
        pg = min((guess_num + a * prior["p_guess"]) / (guess_den + a), MAX_SLIP_OR_GUESS)
        if abs(ll - prev_ll) < EM_TOLERANCE * max(1.0, abs(ll)):
            break
        prev_ll = ll

    return {
        "p_init": round(p0, 4),
        "p_transit": round(pt, 4),
        "p_slip": round(ps, 4),
        "p_guess": round(pg, 4),
    }


def credit_sequences(events: list[dict]) -> list[list[float]]:
    """Group learning_events rows (already ordered by time) into per-student
    credit sequences, autonomous attempts only."""
    by_user: dict[str, list[float]] = {}
    for e in events:
        if e.get("is_assisted") or e.get("credit") is None:
            continue
        by_user.setdefault(e["user_id"], []).append(float(e["credit"]))
    return list(by_user.values())


def compute_empirical_kc_params(states: list[dict]) -> dict | None:
    """Aggregate per-student states into empirical KC parameters.

    Returns None when there is not enough data to update safely (avoids
    overfitting on a handful of students).

    Returns a dict with empirical_difficulty, interactions_count, and optionally
    lambda_decay when personal lambdas are available.
    """
    if not states or len(states) < MIN_STUDENTS_FOR_KC_CALIBRATION:
        return None

    eligible = [s for s in states if (s.get("interactions_on_kc") or 0) >= MIN_INTERACTIONS_PER_STUDENT]
    if not eligible:
        return None

    struggling = [s for s in eligible if (s.get("mastery_score_raw") or 0.0) < 0.5]
    empirical_difficulty = len(struggling) / len(eligible)

    personal_lambdas = [s["lambda_personal"] for s in states if s.get("lambda_personal")]
    empirical_lambda = (
        sum(personal_lambdas) / len(personal_lambdas) if personal_lambdas else None
    )

    result = {
        "empirical_difficulty": round(empirical_difficulty, 4),
        "interactions_count": sum((s.get("interactions_on_kc") or 0) for s in states),
    }
    if empirical_lambda is not None:
        result["lambda_decay"] = round(empirical_lambda, 5)
    return result
