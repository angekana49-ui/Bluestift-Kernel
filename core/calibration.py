"""Self-calibration of living parameters.

The Kernel starts on literature priors and refines parameters from real data:
  - per-student personal lambda (observed forgetting),
  - per-KC empirical difficulty and lambda (aggregated over students).

These run in the background and never block /analyze.
"""
from __future__ import annotations

import math

# Bounds for an observed lambda — values outside are treated as noise.
LAMBDA_MIN = 0.001
LAMBDA_MAX = 0.2

# Blend weight for newly observed lambda vs. the previous estimate.
OBSERVED_WEIGHT = 0.7
PRIOR_WEIGHT = 0.3

# Calibration gates.
MIN_STUDENTS_FOR_KC_CALIBRATION = 10
MIN_INTERACTIONS_PER_STUDENT = 5


def calibrate_personal_lambda(
    k_raw_before: float,
    k_raw_after: float,
    delta_days: float,
    current_lambda: float,
) -> float:
    """Estimate a student's personal decay rate from observed forgetting.

    From K_after ~= K_before * e^(-lambda * delta_days):
        lambda = -ln(K_after / K_before) / delta_days

    The result is bounded and blended with the current estimate to avoid jumps.
    """
    if k_raw_before <= 0 or k_raw_after <= 0 or delta_days < 1:
        return current_lambda  # not enough signal

    try:
        observed = -math.log(k_raw_after / k_raw_before) / delta_days
    except (ValueError, ZeroDivisionError):
        return current_lambda

    observed = max(LAMBDA_MIN, min(LAMBDA_MAX, observed))
    return OBSERVED_WEIGHT * observed + PRIOR_WEIGHT * current_lambda


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
