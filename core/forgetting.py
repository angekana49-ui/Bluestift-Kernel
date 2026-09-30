"""Exponential forgetting / decay of mastery.

    K_effective = floor + (K_raw - floor) * exp(-lambda * delta_days)

K is a probability of mastery, so it decays towards what we would believe
about this student with no evidence at all — the KC's p_init — never to 0.
Decaying to 0 claimed certainty that the student does NOT know, which put a
student who once learned the concept below one who never saw it. A K already
below the floor is left alone: time passing does not make a failure better.
(Forgetting in BKT: Qiu et al. 2011; Khajah, Lindsey & Mozer 2016.)

Lambda is a *living* parameter resolved with a 3-level priority:
  1. the student's personal lambda on this KC (most precise),
  2. the KC's empirical lambda once enough interactions exist,
  3. the literature prior by KC type.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from core.bkt import BKT_PRIORS

# [design] Decay rates per day by KC type — half-lives of ~69 (procedural),
# ~35 (conceptual) and ~14 (declarative) days. The ordering (facts fade faster
# than procedures) follows the retention literature; the values themselves are
# ours, starting points to be replaced by calibration (see PARAMETERS.md).
LAMBDA_PRIORS = {
    "procedural": 0.01,
    "declarative": 0.05,
    "conceptual": 0.02,
}

# [design] Minimum interactions before a KC's empirical lambda is trusted over the prior.
MIN_INTERACTIONS_FOR_EMPIRICAL = 10


def get_lambda(concept_node: dict, student_state: dict | None = None) -> float:
    """Resolve the decay rate for a (KC, student) pair by priority."""
    node = concept_node or {}

    # Level 1 — student-specific empirical lambda.
    if student_state and student_state.get("lambda_personal"):
        return student_state["lambda_personal"]

    # Level 2 — KC empirical lambda, trusted only with enough data.
    if node.get("lambda_decay") and node.get("interactions_count", 0) > MIN_INTERACTIONS_FOR_EMPIRICAL:
        return node["lambda_decay"]

    # Level 3 — literature prior by type.
    return LAMBDA_PRIORS.get(node.get("type_kc", "conceptual"), 0.02)


def _parse_ts(value) -> datetime:
    """Coerce a timestamp (datetime or ISO string) into a tz-aware datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        # Supabase returns ISO strings, sometimes with a trailing 'Z'.
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_since(timestamp) -> float:
    """Whole-and-fractional days elapsed since a timestamp (datetime or ISO str)."""
    if timestamp is None:
        return 0.0
    delta = (datetime.now(timezone.utc) - _parse_ts(timestamp)).total_seconds() / 86400.0
    return max(0.0, delta)


def compute_effective_mastery(
    k_raw: float,
    kc_type: str,
    last_interaction_at,
    lambda_override: float | None = None,
    floor: float | None = None,
) -> float:
    """Apply exponential decay to a raw mastery score.

    Args:
        k_raw: stored mastery probability.
        kc_type: KC type used to pick the prior lambda when no override given.
        last_interaction_at: datetime or ISO string of the last observation.
        lambda_override: explicit lambda (already resolved via `get_lambda`).
        floor: what K decays towards — the KC's p_init. Defaults to the
            literature prior when the caller has no node at hand.
    """
    k_raw = max(0.0, min(1.0, k_raw))
    if last_interaction_at is None:
        return k_raw
    floor = BKT_PRIORS["p_init"] if floor is None else floor
    if k_raw <= floor:
        return k_raw

    lambda_val = (
        lambda_override
        if lambda_override is not None
        else LAMBDA_PRIORS.get(kc_type, 0.02)
    )
    last_dt = _parse_ts(last_interaction_at)
    delta_days = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400.0
    if delta_days < 0:
        delta_days = 0.0
    return floor + (k_raw - floor) * math.exp(-lambda_val * delta_days)
