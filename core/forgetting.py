"""Exponential forgetting / decay of mastery.

K_effective = K_raw * exp(-lambda * delta_days)

Lambda is a *living* parameter resolved with a 3-level priority:
  1. the student's personal lambda on this KC (most precise),
  2. the KC's empirical lambda once enough interactions exist,
  3. the literature prior by KC type.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Literature priors by KC type.
LAMBDA_PRIORS = {
    "procedural": 0.01,
    "declarative": 0.05,
    "conceptual": 0.02,
}

# Minimum interactions before a KC's empirical lambda is trusted over the prior.
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


def compute_effective_mastery(
    k_raw: float,
    kc_type: str,
    last_interaction_at,
    lambda_override: float | None = None,
) -> float:
    """Apply exponential decay to a raw mastery score.

    Args:
        k_raw: stored mastery probability.
        kc_type: KC type used to pick the prior lambda when no override given.
        last_interaction_at: datetime or ISO string of the last strong signal.
        lambda_override: explicit lambda (already resolved via `get_lambda`).
    """
    if last_interaction_at is None:
        return max(0.0, min(1.0, k_raw))

    lambda_val = (
        lambda_override
        if lambda_override is not None
        else LAMBDA_PRIORS.get(kc_type, 0.02)
    )
    last_dt = _parse_ts(last_interaction_at)
    delta_days = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400.0
    if delta_days < 0:
        delta_days = 0.0
    return max(0.0, k_raw * math.exp(-lambda_val * delta_days))
