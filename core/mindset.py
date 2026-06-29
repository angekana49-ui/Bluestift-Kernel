"""Mindset score M.

A sigmoid blend of behavioural signals, clamped away from {0,1} because — per
Dweck — a fully fixed or fully growth mindset is never a settled fact.
"""
from __future__ import annotations

import math

# Weights (sum of magnitudes ~ 1). Abandon hurts, persistence/time/quality help.
W_ABANDON = -0.35
W_PERSISTENCE = 0.30
W_TIME = 0.20
W_QUALITY = 0.15

M_FLOOR = 0.05
M_CEIL = 0.95


def compute_mindset_score(
    abandon_rate: float,
    persistence_score: float,
    time_on_task: float,
    interaction_quality: float,
) -> float:
    """Blend behavioural signals into a mindset score in [0.05, 0.95]."""
    m_raw = (
        W_ABANDON * abandon_rate
        + W_PERSISTENCE * persistence_score
        + W_TIME * time_on_task
        + W_QUALITY * interaction_quality
    )
    m_score = 1.0 / (1.0 + math.exp(-m_raw))
    return max(M_FLOOR, min(M_CEIL, m_score))


def classify_mindset(m_score: float) -> str:
    """Map a mindset score to a coarse label."""
    if m_score >= 0.66:
        return "growth"
    if m_score <= 0.4:
        return "fixed"
    return "mixed"
