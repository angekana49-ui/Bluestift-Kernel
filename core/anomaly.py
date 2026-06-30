"""Pedagogical-safety anomaly detection.

Pure detectors over a single /analyze context. Each returns an alert dict
(alert_type, alert_severity, alert_details, concept) or None. The orchestrator
`detect_anomalies` runs them all and returns the list of fired alerts.

Patterns and signatures follow the Bluestift corpus (§1.5): false mastery
(Corbett), passive dependency (Bastani/Sweller), re-emergence errors,
cognitive overload (Sweller), fixed mindset (Dweck). OOD and long-window
temporal inconsistency need population/history baselines and are left for later.
"""
from __future__ import annotations

# Thresholds (tunable, later calibrated on real data).
FALSE_MASTERY_K = 0.9          # looks mastered...
SLIP_HIGH = 0.15              # ...but slips too often -> false mastery
FIXED_MINDSET_M = 0.4         # persistent low mindset
OVERLOAD_FAILURE_RATE = 0.5   # majority of attempts fail
PASSIVE_ASSISTED_RATIO = 0.5  # mostly leaning on RAYA
MASTERED_PREREQ = 0.7         # a prerequisite that "looks solid"
FAILING = 0.4                 # a KC the student is failing


def _alert(alert_type: str, severity: str, details: dict, concept: str | None = None) -> dict:
    return {
        "alert_type": alert_type,
        "alert_severity": severity,
        "alert_details": details,
        "concept": concept,
    }


def detect_false_mastery(label: str, k_effective: float, p_slip: float) -> dict | None:
    """K is high but the slip rate is high — mastery that won't hold (Corbett)."""
    if k_effective >= FALSE_MASTERY_K and p_slip > SLIP_HIGH:
        return _alert(
            "false_mastery", "high",
            {"k_effective": round(k_effective, 4), "p_slip": round(p_slip, 4)},
            concept=label,
        )
    return None


def detect_passive_dependency(attempts: list[dict]) -> dict | None:
    """Fast, assisted, error-free progress = leaning on the tutor, not learning."""
    if len(attempts) < 2:
        return None
    n = len(attempts)
    assisted = sum(1 for a in attempts if a.get("is_assisted"))
    fast = sum(1 for a in attempts if a.get("response_time_estimate") == "fast")
    successes = sum(1 for a in attempts if a.get("outcome") == "success")
    if (
        assisted / n >= PASSIVE_ASSISTED_RATIO
        and successes == n            # no intermediate errors
        and fast / n >= 0.5
    ):
        return _alert(
            "passive_dependency", "medium",
            {"attempts": n, "assisted": assisted, "fast": fast},
        )
    return None


def detect_cognitive_overload(attempts: list[dict], blocage_type: str | None) -> dict | None:
    """A majority of failures with a conceptual block — working memory overloaded."""
    if not attempts:
        return None
    failures = sum(1 for a in attempts if a.get("outcome") == "failure")
    if failures / len(attempts) >= OVERLOAD_FAILURE_RATE and blocage_type == "conceptual":
        return _alert(
            "cognitive_overload", "medium",
            {"failure_rate": round(failures / len(attempts), 3)},
        )
    return None


def detect_fixed_mindset(m_score: float | None) -> dict | None:
    """A low mindset signal — protect M (intervene on mindset, not on K) (Dweck)."""
    if m_score is not None and m_score <= FIXED_MINDSET_M:
        return _alert("fixed_mindset", "medium", {"m_score": round(m_score, 4)})
    return None


def detect_re_emergence(label: str, k_effective: float, prereq_masteries: list[float]) -> dict | None:
    """Foundations look solid but the dependent KC fails — a sub-optimal rule."""
    if not prereq_masteries:
        return None
    if k_effective < FAILING and all(m >= MASTERED_PREREQ for m in prereq_masteries):
        return _alert(
            "re_emergence_error", "medium",
            {"k_effective": round(k_effective, 4), "prereqs": len(prereq_masteries)},
            concept=label,
        )
    return None


def detect_anomalies(
    kc_records: list[dict],
    attempts: list[dict],
    blocage_type: str | None,
    m_score: float | None,
) -> list[dict]:
    """Run every detector and collect the fired alerts.

    Args:
        kc_records: per-KC dicts with label, k_effective, p_slip, prereq_masteries.
        attempts: extracted attempts for the conversation.
        blocage_type: detected block type for the conversation.
        m_score: the student's current mindset score.
    """
    alerts: list[dict] = []

    for rec in kc_records:
        for alert in (
            detect_false_mastery(rec["label"], rec["k_effective"], rec["p_slip"]),
            detect_re_emergence(rec["label"], rec["k_effective"], rec.get("prereq_masteries", [])),
        ):
            if alert:
                alerts.append(alert)

    for alert in (
        detect_passive_dependency(attempts),
        detect_cognitive_overload(attempts, blocage_type),
        detect_fixed_mindset(m_score),
    ):
        if alert:
            alerts.append(alert)

    return alerts
