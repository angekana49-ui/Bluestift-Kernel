"""Pedagogical-safety anomaly detection.

Pure detectors over a single /analyze context. Each returns an alert dict
(alert_type, alert_severity, alert_details, concept) or None. The orchestrator
`detect_anomalies` runs them all and returns the list of fired alerts.

Patterns and signatures follow the Bluestift corpus (§1.5): false mastery
(Corbett), passive dependency (Bastani/Sweller), re-emergence errors,
cognitive overload (Sweller), fixed mindset (Dweck).

Two detectors read beyond a single conversation and so take extra arguments:
`inconsistency_high` needs the KC's trajectory history, and `ood_distribution`
needs the population baseline each KC carries once calibrated. Both stay pure —
the caller does the loading.
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

# --- Temporal inconsistency (Hooshyar) ------------------------------------- #
INCONSISTENCY_WINDOW = 20      # the corpus' window: the last 20 snapshots
# The corpus states the metric over 20 interactions. Requiring a full 20 before
# the metric may fire means it never fires for a first cohort — the students who
# most need the safety net. We compute it from 6 snapshots up, over whatever part
# of the window exists, and carry the count in the alert so a reader can weigh it.
MIN_SNAPSHOTS_FOR_INCONSISTENCY = 6
INCONSISTENCY_HIGH = 0.40      # > 0.40 -> unstable knowledge state

# --- Out-of-distribution (Amodei/Goodhart) --------------------------------- #
# BKT priors come from North-American and Estonian data. A deployment whose
# students diverge from that distribution fails silently: the numbers stay
# plausible while describing nobody. We compare a student against the *local*
# population baseline each KC accumulates, not against the priors.
MIN_KCS_FOR_OOD = 3            # one odd KC is a bad day, not a distribution
OOD_DEVIATION = 0.4            # mean signed deviation from the local baseline


def _alert(alert_type: str, severity: str, details: dict, concept: str | None = None) -> dict:
    return {
        "alert_type": alert_type,
        "alert_severity": severity,
        "alert_details": details,
        "concept": concept,
    }


def volatility_score(series: list[float]) -> float:
    """Mean absolute step between consecutive mastery snapshots.

    How much the estimate moves per interaction, regardless of direction. A
    student converging on mastery has a low score; one bouncing has a high one.
    """
    if len(series) < 2:
        return 0.0
    steps = [abs(b - a) for a, b in zip(series, series[1:])]
    return round(sum(steps) / len(steps), 4)


def temporal_inconsistency(series: list[float]) -> float:
    """How much of the movement in a mastery series is churn rather than progress.

    Total variation vs net displacement: a monotonic climb spends all its
    movement on progress and scores 0; a series that ends where it started after
    swinging up and down scores 1. This is the stability metric behind
    `inconsistency_high` — an unstable state, not a low one.
    """
    if len(series) < 2:
        return 0.0
    total_variation = sum(abs(b - a) for a, b in zip(series, series[1:]))
    if total_variation == 0:
        return 0.0  # perfectly flat: stable, not inconsistent
    net = abs(series[-1] - series[0])
    return round(max(0.0, 1.0 - net / total_variation), 4)


def detect_inconsistency_high(label: str, series: list[float]) -> dict | None:
    """A mastery estimate that oscillates instead of settling (Hooshyar).

    `series` is the KC's k_raw snapshots, oldest first. High inconsistency means
    the student's knowledge state is unstable — the estimate itself is unreliable,
    so downstream decisions built on it (sequencing, "mastered") are too.
    """
    window = series[-INCONSISTENCY_WINDOW:]
    if len(window) < MIN_SNAPSHOTS_FOR_INCONSISTENCY:
        return None
    rate = temporal_inconsistency(window)
    if rate <= INCONSISTENCY_HIGH:
        return None
    return _alert(
        "inconsistency_high", "medium",
        {
            "inconsistency_rate": rate,
            "volatility_score": volatility_score(window),
            "interactions_count": len(window),
        },
        concept=label,
    )


def detect_ood_distribution(kc_records: list[dict]) -> dict | None:
    """The student doesn't behave like the population the parameters describe.

    For every KC that carries a calibrated `empirical_difficulty` (the local
    fraction of students who struggle on it), compare what the baseline expects
    against what this student did. A large *signed* mean deviation across several
    KCs means the calibrated distribution doesn't describe this student: their
    mastery estimates are being computed with the wrong priors.

    Direction matters and is reported: `below_population` is the dangerous one —
    a student silently failing under parameters tuned for someone else.
    """
    deviations: list[float] = []
    for rec in kc_records:
        baseline = rec.get("population_difficulty")
        if baseline is None:
            continue  # KC not calibrated yet: no local baseline to compare against
        struggling = 1.0 if rec["k_effective"] < 0.5 else 0.0
        deviations.append(struggling - baseline)

    if len(deviations) < MIN_KCS_FOR_OOD:
        return None
    mean_dev = sum(deviations) / len(deviations)
    if abs(mean_dev) < OOD_DEVIATION:
        return None
    return _alert(
        "ood_distribution", "high" if mean_dev > 0 else "medium",
        {
            "mean_deviation": round(mean_dev, 4),
            "direction": "below_population" if mean_dev > 0 else "above_population",
            "kcs_compared": len(deviations),
        },
    )


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
        kc_records: per-KC dicts with label, k_effective, p_slip, and optionally
            prereq_masteries, trajectory (k_raw snapshots, oldest first) and
            population_difficulty (the KC's calibrated empirical difficulty).
            The last two are what the history- and population-aware detectors
            need; a record without them simply skips those.
        attempts: extracted attempts for the conversation.
        blocage_type: detected block type for the conversation.
        m_score: the student's current mindset score.
    """
    alerts: list[dict] = []

    for rec in kc_records:
        for alert in (
            detect_false_mastery(rec["label"], rec["k_effective"], rec["p_slip"]),
            detect_re_emergence(rec["label"], rec["k_effective"], rec.get("prereq_masteries", [])),
            detect_inconsistency_high(rec["label"], rec.get("trajectory", [])),
        ):
            if alert:
                alerts.append(alert)

    for alert in (
        detect_passive_dependency(attempts),
        detect_cognitive_overload(attempts, blocage_type),
        detect_fixed_mindset(m_score),
        detect_ood_distribution(kc_records),
    ):
        if alert:
            alerts.append(alert)

    return alerts
