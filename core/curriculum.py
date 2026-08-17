"""The School → AI → Student channel.

A school does not only read reports about its students; it calibrates what the
Kernel does with them. `schools.school_curriculum_layers` carries four kinds of
layer (see migration 010) and this module turns them into decisions:

    curriculum    -> which concepts belong to the official program
    kc_priorities -> weights that reorder the remediation sequence
    objectives    -> mastery targets with deadlines, reported against real state
    custom_rules  -> passed through for RAYA's prompt; not the Kernel's business

Everything here is pure: the caller loads the rows, this parses and applies them.
A malformed payload degrades to "no layer" rather than raising — a school's bad
JSON must never take down a student's analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# A target not yet met, whose deadline falls inside this window, is at risk.
AT_RISK_DAYS = 14

# Bounds on a school-supplied priority multiplier. A school can express "teach
# this first", not rewrite the graph: an unbounded weight would let a bad payload
# override the Kernel's own reasoning entirely.
MIN_PRIORITY = 0.1
MAX_PRIORITY = 5.0
DEFAULT_PRIORITY = 1.0


def normalize(label: str) -> str:
    """Match kc_registry's canonical label form, so schools can write prose."""
    return str(label).strip().lower().replace(" ", "_")


@dataclass
class CurriculumLayers:
    """The active layers for one school, parsed and ready to apply."""

    school_id: str | None = None
    concepts: set[str] = field(default_factory=set)          # curriculum
    weights: dict[str, float] = field(default_factory=dict)  # kc_priorities
    objectives: list[dict] = field(default_factory=list)     # objectives
    rules: list[str] = field(default_factory=list)           # custom_rules

    @property
    def is_empty(self) -> bool:
        return not (self.concepts or self.weights or self.objectives or self.rules)

    def applied(self) -> list[str]:
        """Which layers actually carry something — surfaced in the response."""
        names = []
        if self.concepts:
            names.append("curriculum")
        if self.weights:
            names.append("kc_priorities")
        if self.objectives:
            names.append("objectives")
        if self.rules:
            names.append("custom_rules")
        return names


def parse_layers(rows: list[dict], school_id: str | None = None) -> CurriculumLayers:
    """Parse active layer rows into one merged, validated structure.

    Rows of the same type merge (a school can split its program across several
    rows). Anything unparseable in a row is skipped, keeping the rest.
    """
    layers = CurriculumLayers(school_id=school_id)

    for row in rows or []:
        payload = row.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        layer_type = row.get("layer_type") or "curriculum"

        if layer_type == "curriculum":
            for concept in _as_list(payload.get("concepts")):
                if isinstance(concept, str) and concept.strip():
                    layers.concepts.add(normalize(concept))
            # Legacy rows (pre-010) only ever carried a bare concept list.
            for concept in _as_list(row.get("concept_ids")):
                if isinstance(concept, str) and concept.strip():
                    layers.concepts.add(normalize(concept))

        elif layer_type == "kc_priorities":
            weights = payload.get("weights")
            if isinstance(weights, dict):
                for concept, weight in weights.items():
                    try:
                        value = float(weight)
                    except (TypeError, ValueError):
                        continue
                    layers.weights[normalize(concept)] = _clamp_priority(value)

        elif layer_type == "objectives":
            for target in _as_list(payload.get("targets")):
                parsed = _parse_target(target)
                if parsed:
                    layers.objectives.append(parsed)

        elif layer_type == "custom_rules":
            for rule in _as_list(payload.get("rules")):
                if isinstance(rule, str) and rule.strip():
                    layers.rules.append(rule.strip())

    return layers


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _clamp_priority(value: float) -> float:
    return max(MIN_PRIORITY, min(MAX_PRIORITY, value))


def _parse_target(target) -> dict | None:
    if not isinstance(target, dict):
        return None
    concept = target.get("concept")
    if not isinstance(concept, str) or not concept.strip():
        return None
    try:
        mastery = float(target.get("mastery", 0.8))
    except (TypeError, ValueError):
        mastery = 0.8
    return {
        "concept": normalize(concept),
        "mastery": max(0.0, min(1.0, mastery)),
        "due_at": target.get("due_at") if isinstance(target.get("due_at"), str) else None,
    }


def priority_of(label: str, weights: dict[str, float]) -> float:
    """The school's multiplier for a concept; neutral when it named none."""
    return weights.get(normalize(label), DEFAULT_PRIORITY)


def prioritize(labels: list[str], weights: dict[str, float]) -> list[str]:
    """Order concepts by the school's priority, keeping the original order on ties.

    A stable sort on the negated weight: concepts the school pushed up come
    first, everything else keeps the sequence the Kernel derived. The school
    reorders what to teach; it does not get to invent the dependency chain.
    """
    if not weights:
        return list(labels)
    return sorted(labels, key=lambda label: -priority_of(label, weights))


def objective_report(
    objectives: list[dict],
    mastery_by_label: dict[str, float],
    now: datetime | None = None,
) -> list[dict]:
    """Report each school objective against the student's real mastery.

    Statuses: `met` (target reached), `overdue` (deadline passed, not reached),
    `at_risk` (deadline within AT_RISK_DAYS, not reached), `pending` (still has
    time), and `unknown` (the student has no evidence on that concept yet —
    reported rather than silently counted as failure).
    """
    now = now or datetime.now(timezone.utc)
    report = []

    for objective in objectives:
        concept = objective["concept"]
        target = objective["mastery"]
        observed = mastery_by_label.get(concept)
        due_at = _parse_due(objective.get("due_at"))

        if observed is None:
            status = "unknown"
        elif observed >= target:
            status = "met"
        elif due_at is None:
            status = "pending"
        elif due_at < now:
            status = "overdue"
        elif (due_at - now).days <= AT_RISK_DAYS:
            status = "at_risk"
        else:
            status = "pending"

        report.append(
            {
                "concept": concept,
                "target_mastery": round(target, 4),
                "observed_mastery": round(observed, 4) if observed is not None else None,
                "due_at": objective.get("due_at"),
                "status": status,
            }
        )

    return report


def _parse_due(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
