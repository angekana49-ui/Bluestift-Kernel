"""Pydantic models for the Bluestift Cognitive Kernel API.

All request/response contracts live here. Validation is strict: the Kernel
refuses malformed inputs rather than guessing.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums / shared literals
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class BlocageType(str, Enum):
    conceptual = "conceptual"
    linguistic = "linguistic"
    ambiguous = "ambiguous"
    none = "none"


KCStatus = Literal["mastered", "partial", "gap", "unknown"]


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    kernel: str = "bluestift-cognitive-kernel"


# --------------------------------------------------------------------------- #
# /analyze
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: Role
    # Upper bound caps payload size (anti-abuse: prevents huge LLM cost blowups).
    content: str = Field(..., min_length=1, max_length=8000)


class AnalyzeRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, description="UUID of the student")
    conversation_history: list[Message] = Field(..., min_length=1, max_length=200)
    subject: str = Field(default="MATH", max_length=64)
    level: str = Field(default="unknown", max_length=64)
    trigger: str = Field(default="post_conversation", max_length=64)
    # Diagnose-only mode. A caller that already sent its graded attempts through
    # /update_concept_state must set this to False: the same evidence committed
    # twice would count twice in BKT and inflate mastery. The diagnosis (root
    # gap, path, alerts) is returned either way.
    commit_state: bool = True


class MasteryEntry(BaseModel):
    k_raw: float
    k_effective: float
    status: KCStatus


class AnalyzeResponse(BaseModel):
    request_id: str
    user_id: str
    root_gap: Optional[str] = None
    root_concept_id: Optional[str] = None
    detection_path: list[str] = Field(default_factory=list)
    mastery_map: dict[str, MasteryEntry] = Field(default_factory=dict)
    confidence: float = 0.0
    summary: str = ""
    recommended_path: list[str] = Field(default_factory=list)
    alerts: list[dict] = Field(default_factory=list)
    # Present only for a student who belongs to a school that has set layers:
    # {school_id, layers_applied, objectives, root_gap_in_program, rules}.
    # See core/curriculum.py and migration 010.
    curriculum: Optional[dict] = None
    kernel_version: str
    llm_used: str


# --------------------------------------------------------------------------- #
# /load_profile
# --------------------------------------------------------------------------- #
class LoadProfileRequest(BaseModel):
    user_id: str


class ConceptStateOut(BaseModel):
    concept_id: str
    label: str
    k_raw: float
    k_effective: float
    v_score: float
    p_score: float
    status: KCStatus
    last_interaction_at: Optional[datetime] = None


class MindsetOut(BaseModel):
    m_score: float
    detected_mindset: str


class LoadProfileResponse(BaseModel):
    user_id: str
    concept_states: list[ConceptStateOut] = Field(default_factory=list)
    mindset: Optional[MindsetOut] = None
    last_kernel_update: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# /update_concept_state
# --------------------------------------------------------------------------- #
class UpdateConceptStateRequest(BaseModel):
    """A graded attempt on one KC.

    The caller identifies the KC either by `concept_id` (a `kernel.concept_nodes`
    UUID it already holds) or by `concept_label` — a plain concept name, which the
    Kernel canonicalizes and creates on the fly if unknown, exactly as `/analyze`
    does. Callers like RAYA grade a question long before they know a KC's UUID, so
    the label path is the normal one; they can cache the `concept_id` the response
    returns and use it directly next time.
    """

    user_id: str
    concept_id: Optional[str] = None
    concept_label: Optional[str] = Field(default=None, max_length=128)
    subject: str = Field(default="MATH", max_length=64)
    level: str = Field(default="unknown", max_length=64)
    partial_credit_score: float = Field(..., ge=0.0, le=1.0)
    is_assisted: bool = False
    response_time_ms: Optional[int] = Field(default=None, ge=0)
    blocage_type: BlocageType = BlocageType.none

    @model_validator(mode="after")
    def _require_a_concept(self) -> "UpdateConceptStateRequest":
        if not self.concept_id and not (self.concept_label or "").strip():
            raise ValueError("either concept_id or concept_label is required")
        return self


class UpdateConceptStateResponse(BaseModel):
    user_id: str
    concept_id: str
    # The canonical label of the KC that was updated. Worth returning even when
    # the caller passed a concept_id: label canonicalization means what it sent
    # ("Dérivées") and what the Kernel stored ("derivees") can differ.
    label: str = ""
    k_raw: float
    k_effective: float
    p_score: float
    status: KCStatus
    updated: bool


# --------------------------------------------------------------------------- #
# /load_alerts  /resolve_alert
# --------------------------------------------------------------------------- #
AlertSeverity = Literal["low", "medium", "high"]


class LoadAlertsRequest(BaseModel):
    """Read pedagogical-safety alerts, for one student or for a whole school.

    Exactly one scope. `user_id` is the student's own view (and the only scope a
    student's token can reach); `school_id` is the staff dashboard and is
    service-only — see the route for why the Kernel cannot authorize a teacher
    itself.
    """

    user_id: Optional[str] = None
    school_id: Optional[str] = None
    # Resolved alerts stay out by default: a dashboard's job is what still needs
    # attention. Pass true to review history.
    include_resolved: bool = False
    severity: Optional[AlertSeverity] = None
    since: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def _exactly_one_scope(self) -> "LoadAlertsRequest":
        if bool(self.user_id) == bool(self.school_id):
            raise ValueError("exactly one of user_id or school_id is required")
        return self


class AlertOut(BaseModel):
    id: str
    user_id: Optional[str] = None
    concept_id: Optional[str] = None
    # Resolved from the graph: an alert row stores a UUID, and no one can read a
    # UUID. Empty when the alert isn't about a specific KC.
    concept_label: str = ""
    alert_type: str
    alert_severity: str
    alert_details: dict = Field(default_factory=dict)
    inconsistency_rate: Optional[float] = None
    volatility_score: Optional[float] = None
    interactions_count: Optional[int] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class LoadAlertsResponse(BaseModel):
    scope: Literal["user", "school"]
    user_id: Optional[str] = None
    school_id: Optional[str] = None
    # How many students the school scope actually covered. A school that expects
    # 300 and sees 12 has a roster problem, not a quiet week.
    students_in_scope: int = 1
    alerts: list[AlertOut] = Field(default_factory=list)
    counts_by_type: dict[str, int] = Field(default_factory=dict)
    counts_by_severity: dict[str, int] = Field(default_factory=dict)
    # True when the limit cut the list short — so a UI never implies it is
    # showing everything when it isn't.
    truncated: bool = False


class ResolveAlertRequest(BaseModel):
    alert_id: str
    # Who acknowledged it. Free text (a teacher's name or id) because the Kernel
    # has no staff directory — it records the claim, the app vouches for it.
    resolved_by: str = Field(..., min_length=1, max_length=128)
    # False reopens an alert closed by mistake.
    resolved: bool = True


class ResolveAlertResponse(BaseModel):
    alert_id: str
    resolved: bool
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# /seed_kcs
# --------------------------------------------------------------------------- #
class SeedResponse(BaseModel):
    seeded: bool
    nodes_inserted: int
    edges_inserted: int
    message: str
