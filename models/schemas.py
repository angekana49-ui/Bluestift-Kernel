"""Pydantic models for the Bluestift Cognitive Kernel API.

All request/response contracts live here. Validation is strict: the Kernel
refuses malformed inputs rather than guessing.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    content: str = Field(..., min_length=1)


class AnalyzeRequest(BaseModel):
    user_id: str = Field(..., description="UUID of the student")
    conversation_history: list[Message] = Field(..., min_length=1)
    subject: str = Field(default="MATH")
    level: str = Field(default="unknown")
    trigger: str = Field(default="post_conversation")


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
    user_id: str
    concept_id: str
    partial_credit_score: float = Field(..., ge=0.0, le=1.0)
    is_assisted: bool = False
    response_time_ms: Optional[int] = Field(default=None, ge=0)
    blocage_type: BlocageType = BlocageType.none


class UpdateConceptStateResponse(BaseModel):
    user_id: str
    concept_id: str
    k_raw: float
    k_effective: float
    p_score: float
    status: KCStatus
    updated: bool


# --------------------------------------------------------------------------- #
# /seed_kcs
# --------------------------------------------------------------------------- #
class SeedResponse(BaseModel):
    seeded: bool
    nodes_inserted: int
    edges_inserted: int
    message: str
