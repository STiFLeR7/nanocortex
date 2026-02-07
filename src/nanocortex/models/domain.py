"""Core domain models for the Unified AI System.

Every struct is immutable (frozen) to prevent accidental mutation in pipelines.
No field exposes raw API keys or PII—models are safe to serialize to audit logs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Perception Layer ──────────────────────────────────────────────


class BoundingBox(BaseModel, frozen=True):
    x: float
    y: float
    width: float
    height: float
    page: int = 0


class ExtractedText(BaseModel, frozen=True):
    text: str
    confidence: float = 1.0
    bbox: BoundingBox | None = None
    source_page: int = 0


class ExtractedImage(BaseModel, frozen=True):
    image_id: str = Field(default_factory=_new_id)
    page: int = 0
    bbox: BoundingBox | None = None
    image_bytes_b64: str = ""  # base64-encoded image content
    description: str = ""


class DocumentIngestion(BaseModel, frozen=True):
    doc_id: str = Field(default_factory=_new_id)
    filename: str
    mime_type: str = "application/pdf"
    pages: int = 0
    texts: tuple[ExtractedText, ...] = ()
    images: tuple[ExtractedImage, ...] = ()
    ingested_at: datetime = Field(default_factory=_utcnow)


# ── Knowledge & Retrieval Layer ───────────────────────────────────


class Citation(BaseModel, frozen=True):
    doc_id: str
    page: int
    bbox: BoundingBox | None = None
    image_id: str | None = None
    snippet: str = ""


class RetrievalResult(BaseModel, frozen=True):
    chunk_id: str = Field(default_factory=_new_id)
    text: str
    score: float = 0.0
    citations: tuple[Citation, ...] = ()
    modality: str = "text"  # "text" | "image"


class RetrievalResponse(BaseModel, frozen=True):
    query: str
    results: tuple[RetrievalResult, ...] = ()
    strategy: str = "hybrid"  # "bm25" | "vector" | "hybrid"


# ── Reasoning & Control Layer ─────────────────────────────────────


class PolicyVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


class PolicyRule(BaseModel, frozen=True):
    rule_id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""
    condition: str = ""  # human-readable condition
    verdict: PolicyVerdict = PolicyVerdict.ALLOW


class PolicyEvaluation(BaseModel, frozen=True):
    rule: PolicyRule
    matched: bool
    verdict: PolicyVerdict
    explanation: str = ""


class AgentState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class Decision(BaseModel, frozen=True):
    decision_id: str = Field(default_factory=_new_id)
    query: str
    answer: str
    evidence: tuple[RetrievalResult, ...] = ()
    policy_evaluations: tuple[PolicyEvaluation, ...] = ()
    agent_state: AgentState = AgentState.COMPLETED
    model_used: str = ""
    auditor_model: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


# ── Adaptation & Learning Layer ───────────────────────────────────


class OutcomeRating(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    HALLUCINATION = "hallucination"


class FeedbackRecord(BaseModel, frozen=True):
    feedback_id: str = Field(default_factory=_new_id)
    decision_id: str
    rating: OutcomeRating
    corrected_answer: str = ""
    explanation: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class LearningAdjustment(BaseModel, frozen=True):
    adjustment_id: str = Field(default_factory=_new_id)
    trigger_feedback_id: str
    adjustment_type: str = ""  # "retrieval_weight" | "policy_rule" | "prompt_patch"
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    applied_at: datetime = Field(default_factory=_utcnow)


# ── Audit Layer ───────────────────────────────────────────────────


class AuditEvent(BaseModel, frozen=True):
    event_id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utcnow)
    layer: str = ""  # perception | knowledge | reasoning | learning
    event_type: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    decision_id: str | None = None
    actor: str = "system"  # "system" | "human" | model name


class HumanOverride(BaseModel, frozen=True):
    override_id: str = Field(default_factory=_new_id)
    decision_id: str
    original_answer: str
    overridden_answer: str
    reason: str = ""
    overridden_at: datetime = Field(default_factory=_utcnow)
