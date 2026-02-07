"""System orchestrator: wires all layers together for the vertical slice.

This is the single entry point that demonstrates:
1. PDF ingestion (with images)
2. Multimodal retrieval query
3. Policy rule enforcement
4. Human-in-the-loop pause
5. Learning feedback loop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanocortex.audit.logger import AuditLogger
from nanocortex.config import Settings, get_settings
from nanocortex.knowledge.retriever import KnowledgeStore
from nanocortex.learning.feedback import LearningLoop
from nanocortex.models.domain import (
    AgentState,
    Decision,
    FeedbackRecord,
    HumanOverride,
    OutcomeRating,
    PolicyRule,
    PolicyVerdict,
)
from nanocortex.perception.ingestion import IngestionPipeline
from nanocortex.reasoning.agent import DecisionAgent
from nanocortex.reasoning.policy import PolicyEngine

logger = logging.getLogger(__name__)


class NanoCortex:
    """Top-level orchestrator that wires all five layers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # Layer 5: Audit (cross-cutting)
        self.audit = AuditLogger(self._settings.audit_dir)

        # Layer 1: Perception
        self.perception = IngestionPipeline(self.audit)

        # Layer 2: Knowledge
        self.knowledge = KnowledgeStore(self.audit)

        # Layer 3: Reasoning
        self.policy_engine = PolicyEngine(self.audit)
        self.agent = DecisionAgent(self._settings, self.policy_engine, self.audit)

        # Layer 4: Learning
        self.learning = LearningLoop(self.audit, self._settings.data_dir)

        # Install default policies
        self._install_default_policies()

        self.audit.log(layer="system", event_type="system_initialized")

    def _install_default_policies(self) -> None:
        """Install baseline policy rules."""
        self.policy_engine.add_rule(PolicyRule(
            name="no_hallucination",
            description="Deny answers with no evidence backing",
            condition="no_evidence",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))
        self.policy_engine.add_rule(PolicyRule(
            name="low_confidence",
            description="Require approval when evidence score is low",
            condition="min_score:0.01",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))

    async def ingest(self, file_path: str | Path) -> dict[str, Any]:
        """Ingest a document and index it for retrieval."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            doc = self.perception.ingest_pdf(path)
        else:
            doc = self.perception.ingest_image(path)

        chunks_added = self.knowledge.index_document(doc)

        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "pages": doc.pages,
            "text_blocks": len(doc.texts),
            "images": len(doc.images),
            "chunks_indexed": chunks_added,
        }

    async def query(
        self,
        question: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a full query through retrieval -> reasoning -> decision."""
        evidence = self.knowledge.retrieve(question, top_k=top_k, strategy=strategy)
        decision = await self.agent.decide(question, evidence, context)

        return {
            "decision_id": decision.decision_id,
            "query": decision.query,
            "answer": decision.answer,
            "state": decision.agent_state.value,
            "model_used": decision.model_used,
            "auditor_model": decision.auditor_model,
            "evidence_count": len(decision.evidence),
            "evidence": [
                {
                    "text": r.text[:200],
                    "score": r.score,
                    "modality": r.modality,
                    "citations": [c.model_dump() for c in r.citations],
                }
                for r in decision.evidence
            ],
            "policy_evaluations": [
                {
                    "rule": e.rule.name,
                    "matched": e.matched,
                    "verdict": e.verdict.value,
                }
                for e in decision.policy_evaluations
            ],
        }

    def approve_decision(self, decision_id: str) -> dict[str, Any]:
        """Approve a pending decision."""
        result = self.agent.approve(decision_id)
        if result is None:
            return {"error": "No pending decision with that ID"}
        return {
            "decision_id": result.decision_id,
            "answer": result.answer,
            "state": result.agent_state.value,
        }

    def reject_decision(self, decision_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a pending decision."""
        result = self.agent.reject(decision_id, reason)
        if result is None:
            return {"error": "No pending decision with that ID"}
        return {
            "decision_id": result.decision_id,
            "answer": result.answer,
            "state": result.agent_state.value,
        }

    def submit_feedback(
        self,
        decision_id: str,
        rating: str,
        corrected_answer: str = "",
        explanation: str = "",
    ) -> dict[str, Any]:
        """Submit feedback on a decision for the learning loop."""
        feedback = FeedbackRecord(
            decision_id=decision_id,
            rating=OutcomeRating(rating),
            corrected_answer=corrected_answer,
            explanation=explanation,
        )
        recorded = self.learning.record_feedback(feedback)
        return {
            "feedback_id": recorded.feedback_id,
            "decision_id": recorded.decision_id,
            "rating": recorded.rating.value,
        }

    def get_audit_trail(self, decision_id: str | None = None) -> list[dict[str, Any]]:
        """Get audit events, optionally filtered by decision."""
        events = self.audit.get_events(decision_id=decision_id)
        return [e.model_dump(mode="json") for e in events]

    def get_learning_stats(self) -> dict[str, Any]:
        """Get current learning metrics."""
        return {
            "accuracy": self.learning.compute_accuracy(),
            "feedback_count": self.learning.feedback_count,
            "adjustment_count": self.learning.adjustment_count,
            "mistake_patterns": self.learning.mistake_patterns,
            "adjustments": [a.model_dump(mode="json") for a in self.learning.get_adjustments()],
        }
