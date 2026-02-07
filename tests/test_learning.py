"""Unit tests for the Learning layer (feedback loop)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanocortex.audit.logger import AuditLogger
from nanocortex.learning.feedback import LearningLoop
from nanocortex.models.domain import (
    Decision,
    FeedbackRecord,
    OutcomeRating,
)


class TestLearningLoop:
    """Tests for LearningLoop."""

    def test_record_feedback(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test recording feedback."""
        loop = LearningLoop(audit_logger, temp_dir)

        feedback = FeedbackRecord(
            decision_id="dec123",
            rating=OutcomeRating.CORRECT,
            explanation="Answer was accurate",
        )
        recorded = loop.record_feedback(feedback)

        assert recorded.decision_id == "dec123"
        assert recorded.rating == OutcomeRating.CORRECT
        assert loop.feedback_count == 1

    def test_get_feedback_for_decision(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test retrieving feedback for a specific decision."""
        loop = LearningLoop(audit_logger, temp_dir)

        loop.record_feedback(FeedbackRecord(
            decision_id="dec1",
            rating=OutcomeRating.CORRECT,
        ))
        loop.record_feedback(FeedbackRecord(
            decision_id="dec2",
            rating=OutcomeRating.INCORRECT,
        ))
        loop.record_feedback(FeedbackRecord(
            decision_id="dec1",
            rating=OutcomeRating.PARTIALLY_CORRECT,
        ))

        fb = loop.get_feedback_for_decision("dec1")
        assert len(fb) == 2
        assert all(f.decision_id == "dec1" for f in fb)

    def test_mistake_patterns_tracking(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test that mistakes are tracked."""
        loop = LearningLoop(audit_logger, temp_dir)

        loop.record_feedback(FeedbackRecord(
            decision_id="d1",
            rating=OutcomeRating.HALLUCINATION,
        ))
        loop.record_feedback(FeedbackRecord(
            decision_id="d2",
            rating=OutcomeRating.INCORRECT,
        ))
        loop.record_feedback(FeedbackRecord(
            decision_id="d3",
            rating=OutcomeRating.HALLUCINATION,
        ))

        patterns = loop.mistake_patterns
        assert patterns.get("hallucination", 0) == 2
        assert patterns.get("incorrect", 0) == 1

    def test_compute_accuracy(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test accuracy computation."""
        loop = LearningLoop(audit_logger, temp_dir)

        loop.record_feedback(FeedbackRecord(decision_id="d1", rating=OutcomeRating.CORRECT))
        loop.record_feedback(FeedbackRecord(decision_id="d2", rating=OutcomeRating.CORRECT))
        loop.record_feedback(FeedbackRecord(decision_id="d3", rating=OutcomeRating.INCORRECT))
        loop.record_feedback(FeedbackRecord(decision_id="d4", rating=OutcomeRating.PARTIALLY_CORRECT))

        stats = loop.compute_accuracy()

        assert stats["total"] == 4
        # 2 correct + 0.5 * 1 partial = 2.5 / 4 = 0.625
        assert stats["accuracy"] == 0.625
        assert stats["breakdown"]["correct"] == 2

    def test_adjustment_generation(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test that adjustments are generated after enough mistakes."""
        loop = LearningLoop(audit_logger, temp_dir)

        # Record 3 hallucinations to trigger adjustment
        for i in range(3):
            loop.record_feedback(FeedbackRecord(
                decision_id=f"d{i}",
                rating=OutcomeRating.HALLUCINATION,
            ))

        adjustments = loop.get_adjustments()
        assert len(adjustments) >= 1
        assert adjustments[0].adjustment_type == "retrieval_weight"

    def test_save_and_load_state(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test state persistence."""
        loop1 = LearningLoop(audit_logger, temp_dir)
        loop1.record_feedback(FeedbackRecord(
            decision_id="d1",
            rating=OutcomeRating.CORRECT,
        ))
        loop1.save_state()

        # Create new loop and load
        loop2 = LearningLoop(audit_logger, temp_dir)
        assert loop2.feedback_count == 0

        loaded = loop2.load_state()
        assert loaded is True
        assert loop2.feedback_count == 1

    def test_evaluate_decision(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test automated decision evaluation."""
        loop = LearningLoop(audit_logger, temp_dir)

        decision = Decision(
            decision_id="d1",
            query="What is the capital?",
            answer="Paris",
        )

        # Exact match
        fb1 = loop.evaluate_decision(decision, "Paris")
        assert fb1.rating == OutcomeRating.CORRECT

        # Partial match
        fb2 = loop.evaluate_decision(decision, "The capital is Paris")
        assert fb2.rating == OutcomeRating.PARTIALLY_CORRECT
