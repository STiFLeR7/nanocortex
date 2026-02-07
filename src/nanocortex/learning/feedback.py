"""Learning loop: outcome evaluation and behavioral correction without retraining.

This layer:
- Records feedback on decisions (correct, incorrect, hallucination, etc.)
- Tracks mistake patterns across runs
- Produces adjustments (retrieval weight tweaks, prompt patches, new policy rules)
- All adjustments are observable and auditable
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from nanocortex.audit.logger import AuditLogger
from nanocortex.models.domain import (
    Decision,
    FeedbackRecord,
    LearningAdjustment,
    OutcomeRating,
)

logger = logging.getLogger(__name__)


class LearningLoop:
    """Post-run evaluation and behavioral improvement without model retraining."""

    def __init__(self, audit: AuditLogger, data_dir: Path | str = "./data") -> None:
        self._audit = audit
        self._data_dir = Path(data_dir)
        self._feedback: list[FeedbackRecord] = []
        self._adjustments: list[LearningAdjustment] = []
        self._mistake_counts: Counter[str] = Counter()

    @property
    def feedback_count(self) -> int:
        return len(self._feedback)

    @property
    def adjustment_count(self) -> int:
        return len(self._adjustments)

    @property
    def mistake_patterns(self) -> dict[str, int]:
        return dict(self._mistake_counts)

    def record_feedback(self, feedback: FeedbackRecord) -> FeedbackRecord:
        """Record feedback on a decision outcome."""
        self._feedback.append(feedback)

        if feedback.rating in (OutcomeRating.INCORRECT, OutcomeRating.HALLUCINATION):
            self._mistake_counts[feedback.rating.value] += 1

        self._audit.log(
            layer="learning",
            event_type="feedback_recorded",
            payload={
                "decision_id": feedback.decision_id,
                "rating": feedback.rating.value,
                "has_correction": bool(feedback.corrected_answer),
            },
            decision_id=feedback.decision_id,
        )

        # Auto-generate adjustments when patterns emerge
        self._check_for_adjustments(feedback)

        return feedback

    def get_feedback_for_decision(self, decision_id: str) -> list[FeedbackRecord]:
        return [f for f in self._feedback if f.decision_id == decision_id]

    def get_adjustments(self) -> list[LearningAdjustment]:
        return list(self._adjustments)

    def evaluate_decision(self, decision: Decision, expected: str) -> FeedbackRecord:
        """Automated evaluation: compare decision answer against expected output."""
        answer = decision.answer.lower().strip()
        expected_lower = expected.lower().strip()

        if answer == expected_lower:
            rating = OutcomeRating.CORRECT
        elif expected_lower in answer or answer in expected_lower:
            rating = OutcomeRating.PARTIALLY_CORRECT
        elif not decision.evidence:
            rating = OutcomeRating.HALLUCINATION
        else:
            rating = OutcomeRating.INCORRECT

        feedback = FeedbackRecord(
            decision_id=decision.decision_id,
            rating=rating,
            corrected_answer=expected if rating != OutcomeRating.CORRECT else "",
            explanation=f"Automated evaluation: {rating.value}",
        )

        return self.record_feedback(feedback)

    def compute_accuracy(self) -> dict[str, Any]:
        """Compute accuracy metrics over all feedback."""
        total = len(self._feedback)
        if total == 0:
            return {"total": 0, "accuracy": 0.0, "breakdown": {}}

        counts: Counter[str] = Counter()
        for f in self._feedback:
            counts[f.rating.value] += 1

        correct = counts.get("correct", 0)
        partial = counts.get("partially_correct", 0)

        return {
            "total": total,
            "accuracy": round((correct + 0.5 * partial) / total, 4),
            "breakdown": dict(counts),
        }

    def _check_for_adjustments(self, feedback: FeedbackRecord) -> None:
        """Generate adjustments when mistake patterns cross thresholds."""
        hallucinations = self._mistake_counts.get("hallucination", 0)
        incorrect = self._mistake_counts.get("incorrect", 0)

        # If hallucinations reach threshold, add a retrieval weight adjustment
        if hallucinations > 0 and hallucinations % 3 == 0:
            adj = LearningAdjustment(
                trigger_feedback_id=feedback.feedback_id,
                adjustment_type="retrieval_weight",
                description=(
                    f"Increasing retrieval confidence threshold after "
                    f"{hallucinations} hallucinations detected"
                ),
                parameters={"min_score_threshold": 0.1 * (hallucinations // 3)},
            )
            self._adjustments.append(adj)
            self._audit.log(
                layer="learning",
                event_type="adjustment_created",
                payload=adj.model_dump(mode="json"),
            )

        # If incorrect answers accumulate, suggest prompt patch
        if incorrect > 0 and incorrect % 5 == 0:
            adj = LearningAdjustment(
                trigger_feedback_id=feedback.feedback_id,
                adjustment_type="prompt_patch",
                description=(
                    f"Suggesting stricter evidence grounding after "
                    f"{incorrect} incorrect answers"
                ),
                parameters={"patch": "require_exact_citation"},
            )
            self._adjustments.append(adj)
            self._audit.log(
                layer="learning",
                event_type="adjustment_created",
                payload=adj.model_dump(mode="json"),
            )

    def save_state(self) -> Path:
        """Persist learning state to disk for cross-run continuity."""
        state_dir = self._data_dir / "learning"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "state.json"

        state = {
            "feedback": [f.model_dump(mode="json") for f in self._feedback],
            "adjustments": [a.model_dump(mode="json") for a in self._adjustments],
            "mistake_counts": dict(self._mistake_counts),
        }

        with open(state_file, "w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2, default=str)

        return state_file

    def load_state(self) -> bool:
        """Load persisted learning state."""
        state_file = self._data_dir / "learning" / "state.json"
        if not state_file.exists():
            return False

        with open(state_file, encoding="utf-8") as fp:
            state = json.load(fp)

        self._feedback = [FeedbackRecord(**f) for f in state.get("feedback", [])]
        self._adjustments = [LearningAdjustment(**a) for a in state.get("adjustments", [])]
        self._mistake_counts = Counter(state.get("mistake_counts", {}))

        return True
