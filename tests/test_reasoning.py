"""Unit tests for the Reasoning layer (agent + policy)."""

from __future__ import annotations

import pytest

from nanocortex.audit.logger import AuditLogger
from nanocortex.config import Settings
from nanocortex.models.domain import (
    PolicyRule,
    PolicyVerdict,
    RetrievalResponse,
    RetrievalResult,
)
from nanocortex.reasoning.agent import DecisionAgent
from nanocortex.reasoning.policy import PolicyEngine


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_add_and_list_rules(self, audit_logger: AuditLogger):
        """Test adding and listing policy rules."""
        engine = PolicyEngine(audit_logger)

        rule = PolicyRule(
            name="test_rule",
            condition="no_evidence",
            verdict=PolicyVerdict.DENY,
        )
        engine.add_rule(rule)

        assert len(engine.rules) == 1
        assert engine.rules[0].name == "test_rule"

    def test_no_evidence_condition(self, audit_logger: AuditLogger):
        """Test no_evidence condition matching."""
        engine = PolicyEngine(audit_logger)
        engine.add_rule(PolicyRule(
            name="require_evidence",
            condition="no_evidence",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))

        # Empty evidence should match
        empty_evidence = RetrievalResponse(query="test", results=())
        evals = engine.evaluate("test query", empty_evidence)

        assert len(evals) == 1
        assert evals[0].matched is True
        assert evals[0].verdict == PolicyVerdict.NEEDS_APPROVAL

    def test_contains_condition(self, audit_logger: AuditLogger):
        """Test contains condition matching."""
        engine = PolicyEngine(audit_logger)
        engine.add_rule(PolicyRule(
            name="sensitive_topic",
            condition="contains:financial|medical",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))

        evidence = RetrievalResponse(query="test", results=())

        # Should match
        evals1 = engine.evaluate("What is the financial impact?", evidence)
        assert evals1[0].matched is True

        # Should not match
        evals2 = engine.evaluate("What is the weather today?", evidence)
        assert evals2[0].matched is False

    def test_min_score_condition(self, audit_logger: AuditLogger):
        """Test min_score condition matching."""
        engine = PolicyEngine(audit_logger)
        engine.add_rule(PolicyRule(
            name="low_confidence",
            condition="min_score:0.5",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))

        low_score = RetrievalResponse(
            query="test",
            results=(RetrievalResult(text="content", score=0.3),),
        )
        high_score = RetrievalResponse(
            query="test",
            results=(RetrievalResult(text="content", score=0.8),),
        )

        # Low score should match (score < threshold)
        evals1 = engine.evaluate("query", low_score)
        assert evals1[0].matched is True

        # High score should not match
        evals2 = engine.evaluate("query", high_score)
        assert evals2[0].matched is False

    def test_check_allowed_deny(self, audit_logger: AuditLogger):
        """Test aggregate verdict with DENY."""
        engine = PolicyEngine(audit_logger)
        engine.add_rule(PolicyRule(
            name="always_deny",
            condition="no_evidence",
            verdict=PolicyVerdict.DENY,
        ))

        evidence = RetrievalResponse(query="test", results=())
        evals = engine.evaluate("query", evidence)
        verdict = engine.check_allowed(evals)

        assert verdict == PolicyVerdict.DENY


class TestDecisionAgent:
    """Tests for DecisionAgent."""

    @pytest.mark.asyncio
    async def test_decide_with_fallback(self, settings: Settings, audit_logger: AuditLogger):
        """Test decision making with fallback (no LLM)."""
        policy = PolicyEngine(audit_logger)
        agent = DecisionAgent(settings, policy, audit_logger)

        evidence = RetrievalResponse(
            query="What is solar energy?",
            results=(
                RetrievalResult(text="Solar energy comes from the sun.", score=0.9),
            ),
        )

        decision = await agent.decide("What is solar energy?", evidence)

        assert decision.query == "What is solar energy?"
        assert decision.answer  # Non-empty
        assert len(decision.evidence) > 0

    @pytest.mark.asyncio
    async def test_decide_needs_approval(self, settings: Settings, audit_logger: AuditLogger):
        """Test that policy can trigger waiting_approval state."""
        policy = PolicyEngine(audit_logger)
        policy.add_rule(PolicyRule(
            name="always_approve",
            condition="no_evidence",
            verdict=PolicyVerdict.NEEDS_APPROVAL,
        ))

        agent = DecisionAgent(settings, policy, audit_logger)
        empty_evidence = RetrievalResponse(query="test", results=())

        decision = await agent.decide("unknown question", empty_evidence)

        assert decision.agent_state.value == "waiting_approval"
