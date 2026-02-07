"""Stateful decision agent with multi-model orchestration.

Architecture:
- GPT-5.2 Codex (orchestrator): generates the answer from evidence
- Claude Opus 4.6 (auditor): reviews the answer for hallucinations and quality
- KimiK 2.5 (ingestion helper): used only for document digestion, not decisions

The agent transitions through states: RUNNING -> WAITING_APPROVAL -> COMPLETED/FAILED.
If human-in-the-loop is enabled and a policy requires approval, the agent pauses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from nanocortex.audit.logger import AuditLogger
from nanocortex.config import Settings
from nanocortex.models.domain import (
    AgentState,
    Decision,
    HumanOverride,
    PolicyVerdict,
    RetrievalResponse,
)
from nanocortex.reasoning.policy import PolicyEngine

logger = logging.getLogger(__name__)


class DecisionAgent:
    """Stateful agent that produces audited, policy-checked decisions."""

    def __init__(
        self,
        settings: Settings,
        policy_engine: PolicyEngine,
        audit: AuditLogger,
    ) -> None:
        self._settings = settings
        self._policy = policy_engine
        self._audit = audit
        self._state = AgentState.RUNNING
        self._pending_decision: Decision | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def pending_decision(self) -> Decision | None:
        return self._pending_decision

    async def decide(
        self,
        query: str,
        evidence: RetrievalResponse,
        context: dict[str, str] | None = None,
    ) -> Decision:
        """Run the full decision pipeline: generate -> audit -> policy check."""
        self._state = AgentState.RUNNING

        # 1. Evaluate policies
        evaluations = self._policy.evaluate(query, evidence, context)
        aggregate_verdict = self._policy.check_allowed(evaluations)

        # 2. If policy denies outright, fail
        if aggregate_verdict == PolicyVerdict.DENY:
            self._state = AgentState.FAILED
            decision = Decision(
                query=query,
                answer="[DENIED] Policy violation: action not permitted.",
                evidence=evidence.results,
                policy_evaluations=tuple(evaluations),
                agent_state=AgentState.FAILED,
            )
            self._audit.log(
                layer="reasoning",
                event_type="decision_denied",
                payload={"query": query},
                decision_id=decision.decision_id,
            )
            return decision

        # 3. Generate answer from orchestrator
        answer = await self._call_orchestrator(query, evidence)

        # 4. Audit with the auditor model
        audit_result = await self._call_auditor(query, answer, evidence)

        # 5. Build decision
        decision = Decision(
            query=query,
            answer=answer,
            evidence=evidence.results,
            policy_evaluations=tuple(evaluations),
            agent_state=AgentState.COMPLETED,
            model_used=self._settings.orchestrator.model,
            auditor_model=self._settings.auditor.model,
        )

        # 6. If policy requires approval, pause
        if aggregate_verdict == PolicyVerdict.NEEDS_APPROVAL and self._settings.enable_human_in_loop:
            self._state = AgentState.WAITING_APPROVAL
            self._pending_decision = decision
            decision = Decision(
                **{
                    **decision.model_dump(),
                    "agent_state": AgentState.WAITING_APPROVAL,
                    "answer": f"[AWAITING APPROVAL] {answer}",
                }
            )
            self._audit.log(
                layer="reasoning",
                event_type="decision_pending_approval",
                payload={"query": query, "audit_result": audit_result},
                decision_id=decision.decision_id,
            )
            return decision

        # 7. Completed
        self._state = AgentState.COMPLETED
        self._audit.log(
            layer="reasoning",
            event_type="decision_completed",
            payload={"query": query, "audit_result": audit_result},
            decision_id=decision.decision_id,
        )
        return decision

    def approve(self, decision_id: str) -> Decision | None:
        """Approve a pending decision. Returns the approved decision or None."""
        if (
            self._pending_decision is None
            or self._pending_decision.decision_id != decision_id
        ):
            return None

        approved = Decision(
            **{
                **self._pending_decision.model_dump(),
                "agent_state": AgentState.COMPLETED,
                "answer": self._pending_decision.answer.replace("[AWAITING APPROVAL] ", ""),
            }
        )
        self._state = AgentState.COMPLETED
        self._pending_decision = None

        self._audit.log(
            layer="reasoning",
            event_type="decision_approved",
            decision_id=approved.decision_id,
            actor="human",
        )
        return approved

    def reject(self, decision_id: str, reason: str = "") -> Decision | None:
        """Reject a pending decision."""
        if (
            self._pending_decision is None
            or self._pending_decision.decision_id != decision_id
        ):
            return None

        rejected = Decision(
            **{
                **self._pending_decision.model_dump(),
                "agent_state": AgentState.FAILED,
                "answer": f"[REJECTED] {reason}" if reason else "[REJECTED]",
            }
        )
        self._state = AgentState.FAILED
        self._pending_decision = None

        self._audit.log(
            layer="reasoning",
            event_type="decision_rejected",
            payload={"reason": reason},
            decision_id=rejected.decision_id,
            actor="human",
        )
        return rejected

    def override(self, decision_id: str, new_answer: str, reason: str = "") -> HumanOverride | None:
        """Apply a human override to a completed decision."""
        override_record = HumanOverride(
            decision_id=decision_id,
            original_answer="",
            overridden_answer=new_answer,
            reason=reason,
        )
        self._audit.log_override(override_record)
        return override_record

    # ── LLM calls ─────────────────────────────────────────────────

    async def _call_orchestrator(self, query: str, evidence: RetrievalResponse) -> str:
        """Call the primary orchestrator (GPT-5.2 Codex) to generate an answer."""
        cfg = self._settings.orchestrator
        if not cfg.api_key:
            return self._fallback_answer(query, evidence)

        evidence_text = "\n".join(
            f"[{r.modality}] (score={r.score}) {r.text}" for r in evidence.results
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a decision-making AI. Answer the query using ONLY the provided evidence. "
                    "If the evidence is insufficient, say so explicitly. Never hallucinate."
                ),
            },
            {
                "role": "user",
                "content": f"Evidence:\n{evidence_text}\n\nQuery: {query}",
            },
        ]

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{cfg.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                    json={"model": cfg.model, "messages": messages, "temperature": 0},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("Orchestrator call failed: %s", exc)
            return self._fallback_answer(query, evidence)

    async def _call_auditor(self, query: str, answer: str, evidence: RetrievalResponse) -> str:
        """Call the auditor (Claude Opus 4.6) to review the answer."""
        cfg = self._settings.auditor
        if not cfg.api_key:
            return "audit_skipped:no_api_key"

        evidence_text = "\n".join(
            f"[{r.modality}] {r.text}" for r in evidence.results
        )
        prompt = (
            f"Review this answer for hallucinations, accuracy, and completeness.\n\n"
            f"Query: {query}\n\nAnswer: {answer}\n\nEvidence:\n{evidence_text}\n\n"
            f"Respond with: PASS (if grounded), FAIL (if hallucinated), or PARTIAL (if incomplete)."
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{cfg.base_url}/v1/messages",
                    headers={
                        "x-api-key": cfg.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": cfg.model,
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
        except Exception as exc:
            logger.warning("Auditor call failed: %s", exc)
            return "audit_skipped:api_error"

    @staticmethod
    def _fallback_answer(query: str, evidence: RetrievalResponse) -> str:
        """Deterministic fallback when no LLM is available."""
        if not evidence.results:
            return "No evidence found. Cannot answer without grounded data."

        top = evidence.results[0]
        citations = ", ".join(
            f"[doc={c.doc_id}, page={c.page}]" for c in top.citations
        )
        return (
            f"Based on available evidence {citations}: {top.text[:500]}"
        )
