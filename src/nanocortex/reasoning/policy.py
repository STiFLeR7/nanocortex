"""Policy engine: externalized rules that govern agent decisions.

Policies are evaluated before any action is taken. If any policy yields
DENY or NEEDS_APPROVAL, the agent state transitions accordingly.
Rules are data, not code—they can be loaded from config files.
"""

from __future__ import annotations

import logging
import re

from nanocortex.audit.logger import AuditLogger
from nanocortex.models.domain import (
    PolicyEvaluation,
    PolicyRule,
    PolicyVerdict,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Evaluate a set of policy rules against a query + evidence context."""

    def __init__(self, audit: AuditLogger) -> None:
        self._audit = audit
        self._rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> list[PolicyRule]:
        return list(self._rules)

    def evaluate(
        self,
        query: str,
        evidence: RetrievalResponse,
        context: dict[str, str] | None = None,
    ) -> list[PolicyEvaluation]:
        """Evaluate all rules against the given query and evidence."""
        evaluations: list[PolicyEvaluation] = []
        context = context or {}

        for rule in self._rules:
            matched = self._check_condition(rule, query, evidence, context)
            eval_result = PolicyEvaluation(
                rule=rule,
                matched=matched,
                verdict=rule.verdict if matched else PolicyVerdict.ALLOW,
                explanation=f"Rule '{rule.name}' {'matched' if matched else 'did not match'}",
            )
            evaluations.append(eval_result)

        self._audit.log(
            layer="reasoning",
            event_type="policy_evaluation",
            payload={
                "query": query,
                "rules_checked": len(self._rules),
                "rules_matched": sum(1 for e in evaluations if e.matched),
                "verdicts": [e.verdict.value for e in evaluations],
            },
        )

        return evaluations

    def check_allowed(self, evaluations: list[PolicyEvaluation]) -> PolicyVerdict:
        """Compute the aggregate verdict from evaluations."""
        for ev in evaluations:
            if ev.matched and ev.verdict == PolicyVerdict.DENY:
                return PolicyVerdict.DENY
        for ev in evaluations:
            if ev.matched and ev.verdict == PolicyVerdict.NEEDS_APPROVAL:
                return PolicyVerdict.NEEDS_APPROVAL
        return PolicyVerdict.ALLOW

    def _check_condition(
        self,
        rule: PolicyRule,
        query: str,
        evidence: RetrievalResponse,
        context: dict[str, str],
    ) -> bool:
        """Evaluate a rule's condition string against the context.

        Supported conditions:
        - "no_evidence": matches when retrieval returns zero results
        - "contains:<pattern>": matches when query contains the pattern
        - "min_score:<threshold>": matches when top evidence score < threshold
        - "context:<key>=<value>": matches when context[key] == value
        """
        cond = rule.condition.strip()

        if cond == "no_evidence":
            return len(evidence.results) == 0

        if cond.startswith("contains:"):
            pattern = cond[len("contains:"):].strip()
            return bool(re.search(pattern, query, re.IGNORECASE))

        if cond.startswith("min_score:"):
            try:
                threshold = float(cond[len("min_score:"):].strip())
            except ValueError:
                return False
            top_score = evidence.results[0].score if evidence.results else 0.0
            return top_score < threshold

        if cond.startswith("context:"):
            kv = cond[len("context:"):].strip()
            if "=" in kv:
                key, value = kv.split("=", 1)
                return context.get(key.strip(), "") == value.strip()

        return False
