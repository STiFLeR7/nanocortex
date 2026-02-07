"""Audit trail: append-only, file-backed event log.

Design:
- One JSON-Lines file per day in the audit directory.
- Every event is an AuditEvent; the logger never drops events.
- Human overrides are recorded alongside system events.
- No PII or API keys ever reach this log (enforced by model design).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanocortex.models.domain import AuditEvent, HumanOverride

logger = logging.getLogger(__name__)


class AuditLogger:
    """Thread-safe, append-only audit logger backed by JSON-Lines files."""

    def __init__(self, audit_dir: Path | str) -> None:
        self._dir = Path(audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    # ── public API ────────────────────────────────────────────────

    def log(
        self,
        layer: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        decision_id: str | None = None,
        actor: str = "system",
    ) -> AuditEvent:
        event = AuditEvent(
            layer=layer,
            event_type=event_type,
            payload=payload or {},
            decision_id=decision_id,
            actor=actor,
        )
        self._append(event)
        return event

    def log_override(self, override: HumanOverride) -> AuditEvent:
        return self.log(
            layer="reasoning",
            event_type="human_override",
            payload=override.model_dump(mode="json"),
            decision_id=override.decision_id,
            actor="human",
        )

    def get_events(
        self,
        decision_id: str | None = None,
        layer: str | None = None,
    ) -> list[AuditEvent]:
        with self._lock:
            events = list(self._events)
        if decision_id:
            events = [e for e in events if e.decision_id == decision_id]
        if layer:
            events = [e for e in events if e.layer == layer]
        return events

    def get_decision_trace(self, decision_id: str) -> list[AuditEvent]:
        return self.get_events(decision_id=decision_id)

    # ── internals ─────────────────────────────────────────────────

    def _append(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
        self._write_to_file(event)
        logger.debug("audit event: %s/%s", event.layer, event.event_type)

    def _write_to_file(self, event: AuditEvent) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._dir / f"audit-{today}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.model_dump(mode="json"), default=str) + "\n")
        except OSError:
            logger.exception("Failed to write audit event to %s", path)
