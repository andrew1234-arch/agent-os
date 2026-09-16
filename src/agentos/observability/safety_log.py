"""Independent safety-event log stream.

Safety events are written to ``~/.agentos/logs/safety-YYYYMMDD.jsonl`` — a
separate file from the decision log. Schemas never merge: a reader of one
file does not need to parse the other.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import structlog

from agentos.paths import default_agentos_home

log = structlog.get_logger(__name__)


class SafetyEventType(StrEnum):
    """Closed enum of safety-event categories."""

    REFUSED_TOOL = "refused_tool"
    TRUNCATED_OUTPUT = "truncated_output"
    RATE_LIMIT = "rate_limit"
    INJECTION_BLOCKED = "injection_blocked"
    TIER_DENIED = "tier_denied"
    SANDBOX_VIOLATION = "sandbox_violation"


@dataclass
class SafetyEvent:
    """One row in the safety-event log."""

    event_type: SafetyEventType
    session_id: str
    reason: str
    ts: str
    tool_name: str | None = None


def _default_log_dir() -> Path:
    """Resolve the safety-log directory (shares env override with decisions)."""

    return Path(os.environ.get("AGENTOS_LOG_DIR", str(default_agentos_home() / "logs")))


def write_safety_event(
    event: SafetyEvent,
    log_dir: Path | None = None,
) -> Path:
    """Append ``event`` to the safety-log file; return the path written to."""

    log_dir = log_dir or _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y%m%d")
    path = log_dir / f"safety-{day}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return path


def record_safety_event(
    event_type: SafetyEventType,
    *,
    session_id: str,
    reason: str,
    tool_name: str | None = None,
    log_dir: Path | None = None,
) -> None:
    """Best-effort convenience wrapper around :func:`write_safety_event`.

    Safety-event logging observes a tool call; it must never be the reason
    that call fails. Mirrors the same never-break-the-turn guarantee
    ``observability.metrics.record_metric`` and
    ``observability.turn_call_log.TurnCallLogger.write`` already give their
    own callers on the same dispatch hot path.
    """
    try:
        write_safety_event(
            SafetyEvent(
                event_type=event_type,
                session_id=session_id,
                reason=reason,
                ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                tool_name=tool_name,
            ),
            log_dir,
        )
    except Exception as exc:  # pragma: no cover - observability must not break turns
        log.debug("safety_log.record_error", event_type=str(event_type), error=str(exc))
