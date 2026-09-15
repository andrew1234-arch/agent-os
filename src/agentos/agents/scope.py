"""Agent scope — per-agent data directory resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import structlog

from agentos.session.keys import normalize_agent_id

log = structlog.get_logger(__name__)


def default_agentos_home() -> Path:
    """Return the user-level AgentOS home directory.

    Delegates to :func:`agentos.paths.default_agentos_home` so the
    ``AGENTOS_STATE_DIR`` env override is honored uniformly.
    """
    from agentos.paths import default_agentos_home as _impl

    return _impl()


def default_state_dir() -> Path:
    """Return the default internal state directory."""
    return default_agentos_home() / "state"


def default_workspace_dir() -> Path:
    """Return the default visible agent workspace directory."""
    return default_agentos_home() / "workspace"


def _config_value(config: object | None, name: str) -> str | None:
    value = getattr(config, name, None) if config is not None else None
    return value if isinstance(value, str) and value else None


def _field_value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _string_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _configured_agent_field(
    config: object | None,
    normalized_agent_id: str,
    field_name: str,
) -> object | None:
    if config is None or normalized_agent_id == "main":
        return None
    agents = getattr(config, "agents", None)
    if isinstance(agents, Mapping):
        entry = agents.get(normalized_agent_id)
        if entry is None or not bool(_field_value(entry, "enabled", True)):
            return None
        return _field_value(entry, field_name)
    if not isinstance(agents, list | tuple):
        return None
    for entry in agents:
        if normalize_agent_id(str(_field_value(entry, "id", ""))) != normalized_agent_id:
            continue
        if not bool(_field_value(entry, "enabled", True)):
            return None
        return _field_value(entry, field_name)
    return None


def _configured_agent_workspace(config: object | None, normalized_agent_id: str) -> Path | None:
    workspace = _configured_agent_field(config, normalized_agent_id, "workspace")
    return Path(str(workspace)).expanduser() if workspace else None


def _configured_agent_model(config: object | None, normalized_agent_id: str) -> str | None:
    return _string_value(_configured_agent_field(config, normalized_agent_id, "model"))


def resolve_agent_state_dir(
    agent_id: str,
    config: object | None = None,
    *,
    state_dir: str | Path | None = None,
) -> Path:
    """Return the internal state directory for an agent."""
    base = state_dir or _config_value(config, "state_dir") or default_state_dir()
    return Path(base) / "agents" / normalize_agent_id(agent_id)


def resolve_agent_workspace_dir(
    agent_id: str,
    config: object | None = None,
    *,
    state_dir: str | Path | None = None,
) -> Path:
    """Return the visible workspace directory for an agent.

    When no workspace is configured, fall back to AgentOS's user-level
    workspace directory instead of internal state.
    """
    workspace = _config_value(config, "workspace_dir")
    normalized = normalize_agent_id(agent_id)
    configured = _configured_agent_workspace(config, normalized)
    if configured is not None:
        return configured
    if workspace:
        root = Path(workspace)
        return root if normalized == "main" else root / "agents" / normalized
    root = default_workspace_dir()
    return root if normalized == "main" else root / "agents" / normalized


def resolve_agent_model(
    agent_id: str,
    config: object | None = None,
    *,
    explicit_model: object | None = None,
    session_model: object | None = None,
) -> str | None:
    """Resolve the model for an agent turn.

    Precedence is explicit call override, persisted session model, then the
    durable agent registry default from ``config.agents``.
    """
    return (
        _string_value(explicit_model)
        or _string_value(session_model)
        or _configured_agent_model(config, normalize_agent_id(agent_id))
    )


def resolve_agent_memory_source_dir(
    agent_id: str,
    config: object | None = None,
    *,
    source: str = "state",
    state_dir: str | Path | None = None,
) -> Path:
    """Return the Markdown memory source root for an agent."""
    if source == "workspace":
        return resolve_agent_workspace_dir(agent_id, config, state_dir=state_dir)
    if source == "state":
        return resolve_agent_state_dir(agent_id, config, state_dir=state_dir)
    raise ValueError("memory source must be 'state' or 'workspace'")


def resolve_agent_data_dir(agent_id: str, base: str | Path | None = None) -> Path:
    """Return the data directory for an agent: {base}/agents/{normalized_id}/"""
    return resolve_agent_state_dir(agent_id, state_dir=base)


def resolve_agent_memory_db(agent_id: str, base: str | Path | None = None) -> Path:
    """Return the memory DB path for an agent."""
    return resolve_agent_data_dir(agent_id, base) / "memory.db"


def resolve_agent_memory_dir(agent_id: str, base: str | Path | None = None) -> Path:
    """Return the memory files directory for an agent."""
    return resolve_agent_data_dir(agent_id, base) / "memory"


def maybe_migrate_legacy_memory(base: str = "data") -> None:
    """One-time migration: move data/memory.db + data/memory/ → data/agents/main/.

    Runs on every startup and must be safely resumable: each of the four
    legacy artifacts (memory.db, its -wal/-shm sidecars, and the memory/
    directory) is checked and moved independently against its own
    destination, rather than inferring "already migrated" from memory.db
    alone. The four renames are not atomic as a group, so a process killed
    partway through (OOM, forced restart) can leave any subset already moved
    -- a single shared "done" flag based on one artifact would silently and
    permanently strand whichever of the others didn't make it across, since
    every future startup would draw the same wrong conclusion from that one
    artifact and never look at the rest. This also covers an older legacy
    layout with a memory/ directory but no memory.db at all, which the
    original memory.db-gated check skipped entirely.

    Each rename() is guarded against FileNotFoundError so two processes
    racing the same migration (e.g. a supervisor restarting a crashed
    process while the old one is still exiting) don't crash each other.
    """
    legacy_db = Path(base) / "memory.db"
    legacy_dir = Path(base) / "memory"
    target_dir = resolve_agent_data_dir("main", base)

    db_suffixes = ("", "-wal", "-shm")
    legacy_db_paths = [Path(f"{legacy_db}{suffix}") for suffix in db_suffixes]

    if not any(p.exists() for p in legacy_db_paths) and not legacy_dir.is_dir():
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    migrated_any = False
    for suffix, src in zip(db_suffixes, legacy_db_paths):
        dest = target_dir / f"memory.db{suffix}"
        if src.exists() and not dest.exists():
            try:
                src.rename(dest)
                migrated_any = True
            except FileNotFoundError:
                pass

    if legacy_dir.is_dir() and not (target_dir / "memory").exists():
        try:
            legacy_dir.rename(target_dir / "memory")
            migrated_any = True
        except FileNotFoundError:
            pass

    if migrated_any:
        log.info("legacy_memory_migrated", target=str(target_dir))
