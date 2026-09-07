"""Memory state files must never be observed truncated.

``Path.write_text`` opens with ``"w"``, which truncates before the new
content lands. For turn capture that window is hit on every turn, and the
file being rewritten is the whole day's captures for a session -- so a crash
in the window costs the day, not the turn.
"""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentos.memory.atomic_write import atomic_write_text
from agentos.memory.turn_capture import TurnCaptureService


def test_content_lands(tmp_path: Path):
    target = tmp_path / "note.md"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_overwrite_replaces_content(tmp_path: Path):
    target = tmp_path / "note.md"
    target.write_text("old and much longer", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_the_original_is_never_truncated_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The decisive property: a failure mid-write leaves the old file intact.

    A plain write_text would have already emptied it by this point.
    """
    target = tmp_path / "note.md"
    target.write_text("important existing content", encoding="utf-8")

    def die(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", die)
    with pytest.raises(OSError):
        atomic_write_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "important existing content"


def test_failed_write_leaves_no_temp_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "note.md"
    target.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        atomic_write_text(target, "replacement")

    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_parent_directories_are_created(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "note.md"
    atomic_write_text(target, "content")
    assert target.read_text(encoding="utf-8") == "content"


def test_unicode_survives_the_round_trip(tmp_path: Path):
    target = tmp_path / "note.md"
    atomic_write_text(target, "§ tiếng Việt 中文 🎯")
    assert target.read_text(encoding="utf-8") == "§ tiếng Việt 中文 🎯"


def test_existing_permissions_survive_a_rewrite(tmp_path: Path):
    """mkstemp defaults to 0600; a routine rewrite must not silently apply that.

    A file the user made group- or world-readable must stay that way after
    the next write, not get quietly locked down to owner-only.
    """
    target = tmp_path / "note.md"
    target.write_text("original", encoding="utf-8")
    os.chmod(target, 0o644)

    atomic_write_text(target, "rewritten")

    assert stat.S_IMODE(os.stat(target).st_mode) == 0o644
    assert target.read_text(encoding="utf-8") == "rewritten"


def test_new_file_gets_the_safe_default_mode(tmp_path: Path):
    """No prior file means nothing to preserve -- mkstemp's 0600 default stands."""
    target = tmp_path / "note.md"
    atomic_write_text(target, "content")
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


# -- turn capture uses it -------------------------------------------------


def _service(tmp_path: Path) -> TurnCaptureService:
    return TurnCaptureService(
        workspace_dir=tmp_path,
        turns_dir=tmp_path / "turns",
        memory_config=SimpleNamespace(
            auto_capture_enabled=True,
            capture_mode="turn_pair",
            capture_user=True,
            capture_assistant=False,
            capture_max_chars=2000,
            capture_roll_max_chars=50_000,
        ),
    )


async def test_capture_writes_the_turn(tmp_path: Path):
    rel = await _service(tmp_path).capture_turn(
        session_key="s1",
        session_id="id1",
        user_text="first message",
        assistant_text="",
        captured_at=datetime(2026, 7, 26, tzinfo=UTC),
    )
    assert rel is not None
    assert "first message" in (tmp_path / rel).read_text(encoding="utf-8")


async def test_a_failed_capture_does_not_destroy_the_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole day file is rewritten each turn; a failure must not empty it."""
    service = _service(tmp_path)
    when = datetime(2026, 7, 26, tzinfo=UTC)
    rel = await service.capture_turn(
        session_key="s1",
        session_id="id1",
        user_text="turn one",
        assistant_text="",
        captured_at=when,
    )
    assert rel is not None
    path = tmp_path / rel
    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(OSError):
        await service.capture_turn(
            session_key="s1",
            session_id="id1",
            user_text="turn two",
            assistant_text="",
            captured_at=when,
        )
    monkeypatch.undo()

    assert path.read_text(encoding="utf-8") == before
    assert "turn one" in path.read_text(encoding="utf-8")
