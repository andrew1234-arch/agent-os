"""``maybe_migrate_legacy_memory`` must be safely resumable across a kill mid-migration (#2216).

The four legacy artifacts -- ``memory.db``, its ``-wal``/``-shm`` sidecars, and
the ``memory/`` directory -- are moved with four separate, non-atomic
``rename()`` calls. The original "already migrated?" check looked at
``memory.db`` alone: a process killed after ``memory.db`` moved but before
``memory/`` did left the next startup concluding "done" and returning
immediately, without ever looking at whether the sidecars or ``memory/``
made it across. That silently and permanently strands whichever artifact
didn't -- including the real persisted markdown memory files in ``memory/``.

These tests replay that exact kill point deterministically (not relying on
process-kill timing), plus the related latent case of a legacy install with
a ``memory/`` directory but no ``memory.db`` at all, which the original
memory.db-gated check skipped entirely.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agentos.agents.scope import maybe_migrate_legacy_memory, resolve_agent_data_dir


def _legacy_db_and_memory_dir(base: Path) -> tuple[Path, Path]:
    (base / "memory.db").write_text("legacy db content")
    (base / "memory.db-wal").write_text("wal content")
    (base / "memory.db-shm").write_text("shm content")
    mem_dir = base / "memory"
    mem_dir.mkdir()
    (mem_dir / "notes.md").write_text("real user memory content")
    return base / "memory.db", mem_dir


def test_fresh_full_migration_moves_everything(tmp_path: Path) -> None:
    """Guard: passes either way by design -- the ordinary, uninterrupted case."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    target_dir = resolve_agent_data_dir("main", str(base))

    maybe_migrate_legacy_memory(str(base))

    assert not (base / "memory.db").exists()
    assert not (base / "memory.db-wal").exists()
    assert not (base / "memory.db-shm").exists()
    assert not (base / "memory").exists()
    assert (target_dir / "memory.db").read_text() == "legacy db content"
    assert (target_dir / "memory.db-wal").read_text() == "wal content"
    assert (target_dir / "memory.db-shm").read_text() == "shm content"
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"


def test_killed_between_db_move_and_memory_dir_move_resumes_on_retry(tmp_path: Path) -> None:
    """Fails without the fix: the second call sees memory.db already at the
    target, concludes migration is done, and returns -- memory/ (the real
    persisted markdown memory) is stranded at the legacy path forever."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    target_dir = resolve_agent_data_dir("main", str(base))

    # Simulate the exact kill point: memory.db + sidecars already moved,
    # memory/ never got there.
    target_dir.mkdir(parents=True, exist_ok=True)
    (base / "memory.db").rename(target_dir / "memory.db")
    (base / "memory.db-wal").rename(target_dir / "memory.db-wal")
    (base / "memory.db-shm").rename(target_dir / "memory.db-shm")
    assert (base / "memory").is_dir()  # still stranded at this point

    maybe_migrate_legacy_memory(str(base))  # the next startup, retrying

    assert not (base / "memory").exists()
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"


def test_killed_before_wal_sidecar_moved_resumes_on_retry(tmp_path: Path) -> None:
    """A crash between moving memory.db and moving memory.db-wal must not be
    read as "already migrated" either -- each sidecar is its own artifact."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    target_dir = resolve_agent_data_dir("main", str(base))

    target_dir.mkdir(parents=True, exist_ok=True)
    (base / "memory.db").rename(target_dir / "memory.db")
    # -wal, -shm, and memory/ all still at the legacy path.

    maybe_migrate_legacy_memory(str(base))

    assert not (base / "memory.db-wal").exists()
    assert not (base / "memory.db-shm").exists()
    assert not (base / "memory").exists()
    assert (target_dir / "memory.db-wal").read_text() == "wal content"
    assert (target_dir / "memory.db-shm").read_text() == "shm content"
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"


def test_already_fully_migrated_is_a_clean_no_op(tmp_path: Path) -> None:
    """Guard: passes either way by design -- nothing legacy left, nothing to do."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    maybe_migrate_legacy_memory(str(base))  # first, full migration
    target_dir = resolve_agent_data_dir("main", str(base))
    before = (target_dir / "memory.db").read_text()

    maybe_migrate_legacy_memory(str(base))  # second call, as every startup makes

    assert (target_dir / "memory.db").read_text() == before
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"


def test_no_legacy_data_at_all_is_a_clean_no_op(tmp_path: Path) -> None:
    """Guard: a fresh install with no legacy layout at all must not crash or
    create anything."""
    base = tmp_path / "data"
    base.mkdir()

    maybe_migrate_legacy_memory(str(base))

    target_dir = resolve_agent_data_dir("main", str(base))
    assert not target_dir.exists()


def test_memory_dir_without_any_db_still_migrates(tmp_path: Path) -> None:
    """Latent bug beyond the reported one: the original check gated
    everything on memory.db existing, so an older legacy layout with only a
    memory/ directory (no db yet) was skipped entirely and never migrated."""
    base = tmp_path / "data"
    base.mkdir()
    mem_dir = base / "memory"
    mem_dir.mkdir()
    (mem_dir / "notes.md").write_text("memory predating the db layout")
    target_dir = resolve_agent_data_dir("main", str(base))

    maybe_migrate_legacy_memory(str(base))

    assert not mem_dir.exists()
    assert (target_dir / "memory" / "notes.md").read_text() == "memory predating the db layout"


def test_db_without_wal_or_shm_sidecars_migrates_cleanly(tmp_path: Path) -> None:
    """Guard: passes either way by design -- a clean shutdown with no WAL
    checkpoint pending leaves no sidecar files; their absence must not be
    treated as anything having gone wrong."""
    base = tmp_path / "data"
    base.mkdir()
    (base / "memory.db").write_text("legacy db content")
    target_dir = resolve_agent_data_dir("main", str(base))

    maybe_migrate_legacy_memory(str(base))

    assert not (base / "memory.db").exists()
    assert (target_dir / "memory.db").read_text() == "legacy db content"
    assert not (target_dir / "memory.db-wal").exists()
    assert not (target_dir / "memory.db-shm").exists()


def test_concurrent_migration_does_not_crash_on_a_file_already_moved(
    tmp_path: Path,
) -> None:
    """Two processes racing the same migration (e.g. a supervisor restarting
    a crashed process while the old one is still exiting) must not crash
    each other: a rename() that loses the race to a file already gone is
    swallowed, not propagated."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    target_dir = resolve_agent_data_dir("main", str(base))
    target_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a concurrent process having already won the race on memory.db
    # and memory/ between this process's existence checks and its renames.
    real_rename = Path.rename

    def racy_rename(self: Path, target: Path) -> Path:
        if self.name == "memory.db":
            (target_dir / "memory.db").write_text("legacy db content")
            self.unlink()
            raise FileNotFoundError(self)
        if self.name == "memory":
            shutil.move(str(self), str(target_dir / "memory"))
            raise FileNotFoundError(self)
        return real_rename(self, target)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "rename", racy_rename)
        maybe_migrate_legacy_memory(str(base))  # must not raise

    assert (target_dir / "memory.db").read_text() == "legacy db content"
    assert (target_dir / "memory.db-wal").read_text() == "wal content"
    assert (target_dir / "memory.db-shm").read_text() == "shm content"
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"


def test_third_call_after_a_resumed_migration_is_also_a_clean_no_op(
    tmp_path: Path,
) -> None:
    """The interrupted-then-resumed sequence must settle: a third call (the
    startup after the one that resumed it) must not move, re-move, or touch
    anything, and must not error."""
    base = tmp_path / "data"
    base.mkdir()
    _legacy_db_and_memory_dir(base)
    target_dir = resolve_agent_data_dir("main", str(base))
    target_dir.mkdir(parents=True, exist_ok=True)
    (base / "memory.db").rename(target_dir / "memory.db")

    maybe_migrate_legacy_memory(str(base))  # resumes: moves wal/shm/memory
    maybe_migrate_legacy_memory(str(base))  # third call: must be a no-op

    assert (target_dir / "memory.db").read_text() == "legacy db content"
    assert (target_dir / "memory" / "notes.md").read_text() == "real user memory content"
    assert not (base / "memory.db-wal").exists()
    assert not (base / "memory").exists()
