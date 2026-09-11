from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentos.scheduler.persistence import (
    _MAX_LIST_EXECUTIONS_LIMIT,
    JobStore,
    _clamp_list_executions_limit,
)
from agentos.scheduler.types import JobExecution


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-1_000_000, 1),
        (-100, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (20, 20),
        (_MAX_LIST_EXECUTIONS_LIMIT, _MAX_LIST_EXECUTIONS_LIMIT),
        (_MAX_LIST_EXECUTIONS_LIMIT + 1, _MAX_LIST_EXECUTIONS_LIMIT),
        (10_000_000, _MAX_LIST_EXECUTIONS_LIMIT),
    ],
)
def test_clamp_list_executions_limit_boundaries(raw: int, expected: int) -> None:
    assert _clamp_list_executions_limit(raw) == expected


async def _seeded_store(tmp_path: Path, job_id: str, count: int) -> tuple[JobStore, list[datetime]]:
    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    base = datetime.now(UTC)
    started_ats = [base + timedelta(seconds=i) for i in range(count)]
    for started_at in started_ats:
        await store.save_execution(JobExecution(job_id=job_id, success=True, started_at=started_at))
    return store, started_ats


@pytest.mark.asyncio
async def test_list_executions_negative_limit_does_not_return_entire_history(
    tmp_path: Path,
) -> None:
    store, _ = await _seeded_store(tmp_path, "job-1", count=5)
    try:
        rows = await store.list_executions("job-1", limit=-1)
        assert len(rows) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_zero_limit_still_returns_something(tmp_path: Path) -> None:
    store, _ = await _seeded_store(tmp_path, "job-1", count=3)
    try:
        rows = await store.list_executions("job-1", limit=0)
        assert len(rows) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_normal_limit_is_unaffected(tmp_path: Path) -> None:
    store, started_ats = await _seeded_store(tmp_path, "job-1", count=5)
    try:
        rows = await store.list_executions("job-1", limit=3)
        assert len(rows) == 3
        # Most-recent-first: the last 3 of the 5 seeded timestamps, newest first.
        expected_newest_first = list(reversed(started_ats[-3:]))
        assert [r.started_at for r in rows] == expected_newest_first
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_executions_ceiling_holds_against_real_seeded_rows(tmp_path: Path) -> None:
    """Not just the pure function: prove the ceiling holds end-to-end against
    a database that actually has more than _MAX_LIST_EXECUTIONS_LIMIT rows.
    """
    store, _ = await _seeded_store(tmp_path, "job-1", count=_MAX_LIST_EXECUTIONS_LIMIT + 5)
    try:
        rows = await store.list_executions("job-1", limit=_MAX_LIST_EXECUTIONS_LIMIT + 500)
        assert len(rows) == _MAX_LIST_EXECUTIONS_LIMIT
    finally:
        await store.close()
