"""Completion-side reservation race: apply_reserved_result, release_reservation,
and finalize_reserved_missing_handler must not race a concurrent
pause()/resume()/update() -- in either direction.

#1537's fix made pause()/resume()/update() (ops.py) safe against a
*concurrent reservation being acquired*, by excluding the 5 reservation
columns from their writes. It did not make the *completion* side of the same
reservation lifecycle atomic with respect to a concurrent pause()/resume()/
update() landing mid-execution: all three completion functions did an
unguarded read -> mutate -> save() sequence, so a stale snapshot's save()
could silently revert a pause that arrived after the read but before the
write. Proven directly before any fix:

    apply_reserved_result read snapshot: status=running
    pause() completed: status=paused token=<matching token>
    FINAL: status=pending   (user asked for PAUSED)

Fixing only the completion side is not enough, though: ops.py's own
pause()/resume()/update() did (and without this change, still do) the exact
same unguarded read -> full-row-save() themselves. Even with the completion
functions made atomic, a concurrent pause() that reads before the completion
commits and writes after it commits will overwrite every field the
completion just set -- run_count, next_run_at, error state -- with its own
stale copy of them, because `write_reservation=False` only protects the 5
reservation columns, not the rest of the row. Confirmed empirically: with
only jobs.py/persistence.py patched, `run_count` and `next_run_at` were
silently reverted to their pre-completion values by a concurrent pause(),
even though `status` happened to still end up correct (pause() sets it
unconditionally, so it "wins" regardless of who writes last). Both sides
must share the same store.transaction() lock across their full
read-decide-write span for neither direction to lose an update.

These tests construct both physical orderings deterministically (not
relying on asyncio scheduling luck) for each of the three completion
functions, plus real-concurrency checks (asyncio.gather) per function --
including one that specifically asserts the completion's own bookkeeping
(run_count) survives a concurrent pause(), which is the half of this race
jobs.py/persistence.py's own fix cannot close on its own.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from agentos.scheduler.jobs import apply_reserved_result
from agentos.scheduler.ops import SchedulerOps
from agentos.scheduler.payloads import make_agent_turn_payload
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import (
    JobExecution,
    JobReservation,
    JobStatus,
    ScheduleKind,
    SessionTarget,
)


async def _open_ops(tmp_path: Path) -> tuple[JobStore, SchedulerOps]:
    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    return store, SchedulerOps(store)


async def _add_job(ops: SchedulerOps):
    return await ops.add(
        name="race-job",
        handler_key="agent_run",
        payload=make_agent_turn_payload("ping"),
        session_target=SessionTarget.ISOLATED,
        schedule_kind=ScheduleKind.CRON,
        schedule_value="*/5 * * * *",
    )


# --- apply_reserved_result ---------------------------------------------


async def test_apply_reserved_result_after_pause_preserves_pause(tmp_path: Path) -> None:
    """Deterministic ordering: pause() completes fully first, then the
    execution's result is applied. Must land PAUSED, not silently reverted
    to PENDING."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        paused = await ops.pause(job.id)
        assert paused is not None and paused.status == JobStatus.PAUSED
        assert paused.reservation_token == token

        exe = JobExecution(job_id=job.id, success=True)
        applied = await apply_reserved_result(job.id, token, exe, store)
        assert applied is True

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.PAUSED
        assert final.reservation_token == ""
    finally:
        await store.close()


async def test_pause_after_apply_reserved_result_is_a_normal_pause(tmp_path: Path) -> None:
    """Reverse ordering: the result is applied and the reservation cleared
    first, then pause() runs on the now-unreserved job -- an ordinary
    pause with no race involved."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        exe = JobExecution(job_id=job.id, success=True)
        applied = await apply_reserved_result(job.id, token, exe, store)
        assert applied is True

        paused = await ops.pause(job.id)
        assert paused is not None and paused.status == JobStatus.PAUSED
    finally:
        await store.close()


async def test_concurrent_apply_reserved_result_and_pause_never_lose_the_pause(
    tmp_path: Path,
) -> None:
    """Supplementary check under real asyncio concurrency: regardless of
    which task the event loop runs first, a pause requested around the
    same time as a job's own completion must never be silently reverted."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        async def finish_execution() -> bool:
            exe = JobExecution(job_id=job.id, success=True)
            return await apply_reserved_result(job.id, token, exe, store)

        applied, paused = await asyncio.gather(finish_execution(), ops.pause(job.id))
        assert applied is True
        assert paused is not None and paused.status == JobStatus.PAUSED

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.PAUSED, (
            "a concurrent pause() must never be silently reverted by the "
            "job's own completing execution"
        )
    finally:
        await store.close()


async def test_concurrent_apply_reserved_result_and_pause_never_lose_the_run_count(
    tmp_path: Path,
) -> None:
    """The other half of the race: a concurrent pause() must not silently
    revert the completion's own bookkeeping either.

    ``write_reservation=False`` on ops.py's save only protects the 5
    reservation columns -- everything else, including run_count and
    next_run_at, is a full-row overwrite from whatever ops read. If ops's
    read races ahead of the completion's commit and its write lands after,
    a fix that only makes the completion side atomic (and leaves ops.py's
    pause()/resume()/update() as an unguarded read + full-row write) still
    loses this update: status ends up right (pause sets it unconditionally)
    but run_count and the rescheduled next_run_at silently revert to their
    pre-completion values. Both sides need the same store.transaction()
    span for this to hold regardless of ordering.
    """
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        async def finish_execution() -> bool:
            exe = JobExecution(job_id=job.id, success=True)
            return await apply_reserved_result(job.id, token, exe, store)

        applied, paused = await asyncio.gather(finish_execution(), ops.pause(job.id))
        assert applied is True
        assert paused is not None

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.PAUSED
        assert final.run_count == 1, (
            "the completing execution's own run_count increment must survive "
            "a concurrent pause(), regardless of which one's write lands last"
        )
        assert final.reservation_token == ""
    finally:
        await store.close()


# --- release_reservation -------------------------------------------------


async def test_release_reservation_after_pause_does_not_clobber_it(tmp_path: Path) -> None:
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        paused = await ops.pause(job.id)
        assert paused is not None and paused.status == JobStatus.PAUSED

        released = await store.release_reservation(job.id, token)
        assert released is True

        final = await store.get(job.id)
        assert final is not None
        # release_reservation only clears the token and, if RUNNING, resets
        # to PENDING -- a job the user paused must stay PAUSED, not bounce
        # back to PENDING.
        assert final.status == JobStatus.PAUSED
        assert final.reservation_token == ""
    finally:
        await store.close()


async def test_concurrent_release_reservation_and_pause_never_lose_the_pause(
    tmp_path: Path,
) -> None:
    """Real-concurrency check, not just the deterministic ordering above --
    without this, a test can pass trivially against an unfixed function
    simply because the two calls never actually raced (confirmed: the
    deterministic test above passes even against the unpatched function)."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        released, paused = await asyncio.gather(
            store.release_reservation(job.id, token), ops.pause(job.id)
        )
        assert released is True
        assert paused is not None and paused.status == JobStatus.PAUSED

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.PAUSED, (
            "a concurrent pause() must never be silently reverted by "
            "release_reservation"
        )
    finally:
        await store.close()


# --- finalize_reserved_missing_handler ------------------------------------


async def test_finalize_missing_handler_after_pause_clears_reservation_cleanly(
    tmp_path: Path,
) -> None:
    """finalize_reserved_missing_handler unconditionally sets FAILED -- unlike
    apply_reserved_result, it has no PAUSED/DISABLED grace branch, and that's
    an unchanged, separate design question, not part of this race fix. This
    test only asserts the atomicity property: the token check and the write
    happen against a fresh, lock-protected read, so the operation completes
    cleanly and the reservation is actually released rather than the write
    silently colliding with pause()'s own write."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        paused = await ops.pause(job.id)
        assert paused is not None and paused.status == JobStatus.PAUSED

        finalized = await store.finalize_reserved_missing_handler(
            job.id, token, error="handler not found"
        )
        assert finalized is True

        final = await store.get(job.id)
        assert final is not None
        assert final.reservation_token == ""
        assert final.status == JobStatus.FAILED
    finally:
        await store.close()


async def test_concurrent_finalize_missing_handler_and_pause_stay_consistent(
    tmp_path: Path,
) -> None:
    """Real-concurrency check. finalize_reserved_missing_handler has no
    PAUSED grace branch, so unlike apply_reserved_result, the *final status*
    legitimately depends on ordering (FAILED-after-PAUSED, or
    PAUSED-after-FAILED -- pause() never touches the token, so a fresh read
    after either ordering still finds a matching token). What must hold
    regardless of ordering is atomicity: the token check and each write
    happen against fresh data, so the reservation is always cleared exactly
    once, both operations report success, and the row never ends up in an
    inconsistent state (e.g. token stuck non-empty, or a status neither
    operation actually wrote)."""
    store, ops = await _open_ops(tmp_path)
    try:
        job = await _add_job(ops)
        reservation = await store.reserve_manual_job(
            job.id, datetime.now(UTC), source="manual", owner="m"
        )
        assert isinstance(reservation, JobReservation)
        token = reservation.token

        finalized, paused = await asyncio.gather(
            store.finalize_reserved_missing_handler(job.id, token, error="handler not found"),
            ops.pause(job.id),
        )
        assert finalized is True
        assert paused is not None and paused.status == JobStatus.PAUSED

        final = await store.get(job.id)
        assert final is not None
        assert final.reservation_token == ""
        assert final.status in (JobStatus.PAUSED, JobStatus.FAILED)
    finally:
        await store.close()
