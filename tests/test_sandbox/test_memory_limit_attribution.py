"""``run_sandboxed`` must label an RLIMIT_AS violation ``memory_limit``.

Hitting the address-space cap almost never raises a signal: malloc/mmap just
return ``ENOMEM``, which a well-behaved runtime turns into a clean, non-fatal
exit (Python: ``MemoryError`` -> exit 1). The old signal-only mapping only
watched for ``SIGSEGV``, which this failure mode essentially never produces,
so ``reason`` came back ``"ok"`` for a run the sandbox's own limit killed.
"""

from __future__ import annotations

import sys

import pytest

from agentos.safety.sandbox import (
    HAS_RESOURCE,
    REASON_CPU_LIMIT,
    REASON_MEMORY_LIMIT,
    REASON_OK,
    SandboxLimits,
    run_sandboxed,
)

pytestmark = pytest.mark.skipif(
    not HAS_RESOURCE, reason="rlimits are POSIX-only; nothing to attribute without them"
)


def test_one_shot_allocation_over_cap_is_memory_limit() -> None:
    """A single allocation that blows RLIMIT_AS in one call.

    The kernel rejects the mmap before mapping any pages, so this case
    leaves no trace in peak virtual memory at all -- it's the case the
    stderr-marker fallback exists for.
    """
    result = run_sandboxed(
        [sys.executable, "-c", "bytearray(2 * 1024 * 1024 * 1024)"],
        SandboxLimits(cpu_seconds=10, memory_mb=64, wall_seconds=10),
    )

    assert result.returncode != 0
    assert "MemoryError" in result.stderr
    assert result.reason == REASON_MEMORY_LIMIT


def test_gradual_growth_over_cap_is_memory_limit() -> None:
    """Memory that grows incrementally up to the cap shows up in peak VM."""
    script = (
        "chunks = []\n"
        "while True:\n"
        "    chunks.append(bytearray(4 * 1024 * 1024))\n"
    )
    result = run_sandboxed(
        [sys.executable, "-c", script],
        SandboxLimits(cpu_seconds=10, memory_mb=64, wall_seconds=10),
    )

    assert result.returncode != 0
    assert result.reason == REASON_MEMORY_LIMIT


def test_cpu_hard_limit_is_still_cpu_limit_not_memory() -> None:
    """A SIGKILL'd busy-loop must stay attributed to RLIMIT_CPU."""
    result = run_sandboxed(
        [sys.executable, "-c", "x = 0\nwhile True:\n    x += 1"],
        SandboxLimits(cpu_seconds=1, memory_mb=512, wall_seconds=10),
    )

    assert result.returncode == -9
    assert result.reason == REASON_CPU_LIMIT


def test_success_is_still_ok() -> None:
    result = run_sandboxed([sys.executable, "-c", "print('fine')"])

    assert result.returncode == 0
    assert result.reason == REASON_OK


def test_no_false_positive_when_rlimit_was_never_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without HAS_RESOURCE, no RLIMIT_AS was ever set on the child.

    A child that legitimately uses a lot of memory and then fails for an
    unrelated reason must not be mislabeled ``memory_limit`` just because
    it happens to cross the same peak-memory threshold — there was no cap
    to attribute the failure to in the first place.
    """
    from agentos.safety import sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "HAS_RESOURCE", False)
    monkeypatch.setattr(sandbox_mod, "_resource", None)

    script = (
        "chunks = [bytearray(4 * 1024 * 1024) for _ in range(20)]\n"
        "raise RuntimeError('unrelated failure')\n"
    )
    result = run_sandboxed([sys.executable, "-c", script], SandboxLimits(memory_mb=64))

    assert result.returncode != 0
    assert result.reason == REASON_OK
