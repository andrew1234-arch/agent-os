"""Subprocess execution sandbox with CPU / memory / wall / network limits.

CPU / memory caps are POSIX-only: they use :mod:`resource` via
``preexec_fn`` to apply ``setrlimit`` before the child process begins. On
platforms where ``resource`` is unavailable (notably Windows) the command
still runs — the wall-clock timeout is cross-platform — but the rlimits are
skipped and :data:`NOTE_NO_RLIMITS` is attached to
:attr:`SandboxResult.notes` so the degradation is never silent.

Hitting the memory cap (``RLIMIT_AS``) essentially never raises a signal —
malloc/mmap just return ``ENOMEM``, which almost every language runtime
turns into a clean, non-fatal exit. Attributing that back to
``reason='memory_limit'`` requires sampling the child's peak virtual memory
live from ``/proc`` (Linux-only); where that isn't possible,
:data:`NOTE_NO_MEMORY_ATTRIBUTION` is attached instead.

Environment whitelist: by default the sandboxed command sees only
``HOME``, ``PATH`` and ``LANG`` — a deliberate narrow whitelist so
secrets in the parent environment do not leak to shell-invoking tools.

Network scoping is advisory in this module: the ``network`` limit
is recorded on the :class:`SandboxLimits` contract and callers are
expected to honour it (``'deny'`` means no socket operations). The
corresponding test asserts denial behaviour end-to-end via a subprocess
that refuses to perform network I/O when the limit is ``'deny'``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

try:
    import resource as _resource_module  # type: ignore[import-not-found]

    _resource: Any | None = _resource_module
    HAS_RESOURCE = True
except ImportError:  # pragma: no cover — exercised on Windows CI only
    _resource = None
    HAS_RESOURCE = False

#: Whether ``/proc/<pid>/status`` is readable, i.e. we can sample a child's
#: peak virtual memory while it runs. Linux-only; used to attribute clean
#: (non-signal) exits to the ``RLIMIT_AS`` cap — see ``_MemoryMonitor``.
HAS_PROCFS: Final[bool] = sys.platform.startswith("linux")

NetworkScope = Literal["deny", "localhost", "allow"]

REASON_OK: Final[str] = "ok"
REASON_CPU_LIMIT: Final[str] = "cpu_limit"
REASON_MEMORY_LIMIT: Final[str] = "memory_limit"
REASON_WALL_LIMIT: Final[str] = "wall_limit"
REASON_NETWORK_DENY: Final[str] = "network_deny"
REASON_UNSUPPORTED: Final[str] = "unsupported_platform"
REASON_EXEC_FAILED: Final[str] = "exec_failed"

NOTE_NO_RLIMITS: Final[str] = (
    "rlimits not applied on this platform: the POSIX 'resource' module is "
    "unavailable, so only the wall-clock timeout is enforced"
)

NOTE_NO_MEMORY_ATTRIBUTION: Final[str] = (
    "memory_limit cannot be distinguished from an ordinary failure on this "
    "platform: attributing a clean exit to RLIMIT_AS requires sampling "
    "/proc/<pid>/status, which is Linux-only"
)

#: A child that hit RLIMIT_AS rarely dies of a signal -- malloc/mmap just
#: fail with ENOMEM, which almost every language runtime turns into a clean,
#: non-fatal exit (Python: MemoryError -> exit 1; C: NULL from malloc; etc).
#: RLIMIT_AS also bounds *virtual* address space, not RSS, so a failed
#: allocation attempt fails before touching most pages -- getrusage's
#: ru_maxrss after the fact is usually nowhere near the cap even when
#: RLIMIT_AS is exactly what killed the process. The only externally
#: observable signal is the peak *virtual* size the child reached before it
#: died, sampled live from /proc. This threshold allows for the cap not
#: being hit exactly (page-alignment overhead, the allocator's own bookkeeping).
_MEMORY_ATTRIBUTION_THRESHOLD: Final[float] = 0.9

#: How much of its own CPU-time cap a SIGKILL'd child must have burned to be
#: attributed to RLIMIT_CPU rather than to memory pressure (e.g. the kernel
#: OOM killer, which also uses SIGKILL, independently of RLIMIT_AS).
_CPU_ATTRIBUTION_THRESHOLD: Final[float] = 0.5

_DEFAULT_ENV_WHITELIST: Final[tuple[str, ...]] = ("HOME", "PATH", "LANG")

#: Last-resort markers for a runtime's own "I couldn't allocate" report. The
#: kernel checks RLIMIT_AS *before* mapping any pages, so a single large
#: request that blows the cap in one call (``bytearray(2 * 1024**3)`` against
#: a 64MB cap) leaves no trace in peak virtual memory at all -- there is no
#: OS-level evidence to sample. Matched only against the child's last
#: non-blank stderr line, to avoid tripping on these words appearing
#: incidentally earlier in unrelated output.
_MEMORY_ERROR_MARKERS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p)
    for p in (
        r"MemoryError$",  # Python
        r"std::bad_alloc$",  # C++
        r"fatal error: out of memory$",  # Go
        r"FATAL ERROR:.*heap out of memory$",  # Node/V8
        r"Cannot allocate memory$",  # libc strerror(ENOMEM) via perror/bash/etc.
        r"memory allocation of \d+ bytes failed$",  # Rust (aborts after this)
    )
)


def _stderr_shows_allocation_failure(stderr: str) -> bool:
    """Best-effort: does the child's own final message say it couldn't allocate?"""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    return any(pattern.search(last) for pattern in _MEMORY_ERROR_MARKERS)


def _read_peak_vm_kb(pid: int) -> int | None:
    """Return the child's peak virtual memory size (``VmPeak``) in KB, or ``None``."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


class _MemoryMonitor:
    """Samples a child's peak virtual memory via ``/proc`` while it runs.

    Only meaningful when a memory cap was actually applied to the child
    (:data:`HAS_RESOURCE`) on a platform that exposes ``/proc``
    (:data:`HAS_PROCFS`); construct and ``start()`` unconditionally,
    ``stop()`` always returns ``0`` where sampling isn't possible or
    wouldn't mean anything, which callers already treat as "no evidence of
    memory pressure" rather than as a false negative. Without
    ``HAS_RESOURCE`` no ``RLIMIT_AS`` was ever set on the child, so there is
    nothing to attribute a failure to even if the child happens to use a
    lot of memory for an unrelated reason.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._peak_kb = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if HAS_PROCFS and HAS_RESOURCE:
            self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return self._peak_kb

    def _run(self) -> None:
        while not self._stop_event.is_set():
            kb = _read_peak_vm_kb(self._pid)
            if kb is not None and kb > self._peak_kb:
                self._peak_kb = kb
            if self._stop_event.wait(0.005):
                break


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits applied to a sandboxed subprocess."""

    cpu_seconds: int = 30
    memory_mb: int = 512
    wall_seconds: int = 60
    network: NetworkScope = "deny"
    env_whitelist: tuple[str, ...] = _DEFAULT_ENV_WHITELIST


@dataclass
class SandboxResult:
    """Outcome of a :func:`run_sandboxed` call."""

    returncode: int
    stdout: str
    stderr: str
    reason: str = REASON_OK
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    #: Advisory degradations that applied to this run — e.g. the caps in
    #: ``limits`` could not be enforced. Non-fatal, but callers must surface
    #: them: a user who turned the sandbox on has to know what is missing.
    notes: tuple[str, ...] = ()


def _preexec(limits: SandboxLimits):  # pragma: no cover — runs in child
    if not HAS_RESOURCE:
        return None
    resource = cast(Any, _resource)

    def _apply() -> None:
        # CPU seconds — RLIMIT_CPU. Signal SIGXCPU on soft, SIGKILL on hard.
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (limits.cpu_seconds, limits.cpu_seconds),
        )
        # Memory — RLIMIT_AS is the address-space cap. Not all platforms
        # enforce this, but Unix backends set it when available.
        mem_bytes = limits.memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            # Some BSDs / macOS refuse RLIMIT_AS; degrade silently — the
            # wall limit still bounds runaway allocations.
            pass

    return _apply


def _filtered_env(whitelist: Sequence[str]) -> dict[str, str]:
    parent = os.environ
    return {key: parent[key] for key in whitelist if key in parent}


def run_sandboxed(
    cmd: Sequence[str],
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Run ``cmd`` under ``limits`` and return a :class:`SandboxResult`.

    * CPU / memory are applied via ``setrlimit`` in a ``preexec_fn``.
    * Wall time is enforced with :meth:`subprocess.Popen.communicate`'s
      ``timeout`` argument; on timeout we kill the process group and
      return ``reason='wall_limit'``.
    * Network scope ``'deny'`` is recorded and returned verbatim on
      result — the child is expected to consult the limit (tests assert
      this by round-tripping the limits).
    * On platforms without :mod:`resource` (Windows) the command still
      runs under the wall limit; only the rlimits are skipped, and
      :data:`NOTE_NO_RLIMITS` is added to ``notes`` to say so.
    """

    effective = limits or SandboxLimits()
    # ``_preexec`` returns ``None`` without ``resource``, so the same call
    # works everywhere; the note is what keeps the degradation visible.
    notes: tuple[str, ...] = () if HAS_RESOURCE else (NOTE_NO_RLIMITS,)
    if HAS_RESOURCE and not HAS_PROCFS:
        notes = (*notes, NOTE_NO_MEMORY_ATTRIBUTION)

    env = _filtered_env(effective.env_whitelist)

    try:
        proc = subprocess.Popen(  # noqa: S603 — cmd is caller-controlled
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=_preexec(effective),  # noqa: PLW1509 — None off POSIX
            text=True,
        )
    except NotImplementedError as exc:
        # Hosts without process creation at all (WASI / Emscripten builds).
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=str(exc),
            reason=REASON_UNSUPPORTED,
            limits=effective,
            notes=notes,
        )
    except (OSError, ValueError) as exc:
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=str(exc),
            reason=REASON_EXEC_FAILED,
            limits=effective,
            notes=notes,
        )

    monitor = _MemoryMonitor(proc.pid)
    monitor.start()
    cpu_before = _cpu_seconds_used() if HAS_RESOURCE else None

    try:
        stdout, stderr = proc.communicate(timeout=effective.wall_seconds)
    except subprocess.TimeoutExpired:
        monitor.stop()
        proc.kill()
        stdout, stderr = proc.communicate()
        return SandboxResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout or "",
            stderr=stderr or "",
            reason=REASON_WALL_LIMIT,
            limits=effective,
            notes=notes,
        )

    peak_vm_kb = monitor.stop()
    cpu_after = _cpu_seconds_used() if HAS_RESOURCE else None
    cpu_used = cpu_after - cpu_before if cpu_before is not None and cpu_after is not None else None

    reason = _attribute_reason(proc.returncode, effective, peak_vm_kb, cpu_used, stderr or "")

    return SandboxResult(
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        reason=reason,
        limits=effective,
        notes=notes,
    )


def _cpu_seconds_used() -> float:
    """Cumulative CPU time (user + system) of all terminated, reaped children."""
    resource = cast(Any, _resource)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return cast(float, ru.ru_utime + ru.ru_stime)


def _attribute_reason(
    returncode: int,
    limits: SandboxLimits,
    peak_vm_kb: int,
    cpu_used: float | None,
    stderr: str,
) -> str:
    """Map a finished child's exit to a structured reason.

    Two limits, two very different failure shapes:

    * ``RLIMIT_CPU``'s hard cap reliably delivers ``SIGKILL`` (soft cap is
      ``SIGXCPU``), so a signal-based check works.
    * ``RLIMIT_AS`` bounds virtual address space, not RSS — a failed
      allocation just makes malloc/mmap return ``ENOMEM``, which almost
      every language runtime turns into a clean, non-fatal, non-signalled
      exit (Python: ``MemoryError`` -> exit 1). It essentially never
      produces a signal, so it can't be attributed from ``returncode``
      alone. Two complementary layers of evidence are used instead:
      peak virtual memory sampled while the child was alive (see
      :class:`_MemoryMonitor`), which catches memory that grows gradually
      up to the cap; and, since the kernel checks RLIMIT_AS *before*
      mapping anything, a single request that blows the cap in one call
      leaves no trace there at all, so :func:`_stderr_shows_allocation_failure`
      is the last-resort fallback for that case.

    ``SIGKILL`` is itself ambiguous: it's also what the kernel OOM killer
    sends under real memory pressure, independent of ``RLIMIT_AS``. Only
    reclassify a ``SIGKILL`` as memory pressure when there's positive
    evidence *and* the child clearly wasn't burning its CPU budget —
    otherwise keep the existing, tested RLIMIT_CPU attribution.
    """
    # Without HAS_RESOURCE, RLIMIT_AS was never applied to the child at all
    # (see ``_preexec``) -- there is no cap to attribute a failure to, no
    # matter how much memory the child happens to have used.
    cap_kb = limits.memory_mb * 1024
    memory_pressure = HAS_RESOURCE and (
        peak_vm_kb >= cap_kb * _MEMORY_ATTRIBUTION_THRESHOLD
        or _stderr_shows_allocation_failure(stderr)
    )

    if returncode < 0:
        signalled = -returncode
        if signalled in {_signal(9), _signal(24)}:  # SIGKILL / SIGXCPU
            cpu_pressure = (
                signalled == _signal(24)
                or cpu_used is None
                or cpu_used >= limits.cpu_seconds * _CPU_ATTRIBUTION_THRESHOLD
            )
            if memory_pressure and not cpu_pressure:
                return REASON_MEMORY_LIMIT
            return REASON_CPU_LIMIT
        # Any other signal (e.g. SIGSEGV from a genuine crash) is only
        # attributed to the memory cap when there's positive evidence —
        # a bare signal number proves nothing about RLIMIT_AS on its own.
        if memory_pressure:
            return REASON_MEMORY_LIMIT
        return REASON_OK

    if returncode != 0 and memory_pressure:
        return REASON_MEMORY_LIMIT

    return REASON_OK


def _signal(num: int) -> int:
    """Return ``num`` on POSIX, else 0 — keeps the mapping self-contained."""

    if sys.platform.startswith("win"):  # pragma: no cover
        return 0
    return num


__all__ = [
    "HAS_PROCFS",
    "HAS_RESOURCE",
    "NOTE_NO_MEMORY_ATTRIBUTION",
    "NOTE_NO_RLIMITS",
    "REASON_CPU_LIMIT",
    "REASON_EXEC_FAILED",
    "REASON_MEMORY_LIMIT",
    "REASON_NETWORK_DENY",
    "REASON_OK",
    "REASON_UNSUPPORTED",
    "REASON_WALL_LIMIT",
    "NetworkScope",
    "SandboxLimits",
    "SandboxResult",
    "run_sandboxed",
]
