"""Issue #3332: a network fault raised while reading the response *body*
crashed ``muse.py`` instead of honouring its documented JSON-error contract.

The module's own docstring promises: "Every subcommand prints one JSON
object to stdout. Failures print ``{"ok": false, "error": ...}`` and exit
non-zero." ``request()`` only caught ``urllib.error.HTTPError`` and
``urllib.error.URLError`` around the ``urlopen()`` + ``resp.read()`` call.
``urlopen()`` wraps a connect-phase fault (refused connection, DNS failure,
a timeout before headers arrive) in ``URLError``, but a fault raised by
``resp.read()`` *after* ``urlopen()`` has already returned -- the server
accepted the connection, sent headers, then stalled, reset the connection,
or closed early with fewer bytes than its own ``Content-Length`` promised --
surfaces as either a bare ``OSError`` subtype (``TimeoutError``,
``ConnectionResetError``, ...) or an ``http.client.HTTPException`` subtype
(``IncompleteRead``, on a clean early close), neither of which urllib wraps,
so each escaped uncaught through every caller (``cmd_get``, ``cmd_post``)
and out of ``main()``, which only catches ``SystemExit``.

The fix widens the existing ``except urllib.error.URLError`` to ``except
(OSError, http.client.HTTPException)`` (``URLError`` is itself an
``OSError`` subclass), closing the whole class of body-read network fault
in the one place, not just the ``TimeoutError`` instance the issue's own
reproduction happened to hit.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent / "src/agentos/skills/bundled/musebook/scripts/muse.py"
)


@pytest.fixture(scope="module")
def muse() -> ModuleType:
    spec = importlib.util.spec_from_file_location("muse_network_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["muse_network_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _local_server(handler) -> int:
    """Start *handler* on an ephemeral loopback port; return the port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    threading.Thread(target=handler, args=(sock,), daemon=True).start()
    return port


def _stall_mid_body(sock: socket.socket) -> None:
    """Accept, send headers promising a body, then never send it."""
    conn, _ = sock.accept()
    conn.recv(4096)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
    )
    time.sleep(5)
    try:
        conn.close()
    except OSError:
        pass


def _reset_mid_body(sock: socket.socket) -> None:
    """Accept, send a partial body, then hard-reset the connection (RST, not FIN)."""
    conn, _ = sock.accept()
    conn.recv(4096)
    conn.sendall(
        b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{"partial'
    )
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    conn.close()


def _truncate_with_clean_close(sock: socket.socket) -> None:
    """Promise a body via ``Content-Length``, send less, close cleanly (FIN).

    No reset, no timeout -- ``http.client`` itself notices the shortfall and
    raises ``IncompleteRead``, a completely separate exception hierarchy from
    ``OSError`` (only ``RemoteDisconnected`` multiply-inherits from both).
    """
    conn, _ = sock.accept()
    conn.recv(4096)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\nshort"
    )
    conn.close()


def _ok_json(sock: socket.socket) -> None:
    conn, _ = sock.accept()
    conn.recv(4096)
    body = json.dumps({"ok": True, "hello": "world"}).encode()
    conn.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    conn.close()


def _http_400(sock: socket.socket) -> None:
    conn, _ = sock.accept()
    conn.recv(4096)
    body = json.dumps({"error": "bad request"}).encode()
    conn.sendall(
        b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    conn.close()


def test_a_body_read_timeout_returns_the_documented_error_not_a_crash(
    muse: ModuleType,
) -> None:
    """Issue #3332's own reproduction: the exact failure mode filed."""
    port = _local_server(_stall_mid_body)

    result = muse.request(f"http://127.0.0.1:{port}/api/latest.json", timeout=1)

    assert result == {"ok": False, "error": "network: timed out", "status": None}


def test_a_connection_reset_mid_body_also_returns_the_documented_error(
    muse: ModuleType,
) -> None:
    """The fix closes the whole OSError class, not just TimeoutError.

    A dropped connection mid-transfer is exactly as ordinary a network fault
    as a stall, and was equally uncaught before this fix: neither is an
    ``HTTPError`` (no status line problem) nor a ``URLError`` (``urlopen()``
    already returned successfully when the fault happens).
    """
    port = _local_server(_reset_mid_body)

    result = muse.request(f"http://127.0.0.1:{port}/api/latest.json", timeout=2)

    assert result["ok"] is False
    assert result["status"] is None
    assert result["error"].startswith("network:")


def test_a_cleanly_truncated_body_also_returns_the_documented_error(
    muse: ModuleType,
) -> None:
    """The third real shape: a short, cleanly-closed body raises
    ``http.client.IncompleteRead`` rather than any ``OSError`` -- a distinct
    exception hierarchy the fix has to catch separately from the other two.
    """
    port = _local_server(_truncate_with_clean_close)

    result = muse.request(f"http://127.0.0.1:{port}/api/latest.json", timeout=2)

    assert result["ok"] is False
    assert result["status"] is None
    assert result["error"].startswith("network:")
    assert "5 bytes read" in result["error"]


def test_main_prints_json_instead_of_crashing_on_a_stalled_response(
    muse: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real CLI entry point, not just ``request()`` in isolation."""
    port = _local_server(_stall_mid_body)
    monkeypatch.setattr(muse, "BASE_URL", f"http://127.0.0.1:{port}")

    exit_code = muse.main(["get", "--path", "latest.json", "--timeout", "1"])

    assert exit_code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["error"].startswith("network:")


def test_a_successful_response_is_unaffected(muse: ModuleType) -> None:
    port = _local_server(_ok_json)

    result = muse.request(f"http://127.0.0.1:{port}/api/x", timeout=2)

    assert result == {"ok": True, "hello": "world", "status": 200}


def test_an_http_error_body_is_still_read_via_the_httperror_branch(
    muse: ModuleType,
) -> None:
    """``HTTPError`` is checked before the widened ``OSError`` clause and must
    still win: it carries a real status code and body, not a bare transport
    failure."""
    port = _local_server(_http_400)

    result = muse.request(f"http://127.0.0.1:{port}/api/x", timeout=2)

    assert result == {"ok": False, "status": 400, "error": "bad request"}


def test_a_connect_phase_refusal_is_unaffected(muse: ModuleType) -> None:
    """The pre-existing ``URLError`` path (nothing listening) is unchanged --
    ``URLError`` is still an ``OSError``, so it still lands in the same branch,
    still reported via its ``.reason``.

    The exact OS error text (``Connection refused`` on POSIX, a different
    string on Windows) is platform-dependent -- checked here only for the
    shape ``request()`` promises, not the OS's own wording.
    """
    result = muse.request("http://127.0.0.1:1/api/x", timeout=1)

    assert result["ok"] is False
    assert result["status"] is None
    assert result["error"].startswith("network:")
