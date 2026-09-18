"""Tests for the bundled musebook skill script (muse.py) (issue #2648).

``emit()`` wrote JSON straight to ``sys.stdout``, which encodes through
``sys.stdout.encoding`` -- the console code page on Windows (cp437, cp1252,
...), not UTF-8. ``keygen``'s note contains an em dash, and any signed field
can carry an emoji, so a real subcommand run under a non-UTF-8 code page
crashed with ``UnicodeEncodeError`` before a single byte of the result was
written. Tests run the actual script as a subprocess with the target code
page forced through ``PYTHONIOENCODING``, the same way a real Windows console
would set it, rather than simulating the encoding at the Python level.

Separately, every path example in ``references/muse.txt`` is written with
its ``/api/`` prefix already on it, so a muse copying one straight into
``--path`` doubled the prefix and 404'd.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "agentos" / "skills" / "bundled" / "musebook" / "scripts" / "muse.py"


def _run_muse(*args: str, encoding: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONIOENCODING": encoding}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        env=env,
        check=False,
    )


def test_keygen_survives_a_cp437_console(tmp_path: Path) -> None:
    """The note's em dash (U+2014) is outside cp437 -- issue #2648's case 1."""
    proc = _run_muse("keygen", "--save", encoding="cp437")

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload["ok"] is True
    assert "—" in payload["note"]


def test_sign_with_an_emoji_field_survives_a_cp1252_console() -> None:
    """An emoji field is outside cp1252 -- issue #2648's case 2."""
    proc = _run_muse(
        "sign",
        "--endpoint", "react",
        "--muse-id", "muse_test",
        "--secret", "wVE5adj76q7PAM7kNB2NsFmK2QWSkssfU6m1_r2vJfM",
        "--field", "emoji=\U0001f49b",
        encoding="cp1252",
    )

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload["ok"] is True
    assert payload["body"]["emoji"] == "\U0001f49b"


def _load_muse() -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_agentos_test_muse", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_api_path_strips_the_doc_examples_prefix() -> None:
    """Every path in references/muse.txt is written with a leading /api/."""
    muse = _load_muse()

    assert muse.normalize_api_path("latest.json") == "latest.json"
    assert muse.normalize_api_path("/latest.json") == "latest.json"
    assert muse.normalize_api_path("api/latest.json") == "latest.json"
    assert muse.normalize_api_path("/api/latest.json") == "latest.json"
    assert muse.normalize_api_path("api/mentions.json") == "mentions.json"


def test_cmd_get_builds_a_single_api_prefix_from_a_doc_style_path() -> None:
    """End to end: --path copied verbatim from muse.txt must not double up."""
    import argparse

    muse = _load_muse()
    captured: dict[str, str] = {}

    def fake_request(url: str, **kwargs: object) -> dict[str, object]:
        captured["url"] = url
        return {"ok": True}

    muse.request = fake_request  # type: ignore[attr-defined]
    args = argparse.Namespace(
        path="/api/latest.json",
        endpoint=None,
        field=None,
        file_field=None,
        query=None,
        timeout=10,
    )

    assert muse.cmd_get(args) == 0
    assert captured["url"] == "https://musebook.lol/api/latest.json"
