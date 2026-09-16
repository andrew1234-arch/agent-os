"""``_run_git`` must raise ``ToolError``, not ``RuntimeError`` (#2480).

Every other builtin tool module (filesystem.py, artifacts.py,
session_search.py, messaging.py, ...) raises ``agentos.tools.types.ToolError``
for an expected tool-execution failure; git.py didn't even import it. A
direct caller that follows that repo-wide convention -- catching
``ToolError`` around a tool call, the way ``_diff_revision`` itself already
did for ``_run_git``'s own failures before this fix -- would not catch a bare
``RuntimeError``.

The issue's second repro (``git_status(workdir="/invalid/directory")``)
claimed this raised ``RuntimeError``; verified directly that it actually
raises ``FileNotFoundError`` from ``asyncio.create_subprocess_exec`` itself,
before ``_run_git``'s own two ``raise`` sites are ever reached -- a third,
previously-unwrapped failure surface. Both are fixed here.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentos.sandbox.config import SandboxSettings
from agentos.sandbox.integration import configure_runtime, reset_runtime
from agentos.tools.builtin import git
from agentos.tools.types import ToolContext, ToolError, current_tool_context


def _git(repo: Path, *args: str) -> None:
    missing = str(repo.parent / "no-such-gitconfig")
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": missing,
            "GIT_CONFIG_SYSTEM": missing,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )


@pytest.fixture
def repo_with_clean_tree(tmp_path: Path) -> Iterator[Path]:
    """One commit, nothing left to stage -- ``git commit`` exits 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    # Local repo config, not just the env vars ``_git`` passes for its own
    # invocations: the ``git.git_commit`` call below runs through the ambient
    # environment (no sandbox), which has no git identity configured on a
    # bare CI runner, so without this it fails on "Author identity unknown"
    # before ever reaching the clean-tree check this test is about.
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")

    configure_runtime(
        SandboxSettings(sandbox=False, security_grading=False, allow_legacy_mode=True),
        workspace=repo,
    )
    token = current_tool_context.set(ToolContext(workspace_dir=str(repo)))
    try:
        yield repo
    finally:
        current_tool_context.reset(token)
        reset_runtime()


async def test_git_commit_with_clean_tree_raises_tool_error(
    repo_with_clean_tree: Path,
) -> None:
    """Issue's own repro: ``nothing to commit, working tree clean`` (exit 1)."""
    with pytest.raises(ToolError) as excinfo:
        await git.git_commit(message="test commit")

    assert "nothing to commit" in str(excinfo.value)


async def test_git_commit_with_clean_tree_does_not_raise_runtime_error(
    repo_with_clean_tree: Path,
) -> None:
    """A caller that follows the repo-wide ``except ToolError`` convention
    must not see a bare ``RuntimeError`` slip past it."""
    try:
        await git.git_commit(message="test commit")
    except ToolError:
        pass
    else:
        pytest.fail("expected git_commit to raise on a clean working tree")


async def test_run_git_nonzero_exit_raises_tool_error(tmp_path: Path) -> None:
    """Direct unit on ``_run_git``'s own nonzero-exit branch, no sandbox."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    with pytest.raises(ToolError):
        await git._run_git("not-a-real-subcommand", cwd=str(repo))


async def test_git_status_invalid_workdir_raises_tool_error(tmp_path: Path) -> None:
    """The issue's second repro. The real failure is ``FileNotFoundError``
    from the subprocess launch itself (cwd does not exist) -- verified this
    is what actually happens, not a ``RuntimeError`` from the exit-code
    check, which is never reached for a cwd that doesn't exist."""
    configure_runtime(
        SandboxSettings(sandbox=False, security_grading=False, allow_legacy_mode=True),
        workspace=tmp_path,
    )
    token = current_tool_context.set(ToolContext(workspace_dir=str(tmp_path)))
    try:
        with pytest.raises(ToolError):
            await git.git_status(workdir=str(tmp_path / "does-not-exist"))
    finally:
        current_tool_context.reset(token)
        reset_runtime()


async def test_run_git_invalid_cwd_is_not_a_bare_os_error(tmp_path: Path) -> None:
    """Guard: the OSError from launching git against a missing cwd must be
    wrapped, not left to propagate as a raw FileNotFoundError."""
    with pytest.raises(ToolError):
        await git._run_git("status", cwd=str(tmp_path / "does-not-exist"))
