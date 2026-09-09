"""Helpers for model text that encodes tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_PLAIN_JSON_TOOL_CALL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*(\{.*\})\s*$",
    re.DOTALL,
)
_PLAIN_JSON_TOOL_PREFIX_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.:-]*)\s*(?=\{)",
)
# --- shared text-tool-protocol vocabulary ---------------------------------
#
# One inventory, used for BOTH jobs: hiding protocol markup from the user and
# executing the calls encoded in it. Issue #1514 was a drift between two
# copies of this list -- the suppressor here knew <tvoe_calls>, DSML's
# pipe-prefixed tags and a bare <invoke>, while the provider-side extractor
# keyed on the literal <minimax:tool_call> wrapper alone. Every variant only
# the suppressor knew was therefore hidden from the user and never executed:
# a silent, total loss of the tool call, with nothing on screen to hint at it.
# What gets hidden and what gets executed must come from the same definition,
# so the parser below is built from these same constants.
_DSML_PREFIX = r"[|｜]\s*DSML\s*[|｜]\s*"


def _tag(name: str) -> str:
    """A tag name, bare or DSML-pipe-prefixed (ASCII or fullwidth pipe)."""
    return rf"(?:{name}|{_DSML_PREFIX}{name})"


_INVOKE_TAG = _tag("invoke")
_PARAMETER_TAG = _tag("parameter")

_TEXT_PROTOCOL_MARKER_RE = re.compile(
    (
        r"<\s*(?:minimax:tool_call|tool_calls?|tvoe_calls|invoke\b|"
        r"parameter\b|effect_calls\b|details\b|angle\s+brackets\b|"
        r"[|｜]\s*DSML\s*[|｜]\s*(?:tool_calls?|invoke\b|parameter\b))"
    ),
    re.IGNORECASE,
)
# Deliberately narrower than the parser's parameter pattern below: this one
# is a leak-detection heuristic ("does this suffix look like protocol?") and
# keys on parameter names that only ever appear in real tool calls. The
# parser must accept any name. Only the tag vocabulary is shared.
_TEXT_PROTOCOL_PARAMETER_RE = re.compile(
    rf"<\s*{_PARAMETER_TAG}\s+name\s*=\s*[\"'](?:path|content|command|code|patch|sheets)[\"']",
    re.IGNORECASE,
)
_TEXT_PROTOCOL_INVOKE_RE = re.compile(
    rf"<\s*{_INVOKE_TAG}\s+name\s*=\s*[\"'][A-Za-z_][A-Za-z0-9_.:-]*[\"']",
    re.IGNORECASE,
)
_TEXT_PROTOCOL_HTML_RE = re.compile(
    r"<!doctype\s+html\b|<html\b|</html\s*>",
    re.IGNORECASE,
)
_TEXT_PROTOCOL_CLOSE_RE = re.compile(
    (
        r"</\s*(?:invoke|[|｜]\s*DSML\s*[|｜]\s*invoke)\s*>|"
        r"</\s*(?:tool_calls?|tvoe_calls|[|｜]\s*DSML\s*[|｜]\s*tool_calls?)\s*>"
    ),
    re.IGNORECASE,
)
_TEXT_PROTOCOL_STANDALONE_MARKER_RE = re.compile(
    (
        r"<\s*(?:parameter|effect_calls|tool_calls?|tvoe_calls|angle\s+brackets|"
        r"[|｜]\s*DSML\s*[|｜]\s*(?:tool_calls?|invoke|parameter))\b"
    ),
    re.IGNORECASE,
)
_TEXT_PROTOCOL_DETAILS_SUMMARY_RE = re.compile(
    r"<\s*details\s*>\s*<\s*summary\s*>\s*View areas around line\b",
    re.IGNORECASE,
)
_TEXT_PROTOCOL_PREFIXES = (
    "<minimax:tool_call",
    "<tool_call",
    "<tool_calls",
    "<tvoe_calls",
    "<|dsml|tool_call",
    "<|dsml|tool_calls",
    "<|dsml|invoke",
    "<|dsml|parameter",
    "<｜dsml｜tool_call",
    "<｜dsml｜tool_calls",
    "<｜dsml｜invoke",
    "<｜dsml｜parameter",
    "<invoke",
    "<parameter",
    "<effect_calls",
    "<details",
    "<summary",
    "<angle brackets",
)
_MAX_TEXT_PROTOCOL_PREFIX_LEN = max(len(prefix) for prefix in _TEXT_PROTOCOL_PREFIXES)


def _find_trailing_tool_call_start(text: str, tool_name: str) -> int | None:
    decoder = json.JSONDecoder()
    for match in reversed(list(_PLAIN_JSON_TOOL_PREFIX_RE.finditer(text))):
        if match.group(1) != tool_name:
            continue
        try:
            arguments, end = decoder.raw_decode(text, match.end())
        except json.JSONDecodeError:
            continue
        if text[end:].strip():
            continue
        if not isinstance(arguments, dict):
            continue
        return match.start()
    return None


def strip_synthetic_tool_call_text(text: str, tool_name: str) -> str:
    """Remove trailing machine-readable tool-call text synthesized into a tool call."""

    if not text:
        return text

    if "<minimax:tool_call>" in text:
        return ""

    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            candidate = lines[index]
            break
    else:
        return text

    match = _PLAIN_JSON_TOOL_CALL_RE.match(candidate)
    if match is None or match.group(1) != tool_name:
        start = _find_trailing_tool_call_start(text, tool_name)
        if start is None:
            return text
        return text[:start].rstrip()

    prefix = "\n".join(lines[:index]).rstrip()
    return prefix


def _looks_like_text_tool_protocol_suffix(suffix: str) -> bool:
    if re.search(r"<\s*minimax:tool_call\s*>", suffix, re.IGNORECASE):
        return True
    if _TEXT_PROTOCOL_STANDALONE_MARKER_RE.search(suffix):
        return True
    if _TEXT_PROTOCOL_DETAILS_SUMMARY_RE.search(suffix):
        return True
    if _TEXT_PROTOCOL_PARAMETER_RE.search(suffix):
        return True
    if _TEXT_PROTOCOL_INVOKE_RE.search(suffix) and _TEXT_PROTOCOL_CLOSE_RE.search(suffix):
        return True
    if _TEXT_PROTOCOL_HTML_RE.search(suffix) and _TEXT_PROTOCOL_INVOKE_RE.search(suffix):
        return True
    return False


def strip_protocol_text_leak(text: str) -> str:
    """Remove text-encoded tool protocol that should not be user-visible."""

    if not text:
        return text

    for marker in _TEXT_PROTOCOL_MARKER_RE.finditer(text):
        suffix = text[marker.start() :]
        if _looks_like_text_tool_protocol_suffix(suffix):
            return text[: marker.start()].rstrip()
    return text


def _find_protocol_marker_start(text: str) -> int | None:
    marker = _TEXT_PROTOCOL_MARKER_RE.search(text)
    return marker.start() if marker is not None else None


def _find_protocol_prefix_suffix_start(text: str) -> int | None:
    lower_text = text.lower()
    start = max(0, len(text) - _MAX_TEXT_PROTOCOL_PREFIX_LEN)
    for index in range(start, len(text)):
        suffix = lower_text[index:]
        if any(prefix.startswith(suffix) for prefix in _TEXT_PROTOCOL_PREFIXES):
            return index
    return None


def _split_visible_prefix_before_protocol_candidate(text: str) -> tuple[str, str]:
    marker_start = _find_protocol_marker_start(text)
    if marker_start is None:
        marker_start = _find_protocol_prefix_suffix_start(text)
    if marker_start is None:
        return text, ""

    prefix = text[:marker_start]
    visible_prefix = prefix.rstrip()
    return visible_prefix, text[len(visible_prefix) :]


class ProtocolTextLeakGuard:
    """Stateful guard for streamed text that may contain tool protocol."""

    def __init__(self) -> None:
        self._pending = ""
        self._suppressed = False

    def push(self, text: str) -> str:
        if not text or self._suppressed:
            return ""

        combined = self._pending + text
        self._pending = ""
        cleaned = strip_protocol_text_leak(combined)
        if cleaned != combined:
            self._suppressed = True
            return cleaned

        visible, pending = _split_visible_prefix_before_protocol_candidate(combined)
        self._pending = pending
        return visible

    def flush(self) -> str:
        if self._suppressed:
            self._pending = ""
            self._suppressed = False
            return ""
        pending = self._pending
        self._pending = ""
        return strip_protocol_text_leak(pending)

    def flush_before_tool_use(self) -> str:
        if self._suppressed:
            self._pending = ""
            self._suppressed = False
            return ""
        return self.flush()


@dataclass(frozen=True)
class TextToolInvocation:
    """A tool call a model encoded in its text instead of the tool-call API."""

    name: str
    arguments: dict[str, Any]


# Keyed on a well-formed <invoke name="..."> ... </invoke> pair rather than on
# the wrapper around it. Which wrapper a model happens to emit --
# <minimax:tool_call>, <tvoe_calls>, a DSML pipe tag, or nothing at all -- says
# nothing about whether the call inside is real, and gating on the wrapper is
# what lost every non-MiniMax variant in #1514.
_INVOCATION_RE = re.compile(
    rf'<\s*{_INVOKE_TAG}\s+name\s*=\s*"([^"]+)"\s*>(.*?)<\s*/\s*{_INVOKE_TAG}\s*>',
    re.DOTALL | re.IGNORECASE,
)
_INVOCATION_PARAMETER_RE = re.compile(
    rf'<\s*{_PARAMETER_TAG}\s+name\s*=\s*"([^"]+)"([^>]*)>(.*?)<\s*/\s*{_PARAMETER_TAG}\s*>',
    re.DOTALL | re.IGNORECASE,
)
# DSML marks each parameter with string="true"/"false" to say whether the body
# is a literal or JSON to decode. Ignoring it hands a tool an escaped JSON
# string where its schema expects structured data -- create_xlsx receives a
# string instead of a list of rows and fails on a call that was well-formed.
_STRING_ATTR_RE = re.compile(r'string\s*=\s*"\s*(true|false)\s*"', re.IGNORECASE)


def _parameter_value(raw: str, attributes: str) -> Any:
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    attr = _STRING_ATTR_RE.search(attributes)
    if attr is None or attr.group(1).lower() != "false":
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # A model that mislabels a literal as JSON should not lose the call;
        # the tool's own schema validation is the better place to reject it.
        return raw


def parse_text_tool_invocations(text: str) -> list[TextToolInvocation]:
    """Extract tool calls a model encoded as text markup.

    Recognises exactly the invoke/parameter tag forms the leak suppressor in
    this module hides, because both are built from the same tag constants.
    """
    if not text:
        return []

    invocations: list[TextToolInvocation] = []
    for invoke in _INVOCATION_RE.finditer(text):
        name = invoke.group(1).strip()
        if not name:
            continue
        arguments: dict[str, Any] = {}
        for parameter in _INVOCATION_PARAMETER_RE.finditer(invoke.group(2)):
            key = parameter.group(1).strip()
            if key:
                arguments[key] = _parameter_value(parameter.group(3), parameter.group(2))
        invocations.append(TextToolInvocation(name=name, arguments=arguments))
    return invocations


def strip_synthetic_tool_call_suffix(text: str, tool_names: list[str]) -> str:
    """Remove text-encoded tool calls for any of the supplied synthetic tools."""

    cleaned = text
    for tool_name in tool_names:
        cleaned = strip_synthetic_tool_call_text(cleaned, tool_name)
    return cleaned
