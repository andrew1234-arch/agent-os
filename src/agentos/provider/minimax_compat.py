"""Compatibility parser for text-encoded ``<invoke>``/``<parameter>`` tool calls.

Originally scoped to MiniMax's native ``<minimax:tool_call>``-wrapped XML.
Widened to also cover two other real, previously-observed wrapper variants
that ``engine.tool_text_compat`` already recognized and hid from the user
without this module ever extracting and executing them: a ``tvoe_calls``
typo-wrapper, and DSML's pipe-prefixed ``<｜DSML｜invoke>``/``<｜DSML｜parameter>``
tags with an explicit ``string="true"/"false"`` value-typing attribute.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentos.engine.tool_text_compat import contains_invoke_protocol_markup

_WRAPPER_MARKER = re.compile(r"<\s*minimax:tool_call\s*>", re.IGNORECASE)

# Optional DSML pipe-prefix immediately before a tag name, in either the
# ASCII ``|`` or fullwidth ``｜`` form models have been observed emitting.
_DSML_PREFIX = r"(?:[|｜]\s*DSML\s*[|｜]\s*)?"

_INVOKE_RE = re.compile(
    rf'<\s*{_DSML_PREFIX}invoke\s+name\s*=\s*"([^"]+)"\s*>(.*?)'
    rf"<\s*/\s*{_DSML_PREFIX}invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    rf'<\s*{_DSML_PREFIX}parameter\s+name\s*=\s*"([^"]+)"'
    rf'(?:\s+string\s*=\s*"(true|false)")?'
    rf"\s*>(.*?)<\s*/\s*{_DSML_PREFIX}parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class MinimaxToolCall:
    name: str
    arguments: dict[str, Any]


def contains_minimax_protocol(text: str) -> bool:
    """Return True when text contains a text-encoded ``<invoke>`` tool call.

    True for the original MiniMax wrapper, and for any other wrapper spelling
    (``tvoe_calls``, DSML's pipe-prefixed tags, or no wrapper at all) that
    ``engine.tool_text_compat`` already treats as leaked tool-call protocol --
    see ``contains_invoke_protocol_markup`` for why these must stay in sync.
    """
    return bool(_WRAPPER_MARKER.search(text)) or contains_invoke_protocol_markup(text)


def _coerce_param_value(raw_value: str, is_json: bool) -> Any:
    value = raw_value
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    if not is_json:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Malformed model output should degrade to the literal text rather
        # than crash extraction of an otherwise-valid tool call.
        return value


def parse_minimax_tool_calls(text: str) -> list[MinimaxToolCall]:
    """Extract text-encoded ``<invoke>``/``<parameter>`` tool invocations."""
    if not contains_minimax_protocol(text):
        return []

    calls: list[MinimaxToolCall] = []
    for invoke_match in _INVOKE_RE.finditer(text):
        name = invoke_match.group(1).strip()
        body = invoke_match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _PARAM_RE.finditer(body):
            key = param_match.group(1).strip()
            is_json = param_match.group(2) == "false"
            arguments[key] = _coerce_param_value(param_match.group(3), is_json)
        if name:
            calls.append(MinimaxToolCall(name=name, arguments=arguments))
    return calls
