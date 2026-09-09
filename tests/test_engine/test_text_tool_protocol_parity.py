"""Execution and suppression must recognise the same protocol markup.

Regression coverage for #1514. ``_synthesize_text_tool_events`` gated
extraction on the literal ``<minimax:tool_call>`` wrapper, while the leak
suppressor in ``engine.tool_text_compat`` already knew the ``<tvoe_calls>``
typo-wrapper, DSML's pipe-prefixed tags and a bare ``<invoke>``. Every
variant only the suppressor knew was hidden from the user and never
executed: the call was lost in silence, because nothing on screen suggested
a tool had been requested at all.

These tests assert the two behaviours over the same inputs, so a variant
added to one side and not the other fails here instead of losing tool calls.
"""

from __future__ import annotations

import pytest

from agentos.engine.tool_text_compat import (
    parse_text_tool_invocations,
    strip_protocol_text_leak,
)

_WRITE_FILE_BODY = '<invoke name="write_file"><parameter name="path">a.txt</parameter></invoke>'

_VARIANTS = {
    "minimax_wrapper": f"<minimax:tool_call>{_WRITE_FILE_BODY}</minimax:tool_call>",
    "tvoe_calls_wrapper": f"<tvoe_calls>{_WRITE_FILE_BODY}</tvoe_calls>",
    "tool_calls_wrapper": f"<tool_calls>{_WRITE_FILE_BODY}</tool_calls>",
    "no_wrapper": _WRITE_FILE_BODY,
    "dsml_ascii_pipe": (
        '<|DSML|invoke name="write_file">'
        '<|DSML|parameter name="path">a.txt</|DSML|parameter>'
        "</|DSML|invoke>"
    ),
    "dsml_fullwidth_pipe": (
        '<｜DSML｜invoke name="write_file">'
        '<｜DSML｜parameter name="path">a.txt</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
    ),
}


@pytest.mark.parametrize("markup", _VARIANTS.values(), ids=list(_VARIANTS))
def test_every_parsed_variant_is_also_suppressed(markup: str) -> None:
    """Anything executable must be markup the user never sees."""
    assert parse_text_tool_invocations(markup), "variant not recognised for execution"
    assert strip_protocol_text_leak(f"Sure, writing that now.{markup}") == (
        "Sure, writing that now."
    )


@pytest.mark.parametrize("markup", _VARIANTS.values(), ids=list(_VARIANTS))
def test_every_variant_yields_the_same_call(markup: str) -> None:
    """The wrapper a model happens to emit must not change the call."""
    invocations = parse_text_tool_invocations(markup)

    assert len(invocations) == 1
    assert invocations[0].name == "write_file"
    assert invocations[0].arguments == {"path": "a.txt"}


def test_dsml_string_false_parameter_is_decoded_as_json() -> None:
    """DSML labels non-literal bodies; a tool's schema expects the decoded form.

    Passing the raw text through would hand ``create_xlsx`` an escaped JSON
    string where its schema expects a list of rows, failing a call that was
    perfectly well-formed.
    """
    markup = (
        '<|DSML|invoke name="create_xlsx">'
        '<|DSML|parameter name="sheets" string="false">[[1, 2]]</|DSML|parameter>'
        '<|DSML|parameter name="path" string="true">out.xlsx</|DSML|parameter>'
        "</|DSML|invoke>"
    )

    arguments = parse_text_tool_invocations(markup)[0].arguments

    assert arguments == {"sheets": [[1, 2]], "path": "out.xlsx"}


def test_mislabelled_json_parameter_falls_back_to_the_literal() -> None:
    """A model that mislabels a literal must not lose the whole call."""
    markup = (
        '<invoke name="write_file">'
        '<parameter name="content" string="false">not json at all</parameter>'
        "</invoke>"
    )

    assert parse_text_tool_invocations(markup)[0].arguments == {
        "content": "not json at all"
    }


def test_ordinary_prose_is_neither_parsed_nor_stripped() -> None:
    text = "I would use write_file here, but the path is unclear."

    assert parse_text_tool_invocations(text) == []
    assert strip_protocol_text_leak(text) == text


def test_multiple_invocations_in_one_text_are_all_parsed() -> None:
    """A model may batch several calls into one message."""
    markup = (
        "<minimax:tool_call>"
        '<invoke name="write_file"><parameter name="path">a.txt</parameter></invoke>'
        '<invoke name="read_file"><parameter name="path">b.txt</parameter></invoke>'
        "</minimax:tool_call>"
    )

    invocations = parse_text_tool_invocations(markup)

    assert [(i.name, i.arguments) for i in invocations] == [
        ("write_file", {"path": "a.txt"}),
        ("read_file", {"path": "b.txt"}),
    ]


def test_an_unoffered_tool_name_still_parses_as_protocol() -> None:
    """Parsing and authorization are separate concerns.

    The caller drops any name the turn did not offer. Parsing must still
    report the invocation, because the text is protocol either way: the
    plain-JSON fallback in _synthesize_text_tool_events keys on whether
    markup was found at all, so text carrying a well-formed invoke for an
    unavailable tool must not be re-read as a bare JSON tool call.
    """
    markup = '<invoke name="not_a_registered_tool"><parameter name="x">1</parameter></invoke>'

    invocations = parse_text_tool_invocations(markup)

    assert [i.name for i in invocations] == ["not_a_registered_tool"]
