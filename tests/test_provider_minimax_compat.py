"""Unit coverage for provider.minimax_compat -- previously had none at all.

That gap is part of how the underlying bug (this module only recognizing the
literal "<minimax:tool_call>" wrapper while engine.tool_text_compat already
recognized several other real wrapper variants as tool-call protocol) went
unreported: nothing exercised this module's detection/parsing against any
wrapper other than the one it was originally written for.
"""

from __future__ import annotations

from agentos.provider.minimax_compat import contains_minimax_protocol, parse_minimax_tool_calls


def test_original_minimax_wrapper_still_parses() -> None:
    """Backward compatibility: the wrapper this module was built for."""
    text = (
        '<minimax:tool_call><invoke name="web_search">'
        '<parameter name="query">agentos</parameter>'
        "</invoke></minimax:tool_call>"
    )

    calls = parse_minimax_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "agentos"}


def test_plain_text_with_no_protocol_markup_is_not_detected() -> None:
    plain = 'Here is the answer.\n\nweb_search{"query": "agentos"}'
    assert contains_minimax_protocol(plain) is False
    assert parse_minimax_tool_calls("Just some ordinary prose about tools.") == []


def test_tvoe_calls_typo_wrapper_is_detected_and_parsed() -> None:
    """A real, previously-observed typo-wrapper variant of the same protocol."""
    text = (
        "Let me write the dashboard now.\n\n"
        '<tvoe_calls><invoke name="write_file">'
        '<parameter name="path">index.html</parameter>'
        '<parameter name="content"><!DOCTYPE html><html><body>app</body></html>'
        "</parameter></invoke></tvoe_calls>"
    )

    assert contains_minimax_protocol(text) is True
    calls = parse_minimax_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {
        "path": "index.html",
        "content": "<!DOCTYPE html><html><body>app</body></html>",
    }


def test_bare_invoke_with_no_wrapper_at_all_is_detected_and_parsed() -> None:
    """Detection is keyed on the invoke/close shape, not on any specific wrapper."""
    text = '<invoke name="web_search"><parameter name="query">news</parameter></invoke>'

    assert contains_minimax_protocol(text) is True
    calls = parse_minimax_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "news"}


def test_dsml_wrapper_is_detected_and_string_true_stays_a_literal_string() -> None:
    text = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="create_xlsx">'
        '<｜DSML｜parameter name="name" string="true">'
        "bean-sprout-daily-record-sheet.xlsx"
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    )

    assert contains_minimax_protocol(text) is True
    calls = parse_minimax_tool_calls(text)
    assert calls[0].arguments["name"] == "bean-sprout-daily-record-sheet.xlsx"
    assert isinstance(calls[0].arguments["name"], str)


def test_dsml_string_false_is_json_decoded_into_its_real_shape() -> None:
    """string="false" means "this value is JSON", not "keep it as a JSON string"."""
    text = (
        '<｜DSML｜invoke name="create_xlsx">'
        '<｜DSML｜parameter name="sheets" string="false">'
        '[{"name":"Record Sheet","rows":[["Day","Height"]]}]'
        "</｜DSML｜parameter></｜DSML｜invoke>"
    )

    calls = parse_minimax_tool_calls(text)

    assert calls[0].arguments["sheets"] == [{"name": "Record Sheet", "rows": [["Day", "Height"]]}]
    assert isinstance(calls[0].arguments["sheets"], list)


def test_dsml_ascii_pipe_variant_is_also_recognized() -> None:
    """The pipe character has been observed in both fullwidth and ASCII form."""
    text = (
        '<|DSML|invoke name="create_xlsx">'
        '<|DSML|parameter name="sheets" string="false">[]</|DSML|parameter>'
        "</|DSML|invoke>"
    )

    calls = parse_minimax_tool_calls(text)

    assert calls[0].arguments["sheets"] == []


def test_malformed_json_in_a_string_false_parameter_degrades_to_raw_text() -> None:
    """A malformed value must not crash extraction of an otherwise-valid call."""
    text = (
        '<｜DSML｜invoke name="create_xlsx">'
        '<｜DSML｜parameter name="sheets" string="false">not actually json'
        "</｜DSML｜parameter></｜DSML｜invoke>"
    )

    calls = parse_minimax_tool_calls(text)

    assert calls[0].arguments["sheets"] == "not actually json"


def test_documentation_style_mention_of_an_unregistered_tool_name_still_parses() -> None:
    """Parsing is permissive; the caller's allowlist is what guards execution.

    This module only answers "what invoke blocks does this text contain" --
    it is provider.openai's ``allowed_tool_names`` filter, not this parser,
    that must refuse to act on a tool name nobody registered. This test
    documents that boundary rather than asserting this module filters
    anything itself.
    """
    text = 'Example: <invoke name="not_a_real_tool"><parameter name="x">y</parameter></invoke>'

    calls = parse_minimax_tool_calls(text)

    assert calls[0].name == "not_a_real_tool"


def test_multiple_invocations_in_one_text_are_all_parsed() -> None:
    text = (
        "<minimax:tool_call>"
        '<invoke name="a"><parameter name="x">1</parameter></invoke>'
        '<invoke name="b"><parameter name="y">2</parameter></invoke>'
        "</minimax:tool_call>"
    )

    calls = parse_minimax_tool_calls(text)

    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].arguments == {"x": "1"}
    assert calls[1].arguments == {"y": "2"}
