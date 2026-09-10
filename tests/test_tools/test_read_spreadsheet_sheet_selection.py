"""Sheet selection in read_spreadsheet.

Regression coverage for #1569: _select_spreadsheet_sheets used to check the
positional-index interpretation of the argument before looking for a sheet
with that exact name, so a workbook containing a sheet literally named "1"
could never reach it -- sheet="1" always resolved to positional index 0.
"""

from __future__ import annotations

import pytest

from agentos.tools.builtin.filesystem import _select_spreadsheet_sheets
from agentos.tools.types import ToolError

# (name, rows, total_rows) -- rows is a 1-indexed sparse map, matching what
# _read_xlsx_sheets / _read_delimited_rows produce.
_SHEETS: list[tuple[str, dict[int, list[str]], int]] = [
    ("0", {1: ["a"]}, 1),
    ("Summary", {1: ["b"]}, 1),
    ("1", {1: ["c"]}, 1),
]


def test_numeric_sheet_name_is_reachable_by_exact_name() -> None:
    """A sheet literally named "1" must not be shadowed by the positional
    interpretation of "1" (which would otherwise resolve to index 0,
    i.e. the sheet named "0").
    """
    result = _select_spreadsheet_sheets(_SHEETS, "1")

    assert [name for name, _, _ in result] == ["1"]
    assert result[0][1] == {1: ["c"]}


def test_numeric_sheet_name_reachable_when_requested_as_int() -> None:
    """The int form must resolve the same way as the string form -- a model
    could pass either depending on how it serializes the tool call.
    """
    result = _select_spreadsheet_sheets(_SHEETS, 1)

    assert [name for name, _, _ in result] == ["1"]


def test_zero_named_sheet_is_reachable() -> None:
    result = _select_spreadsheet_sheets(_SHEETS, "0")

    assert [name for name, _, _ in result] == ["0"]
    assert result[0][1] == {1: ["a"]}


def test_positional_index_still_works_when_no_sheet_has_that_name() -> None:
    """Fallback to positional index must survive the reorder for the
    ordinary case -- no sheet named "2" exists, so "2" means "the 2nd sheet".
    """
    result = _select_spreadsheet_sheets(_SHEETS, "2")

    assert [name for name, _, _ in result] == ["Summary"]


def test_case_insensitive_name_match_still_beats_positional_fallback() -> None:
    sheets: list[tuple[str, dict[int, list[str]], int]] = [
        ("Q1", {1: ["x"]}, 1),
        ("Q2", {1: ["y"]}, 1),
    ]

    result = _select_spreadsheet_sheets(sheets, "q1")

    assert [name for name, _, _ in result] == ["Q1"]


def test_unknown_sheet_raises_with_available_names() -> None:
    with pytest.raises(ToolError, match="Sheet not found: nope"):
        _select_spreadsheet_sheets(_SHEETS, "nope")


def test_none_or_empty_returns_all_sheets() -> None:
    assert _select_spreadsheet_sheets(_SHEETS, None) == _SHEETS
    assert _select_spreadsheet_sheets(_SHEETS, "") == _SHEETS
