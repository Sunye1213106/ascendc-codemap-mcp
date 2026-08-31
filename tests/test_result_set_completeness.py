# -*- coding: utf-8 -*-
"""Say whether the answer is all of it.

Six of eleven review agents hedged a conclusion or re-derived a list because
nothing told them whether what came back was the whole set. The tool knew:
search carries a total and a cursor, resolve carries `coverage`. None of it was
rendered, so a complete answer and a first page looked identical.

The two halves have to agree. Resolve points at search when its own list is
partial, so search declaring `complete` is what actually ends the question.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)
_REL_UO = Path(".ascendc-codemap/arch35/FlashAttentionScoreGrad.arch35.uo")


def _operator() -> Path | None:
    for root in _CANDIDATES:
        if (root / _REL_UO).is_file():
            return root
    return None


FAG = _operator()

pytestmark = pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")

#: A hit under its unit heading. Ungrouped rows carry no indent and are counted
#: by the header too, so the comparison allows for them.
_HIT_LINE_RE = re.compile(r"^\s+\d+:")
_LEFTOVER_SLACK = 120


def _data(**kw) -> dict:
    status(project=str(FAG), architecture="arch35")
    payload = query(project=str(FAG), architecture="arch35", **kw)
    return payload.get("data") or {}


def _text(**kw) -> str:
    return str(_data(**kw).get("text") or "")


def _header(pattern: str) -> str:
    return _text(pattern=pattern, operation="search").splitlines()[0]


def _coverage(symbol: str) -> str:
    lines = [ln for ln in _text(symbol=symbol).splitlines() if ln.startswith("Coverage:")]
    assert lines, f"no coverage line for {symbol}"
    return lines[0]


def test_a_small_result_set_is_declared_complete() -> None:
    """The one case a reviewer most needs: this really is all of them."""
    head = _header("SelectDeterBandSchedule")
    assert head.endswith("complete"), head


def test_a_partly_shown_result_set_says_so_and_how_to_continue() -> None:
    """Every unit is listed, so the way on is a unit, not a page.

    Leaving the withheld count to be summed from the per-unit notes put the
    arithmetic on the reader, and one broad pattern folded a hundred lines
    across twenty-four units, so the header states the total itself.
    """
    head = _header("deterMaxRound")
    assert "showing" in head, head
    assert "every unit listed" in head, head
    assert re.search(r"\d+ more folded into the units below", head), head
    assert "resolve a unit" in head, head


def test_search_does_not_offer_a_cursor_that_returns_the_same_view() -> None:
    """Units are grouped over every match, so a page offset moves nothing.

    Following the advertised cursor used to hand back the same twenty-eight
    units under a new offset, which reads as a pager that will not advance.
    """
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="search",
        pattern="vregReduceSum",
    )
    data = payload.get("data") or {}
    units = [u for u in (data.get("units") or []) if isinstance(u, dict)]
    covered = sum(len(u.get("hits") or []) for u in units) + len(data.get("leftover") or [])
    assert covered >= int(data.get("total") or 0), "units no longer cover every match"
    assert not data.get("next_cursor"), "a cursor was offered over a complete unit view"


def test_a_header_counts_only_the_lines_the_reader_receives() -> None:
    """The count used to be taken before the payload clipper cut the body."""
    text = _text(pattern="i", operation="search")
    head = text.splitlines()[0]
    claimed = re.search(r"showing (\d+)", head)
    assert claimed, head
    delivered = sum(1 for ln in text.splitlines() if _HIT_LINE_RE.match(ln))
    assert int(claimed.group(1)) <= delivered + _LEFTOVER_SLACK, f"{head} over {delivered}"
    assert "truncated" not in text.splitlines()[-1].lower(), text.splitlines()[-1]


def test_a_unit_showing_part_of_its_hits_counts_the_rest() -> None:
    """Three of nine hits used to render exactly like a unit that had three."""
    text = _text(pattern="CrossCore", operation="search")
    assert "more in this unit" in text, text.splitlines()[0]


def test_a_unique_definition_is_not_reported_as_an_unchecked_first_hit() -> None:
    assert "the only definition" in _coverage("SelectBlockSchedule")


def test_a_partial_definition_list_points_at_the_query_that_completes_it() -> None:
    line = _coverage("isBn2MultiBlk")
    assert "first page only" in line, line
    assert "search pattern=isBn2MultiBlk" in line, line


def test_a_complete_definition_list_says_all_listed() -> None:
    assert "all listed" in _coverage("deterMaxRound")


def test_a_line_past_a_files_end_is_not_reported_as_a_missing_file() -> None:
    """The file is indexed; the line is not. Those need different answers.

    Told the file was absent, a reader stopped using file+line altogether and
    took two calls to get back to a locator; told the line was, they pass a
    line that exists.
    """
    text = _text(
        operation="resolve",
        file="op_kernel/arch35/flash_attention_score_grad_kernel.h",
        line=919,
    )
    assert "is in the snapshot" in text, text[:300]
    assert "ends at line 759" in text, text[:300]
    assert "not in snapshot" not in text, text[:300]


def test_trimming_context_to_fit_is_admitted_not_hidden() -> None:
    """`completeness: COMPLETE` alongside `truncated: true` told the reader nothing."""
    assert "trimmed to fit" in _coverage("deterMaxRound")
    # The untrimmed case has to say what it is claiming. "Lists complete" was
    # read as a warranty over every section, including ones with their own cap.
    assert "nothing dropped to fit" in _coverage("ProcessDqkv")


def test_an_admitted_trim_names_the_list_it_cut() -> None:
    """"Some lists were trimmed" costs as much as saying nothing.

    One reader, told only that something had been shed, rebuilt a caller list
    that had come back complete -- five calls to recover nothing. Naming the
    list that was cut is the whole value of admitting the cut.
    """
    data = _data(symbol="ProcessVec3")
    cut = list(data.get("trimmed_lists") or [])
    assert cut, "fixture no longer trims anything; pick a card that does"
    line = _coverage("ProcessVec3")
    # The clause names exactly the lists the packer shed -- no more, no fewer.
    assert f"{', '.join(cut)} trimmed to fit" in line, (cut, line)
    assert "the rest complete" in line, line


def test_a_card_that_sheds_context_keeps_its_body_and_says_which_it_shed() -> None:
    """The callee list is recoverable by resolving any name in it; a body is not."""
    line = _coverage("ProcessVec3")
    assert "trimmed to fit" in line, line
    text = _text(symbol="ProcessVec3")
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\|", text, re.M)]
    assert len(nums) >= 84, f"body cut to {len(nums)} lines to save a list"
