# -*- coding: utf-8 -*-
"""Asking for a function has to return the function.

Ten of eleven review agents hit this. Three of them spent a call re-fetching a
body that was already complete, because the coverage note counted rendered rows
against span length and the index stores no row for a blank line. The rest lost
the tail of a real body to a budget that was mostly spent elsewhere:

  - the payload carried a markdown mirror of its own cards, so every fact was
    charged twice and the body paid the difference;
  - `calls` was not shed with the other lists of places-to-look-next;
  - the per-file source cap, there so one file cannot eat a multi-hit card,
    also applied to cards showing a single definition.
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


def _card(symbol: str) -> str:
    status(project=str(FAG), architecture="arch35")
    payload = query(project=str(FAG), architecture="arch35", symbol=symbol)
    return str((payload.get("data") or {}).get("text") or "")


def _resolve_site(file: str, line: int) -> str:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        file=file,
        line=line,
    )
    return str((payload.get("data") or {}).get("text") or "")


@pytest.mark.parametrize(
    "symbol",
    [
        "ReduceSinkVF",            # 30 lines, was reported as 28 of 30
        "SelectBlockSchedule",     # 52 lines, was 48 of 52
        "GetSparseBlockInfoBn2",   # 72 lines, was 64 of 72
        "ProcessVec3",             # 84 lines, was 24 of 84
        "DoPreTiling",             # 111 lines, was 24 of 111
        "ProcessDqkv",             # 114 lines, was 109 of 114
        "CalDeterMaxLoopNum",      # 153 lines, was capped at 120
    ],
)
def test_a_function_under_the_hard_cap_comes_back_whole(symbol: str) -> None:
    text = _card(symbol)
    assert "showing" not in text, f"{symbol} still truncated: " + next(
        ln for ln in text.splitlines() if "showing" in ln
    )


def test_a_body_is_contiguous_so_a_reader_can_trust_the_numbering() -> None:
    text = _card("ReduceSinkVF")
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\|", text, re.M)]
    assert nums, "no numbered source rows"
    assert nums == list(range(nums[0], nums[-1] + 1)), f"gaps in body: {nums}"


def test_a_body_over_the_hard_cap_says_how_to_read_the_rest() -> None:
    """240 lines is a deliberate ceiling. Silence about it is not."""
    text = _card("CalGQABandIndex")
    note = [ln for ln in text.splitlines() if "showing" in ln]
    assert note, "a clipped body must say so"
    assert "resolve file=" in text


def test_the_way_on_names_the_line_the_body_stopped_at() -> None:
    """Offering the whole span invites a line the reader already has.

    A site resolve answers with the window it just returned, so "any part of
    2379-2663" cost one reader eight calls and made another stop at the cut.
    The line after the last one shown is the only one that moves.
    """
    text = _card("CalGQABandIndex")
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\|", text, re.M)]
    note = next(ln for ln in text.splitlines() if "showing" in ln and "resolve" in ln)
    assert f"line={max(nums) + 1}" in note, note
    # And that line has to actually return what it promises.
    tail = _resolve_site("op_kernel/arch35/deter.h", max(nums) + 1)
    tail_nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\|", tail, re.M)]
    assert tail_nums and min(tail_nums) == max(nums) + 1, tail_nums[:5]


def test_a_write_site_in_another_file_keeps_its_whole_expression() -> None:
    """Clipping is free only when the card prints the line somewhere else.

    A field's writers live in files the card never shows, so the echo is the
    only copy; one cut mid-call hid the `* g` that the question was about.
    """
    text = _card("deterMaxRound")
    row = next(ln for ln in text.splitlines() if "CalcleTNDDenseDeterParam" in ln)
    assert "…" not in row, row
    assert "s1Max * fBaseParams.g" in row, row


def test_a_long_callee_list_does_not_cost_the_body_its_lines() -> None:
    """`calls` was protected from every shed and still charged to the budget.

    On a host entry point with many callees it held three quarters of the
    payload, and the body -- the part no other query recovers -- was cut to 60
    of 341 lines to pay for it. Protection is there so the key survives, which
    trimming the lists inside it does not threaten.
    """
    text = _card("GetShapeAttrsInfo")
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\|", text, re.M)]
    assert len(nums) >= 240, f"body cut to {len(nums)} lines"
    assert nums == list(range(nums[0], nums[-1] + 1)), "gaps in body"
    assert "resolve file=" in text, "a clipped body must say how to read the rest"


def test_the_rendered_text_is_not_charged_against_the_fact_budget() -> None:
    from ascendc_codemap_mcp.engine.query.sql import _payload_size

    facts = {"cards": [{"snippet": "x" * 500}]}
    assert _payload_size(facts) == _payload_size({**facts, "text": "y" * 5000})
