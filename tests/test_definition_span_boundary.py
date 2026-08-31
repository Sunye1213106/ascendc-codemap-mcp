# -*- coding: utf-8 -*-
"""Definition spans must not overlap, and a card must not open in its neighbour.

Extraction keyed function records by short name and started a span at the name
token, so an out-of-class template method began a few lines above itself --
inside the function before it. `resolve` then answered with source the caller
did not ask for, and the header claimed a range that was never that function's.
"""
from __future__ import annotations

import sqlite3
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
DEF_KINDS = ("FUNCTION", "METHOD", "KERNEL")

pytestmark = pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")


def _text(payload: dict) -> str:
    return str((payload.get("data") or {}).get("text") or "")


def test_definition_spans_do_not_overlap_siblings() -> None:
    conn = sqlite3.connect(str(FAG / _REL_UO))
    try:
        rows = conn.execute(
            f"""
            SELECT file, name, line_start, line_end FROM entity
            WHERE kind IN ({','.join('?' for _ in DEF_KINDS)})
              AND IFNULL(line_start, 0) > 0 AND IFNULL(line_end, 0) > line_start
            """,
            DEF_KINDS,
        ).fetchall()
    finally:
        conn.close()

    by_file: dict[str, list[tuple[int, int, str]]] = {}
    for file, name, start, end in rows:
        by_file.setdefault(str(file), []).append((int(start), int(end), str(name)))

    bad: list[str] = []
    for file, spans in by_file.items():
        spans.sort()
        for index in range(1, len(spans)):
            p_start, p_end, p_name = spans[index - 1]
            c_start, c_end, c_name = spans[index]
            nested = c_start >= p_start and c_end <= p_end
            if c_start <= p_end and not nested:
                bad.append(f"{file}: {p_name} {p_start}-{p_end} vs {c_name} {c_start}-{c_end}")
    assert not bad, "overlapping definition spans:\n" + "\n".join(bad[:10])


def test_resolve_at_closing_brace_stays_in_the_function() -> None:
    header = "op_kernel/arch35/flash_attention_score_grad_kernel_deter.h"
    status(project=str(FAG), architecture="arch35")

    body = _text(query(project=str(FAG), architecture="arch35", file=header, line=490))
    assert "CalDenseDeterIndex" in body.splitlines()[0]

    tail = _text(query(project=str(FAG), architecture="arch35", file=header, line=494))
    assert "CalDenseDeterIndex" in tail.splitlines()[0]

    nxt = _text(query(project=str(FAG), architecture="arch35", file=header, line=497))
    assert "CalBandDeterIndex" in nxt.splitlines()[0]
    # The previous function's last lines must not be quoted as this one's body.
    source = nxt.split("Source", 1)[-1]
    assert "493|" not in source
    assert "494|" not in source


def test_resolve_symbol_returns_the_whole_function_body() -> None:
    status(project=str(FAG), architecture="arch35")
    text = _text(
        query(project=str(FAG), architecture="arch35", symbol="GetDTemplateType")
    )
    # One call has to be enough to read the branch that answers the question.
    assert "hasRope" in text
    assert "NUM768" in text
    assert text.count("|") > 15


def test_enclosing_definition_beats_a_call_recorded_on_the_same_line() -> None:
    status(project=str(FAG), architecture="arch35")
    text = _text(
        query(
            project=str(FAG),
            architecture="arch35",
            file="op_kernel/arch35/flash_attention_score_grad_kernel_base.h",
            line=243,
        )
    )
    assert text.splitlines()[0].endswith("Process")
