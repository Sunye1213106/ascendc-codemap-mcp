# -*- coding: utf-8 -*-
"""Claims a card must not make when the source does not support them.

Each case here was a wrong statement a reader acted on, not a missing one:
a caller that cannot call, a condition attached to the wrong half of a
branch, a call line counted as a definition. A card that stays silent costs
one more query; a card that is confidently wrong costs the conclusion.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.engine.passes.kernel_tiling_closure import _param_names
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel


def _resolve(op: Path, **kw) -> str:
    status(project=str(op), architecture="arch35")
    payload = query(project=str(op), architecture="arch35", operation="resolve", **kw)
    return str((payload.get("data") or {}).get("text") or "")


def _section(text: str, head: str) -> str:
    """Just the rows under one heading; sections end at the blank line."""
    if head not in text:
        return ""
    return text.split(head, 1)[-1].split("\n\n", 1)[0]


def test_caller_site_outside_its_body_is_dropped(tmp_path: Path) -> None:
    """A call on line 773 is not made by a function that ends on 763."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _insert_entity(
            conn, eid="target", kind="METHOD", name="Helper::Run",
            file="op_kernel/k.h", line=900, line_end=910,
        )
        _insert_entity(
            conn, eid="near", kind="METHOD", name="SpecialS2Index",
            file="op_kernel/k.h", line=751, line_end=763,
        )
        _insert_entity(
            conn, eid="real", kind="METHOD", name="RealCaller",
            file="op_kernel/k.h", line=765, line_end=800,
        )
        _insert_rel(
            conn, rid="bad", kind="CALLS", src="near", dst="target",
            file="op_kernel/k.h", line=773,
        )
        _insert_rel(
            conn, rid="good", kind="CALLS", src="real", dst="target",
            file="op_kernel/k.h", line=773,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id IN ('bad','good')")
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="Helper::Run")
    called_by = _section(text, "**Called by**")
    assert "RealCaller" in called_by
    assert "SpecialS2Index" not in called_by


def test_possible_callers_excludes_things_that_cannot_call(tmp_path: Path) -> None:
    """A queue member offered as a candidate caller is a wrong answer."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _insert_entity(
            conn, eid="target", kind="METHOD", name="Sink::Process",
            file="op_kernel/k.h", line=100, line_end=120,
        )
        _insert_entity(
            conn, eid="q", kind="QUEUE", name="inQueuePing",
            file="op_kernel/k.h", line=144,
        )
        _insert_entity(
            conn, eid="fn", kind="FUNCTION", name="MaybeCaller",
            file="op_kernel/k.h", line=200, line_end=220,
        )
        for rid, src in (("pq", "q"), ("pf", "fn")):
            _insert_rel(
                conn, rid=rid, kind="CALLS", src=src, dst="target",
                file="op_kernel/k.h", line=205,
            )
        conn.execute("UPDATE relation SET status='partial' WHERE id IN ('pq','pf')")
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="Sink::Process")
    assert "MaybeCaller" in text
    assert "inQueuePing" not in text.split("**Possible callers", 1)[-1]


def test_two_call_sites_keep_their_own_condition(tmp_path: Path) -> None:
    """One name called in both halves of a branch has two answers, not one."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _insert_entity(
            conn, eid="target", kind="METHOD", name="Base::Compute",
            file="op_kernel/k.h", line=300, line_end=320,
        )
        _insert_entity(
            conn, eid="caller", kind="METHOD", name="Base::Run",
            file="op_kernel/k.h", line=50, line_end=200,
        )
        conn.execute(
            "INSERT INTO relation(id, kind, src, dst, status, confidence, data) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                "c1", "CALLS", "caller", "target", "confirmed", 1.0,
                '{"file": "op_kernel/k.h", "line": 61, "sites": ['
                '{"file": "op_kernel/k.h", "line": 61, "guard": "IS_DROP"},'
                '{"file": "op_kernel/k.h", "line": 125, "guard": "!IS_DROP"}]}',
            ),
        )
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="Base::Compute")
    body = _section(text, "**Called by**")
    assert "when IS_DROP" in body
    assert "when !IS_DROP" in body
    # Both polarities on one line would mean the card could not say which site
    # is which, which is the failure this guards against.
    for line in body.splitlines():
        assert not ("when IS_DROP" in line and "!IS_DROP" in line)


def test_caller_in_another_file_is_qualified(tmp_path: Path) -> None:
    """`@773` reads as the card's own file, so a foreign caller must name its."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _insert_entity(
            conn, eid="target", kind="METHOD", name="Base::Compute",
            file="op_kernel/base.h", line=300, line_end=320,
        )
        _insert_entity(
            conn, eid="caller", kind="METHOD", name="Entry::Run",
            file="op_kernel/entry.h", line=50, line_end=200,
        )
        _insert_rel(
            conn, rid="c1", kind="CALLS", src="caller", dst="target",
            file="op_kernel/entry.h", line=61,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='c1'")
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="Base::Compute")
    assert "entry.h:61" in _section(text, "**Called by**")


def test_call_lines_are_not_definition_sites(tmp_path: Path) -> None:
    """An API the tree only calls has no definitions to compare."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _add_source_lines(
            conn,
            [("op_kernel/k.h", n, "    CrossCoreSetFlag<0, PIPE_MTE3>(id0_);")
             for n in (10, 20, 30)],
        )
        for i, line in enumerate((10, 20, 30), start=1):
            _insert_entity(
                conn, eid=f"op{i}", kind="OPERATION", name="CrossCoreSetFlag",
                file="op_kernel/k.h", line=line,
            )
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="CrossCoreSetFlag")
    assert "definition sites" not in text


def test_alias_target_is_not_a_second_definition(tmp_path: Path) -> None:
    """`using T = conditional_t<…, A, B>` mentions A, it does not declare it."""
    op = tmp_path / "toy_op"
    op.mkdir()
    conn = sqlite3.connect(str(write_uo_fixture(op)))
    try:
        _insert_entity(
            conn, eid="decl", kind="TYPE", name="MutexBuffersPolicyDB",
            file="op_kernel/policy.h", line=40, line_end=90,
            data='{"cpp_kind": "class"}',
        )
        _insert_entity(
            conn, eid="ref", kind="TYPE", name="MutexBuffersPolicyDB",
            file="op_kernel/block_cube.h", line=106,
            data='{"reference_only": true, "role": "source_type"}',
        )
        conn.commit()
    finally:
        conn.close()
    text = _resolve(op, symbol="MutexBuffersPolicyDB")
    assert "2 definition sites" not in text


def test_parameter_shadows_same_named_tiling_field() -> None:
    """A branch on a parameter is not a branch on the field it collides with."""
    head = (
        "__aicore__ inline void CalTNDDenseIndex(uint32_t deterMaxRound, "
        "const T &in, int64_t stride = 0) {"
    )
    names = _param_names(head, "CalTNDDenseIndex")
    assert "deterMaxRound" in names
    assert "stride" in names
    assert "in" in names
