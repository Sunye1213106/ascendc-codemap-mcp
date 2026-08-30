# -*- coding: utf-8 -*-
"""resolve(file,line) keeps a stable semantic unit; facets stay conservative."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel

_HOST = "op_host/arch35/common_regbase.cpp"
_KERNEL = "op_kernel/arch35/kernel_deter.h"
_HEADER = "../common/op_host/fia_tiling_templates_registry.h"
_OTHER = "op_kernel/arch35/other.cpp"


def _data(payload: dict) -> dict:
    return payload.get("data") or {}


def _text(payload: dict) -> str:
    return str(_data(payload).get("text") or "")


def _resolve(op: Path, file: str, line: int, **kwargs) -> dict:
    return query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file=file,
        line=line,
        **kwargs,
    )


def _span_of(payload: dict) -> tuple[int, int]:
    data = _data(payload)
    start = int(data.get("unit_start") or 0)
    end = int(data.get("unit_end") or 0)
    return start, end


def test_resolve_140_line_function_is_stable_across_interior_lines(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    start, end = 598, 737
    rows = [(_KERNEL, start, "int64_t CalDeterMaxLoopNum()")]
    rows.append((_KERNEL, start + 1, "{"))
    for n in range(start + 2, end):
        rows.append((_KERNEL, n, f"    body_{n};"))
    rows.append((_KERNEL, end, "}"))
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        _insert_entity(
            conn,
            eid="fn_cal",
            kind="METHOD",
            name="CalDeterMaxLoopNum",
            file=_KERNEL,
            line=start,
            line_end=end,
        )
        _insert_entity(
            conn,
            eid="fld",
            kind="TILING_FIELD",
            name="deterLoopMax",
            file=_KERNEL,
            line=720,
        )
        _insert_rel(
            conn,
            rid="w1",
            kind="WRITES",
            src="fn_cal",
            dst="fld",
            file=_KERNEL,
            line=720,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w1'",
            (json.dumps({"file": _KERNEL, "line": 720, "rhs": "bandInfo.rm2"}),),
        )
        _insert_rel(
            conn,
            rid="c1",
            kind="CONTROLS",
            src="fld",
            dst="fn_cal",
            file=_KERNEL,
            line=720,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='c1'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    spans = []
    texts = []
    for loc in (642, 668, 680, 702, 598):
        payload = _resolve(op, _KERNEL, loc)
        spans.append(_span_of(payload))
        texts.append(_text(payload))

    assert len(set(spans)) == 1
    assert spans[0] == (start, end)
    for text in texts:
        assert "CalDeterMaxLoopNum" in text
        assert f"{start}|" in text
        assert f"{end}|" in text
        assert "Controls" not in text
        assert "Used by" not in text
        assert "Showing" not in text


def test_resolve_large_function_keeps_identity_and_aligned_tile(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    start, end = 100, 499
    rows = [(_KERNEL, start, "void BigFn()")]
    rows.append((_KERNEL, start + 1, "{"))
    for n in range(start + 2, end):
        rows.append((_KERNEL, n, f"    big_{n};"))
    rows.append((_KERNEL, end, "}"))
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        _insert_entity(
            conn,
            eid="fn_big",
            kind="FUNCTION",
            name="BigFn",
            file=_KERNEL,
            line=start,
            line_end=end,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    first = _resolve(op, _KERNEL, 250)
    second = _resolve(op, _KERNEL, 330)
    third = _resolve(op, _KERNEL, 350)
    assert _span_of(first) == _span_of(second)
    assert _span_of(first) == (100, 339)
    assert _span_of(third) == (340, 499)

    for payload in (first, second, third):
        text = _text(payload)
        data = _data(payload)
        assert "BigFn" in text
        assert f"{_KERNEL}:100-499" in text or f"{_KERNEL}:100–499" in text
        assert "Showing" in text
        assert int(data.get("function_start") or 0) == 100
        assert int(data.get("function_end") or 0) == 499


def test_resolve_without_enclosing_expands_beyond_one_line(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    rows = []
    for n in range(50, 121):
        if n == 60:
            rows.append((_HEADER, n, "    std::map<int32_t, FiaTilingClassCase> cases_;"))
        elif n == 94:
            rows.append(
                (
                    _HEADER,
                    n,
                    "    ge::graphStatus DoTilingImpl(gert::TilingContext *context, TilingInfo *tilingInfo)",
                )
            )
        elif n == 95:
            rows.append((_HEADER, n, "    {"))
        elif n == 110:
            rows.append((_HEADER, n, "    }"))
        else:
            rows.append((_HEADER, n, f"    header_body_{n};"))
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    payload = _resolve(op, _HEADER, 94)
    text = _text(payload)
    start, end = _span_of(payload)
    assert end > start
    assert end - start + 1 >= 8
    assert "DoTilingImpl" in text
    assert text.count("|") >= 8


def test_resolve_missing_snapshot_file_does_not_fake_a_one_line_read(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, [(_KERNEL, 10, "void Present() {}")])
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = _resolve(op, "op_host/missing.h", 20)
    text = _text(payload).lower()
    data = _data(payload)
    assert "not in snapshot" in text or "not in snapshot" in str(data.get("hint") or "").lower()
    start, end = _span_of(payload)
    assert not (start == end == 20 and "20|" in _text(payload) and _text(payload).count("|") == 1)


def test_state_changes_ignore_cross_file_line_guards(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    rows = [
        (_HOST, 1660, "void SetSplitAxis()"),
        (_HOST, 1661, "{"),
        (_HOST, 1667, "    bnLimit = 1;"),
        (_HOST, 1701, "    bn2S2RouteLimit = 2;"),
        (_HOST, 1710, "}"),
        (_OTHER, 1660, "void OtherFn()"),
        (_OTHER, 1661, "{"),
        (_OTHER, 1667, "    if (!(r>deterMaxRound)) { leak = 0; }"),
        (_OTHER, 1701, "    if (strcmp(inputLayout,\"SBH\")) { leak2 = 1; }"),
        (_OTHER, 1710, "}"),
    ]
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        _insert_entity(
            conn, eid="fn_a", kind="FUNCTION", name="SetSplitAxis", file=_HOST, line=1660, line_end=1710
        )
        _insert_entity(
            conn, eid="fn_b", kind="FUNCTION", name="OtherFn", file=_OTHER, line=1660, line_end=1710
        )
        _insert_entity(conn, eid="fld_bn", kind="TILING_FIELD", name="bnLimit", file=_HOST, line=1667)
        _insert_entity(
            conn, eid="fld_route", kind="TILING_FIELD", name="bn2S2RouteLimit", file=_HOST, line=1701
        )
        _insert_entity(conn, eid="fld_leak", kind="FIELD", name="leak", file=_OTHER, line=1667)
        _insert_rel(conn, rid="w_bn", kind="WRITES", src="fn_a", dst="fld_bn", file=_HOST, line=1667)
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w_bn'",
            (json.dumps({"file": _HOST, "line": 1667, "rhs": "1"}),),
        )
        _insert_rel(conn, rid="w_route", kind="WRITES", src="fn_a", dst="fld_route", file=_HOST, line=1701)
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w_route'",
            (json.dumps({"file": _HOST, "line": 1701, "rhs": "2"}),),
        )
        _insert_rel(conn, rid="w_leak", kind="WRITES", src="fn_b", dst="fld_leak", file=_OTHER, line=1667)
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w_leak'",
            (json.dumps({"file": _OTHER, "line": 1667, "rhs": "0"}),),
        )
        _insert_entity(
            conn, eid="br_real", kind="BRANCH", name="dropMaskOuter", file=_HOST, line=1667
        )
        _insert_rel(conn, rid="g_real", kind="GUARDED_BY", src="w_bn", dst="br_real", file=_HOST, line=1667)
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='g_real'")
        _insert_entity(
            conn, eid="br_leak", kind="BRANCH", name="!(r>deterMaxRound)", file=_OTHER, line=1667
        )
        _insert_rel(
            conn, rid="g_leak", kind="GUARDED_BY", src="w_leak", dst="br_leak", file=_OTHER, line=1667
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='g_leak'")
        _insert_entity(
            conn, eid="br_sbh", kind="BRANCH", name='strcmp(inputLayout,"SBH")', file=_OTHER, line=1701
        )
        _insert_rel(conn, rid="g_sbh", kind="GUARDED_BY", src="w_leak", dst="br_sbh", file=_OTHER, line=1701)
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='g_sbh'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    text = _text(_resolve(op, _HOST, 1667))
    assert "bnLimit" in text
    assert "dropMaskOuter" in text
    assert "!(r>deterMaxRound)" not in text
    assert "strcmp" not in text
    assert "SBH" not in text


def test_assignments_and_host_kernel_do_not_claim_exhaustive(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fld",
            kind="TILING_FIELD",
            name="isBn2MultiBlk",
            file=_HOST,
            line=45,
        )
        _insert_entity(
            conn, eid="fn_split", kind="FUNCTION", name="SetSplitAxis", file=_HOST, line=32, line_end=59
        )
        _insert_entity(
            conn,
            eid="kn",
            kind="METHOD",
            name="KernelRead",
            file="op_kernel/arch35/k.h",
            line=8,
        )
        _insert_rel(conn, rid="w1", kind="WRITES", src="fn_split", dst="fld", file=_HOST, line=45)
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w1'",
            (json.dumps({"file": _HOST, "line": 45, "rhs": "true"}),),
        )
        _insert_rel(
            conn, rid="rd1", kind="READS", src="kn", dst="fld", file="op_kernel/arch35/k.h", line=8
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='rd1'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    text = _text(
        query(project=str(op), architecture="arch35", operation="resolve", symbol="isBn2MultiBlk")
    )
    assert "Assignments" in text
    assert "1/1" in text
    assert "exhaustive=no" in text
    assert "Host producers" in text or "Kernel consumers" in text
    assert "exhaustive: false" in text or "exhaustive=false" in text
    assert "exhaustive=yes" not in text
    assert "exhaustive: true" not in text
