# -*- coding: utf-8 -*-
"""Low-noise typed search: one fact once, ranked candidates, no NL."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture
from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query


def _insert(
    conn: sqlite3.Connection,
    *,
    eid: str,
    kind: str,
    name: str,
    file: str,
    line: int,
    line_end: int | None = None,
    data: str = "{}",
    snippet: str = "",
) -> None:
    end = line if line_end is None else line_end
    conn.execute(
        "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (eid, kind, name, "extracted", 1.0, file, line, end, data),
    )
    conn.execute(
        "INSERT INTO source_span(id, entity_id, file, line_start, line_end, snippet) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"span:{eid}", eid, file, line, end, snippet or name),
    )


def test_find_render_is_candidates_not_flow_or_impact() -> None:
    text = render_explore_markdown(
        {
            "operation": "find",
            "shape": "find",
            "projection": "locations",
            "completeness": "COMPLETE",
            "total": 105,
            "count": 2,
            "truncated": True,
            "cards": [
                {
                    "name": "MutexBuffersPolicySingleBuffer",
                    "kind": "TYPE",
                    "file": "op_kernel/mutex_buffers_policy.h",
                    "line": 41,
                },
                {
                    "name": "Buffer",
                    "kind": "TYPE",
                    "file": "../common/op_kernel/attn_buffer.h",
                    "line": 166,
                },
            ],
        }
    )
    assert "**Flow**" not in text
    assert "**Impact**" not in text
    assert "**Source**" not in text
    assert "**Used at**" not in text
    assert "Matches:" in text
    assert "105" in text
    assert "MutexBuffersPolicySingleBuffer" in text
    assert "Groups:" in text


def test_resolve_render_skips_unrelated_type_neighbors() -> None:
    text = render_explore_markdown(
        {
            "operation": "resolve",
            "shape": "name",
            "completeness": "COMPLETE",
            "cards": [
                {
                    "name": "kL1Buffer",
                    "kind": "BUFFER",
                    "file": "op_kernel/block_cube.h",
                    "line": 640,
                    "snippet": "640:      MutexBuffer<BufferType::L1> kL1Buffer;",
                }
            ],
            "contract": {
                "seed": {
                    "name": "kL1Buffer",
                    "kind": "BUFFER",
                    "file": "op_kernel/block_cube.h",
                    "line": 640,
                },
                "producers": [],
                "consumers": [
                    {
                        "name": "TensorType",
                        "kind": "TYPE",
                        "file": "../common/op_kernel/attn_buffer.h",
                        "line": 167,
                    },
                    {
                        "name": "PosType",
                        "kind": "TYPE",
                        "file": "op_kernel/common.h",
                        "line": 235,
                    },
                ],
            },
        }
    )
    assert "PosType" not in text
    assert "TensorType" not in text
    assert "**Impact**" not in text
    assert "kL1Buffer" in text
    assert "block_cube.h" in text


def test_contract_render_collapses_tpl_and_does_not_repeat_spans() -> None:
    text = render_explore_markdown(
        {
            "operation": "contract",
            "shape": "name",
            "completeness": "COMPLETE",
            "cards": [
                {
                    "name": "IsFoo",
                    "kind": "TILING_KEY",
                    "file": "op_kernel/template_tiling_key.h",
                    "line": 106,
                    "snippet": "106:                        ASCENDC_TPL_BOOL_DECL(IsFoo, 0, 1),",
                }
            ],
            "contract": {
                "seed": {
                    "name": "IsFoo",
                    "kind": "TILING_KEY",
                    "file": "op_kernel/template_tiling_key.h",
                    "line": 106,
                },
                "producers": [
                    {
                        "name": "dNoEqual",
                        "file": "op_host/tiling.cpp",
                        "line": 1870,
                        "snippet": "1870:      auto dNoEqual = (d1 != d) || hasRope;",
                    }
                ],
                "consumers": [
                    {
                        "name": "ASCENDC_TPL_ARGS_DECL",
                        "file": "op_kernel/template_tiling_key.h",
                        "line": 49,
                    },
                    {
                        "name": "ASCENDC_TPL_BOOL_DECL",
                        "file": "op_kernel/template_tiling_key.h",
                        "line": 51,
                    },
                    {
                        "name": "ASCENDC_TPL_SEL",
                        "file": "op_kernel/template_tiling_key.h",
                        "line": 127,
                    },
                    {
                        "name": "IsFoo",
                        "file": "op_kernel/apt.cpp",
                        "line": 35,
                    },
                    {
                        "name": "IsFoo",
                        "file": "op_kernel/block_vec.h",
                        "line": 726,
                    },
                ],
            },
            "used_at": [
                {
                    "file": "op_kernel/template_tiling_key.h",
                    "line": 106,
                    "snippet": "106:                        ASCENDC_TPL_BOOL_DECL(IsFoo, 0, 1),",
                },
                {
                    "file": "op_kernel/template_tiling_key.h",
                    "line": 106,
                    "snippet": "106:                        ASCENDC_TPL_BOOL_DECL(IsFoo, 0, 1),",
                },
            ],
        }
    )
    assert "ASCENDC_TPL_ARGS_DECL" not in text
    assert "ASCENDC_TPL_SEL" not in text
    assert "**Impact**" not in text
    assert "**Used at**" not in text
    assert "tiling.cpp" in text
    assert "apt.cpp" in text or "block_vec.h" in text
    # The tiling-key declaration is one evidence span, not two Used-at copies.
    assert text.count("ASCENDC_TPL_BOOL_DECL(IsFoo") <= 1


def test_glob_ranks_operator_local_above_common_generic(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert(
            conn,
            eid="gen",
            kind="TYPE",
            name="Buffer",
            file="../common/op_kernel/attn_buffer.h",
            line=166,
        )
        _insert(
            conn,
            eid="genm",
            kind="METHOD",
            name="Buffer",
            file="../common/op_kernel/attn_buffer.h",
            line=166,
        )
        _insert(
            conn,
            eid="loc",
            kind="TYPE",
            name="FooBuffersPolicy",
            file="op_kernel/cube_api/mutex_buffers_policy.h",
            line=41,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        name="*Buffer*",
        limit=8,
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    names = [str(c.get("name") or "") for c in (data.get("cards") or []) if isinstance(c, dict)]
    assert "FooBuffersPolicy" in names
    assert names.index("FooBuffersPolicy") < names.index("Buffer")
    assert payload.get("verdict") == "ANSWERED"
    assert data.get("completeness") == "COMPLETE"
    # TYPE + METHOD at the same span collapse to one candidate.
    buffer_rows = [n for n in names if n == "Buffer"]
    assert len(buffer_rows) <= 1
    assert "Matches:" in text


def test_same_span_variable_and_contract_are_not_ambiguous(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert(
            conn,
            eid="v1",
            kind="VARIABLE",
            name="dNoEqual",
            file="op_host/tiling.cpp",
            line=1870,
            snippet="auto dNoEqual = (d1 != d);",
        )
        _insert(
            conn,
            eid="c1",
            kind="CONTRACT",
            name="dNoEqual",
            file="op_host/tiling.cpp",
            line=1870,
            snippet="auto dNoEqual = (d1 != d);",
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="dNoEqual",
    )
    data = payload.get("data") or {}
    assert data.get("completeness") != "AMBIGUOUS"
    assert data.get("unresolved_reason") != "MULTIPLE_SEEDS"


def test_truncated_find_is_answered_not_partial_complete(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i in range(12):
            _insert(
                conn,
                eid=f"n{i}",
                kind="TYPE",
                name=f"FooBuf{i}",
                file=f"op_kernel/f{i}.h",
                line=10,
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        name="*FooBuf*",
        limit=4,
    )
    data = payload.get("data") or {}
    assert payload.get("verdict") == "ANSWERED"
    assert data.get("completeness") == "COMPLETE"
    text = str(data.get("text") or "")
    assert "Matches:" in text
    assert "12" in text
