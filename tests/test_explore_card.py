# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture


def test_envelope_omits_null_next_cursor() -> None:
    from ascendc_codemap_mcp.service.envelope import envelope

    body = envelope(ok=True, verdict="ANSWERED", layer="template")
    assert "next_cursor" not in body
    assert body["ok"] is True
    assert body["verdict"] == "ANSWERED"


def test_pydantic_envelope_omits_nones() -> None:
    from ascendc_codemap_mcp.service.models import Envelope

    dumped = Envelope.model_validate({"ok": True, "verdict": "ANSWERED"}).model_dump()
    assert "next_cursor" not in dumped
    assert dumped.get("error") is None or "error" not in dumped
    assert dumped["ok"] is True


def test_render_explore_markdown_groups_source_by_file_and_caps() -> None:
    from ascendc_codemap_mcp.engine.query.explore import (
        MAX_EXPLORE_CHARS,
        render_explore_markdown,
    )

    payload = {
        "operation": "contract",
        "shape": "name",
        "completeness": "COMPLETE",
        "cards": [
            {
                "name": "IsPse",
                "kind": "TILING_KEY",
                "file": "op_kernel/key.h",
                "line": 10,
                "snippet": "10:  ASCENDC_TPL_BOOL_DECL(IsPse, 0, 1),",
            }
        ],
        "contract": {
            "producers": [
                {
                    "name": "GetTilingKey",
                    "kind": "METHOD",
                    "file": "op_host/tiling.cpp",
                    "line": 20,
                    "role": "producer",
                    "snippet": "20:    GET_TPL_TILING_KEY(..., pseValue, ...)",
                }
            ],
            "consumers": [
                {
                    "name": "flash_attention_score_grad",
                    "kind": "KERNEL",
                    "file": "op_kernel/vec.h",
                    "line": 30,
                    "role": "consumer",
                    "snippet": "30:    if constexpr (IS_PSE) {",
                }
            ],
        },
    }
    text = render_explore_markdown(payload, verdict="ANSWERED", layer="template")
    assert "**Contract**" in text
    assert "**Flow**" not in text
    assert "**Impact**" not in text
    assert "**Used at**" not in text
    assert "op_kernel/key.h" in text
    assert "op_host/tiling.cpp" in text
    assert text.count("GET_TPL_TILING_KEY") == 1
    assert "do not read" not in text.lower()
    assert len(text) < 8_000

    huge = {
        "cards": [
            {
                "name": f"D{i}",
                "file": f"op_kernel/f{i}.h",
                "line": 1,
                "snippet": f"1: {'x' * 400}",
            }
            for i in range(200)
        ]
    }
    capped = render_explore_markdown(huge)
    assert len(capped) <= MAX_EXPLORE_CHARS + 80


def test_explore_mcp_result_sends_answer_once() -> None:
    """Agent sees markdown in content; structured_content is follow-up only."""
    import json

    from ascendc_codemap_mcp.mcp_adapter import _query_result

    answer = (
        "verdict: ANSWERED  layer: template\n\n"
        "completeness: COMPLETE\n\n"
        "1. GetTilingKey (op_host/tiling.cpp:20)  writes\n"
        "IsPse (op_kernel/key.h:10)\n"
        "op_kernel/key.h\n"
        "10|  ASCENDC_TPL_BOOL_DECL(IsPse, 0, 1),\n"
    )
    payload = {
        "ok": True,
        "verdict": "ANSWERED",
        "layer": "template",
        "codemap": {
            "id": "p:x::op@arch35",
            "alias": "op@arch35",
            "snapshot_id": "cm:abc",
            "architecture": "arch35",
            "path": "D:/op/.ascendc-codemap/arch35/op.arch35.uo",
            "project": "D:/op",
        },
        "data": {
            "text": answer,
            "cards": [{"name": "IsPse", "kind": "TILING_KEY", "file": "key.h", "line": 10}],
            "contract": {
                "producers": [{"name": "GetTilingKey", "file": "tiling.cpp", "line": 20}]
            },
            "completeness": "COMPLETE",
        },
        "evidence": [{"id": "span:e1", "file": "key.h", "line": 10}],
        "coverage": {"returned": 1, "total": 1, "truncated": False, "token_budget": 24000},
        "engine": "codemap_query",
    }
    result = _query_result(payload)
    text = result.content[0].text
    sc = result.structured_content or {}
    assert text == answer
    assert "data" not in sc
    assert "evidence" not in sc
    assert "engine" not in sc
    dumped = json.dumps(sc, ensure_ascii=False)
    assert "ASCENDC_TPL" not in dumped
    assert answer not in dumped
    assert "verdict" not in sc
    assert "layer" not in sc
    assert "ok" not in sc
    assert sc.get("codemap_id") == "p:x::op@arch35"
    assert sc.get("snapshot_id") == "cm:abc"
    assert (sc.get("codemap") or {}).get("id") == "p:x::op@arch35"
    assert (sc.get("codemap") or {}).get("snapshot_id") == "cm:abc"
    assert "path" not in (sc.get("codemap") or {})


def test_explore_mcp_error_is_not_a_second_json_dump() -> None:
    from ascendc_codemap_mcp.mcp_adapter import _query_result

    payload = {
        "ok": False,
        "error": "project is required",
        "error_code": "PROJECT_REQUIRED",
        "data": {"text": "", "cards": []},
    }
    result = _query_result(payload)
    text = result.content[0].text
    sc = result.structured_content or {}
    assert "PROJECT_REQUIRED" in text
    assert "project is required" in text
    assert sc.get("ok") is False
    assert sc.get("error_code") == "PROJECT_REQUIRED"
    assert "data" not in sc
    assert text.count("PROJECT_REQUIRED") == 1


def test_explore_text_has_no_human_coaching() -> None:
    from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown

    payload = {
        "operation": "contract",
        "completeness": "COMPLETE",
        "cards": [
            {
                "name": "IsPse",
                "kind": "TILING_KEY",
                "file": "op_kernel/key.h",
                "line": 10,
                "snippet": "10:  ASCENDC_TPL_BOOL_DECL(IsPse, 0, 1),",
            }
        ],
        "contract": {
            "producers": [
                {
                    "name": "GetTilingKey",
                    "file": "op_host/tiling.cpp",
                    "line": 20,
                    "snippet": "20:    GET_TPL_TILING_KEY(..., pseValue, ...)",
                }
            ],
            "consumers": [
                {
                    "name": "flash_attention_score_grad",
                    "file": "op_kernel/vec.h",
                    "line": 30,
                    "snippet": "30:    if constexpr (IS_PSE) {",
                }
            ],
        },
    }
    text = render_explore_markdown(payload, verdict="ANSWERED", layer="template")
    low = text.lower()
    assert "do not read" not in low
    assert "verbatim" not in low
    assert "locations only" not in low
    assert "already read" not in low
    assert "op_host/tiling.cpp" in text
    assert "GET_TPL_TILING_KEY" in text
    assert "op_kernel/vec.h" in text


def test_index_markdown_lists_dim_names() -> None:
    from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown

    text = render_explore_markdown(
        {
            "shape": "index",
            "dim_names": ["IsTnd", "IsRope", "Deterministic"],
            "hint": "Dim=IsTnd lists that dim. IsTnd=0 is Name=Value.",
        },
        verdict="ANSWERED",
        layer="host",
    )
    assert "**Dims**" in text
    assert "IsTnd" in text
    assert "IsRope" in text
    assert "Deterministic" in text
    assert "none listed" not in text
    assert "Host → TilingKey → Kernel" not in text


def test_name_miss_hint_says_absent_and_lists_dims() -> None:
    from ascendc_codemap_mcp.engine.query.hints import attach_query_hints

    payload: dict = {
        "dim_names": ["IsTnd", "IsRope", "Deterministic"],
        "cards": [],
        "count": 0,
    }
    attach_query_hints(payload, "IsDrop", count=0, mode="name")
    hint = str(payload.get("hint") or "")
    assert "IsDrop" in hint
    assert "compiled dim" in hint.lower()
    assert "IsTnd" in hint
    assert "IsRope" in hint
    assert "trace dim=" in hint
    assert "not proof" not in hint.lower()


def test_kernel_api_markdown_skips_host_tiling_flow() -> None:
    from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown

    text = render_explore_markdown(
        {
            "shape": "name",
            "cards": [
                {
                    "name": "InitBuffer",
                    "kind": "OPERATION",
                    "file": "op_kernel/arch35/cube_api/mutex_buffer_manager.h",
                    "line": 29,
                    "snippet": "29:  void InitBuffer(TBufOffset offset);",
                }
            ],
        },
        verdict="PARTIAL",
        layer="kernel",
    )
    assert "Host → TilingKey → Kernel" not in text
    assert "InitBuffer" in text
    assert "mutex_buffer_manager.h" in text
    assert "none listed" not in text


def test_explore_empty_and_miss_use_declared_dims(tmp_path: Path) -> None:
    import sqlite3

    from ascendc_codemap_mcp.service.control import status
    from ascendc_codemap_mcp.service.query import query

    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "UPDATE entity SET data = ? WHERE name = ?",
            ('{"source_declared": true}', "IsPse"),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    index = query(project=str(op), architecture="arch35", operation="resolve")
    index_text = str((index.get("data") or {}).get("text") or "")
    assert "**Dims**" in index_text
    assert "IsPse" in index_text
    assert "none listed" not in index_text

    miss = query(project=str(op), architecture="arch35", symbol="IsDrop")
    miss_text = str((miss.get("data") or {}).get("text") or "")
    assert miss.get("verdict") == "UNKNOWN"
    assert "compiled dim" in miss_text.lower()
    assert "IsPse" in miss_text
    assert "not proof" not in miss_text.lower()


def test_explore_returns_markdown_without_flattening(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.service.control import status
    from ascendc_codemap_mcp.service.query import query

    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    body = query(project=str(op), architecture="arch35", symbol="IsPse")
    assert body.get("ok") is True
    assert "cards" not in body
    data = body.get("data") or {}
    text = str(data.get("text") or "")
    assert text
    assert "**Definition**" in text or "**Source**" in text or "IsPse" in text
    assert "proof" not in data
    assert len(text) < 8_000
    dumped = str(body)
    assert dumped.count("ASCENDC_TPL") <= 1 or "snippet" not in dumped.lower()


def test_check_macros_omitted_from_flow_and_impact() -> None:
    from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown

    text = render_explore_markdown(
        {
            "operation": "contract",
            "shape": "name",
            "completeness": "COMPLETE",
            "cards": [{"name": "PostTiling", "kind": "METHOD", "file": "op_host/tiling.cpp", "line": 80}],
            "contract": {
                "producers": [
                    {"name": "CheckLogLevel", "kind": "MACRO", "file": "op_host/log.h", "line": 12},
                    {"name": "GetTilingKey", "kind": "METHOD", "file": "op_host/tiling.cpp", "line": 20},
                ],
                "consumers": [
                    {"name": "OP_CHECK", "file": "op_host/check.h", "line": 4},
                    {"name": "flash_attention_score_grad", "file": "op_kernel/vec.h", "line": 30},
                ],
            },
        },
        verdict="ANSWERED",
        layer="host",
    )
    assert "CheckLogLevel" not in text
    assert "OP_CHECK" not in text
    assert "GetTilingKey" in text
    assert "vec.h" in text


def test_clip_source_is_contiguous_for_line_end(tmp_path: Path) -> None:
    import sqlite3

    from tests.conftest import write_uo_fixture
    from tests.test_query_surface import _add_source_lines
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
    from ascendc_codemap_mcp.engine.query.explore import _clip_source

    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    rows = [
        ("tiling.cpp", 1, "// comment 1933"),
        ("tiling.cpp", 2, "SetScheduleMode(Q);"),
        ("tiling.cpp", 3, "SetScheduleMode(K);"),
        ("tiling.cpp", 4, "SetScheduleMode(V);"),
        ("tiling.cpp", 5, "int unused = 0;"),
        ("tiling.cpp", 6, "return;"),
        ("tiling.cpp", 7, "// end 1939"),
    ]
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        conn.commit()
    finally:
        conn.close()
    clip = _clip_source(UoSqlQuery(dest), "tiling.cpp", 1, line_end=7)
    assert "1:" in clip
    assert "2:" in clip
    assert "7:" in clip
    assert "comment 1933" in clip
    assert "end 1939" in clip


def test_tpl_boilerplate_shown_once() -> None:
    from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown

    text = render_explore_markdown(
        {
            "cards": [
                {
                    "name": "IsPse",
                    "file": "op_kernel/key.h",
                    "line": 10,
                    "snippet": "10:  ASCENDC_TPL_ARGS_DECL(",
                },
                {
                    "name": "IsDrop",
                    "file": "op_kernel/key.h",
                    "line": 20,
                    "snippet": "20:  ASCENDC_TPL_ARGS_SEL(",
                },
            ]
        }
    )
    assert text.count("ASCENDC_TPL_ARGS") == 1


def test_arch_file_rank_prefers_op_over_common() -> None:
    from ascendc_codemap_mcp.engine.query.sql import _arch_file_rank

    assert _arch_file_rank("op_host/arch35/tiling.cpp", "arch35") < _arch_file_rank(
        "../common/op_kernel/matmul.h", "arch35"
    )
    assert _arch_file_rank("op_kernel/block_vec.h", "arch35") < _arch_file_rank(
        "common/op_kernel/matmul.h", "arch35"
    )
