# -*- coding: utf-8 -*-
"""Path-bound tests: facb-shaped calls through validate_plan + query.

Assertions bind to the public call shapes the agent actually uses, not to
internal helpers such as _recovery_tokens or build_symbol_bundle.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.query.rg import path_matches
from ascendc_codemap_mcp.engine.query.typed import InvalidQuery, validate_plan
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity

FAG = Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad")
FAG_UO = FAG / r".ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
_HOST = "op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp"


def _text(payload: dict) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return str((data or {}).get("text") or "")


def test_path_matches_globstar_zero_directories() -> None:
    assert path_matches(
        "op_host/flash_attention_score_grad_tiling.cpp", "op_host/**/*.cpp"
    )
    assert path_matches("op_host/arch35/foo.cpp", "op_host/**/*.cpp")
    assert not path_matches("op_kernel/arch35/foo.cpp", "op_host/**/*.cpp")
    assert path_matches("op_host/foo.cpp", "op_host/*.cpp")
    assert path_matches("op_kernel/arch35/foo.cpp", "**/*.cpp")
    assert path_matches("foo.cpp", "**/*.cpp")


def test_search_pattern_and_name_alias_are_public() -> None:
    plan_p = validate_plan(operation="search", pattern="DT_HIFLOAT8")
    plan_n = validate_plan(operation="search", name="DT_HIFLOAT8")
    assert plan_p.pattern == plan_n.pattern == "DT_HIFLOAT8"
    try:
        validate_plan(operation="search")
    except InvalidQuery as exc:
        assert "pattern" in str(exc)
        assert "regex over source lines" in str(exc)
        assert exc.legal_filters == ["file", "kind", "pattern"]
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_resolve_file_plus_name_does_not_return_dims(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="METHOD",
            name="GetSparseUnpadBlockInfo",
            file="op_host/arch35/tiling_normal.cpp",
            line=10,
            line_end=40,
            snippet="void GetSparseUnpadBlockInfo() {}",
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file="op_host/arch35/tiling_normal.cpp",
        name="GetSparseUnpadBlockInfo",
    )
    text = _text(payload)
    assert "**Dims**" not in text
    assert "GetSparseUnpadBlockInfo" in text
    assert payload.get("error_code") != "INVALID_QUERY"


def test_search_globstar_hits_top_level_host_file(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "op_host/flash_attention_score_grad_tiling.cpp",
                    150,
                    "emptyTensorTilingDataRegbase->set_formerDqNum(aivNum);",
                ),
                (
                    "op_host/arch35/nested.cpp",
                    4,
                    "EmptyTensor leftover;",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        pattern="EmptyTensor",
        file="op_host/**/*.cpp",
    )
    text = _text(payload)
    assert "0 matches" not in text.splitlines()[:1]
    assert "flash_attention_score_grad_tiling.cpp" in text


def test_search_recovers_camel_token_on_miss(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [("op_kernel/buffer.h", 10, "class FooBuffer {};")],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="BufferNum",
    )
    text = _text(payload)
    data = payload.get("data") or {}
    assert "FooBuffer" in text
    assert "no match for BufferNum; showing buffer" in text.lower() or (
        "showing buffer" in str(data.get("hint") or "").lower()
    )


def test_resolve_file_line_lists_fields_in_unit(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    host = "op_host/arch35/tiling_common.cpp"
    attrs = {
        "value_defining_sites": [
            {
                "file": host,
                "line": 40,
                "rhs": "true",
                "function": "SetSplitAxis",
            }
        ],
        "host_writer_sites": [
            {"file": host, "line": 40, "rhs": "true", "receiver": "fBaseParams"}
        ],
    }
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fn_split",
            kind="METHOD",
            name="SetSplitAxis",
            file=host,
            line=30,
            line_end=50,
            snippet="void SetSplitAxis() { fBaseParams.isBn2MultiBlk = true; }",
        )
        _insert_entity(
            conn,
            eid="tdf_bn2",
            kind="TILING_FIELD",
            name="isBn2MultiBlk",
            file=host,
            line=8,
            line_end=8,
            data=json.dumps(attrs),
        )
        _add_source_lines(
            conn,
            [
                (host, 30, "void SetSplitAxis() {"),
                (host, 40, "    fBaseParams.isBn2MultiBlk = true;"),
                (host, 50, "}"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file=host,
        line=40,
    )
    text = _text(payload)
    assert "**Dims**" not in text
    assert "Fields in this unit" in text
    assert "isBn2MultiBlk" in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_fag_search_hifloat8_via_pattern() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="search",
        pattern="DT_HIFLOAT8",
    )
    text = _text(payload)
    assert "214" in text
    assert "common_regbase.cpp" in text or "DT_HIFLOAT8" in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_fag_search_empty_tensor_globstar() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="search",
        pattern="EmptyTensor",
        file="op_host/**/*.cpp",
    )
    text = _text(payload)
    assert "flash_attention_score_grad_tiling.cpp" in text
    assert "0 matches" not in text.splitlines()[:1]


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_fag_resolve_site_exposes_isbn2_bundle() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        file=_HOST,
        line=1673,
    )
    text = _text(payload)
    assert "**Dims**" not in text
    assert "Fields in this unit" in text
    assert "isBn2MultiBlk" in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_fag_resolve_file_name_not_dims() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
        name="GetSparseUnpadBlockInfo",
    )
    text = _text(payload)
    assert "**Dims**" not in text
    assert "GetSparseUnpadBlockInfo" in text
