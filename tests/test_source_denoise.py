# -*- coding: utf-8 -*-
"""Source-card identity / callers / fields, and dim catalog discoverability."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.query.explore import _render_unit_fields
from ascendc_codemap_mcp.engine.query.typed import InvalidQuery, validate_plan
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel

_HOST = "op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp"
_NORMAL = "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp"


def _text(payload: dict) -> str:
    return str((payload.get("data") or {}).get("text") or "")


def test_source_line_end_keeps_enclosing_function_title(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (_HOST, 1657, "void SetSplitAxis(const gert::TilingContext *context)"),
                (_HOST, 1658, "{"),
                (_HOST, 1661, "    constexpr int cap = BN2_MAX_D;"),
                (_HOST, 1719, "}"),
                (_HOST, 1740, "// next function"),
            ],
        )
        _insert_entity(
            conn,
            eid="fn_split",
            kind="FUNCTION",
            name="SetSplitAxis",
            file=_HOST,
            line=1657,
            line_end=1719,
        )
        _insert_entity(
            conn,
            eid="c_bn2",
            kind="CONTRACT",
            name="BN2_MAX_D",
            file=_HOST,
            line=1657,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="source",
        file=_HOST,
        line=1657,
        line_end=1740,
    )
    text = _text(payload)
    assert text.splitlines()[0].strip() == "SetSplitAxis"
    assert "not computed for" not in text
    site = text.split("At this site", 1)[-1] if "At this site" in text else ""
    assert "BN2_MAX_D" not in site
    assert "CONTRACT" not in site


def test_source_window_majority_retitles_when_start_is_previous_function(
    tmp_path: Path,
) -> None:
    """A search-style range that starts in the previous function is about the next one."""
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (_HOST, 1614, "bool SetSparseParams() {"),
                (_HOST, 1650, "    return true;"),
                (_HOST, 1655, "}"),
                (_HOST, 1657, "void SetSplitAxis() {"),
                (_HOST, 1673, "    fBaseParams.isBn2MultiBlk = true;"),
                (_HOST, 1710, "}"),
            ],
        )
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="FUNCTION",
            name="SetSparseParams",
            file=_HOST,
            line=1614,
            line_end=1655,
        )
        _insert_entity(
            conn,
            eid="fn_split",
            kind="FUNCTION",
            name="SetSplitAxis",
            file=_HOST,
            line=1657,
            line_end=1710,
        )
        _insert_entity(
            conn,
            eid="fn_tiling",
            kind="FUNCTION",
            name="DoOpTiling",
            file=_NORMAL,
            line=800,
            line_end=830,
        )
        _insert_rel(
            conn,
            rid="call_split",
            kind="CALLS",
            src="fn_tiling",
            dst="fn_split",
            file=_NORMAL,
            line=817,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="source",
        file=_HOST,
        line=1650,
        line_end=1710,
    )
    text = _text(payload)
    assert text.splitlines()[0].strip() == "SetSplitAxis"
    assert "DoOpTiling" in text
    assert "ProcessOptionalInput" not in text


def test_source_callers_are_the_enclosing_function(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (_NORMAL, 800, "ge::graphStatus DoOpTiling() {"),
                (_NORMAL, 819, "    DoSparse();"),
                (_NORMAL, 830, "}"),
                (_NORMAL, 1077, "ge::graphStatus DoSparse() {"),
                (_NORMAL, 1078, "    if (!(DoBn2s2Sparse() && blockOuter >= aicNum)) {"),
                (_NORMAL, 1149, "}"),
            ],
        )
        _insert_entity(
            conn,
            eid="fn_tiling",
            kind="FUNCTION",
            name="DoOpTiling",
            file=_NORMAL,
            line=800,
            line_end=830,
        )
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="FUNCTION",
            name="DoSparse",
            file=_NORMAL,
            line=1077,
            line_end=1149,
        )
        _insert_entity(
            conn,
            eid="br_sparse",
            kind="BRANCH",
            name="!(DoBn2s2Sparse()&&blockOuter>=aicNum)",
            file=_NORMAL,
            line=1077,
        )
        _insert_rel(
            conn,
            rid="call_sparse",
            kind="CALLS",
            src="fn_tiling",
            dst="fn_sparse",
            file=_NORMAL,
            line=819,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="source",
        file=_NORMAL,
        line=1077,
        line_end=1149,
    )
    text = _text(payload)
    assert text.splitlines()[0].strip() == "DoSparse"
    assert "Called by" in text
    assert "DoOpTiling" in text
    assert "819" in text
    assert "not computed for" not in text
    assert "not computed for BN2_MAX_D" not in text


def test_host_source_fields_do_not_expand_kernel_consumers() -> None:
    lines = _render_unit_fields(
        [
            {
                "name": "isBn2MultiBlk",
                "bundle": {
                    "host_value_definitions": [
                        {
                            "file": _HOST,
                            "line": 1673,
                            "rhs": "true",
                            "function": "SetSplitAxis",
                        }
                    ],
                    "kernel_consumers": [{"name": "Init", "line": 40, "file": "op_kernel/k.h"}],
                },
            }
        ],
        window_start=1657,
        window_end=1740,
        file=_HOST,
    )
    text = "\n".join(lines)
    assert "isBn2MultiBlk" in text
    assert "Init" not in text
    assert "kernel" not in text.lower()


def test_trace_dim_catalog_and_unknown_name(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i, name in enumerate(("IsBn2MultiBlk", "IsRope", "DTemplateNum"), start=2):
            _insert_entity(
                conn,
                eid=f"dim{i}",
                kind="TILING_KEY",
                name=name,
                file="op_host/tiling.cpp",
                line=10 + i,
                data=json.dumps({"source_declared": True}),
            )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (1, 'k1', '', '', 'ok')"
        )
        conn.execute("INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'IsBn2MultiBlk', '1')")
        conn.execute("INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'IsRope', '0')")
        conn.execute("INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'DTemplateNum', '192')")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    catalog = query(project=str(op), architecture="arch35", operation="trace", dim="*")
    catalog_text = _text(catalog)
    assert "IsBn2MultiBlk" in catalog_text
    assert "IsRope" in catalog_text
    assert "DTemplateNum" in catalog_text

    miss = query(
        project=str(op),
        architecture="arch35",
        operation="trace",
        dim="DeterBandScheduleMode",
    )
    miss_text = _text(miss)
    assert "not a compiled dim" in miss_text.lower()
    assert "IsBn2MultiBlk" in miss_text
    assert "Empty query" not in miss_text
    assert "entity_id" not in miss_text
    assert "from_symbol" not in miss_text


def test_empty_trace_error_hides_internal_filters(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    payload = query(project=str(op), architecture="arch35", operation="trace")
    assert payload.get("error_code") == "INVALID_QUERY"
    filters = payload.get("legal_filters") or (payload.get("data") or {}).get("legal_filters") or []
    assert "entity_id" not in filters
    assert "from_symbol" not in filters
    with pytest.raises(InvalidQuery) as caught:
        validate_plan(operation="trace")
    assert "entity_id" not in caught.value.legal_filters
    assert "from_symbol" not in caught.value.legal_filters
