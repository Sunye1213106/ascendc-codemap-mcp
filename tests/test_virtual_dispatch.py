# -*- coding: utf-8 -*-
"""Host virtual call families: empty base vs override (Q4 GetSparseUnpadBlockInfo)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.clang_walk import BaseDecl
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.host_virtual_dispatch import enrich_host_virtual_dispatch
from ascendc_codemap_mcp.engine.query.virtual_dispatch import (
    build_family,
    classify_decl,
    render_virtual_dispatch,
)
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import fag_operator_root, write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel

_NORMAL_H = "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.h"
_NORMAL_CPP = "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp"
_VARLEN = "op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp"


def _text(payload: dict) -> str:
    return str((payload.get("data") or {}).get("text") or "")


def test_empty_virtual_decl_is_the_q4_base() -> None:
    flags = classify_decl(
        "virtual ge::graphStatus GetSparseUnpadBlockInfo() {};",
        line_start=95,
        line_end=95,
    )
    assert flags["virtual"] is True
    assert flags["empty"] is True
    assert flags["has_body"] is False


def test_override_body_is_not_the_base() -> None:
    flags = classify_decl(
        "ge::graphStatus GetSparseUnpadBlockInfo() override",
        line_start=953,
        line_end=1044,
    )
    assert flags["override"] is True
    assert flags["has_body"] is True
    assert flags["empty"] is False


def test_same_class_decl_and_def_is_not_a_family() -> None:
    family = build_family(
        [
            {
                "id": "m",
                "kind": "METHOD",
                "name": "Normal::Foo",
                "file": "op_host/normal.h",
                "line": 10,
                "line_end": 10,
                "attrs": {"owner": "Normal"},
                "text": "void Foo();",
            },
            {
                "id": "f",
                "kind": "FUNCTION",
                "name": "Foo",
                "file": "op_host/normal.cpp",
                "line": 40,
                "line_end": 80,
                "attrs": {"owner": "Normal"},
                "text": "void Normal::Foo()",
            },
        ]
    )
    assert family is None


def test_empty_virtual_plus_override_is_a_family() -> None:
    family = build_family(
        [
            {
                "id": "base",
                "kind": "METHOD",
                "name": "FlashAttentionScoreGradTilingNormalRegbase::GetSparseUnpadBlockInfo",
                "file": _NORMAL_H,
                "line": 95,
                "line_end": 95,
                "attrs": {"owner": "FlashAttentionScoreGradTilingNormalRegbase"},
                "text": "virtual ge::graphStatus GetSparseUnpadBlockInfo() {};",
            },
            {
                "id": "over",
                "kind": "FUNCTION",
                "name": "GetSparseUnpadBlockInfo",
                "file": _VARLEN,
                "line": 953,
                "line_end": 1044,
                "attrs": {},
                "text": "ge::graphStatus GetSparseUnpadBlockInfo() override",
            },
        ],
        owners_by_file={_VARLEN: "FlashAttentionScoreGradTilingVarlenRegbase"},
    )
    assert family is not None
    assert family["leaf"] == "GetSparseUnpadBlockInfo"
    assert family["base"][0]["empty"] is True
    assert family["overrides"][0]["line"] == 953
    text = "\n".join(render_virtual_dispatch(family, heading=True))
    assert "empty virtual FlashAttentionScoreGradTilingNormalRegbase" in text
    assert "override FlashAttentionScoreGradTilingVarlenRegbase" in text
    assert "953" in text
    assert "95" in text


def _seed_q4_fixture(op: Path) -> None:
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (_NORMAL_H, 46, "class FlashAttentionScoreGradTilingNormalRegbase {"),
                (_NORMAL_H, 95, "    virtual ge::graphStatus GetSparseUnpadBlockInfo() {};"),
                (_NORMAL_CPP, 819, "    DoSparse();"),
                (_NORMAL_CPP, 1077, "ge::graphStatus DoSparse() {"),
                (_NORMAL_CPP, 1110, "        GetSparseUnpadBlockInfo();"),
                (_NORMAL_CPP, 1149, "}"),
                (_VARLEN, 22, "class FlashAttentionScoreGradTilingVarlenRegbase : public FlashAttentionScoreGradTilingNormalRegbase {"),
                (_VARLEN, 953, "    ge::graphStatus GetSparseUnpadBlockInfo() override"),
                (_VARLEN, 954, "    {"),
                (_VARLEN, 1044, "    }"),
            ],
        )
        _insert_entity(
            conn,
            eid="ty_normal",
            kind="TYPE",
            name="FlashAttentionScoreGradTilingNormalRegbase",
            file=_NORMAL_H,
            line=46,
            data=json.dumps({"cpp_kind": "class"}),
        )
        _insert_entity(
            conn,
            eid="ty_varlen",
            kind="TYPE",
            name="FlashAttentionScoreGradTilingVarlenRegbase",
            file=_VARLEN,
            line=22,
            data=json.dumps({"cpp_kind": "class"}),
        )
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="FUNCTION",
            name="DoSparse",
            file=_NORMAL_CPP,
            line=1077,
            line_end=1149,
        )
        _insert_entity(
            conn,
            eid="fn_unpad",
            kind="FUNCTION",
            name="GetSparseUnpadBlockInfo",
            file=_VARLEN,
            line=953,
            line_end=1044,
        )
        _insert_entity(
            conn,
            eid="m_unpad",
            kind="METHOD",
            name="FlashAttentionScoreGradTilingNormalRegbase::GetSparseUnpadBlockInfo",
            file=_NORMAL_H,
            line=95,
            line_end=95,
            data=json.dumps(
                {
                    "owner": "FlashAttentionScoreGradTilingNormalRegbase",
                    "provenance": "source_runtime_method",
                }
            ),
        )
        _insert_rel(
            conn,
            rid="c_sparse_unpad",
            kind="CALLS",
            src="fn_sparse",
            dst="fn_unpad",
            file=_NORMAL_CPP,
            line=1110,
        )
        conn.commit()
    finally:
        conn.close()


def test_trace_dosparse_names_override_and_empty_base(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _seed_q4_fixture(op)
    status(project=str(op), architecture="arch35")
    text = _text(
        query(project=str(op), architecture="arch35", operation="trace", symbol="DoSparse")
    )
    assert "GetSparseUnpadBlockInfo" in text
    assert "virtual" in text
    assert "override" in text
    assert "FlashAttentionScoreGradTilingVarlenRegbase" in text
    assert "953" in text
    assert "empty virtual" in text
    assert "FlashAttentionScoreGradTilingNormalRegbase" in text
    assert "95" in text


def test_trace_unpad_states_the_empty_virtual(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _seed_q4_fixture(op)
    status(project=str(op), architecture="arch35")
    text = _text(
        query(
            project=str(op),
            architecture="arch35",
            operation="trace",
            symbol="GetSparseUnpadBlockInfo",
        )
    )
    assert "Virtual dispatch" in text
    assert "empty virtual" in text
    assert "normal_regbase.h:95" in text or "95" in text
    assert "override" in text
    assert "DoSparse" in text


def test_source_dosparse_calls_show_virtual_dispatch(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _seed_q4_fixture(op)
    status(project=str(op), architecture="arch35")
    text = _text(
        query(
            project=str(op),
            architecture="arch35",
            operation="source",
            file=_NORMAL_CPP,
            line=1077,
            line_end=1149,
        )
    )
    assert text.splitlines()[0].strip() == "DoSparse"
    assert "GetSparseUnpadBlockInfo" in text
    assert "virtual" in text
    assert "override" in text
    assert "empty virtual" in text


def test_extract_pass_annotates_calls_and_inheritance(tmp_path: Path) -> None:
    root = tmp_path / "toy_op"
    host = root / "op_host" / "arch35"
    host.mkdir(parents=True)
    header = host / "normal_regbase.h"
    impl = host / "varlen_regbase.cpp"
    header.write_text(
        "class FlashAttentionScoreGradTilingNormalRegbase {\n"
        "    virtual ge::graphStatus GetSparseUnpadBlockInfo() {};\n"
        "};\n",
        encoding="utf-8",
    )
    impl.write_text(
        "class FlashAttentionScoreGradTilingVarlenRegbase "
        ": public FlashAttentionScoreGradTilingNormalRegbase {\n"
        "    ge::graphStatus GetSparseUnpadBlockInfo() override { return 0; }\n"
        "};\n",
        encoding="utf-8",
    )
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    normal = cm.upsert(
        EntityKind.TYPE,
        "FlashAttentionScoreGradTilingNormalRegbase",
        file="op_host/arch35/normal_regbase.h",
        line=1,
        attrs={"cpp_kind": "class", "layer": "host"},
    )
    varlen = cm.upsert(
        EntityKind.TYPE,
        "FlashAttentionScoreGradTilingVarlenRegbase",
        file="op_host/arch35/varlen_regbase.cpp",
        line=1,
        attrs={"cpp_kind": "class", "layer": "host"},
    )
    caller = cm.upsert(
        EntityKind.FUNCTION,
        "DoSparse",
        file="op_host/arch35/normal_regbase.cpp",
        line=10,
        line_end=20,
        attrs={"layer": "host"},
    )
    override = cm.upsert(
        EntityKind.FUNCTION,
        "GetSparseUnpadBlockInfo",
        file="op_host/arch35/varlen_regbase.cpp",
        line=2,
        line_end=2,
        attrs={"layer": "host"},
    )
    cm.upsert(
        EntityKind.METHOD,
        "FlashAttentionScoreGradTilingNormalRegbase::GetSparseUnpadBlockInfo",
        file="op_host/arch35/normal_regbase.h",
        line=2,
        line_end=2,
        attrs={
            "layer": "host",
            "owner": "FlashAttentionScoreGradTilingNormalRegbase",
            "provenance": "source_runtime_method",
        },
    )
    rel = cm.link(RelationKind.CALLS, caller.id, override.id, attrs={"file": "op_host/arch35/normal_regbase.cpp", "line": 15})

    class _Host:
        base_decls = [
            BaseDecl(
                derived_name="FlashAttentionScoreGradTilingVarlenRegbase",
                base_name="FlashAttentionScoreGradTilingNormalRegbase",
                file="op_host/arch35/varlen_regbase.cpp",
                line=1,
            )
        ]

    enrich_host_virtual_dispatch(cm, root, architecture="arch35", host_ir=_Host())
    assert rel.attrs.get("virtual") is True
    family = rel.attrs.get("virtual_dispatch") or {}
    assert family.get("leaf") == "GetSparseUnpadBlockInfo"
    assert any(row.get("empty") for row in family.get("base") or [])
    specs = [
        r
        for r in cm.relations.values()
        if r.kind_name() == RelationKind.SPECIALIZES.value
    ]
    assert any(r.src == varlen.id and r.dst == normal.id for r in specs)


@pytest.mark.skipif(fag_operator_root() is None, reason="FAG arch35 .uo missing")
def test_fag_dosparse_virtual_call_names_varlen_override() -> None:
    """ses_fa50 Q4: the call graph used to stop at a collapsed override."""
    fag = fag_operator_root()
    assert fag is not None
    status(project=str(fag), architecture="arch35")
    text = _text(
        query(
            project=str(fag),
            architecture="arch35",
            operation="trace",
            symbol="DoSparse",
        )
    )
    assert "GetSparseUnpadBlockInfo" in text
    assert "virtual" in text
    assert "override" in text
    assert "FlashAttentionScoreGradTilingVarlenRegbase" in text
    assert "953" in text
    assert "empty virtual" in text
    assert "95" in text


@pytest.mark.skipif(fag_operator_root() is None, reason="FAG arch35 .uo missing")
def test_fag_unpad_card_names_empty_virtual_base() -> None:
    fag = fag_operator_root()
    assert fag is not None
    status(project=str(fag), architecture="arch35")
    text = _text(
        query(
            project=str(fag),
            architecture="arch35",
            operation="trace",
            symbol="GetSparseUnpadBlockInfo",
        )
    )
    assert "empty virtual" in text
    assert "FlashAttentionScoreGradTilingNormalRegbase" in text
    assert "override" in text
    assert "953" in text
