# -*- coding: utf-8 -*-
from __future__ import annotations

from ascendc_codemap_mcp.engine.clang_walk import CallSite, PathCond
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.passes.guarded_calls import ingest_guarded_calls
from ascendc_codemap_mcp.engine.query.predicate_ast import parse_predicate


def test_layout_and_zero() -> None:
    ast = parse_predicate("layout == TND && s == 0")
    assert ast.op == "AND"
    assert ast.args[0].op == "EQ"
    assert "0" in ast.literals()
    assert "TND" in ast.enum_values()


def test_core_num_div() -> None:
    ast = parse_predicate("coreNum / 2")
    assert ast.op == "DIV"
    assert ast.args[0].op == "REF"
    assert ast.args[0].value == "coreNum"
    assert ast.args[1].op == "INT"
    assert ast.args[1].value == "2"


def test_strcmp_tnd_zero() -> None:
    ast = parse_predicate('strcmp(inputLayout, "TND")== 0')
    assert ast.op == "EQ"
    assert "0" in ast.literals()
    assert "TND" in ast.literals()
    assert "TND" in ast.enum_values()
    from ascendc_codemap_mcp.engine.query.predicate_ast import ast_matches_literal, ast_matches_value

    attrs = {"literals": ast.literals(), "enum_values": ast.enum_values()}
    assert ast_matches_literal(attrs, "0")
    assert ast_matches_value(attrs, "TND")


def test_guarded_call_stamps_ast() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    site = CallSite(
        caller="DoTiling",
        callee="PreloadCube",
        file="op_host/tiling.cpp",
        line=40,
        path_conditions=(
            PathCond(
                text="layout == TND && s == 0",
                negated=False,
                file="op_host/tiling.cpp",
                line=38,
                kind="if",
            ),
        ),
    )
    ingest_guarded_calls(cm, [site], side="host", architecture="arch35")
    branches = list(cm.by_kind(EntityKind.BRANCH))
    assert branches
    ops = branches[0].attrs.get("operators") or []
    assert "AND" in ops or "EQ" in ops
    lits = branches[0].attrs.get("literals") or []
    assert "0" in lits
