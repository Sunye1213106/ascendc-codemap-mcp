# -*- coding: utf-8 -*-
from __future__ import annotations

from ascendc_codemap_mcp.engine.clang_walk import CallSite, PathCond
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.constexpr_alias import enrich_constexpr_aliases
from ascendc_codemap_mcp.engine.passes.guarded_calls import ingest_guarded_calls
from ascendc_codemap_mcp.engine.passes.host_predicates import enrich_host_predicates
from ascendc_codemap_mcp.engine.passes.kernel_tiling_closure import _tiling_key_at_order


def test_guarded_call_mints_calls_under_guard() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    flag = cm.upsert(EntityKind.VARIABLE, "enableFeature", attrs={"layer": "host"})
    site = CallSite(
        caller="DoTiling",
        callee="PreloadCube",
        file="op_host/tiling.cpp",
        line=40,
        path_conditions=(
            PathCond(text="enableFeature", negated=False, file="op_host/tiling.cpp", line=38, kind="if"),
        ),
    )
    n = ingest_guarded_calls(cm, [site], side="host", architecture="arch35")
    assert n == 1
    kinds = {rel.kind_name() for rel in cm.relations.values()}
    assert RelationKind.CALLS_UNDER_GUARD.value in kinds
    assert RelationKind.CONTROLS.value in kinds
    guarded = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.CALLS_UNDER_GUARD.value
    ]
    assert guarded
    branches = [e for e in cm.by_kind(EntityKind.BRANCH)]
    assert branches
    reads = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.READS.value and rel.src == branches[0].id
    ]
    assert any(rel.dst == flag.id for rel in reads)
    callees = cm.by_name("PreloadCube")
    assert callees
    assert any(
        rel.dst == callees[0].id and rel.src == branches[0].id
        for rel in guarded
    )


def test_unguarded_call_does_not_mint() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    site = CallSite(
        caller="DoTiling",
        callee="AlwaysRun",
        file="op_host/tiling.cpp",
        line=10,
        path_conditions=(),
    )
    assert ingest_guarded_calls(cm, [site], side="host", architecture="arch35") == 0
    assert not any(
        rel.kind_name() == RelationKind.CALLS_UNDER_GUARD.value for rel in cm.relations.values()
    )


def test_bailout_is_not_a_per_call_guard() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    site = CallSite(
        caller="DoTiling",
        callee="Encode",
        file="op_host/tiling.cpp",
        line=80,
        path_conditions=(
            PathCond(
                text="dtype == DT_HIFLOAT8",
                negated=True,
                file="op_host/tiling.cpp",
                line=20,
                kind="bailout",
            ),
        ),
    )
    assert ingest_guarded_calls(cm, [site], side="host", architecture="arch35") == 0


def test_constexpr_alias_materializes_template_arg(tmp_path) -> None:
    op = tmp_path / "toy_op"
    kernel = op / "op_kernel" / "arch35"
    kernel.mkdir(parents=True)
    (kernel / "kernel.h").write_text(
        "template<uint16_t s1TemplateType>\n"
        "struct Kernel {\n"
        "  constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;\n"
        "};\n",
        encoding="utf-8",
    )
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    arg = cm.upsert(
        EntityKind.TEMPLATE_ARG,
        "s1TemplateType",
        attrs={"provenance": "test"},
    )
    enrich_constexpr_aliases(cm, op, architecture="arch35")
    aliases = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.MATERIALIZES_AS.value
    ]
    assert aliases
    assert any(rel.src == arg.id for rel in aliases)
    dst = cm.entities[aliases[0].dst]
    assert dst.name == "CUBE_BASEM"


def test_host_and_predicates_derive_conjuncts() -> None:
    from types import SimpleNamespace

    cm = CodeMap(op_name="toy", architecture="arch35")
    a = cm.upsert(EntityKind.VARIABLE, "splitOk", attrs={"layer": "host"})
    b = cm.upsert(EntityKind.VARIABLE, "dWide", attrs={"layer": "host"})
    flag = cm.upsert(EntityKind.VARIABLE, "enableOut", attrs={"layer": "host"})
    host_ir = SimpleNamespace(
        writes=[],
        local_writes=[
            SimpleNamespace(
                path="enableOut",
                rhs="splitOk && dWide && !isFp32",
                file="op_host/tiling.cpp",
                line=12,
            )
        ],
    )
    enrich_host_predicates(cm, architecture="arch35", host_ir=host_ir)
    derives = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.DERIVES.value and rel.dst == flag.id
    ]
    srcs = {rel.src for rel in derives}
    assert a.id in srcs and b.id in srcs


def test_positional_tiling_key_lookup_ignores_name() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    cm.upsert(
        EntityKind.TILING_KEY,
        "S1TemplateType",
        attrs={"source_declared": True, "decl_order": 0},
    )
    second = cm.upsert(
        EntityKind.TILING_KEY,
        "S2TemplateType",
        attrs={"source_declared": True, "decl_order": 1},
    )
    assert _tiling_key_at_order(cm, 1) is not None
    assert _tiling_key_at_order(cm, 1).id == second.id
    assert _tiling_key_at_order(cm, 1).name != "UnrelatedTemplateName"


def test_compile_policy_conditional_t_binds_buffer(tmp_path) -> None:
    from ascendc_codemap_mcp.engine.passes.compile_policy import enrich_compile_policy

    op = tmp_path / "toy_op"
    kernel = op / "op_kernel" / "arch35"
    kernel.mkdir(parents=True)
    (kernel / "kernel.h").write_text(
        "template<bool flag>\n"
        "struct Kernel {\n"
        "  using Policy = std::conditional_t<flag, ThenBuf, ElseBuf>;\n"
        "  Policy slot;\n"
        "};\n",
        encoding="utf-8",
    )
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    flag = cm.upsert(EntityKind.TEMPLATE_ARG, "flag", attrs={"provenance": "test"})
    buf = cm.upsert(
        EntityKind.BUFFER,
        "slot",
        attrs={"type_name": "Policy", "trace": ["Policy", "ThenBuf"]},
    )
    enrich_compile_policy(cm, op, architecture="arch35")
    controls = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.CONTROLS.value and rel.src == flag.id
    ]
    assert controls
    binds = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.BINDS.value and rel.dst == buf.id
    ]
    assert binds


def test_workspace_size_allocates(tmp_path) -> None:
    from ascendc_codemap_mcp.engine.passes.workspace_abi import enrich_workspace_abi

    op = tmp_path / "toy_op"
    host = op / "op_host"
    host.mkdir(parents=True)
    (host / "tiling.cpp").write_text(
        "void DoTiling() {\n"
        "  uint64_t workspaceSize = 0;\n"
        "  workspaceSize += blockBytes;\n"
        "  tiling.set_inputOffset(workspaceSize);\n"
        "}\n",
        encoding="utf-8",
    )
    (op / "op_kernel" / "arch35").mkdir(parents=True)
    (op / "op_kernel" / "arch35" / "kernel.h").write_text(
        "void Kernel(GM_ADDR workspace) { workspace.SetGlobalBuffer(); }\n",
        encoding="utf-8",
    )
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    field = cm.upsert(EntityKind.TILING_FIELD, "inputOffset", attrs={"provenance": "test"})
    size = cm.upsert(EntityKind.VARIABLE, "blockBytes", attrs={"layer": "host"})
    enrich_workspace_abi(cm, op, architecture="arch35")
    kinds = {rel.kind_name() for rel in cm.relations.values()}
    assert RelationKind.ALLOCATES.value in kinds
    allocs = [
        rel
        for rel in cm.relations.values()
        if rel.kind_name() == RelationKind.ALLOCATES.value
    ]
    assert any(rel.src == size.id or rel.dst == field.id for rel in allocs)


def test_struct_sel_parse_not_a_dim() -> None:
    from ascendc_codemap_mcp.engine.tpl_dsl import is_tiling_struct_sel, parse_args_sel

    src = (
        "ASCENDC_TPL_ARGS_SEL(\n"
        "  ASCENDC_TPL_BOOL_SEL(FLAG, 0),\n"
        "  ASCENDC_TPL_TILING_STRUCT_SEL(ToyTilingData)\n"
        ")\n"
    )
    groups = parse_args_sel(src)
    assert groups
    structs = [s for s in groups[0] if is_tiling_struct_sel(s)]
    dims = [s for s in groups[0] if not is_tiling_struct_sel(s)]
    assert structs and structs[0]["struct"] == "ToyTilingData"
    assert dims and dims[0]["name"] == "FLAG"


def test_consumer_role_from_parent_not_function_name() -> None:
    from ascendc_codemap_mcp.engine.passes.consumer_role import (
        BRANCH_GUARD,
        DATA_MOVE,
        UNKNOWN_ROLE,
        classify_statement,
    )

    assert classify_statement("if (enable) { run(); }", parent_kind="IF_STMT") == BRANCH_GUARD
    assert classify_statement("x.DataCopy(dst, src);", callee="DataCopy") == DATA_MOVE
    assert classify_statement("helper();", callee="PreloadCube") == UNKNOWN_ROLE


def test_impact_closure_follows_calls_under_guard() -> None:
    from ascendc_codemap_mcp.engine.passes.guarded_calls import ingest_guarded_calls
    from ascendc_codemap_mcp.engine.query.closure import semantic_impact_closure_mem
    from ascendc_codemap_mcp.engine.clang_walk import CallSite, PathCond

    cm = CodeMap(op_name="toy", architecture="arch35")
    flag = cm.upsert(EntityKind.VARIABLE, "enableFeature", attrs={"layer": "host"})
    site = CallSite(
        caller="DoTiling",
        callee="PreloadCube",
        file="op_host/tiling.cpp",
        line=40,
        path_conditions=(
            PathCond(text="enableFeature", negated=False, file="op_host/tiling.cpp", line=38, kind="if"),
        ),
    )
    ingest_guarded_calls(cm, [site], side="host", architecture="arch35")
    closed = semantic_impact_closure_mem(cm, [flag.id])
    names = {row["name"] for row in closed["sinks"]}
    assert "PreloadCube" in names or any(
        "PreloadCube" in str(cm.entities[nid].name)
        for nid in closed["nodes"]
        if nid in cm.entities
    )


def test_completeness_false_complete_without_consumer() -> None:
    from ascendc_codemap_mcp.engine.query.completeness import (
        COMPLETE,
        INCOMPLETE,
        fence_contract,
    )

    seed = {"id": "s", "name": "field", "kind": "TILING_FIELD", "file": "a.cpp", "line": 1}
    producer = {"id": "p", "name": "DoTiling", "file": "a.cpp", "line": 2}
    fence = fence_contract(
        seeds=[seed],
        producers=[producer],
        consumers=[],
        sinks=[],
        transport="tiling_data",
    )
    assert fence["completeness"] == INCOMPLETE
    fence2 = fence_contract(
        seeds=[seed],
        producers=[producer],
        consumers=[{"id": "k", "name": "Kernel", "file": "k.h", "line": 10}],
        sinks=[],
        transport="tiling_data",
    )
    assert fence2["completeness"] == COMPLETE


def test_predicate_ast_and_and_div() -> None:
    from ascendc_codemap_mcp.engine.query.predicate_ast import parse_predicate

    ast = parse_predicate("layout == TND && s == 0")
    assert ast.op == "AND"
    assert "EQ" in ast.operators()
    assert "0" in ast.literals()
    assert "layout" in ast.references()
    assert "TND" in ast.enum_values()
    div = parse_predicate("coreNum / 2")
    assert ast.op != "DIV"
    assert div.op == "DIV"
    assert "2" in div.literals()
    assert "coreNum" in div.references()


def test_calls_are_per_site() -> None:
    from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
    from ascendc_codemap_mcp.engine.ir.entity import EntityKind
    from ascendc_codemap_mcp.engine.ir.relation import RelationKind

    cm = CodeMap(op_name="toy", architecture="arch35")
    caller = cm.upsert(EntityKind.FUNCTION, "Caller", attrs={"layer": "host"})
    callee = cm.upsert(EntityKind.FUNCTION, "Callee", attrs={"layer": "host"})
    r1 = cm.link(
        RelationKind.CALLS,
        caller.id,
        callee.id,
        attrs={"file": "op_host/a.cpp", "line": 10},
    )
    r2 = cm.link(
        RelationKind.CALLS,
        caller.id,
        callee.id,
        attrs={"file": "op_host/a.cpp", "line": 20},
    )
    assert r1.id != r2.id
    calls = [r for r in cm.relations.values() if r.kind_name() == RelationKind.CALLS.value]
    assert len(calls) == 2

