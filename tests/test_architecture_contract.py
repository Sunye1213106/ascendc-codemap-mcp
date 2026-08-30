# -*- coding: utf-8 -*-
"""Architecture contract: .uo-only query, owner-aware TYPE identity, no internal ids."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity
from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create, declaration_id
from ascendc_codemap_mcp.engine.query.explore import _clip_source, render_explore_markdown
from ascendc_codemap_mcp.engine.semantics.ascendc_storage import memory_space_from_type_text
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query


_INTERNAL_MARKERS = (
    "SRCPOL::",
    "SRCPOLCOND::",
    "entity_id=",
    "E_TYPE_",
    "REL::",
)


def test_clip_source_reads_snapshot_not_workspace(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    live = op / "op_kernel" / "live.h"
    live.parent.mkdir(parents=True)
    live.write_text("LIVE_TREE_ONLY\n", encoding="utf-8")
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, [("op_kernel/live.h", 1, "SNAPSHOT_LINE")])
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    clip = _clip_source(UoSqlQuery(dest), "op_kernel/live.h", 1, line_end=1)
    assert "SNAPSHOT_LINE" in clip
    assert "LIVE_TREE_ONLY" not in clip


def test_resolve_and_search_markdown_hide_internal_ids(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op, symbol="IS_SMALL_D_PRELOAD")
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="SRCPOL::op_kernel/block.h::TYPE",
            kind="COMPILE_VAR",
            name="IS_SMALL_D_PRELOAD",
            file="op_kernel/block.h",
            line=149,
            snippet="constexpr static bool IS_SMALL_D_PRELOAD = !IS_DROP;",
        )
        _add_source_lines(
            conn,
            [
                (
                    "op_kernel/block.h",
                    149,
                    "constexpr static bool IS_SMALL_D_PRELOAD = !IS_DROP;",
                )
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    resolved = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="IS_SMALL_D_PRELOAD"
    )
    searched = query(
        project=str(op), architecture="arch35", operation="search", name="IS_SMALL_D_PRELOAD"
    )
    blob = str((resolved.get("data") or {}).get("text") or "") + str(
        (searched.get("data") or {}).get("text") or ""
    )
    for marker in _INTERNAL_MARKERS:
        assert marker not in blob
    cards = (resolved.get("data") or {}).get("cards") or []
    assert cards
    assert "id" not in cards[0]


def test_nested_type_identity_is_owner_aware() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    file = "op_kernel/block_cube.h"
    clang = cm.upsert(
        EntityKind.TYPE,
        "QL1BuffSelector::TYPE",
        eid="CLANG::QL1BuffSelector::TYPE",
        attrs={"lexical_owner": "QL1BuffSelector", "provenance": "clang_ast"},
        file=file,
        line=53,
        status="confirmed",
    )
    rebound = bind_or_create(
        cm,
        EntityKind.TYPE,
        "TYPE",
        file=file,
        line=53,
        owner="QL1BuffSelector",
        architecture="arch35",
        attrs={"provenance": "source_compile_policy"},
    )
    assert rebound.id == clang.id
    q = bind_or_create(
        cm, EntityKind.TYPE, "TYPE", file=file, line=55, owner="KL1BuffSelector", architecture="arch35"
    )
    d = bind_or_create(
        cm, EntityKind.TYPE, "TYPE", file=file, line=57, owner="DyL1BuffSelector", architecture="arch35"
    )
    assert q.id != clang.id
    assert d.id != q.id
    assert "QL1BuffSelector" in declaration_id(
        kind=EntityKind.TYPE, architecture="arch35", file=file, owner="QL1BuffSelector", symbol="TYPE"
    )


def test_compile_policy_three_selector_types(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.engine.passes.compile_policy import enrich_compile_policy

    op = tmp_path / "toy_op"
    kernel = op / "op_kernel" / "arch35"
    kernel.mkdir(parents=True)
    (kernel / "block_cube.h").write_text(
        "struct QL1BuffSelector {\n"
        "  using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, A, B>;\n"
        "};\n"
        "struct KL1BuffSelector {\n"
        "  using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, C, D>;\n"
        "};\n"
        "struct DyL1BuffSelector {\n"
        "  using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, E, F>;\n"
        "};\n",
        encoding="utf-8",
    )
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    enrich_compile_policy(cm, op, architecture="arch35")
    types = [e for e in cm.by_kind(EntityKind.TYPE) if str(e.name or "").endswith("::TYPE")]
    names = sorted(e.name for e in types)
    assert "QL1BuffSelector::TYPE" in names
    assert "KL1BuffSelector::TYPE" in names
    assert "DyL1BuffSelector::TYPE" in names
    assert len({e.id for e in types}) >= 3
    assert all("BuffSelector" in e.name for e in types)
    assert all(not str(e.id).startswith("SRCPOL::") for e in types)


def test_local_tensor_is_not_implicit_ub() -> None:
    assert memory_space_from_type_text("LocalTensor<half>") is None
    assert memory_space_from_type_text("LocalTensor<half, TPosition::VECIN>") is not None
    assert memory_space_from_type_text("GlobalTensor<half>") == "GM"


def test_render_omits_empty_facets() -> None:
    text = render_explore_markdown(
        {
            "operation": "resolve",
            "cards": [
                {
                    "name": "IS_SMALL_D_PRELOAD",
                    "kind": "COMPILE_VAR",
                    "file": "op_kernel/block.h",
                    "line": 149,
                    "snippet": "149:  constexpr static bool IS_SMALL_D_PRELOAD = !IS_DROP;",
                    "id": "SRCPOL::block.h::TYPE",
                }
            ],
        }
    )
    assert "SRCPOL::" not in text
    assert "IS_SMALL_D_PRELOAD" in text
    assert "entity_id" not in text


def test_integrity_blocks_dangling_confirmed_edge() -> None:
    from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap
    from ascendc_codemap_mcp.engine.ir.relation import RelationKind

    cm = CodeMap(op_name="toy", architecture="arch35")
    field = cm.upsert(EntityKind.TILING_FIELD, "s1Inner", eid="f1", file="td.h", line=1)
    cm.link(RelationKind.WRITES, "missing_writer", field.id, status="confirmed")
    report = audit_codemap(cm)
    codes = {row["code"] for row in (report.get("integrity_blocking") or [])}
    assert "CONFIRMED_DANGLING_EDGE" in codes


def test_integrity_blocks_leaf_only_srcpol_type() -> None:
    from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap

    cm = CodeMap(op_name="toy", architecture="arch35")
    cm.upsert(
        EntityKind.TYPE,
        "TYPE",
        eid="SRCPOL::op_kernel/block.h::TYPE",
        file="op_kernel/block.h",
        line=1,
        status="confirmed",
    )
    report = audit_codemap(cm)
    codes = {row["code"] for row in (report.get("integrity_blocking") or [])}
    assert "LEAF_ONLY_TYPE_IDENTITY" in codes


def test_backed_by_links_view_to_storage_owner() -> None:
    from ascendc_codemap_mcp.engine.passes.kernel_root_trace import _link_backed_by
    from ascendc_codemap_mcp.engine.ir.relation import RelationKind

    cm = CodeMap(op_name="toy", architecture="arch35")
    view = cm.upsert(EntityKind.BUFFER, "mm1PingL1", eid="v1")
    owner = cm.upsert(EntityKind.BUFFER, "l1Tbuf", eid="o1", attrs={"memory_space": "L1"})
    _link_backed_by(cm, view.id, owner.id, space="L1", via="storage_owner")
    edges = [r for r in cm.relations.values() if r.kind_name() == RelationKind.BACKED_BY.value]
    assert len(edges) == 1
    assert edges[0].src == view.id
    assert edges[0].dst == owner.id
    assert edges[0].attrs.get("physical_space") == "L1"


def test_datacopy_arg_effects_are_dst_write_src_read() -> None:
    from ascendc_codemap_mcp.engine.semantics.registry import arg_effects

    reads, writes = arg_effects("DataCopy", ["dstBuf", "srcBuf"])
    assert writes == ["dstBuf"]
    assert reads == ["srcBuf"]


def test_compatible_flag_guards() -> None:
    from ascendc_codemap_mcp.engine.semantics.ascendc_sync import compatible_flag_guards

    assert compatible_flag_guards("MTE2_V", "MTE2_V")
    assert compatible_flag_guards("", "MTE2_V")
    assert not compatible_flag_guards("MTE2_V", "V_MTE2")


def test_query_engine_has_no_workspace_file_helpers() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ascendc_codemap_mcp" / "engine" / "query"
    sql = (root / "sql.py").read_text(encoding="utf-8")
    explore = (root / "explore.py").read_text(encoding="utf-8")
    assert "def _disk_window" not in sql
    assert "def _file_lines" not in sql
    assert "def _resolve_source_path" not in sql
    assert "Path.read_text" not in explore
    assert "path.open(" not in sql
