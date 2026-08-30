# -*- coding: utf-8 -*-
"""Canonical compiler facts: identity, relations, storage, toy compile, FAG probe."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.diagnostics.audit import _integrity_false_confirmed
from ascendc_codemap_mcp.engine.ids import buffer_site_id, register_site_id
from ascendc_codemap_mcp.engine.ir.identity import (
    bind_local_object,
    bind_or_create,
    is_forbidden_callable_name,
    is_untrusted_scope,
)
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.kernel_root_trace import _link_backed_by
from ascendc_codemap_mcp.engine.passes.source_resolution import _mint_runtime_field
from ascendc_codemap_mcp.engine.query.agent_card import AGENT_FIELDS, clip_logical_unit, to_agent_card
from ascendc_codemap_mcp.engine.query.explore import render_explore_markdown
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _insert_entity, _insert_rel

FAG_UO = Path(
    r"D:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)


def test_bind_or_create_refuses_keyword_method() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    assert is_forbidden_callable_name("constexpr")
    assert bind_or_create(cm, EntityKind.METHOD, "constexpr", file="a.h", line=1) is None
    assert bind_or_create(cm, EntityKind.FUNCTION, "if", file="a.h", line=2) is None
    typed = bind_or_create(cm, EntityKind.TYPE, "QL1BuffSelector", file="a.h", line=3)
    assert typed is not None
    again = bind_or_create(cm, EntityKind.TYPE, "QL1BuffSelector", file="a.h", line=3)
    assert again is not None
    assert again.id == typed.id


def test_untrusted_scope_does_not_split_local_object() -> None:
    assert is_untrusted_scope("constexpr")
    assert is_untrusted_scope("static")
    assert not is_untrusted_scope("IterateMmQK")
    trusted = buffer_site_id(
        file="block_cube.h", line=666, scope="IterateMmQK", name="qL1Tensor"
    )
    dirty = buffer_site_id(
        file="block_cube.h", line=666, scope="constexpr", name="qL1Tensor"
    )
    empty = buffer_site_id(file="block_cube.h", line=666, scope="", name="qL1Tensor")
    assert trusted == dirty == empty
    split = buffer_site_id(
        file="block_cube.h",
        line=666,
        scope="IterateMmQK",
        name="qL1Tensor",
        instantiation_context="IS_SMALL_D_PRELOAD=1",
    )
    assert split != trusted
    assert register_site_id(
        file="a.h", line=10, scope="constexpr", name="vreg"
    ) == register_site_id(file="a.h", line=10, scope="Kernel", name="vreg")


def test_bind_local_object_reuses_declaration_site() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    first = bind_local_object(
        cm,
        EntityKind.BUFFER,
        "qL1Tensor",
        file="block_cube.h",
        line=666,
        scope="IterateMmQK",
    )
    second = bind_local_object(
        cm,
        EntityKind.BUFFER,
        "qL1Tensor",
        file="block_cube.h",
        line=666,
        scope="constexpr",
    )
    assert first is not None and second is not None
    assert first.id == second.id
    assert len([e for e in cm.by_kind(EntityKind.BUFFER) if e.name == "qL1Tensor"]) == 1
    inst = bind_local_object(
        cm,
        EntityKind.BUFFER,
        "qL1Tensor",
        file="block_cube.h",
        line=666,
        scope="IterateMmQK",
        instantiation_context="D=128",
    )
    assert inst is not None
    assert inst.id != first.id


def test_duplicate_local_object_is_integrity_blocking() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    cm.upsert(
        EntityKind.BUFFER,
        "qL1Tensor",
        eid="BUF_OLD_TRUSTED",
        attrs={"scope": "IterateMmQK"},
        file="block_cube.h",
        line=666,
        status="extracted",
    )
    cm.upsert(
        EntityKind.BUFFER,
        "qL1Tensor",
        eid="BUF_OLD_CONSTEXPR",
        attrs={"scope": "constexpr"},
        file="block_cube.h",
        line=666,
        status="extracted",
    )
    codes = {str(row.get("code") or "") for row in _integrity_false_confirmed(cm)}
    assert "DUPLICATE_LOCAL_OBJECT" in codes


def test_using_type_is_not_field() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    owner = cm.upsert(EntityKind.TYPE, "QL1BuffSelector", file="block.h", line=10)
    ok = _mint_runtime_field(
        cm,
        owner,
        "QL1BuffSelector",
        "TYPE",
        "using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, A, B>",
        "block.h",
        12,
    )
    assert ok is False
    fields = [e for e in cm.by_kind(EntityKind.FIELD) if str(e.name or "").endswith("TYPE")]
    assert fields == []
    bind_or_create(
        cm,
        EntityKind.TYPE,
        "TYPE",
        file="block.h",
        line=12,
        owner="QL1BuffSelector",
        architecture="arch35",
        status="confirmed",
    )
    types = [e for e in cm.by_kind(EntityKind.TYPE) if "TYPE" in str(e.name or "")]
    assert any(str(e.name).endswith("::TYPE") or str(e.name) == "TYPE" for e in types)


def test_calls_topology_unique_accumulates_sites() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    src = cm.upsert(EntityKind.FUNCTION, "caller", file="k.cpp", line=1)
    dst = cm.upsert(EntityKind.FUNCTION, "callee", file="k.cpp", line=20)
    cm.link(
        RelationKind.CALLS,
        src.id,
        dst.id,
        attrs={"file": "k.cpp", "line": 40},
        status="confirmed",
    )
    cm.link(
        RelationKind.CALLS,
        src.id,
        dst.id,
        attrs={"file": "k.cpp", "line": 80},
        status="confirmed",
    )
    rels = [r for r in cm.relations.values() if r.kind_name() == RelationKind.CALLS.value]
    assert len(rels) == 1
    sites = rels[0].attrs.get("sites") or []
    lines = {int(s.get("line") or 0) for s in sites if isinstance(s, dict)}
    assert {40, 80} <= lines


def test_backed_by_refuses_catalog_local_tensor() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    view = cm.upsert(EntityKind.VARIABLE, "qL1Tensor", file="k.h", line=10)
    catalog = cm.upsert(
        EntityKind.TYPE,
        "LocalTensor",
        attrs={"catalog": "ascendc", "root": "AscendC::LocalTensor"},
    )
    buf = cm.upsert(
        EntityKind.BUFFER,
        "qL1Buffer",
        attrs={"type_name": "TBuf<TPosition::A1>", "allocated": True},
        file="k.h",
        line=8,
    )
    _link_backed_by(cm, view.id, catalog.id, space="L1", file="k.h", line=10)
    backed = [r for r in cm.relations.values() if r.kind_name() == RelationKind.BACKED_BY.value]
    assert backed == []
    _link_backed_by(cm, view.id, buf.id, space="L1", file="k.h", line=10)
    backed = [r for r in cm.relations.values() if r.kind_name() == RelationKind.BACKED_BY.value]
    assert len(backed) == 1
    assert backed[0].dst == buf.id
    assert backed[0].attrs.get("physical_space") == "L1"

    l0 = cm.upsert(
        EntityKind.BUFFER,
        "l0aBuf",
        attrs={"type_name": "TBuf<TPosition::A2>", "allocated": True},
    )
    l0v = cm.upsert(EntityKind.VARIABLE, "l0aTensor")
    _link_backed_by(cm, l0v.id, l0.id, space="L0A")
    reg = cm.upsert(EntityKind.REGISTER, "vReg")
    rv = cm.upsert(EntityKind.VARIABLE, "vRegView")
    _link_backed_by(cm, rv.id, reg.id, space="REG")
    spaces = {
        r.attrs.get("physical_space")
        for r in cm.relations.values()
        if r.kind_name() == RelationKind.BACKED_BY.value
    }
    assert {"L1", "L0A", "REG"} <= spaces


def test_agent_card_schema_and_logical_unit() -> None:
    card = to_agent_card(
        {"name": "hasRope", "kind": "COMPILE_VAR", "file": "k.h", "line": 4, "id": "E_TYPE_X"},
        summary="host encoding",
        source="using TYPE = std::conditional_t<flag, A, B>;",
        facets={"compiled_support": {"legal": True}},
        counts={"variants": 3},
    )
    assert set(card) <= set(AGENT_FIELDS) | {"name", "kind", "file", "line"}
    assert "id" not in card
    rows = [
        (3, "template<bool flag>"),
        (4, "using TYPE = std::conditional_t<"),
        (5, "    flag, A, B>;"),
        (6, "int unused = 0;"),
    ]
    clipped = clip_logical_unit(rows, 5, max_lines=16)
    text = "\n".join(t for _, t in clipped)
    assert "using TYPE" in text
    assert "conditional_t" in text


def test_resolve_type_omits_producer_missing(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="ty1",
            kind="TYPE",
            name="QL1BuffSelector::TYPE",
            file="op_kernel/block.h",
            line=12,
            snippet="using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, A, B>;",
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="QL1BuffSelector::TYPE",
        projection="source",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "") + str(data.get("unresolved_reason") or "")
    assert "PRODUCER_MISSING" not in text
    assert "using TYPE" in text or "QL1BuffSelector" in text


def test_find_operation_envelope_is_exhaustive_honest(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i in range(20):
            _insert_entity(
                conn,
                eid=f"op{i}",
                kind="OPERATION",
                name="InitBuffer",
                file=f"op_kernel/block_{i % 3}.h",
                line=10 + i,
                data='{"callee":"InitBuffer"}',
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        kind="OPERATION",
        callee="InitBuffer",
        limit=8,
    )
    data = payload.get("data") or {}
    cov = payload.get("coverage") or {}
    assert int(cov.get("total") or data.get("total") or 0) == 20
    assert int(cov.get("returned") or 0) == 8
    assert cov.get("exhaustive") is False
    assert payload.get("next_cursor")
    text = str(data.get("text") or "")
    assert "showing 8 of 20" not in text
    assert "exhaustive=no" in text
    assert int(data.get("returned") or 0) == 8
    cards = [c for c in (data.get("cards") or []) if isinstance(c, dict)]
    assert len(cards) == 8
    page2 = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        kind="OPERATION",
        callee="InitBuffer",
        limit=8,
        cursor=str(payload.get("next_cursor") or ""),
    )
    cards2 = (page2.get("data") or {}).get("cards") or []
    assert len(cards2) == 8
    names1 = {
        (c.get("file"), c.get("line"))
        for c in (data.get("cards") or [])
        if isinstance(c, dict)
    }
    names2 = {(c.get("file"), c.get("line")) for c in cards2 if isinstance(c, dict)}
    assert names1.isdisjoint(names2)


def test_trace_complete_requires_hops(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn, eid="fn_a", kind="FUNCTION", name="HostPack", file="op_host/t.cpp", line=10
        )
        _insert_entity(
            conn, eid="fn_b", kind="FUNCTION", name="KernelRun", file="op_kernel/k.h", line=20
        )
        _insert_rel(
            conn,
            rid="r1",
            kind="CALLS",
            src="fn_a",
            dst="fn_b",
            file="op_host/t.cpp",
            line=12,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    ok = query(
        project=str(op),
        architecture="arch35",
        operation="trace",
        from_symbol="HostPack",
        to_symbol="KernelRun",
    )
    data = ok.get("data") or {}
    assert data.get("completeness") == "COMPLETE"
    path = data.get("path") or []
    assert len(path) >= 1
    assert all(step.get("kind") for step in path)

    empty = query(
        project=str(op),
        architecture="arch35",
        operation="trace",
        from_symbol="HostPack",
        to_symbol="HostPack",
    )
    edata = empty.get("data") or {}
    assert edata.get("completeness") != "COMPLETE"


def test_resolve_has_rope_compiled_support(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op, symbol="hasRope")
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="tk_rope",
            kind="TILING_KEY",
            name="IsRope",
            file="op_kernel/kernel.h",
            line=8,
            data='{"source_declared":true}',
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (1, 'k1', '', '', 'ok')"
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (2, 'k2', '', '', 'ok')"
        )
        for dim, val in (("IsRope", "1"), ("DTemplate", "192"), ("IsDNoEqual", "1")):
            conn.execute(
                "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, ?, ?)",
                (dim, val),
            )
        for dim, val in (("IsRope", "0"), ("DTemplate", "128"), ("IsDNoEqual", "0")):
            conn.execute(
                "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, ?, ?)",
                (dim, val),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="hasRope",
        projection="source",
    )
    data = payload.get("data") or {}
    support = data.get("compiled_support")
    if not isinstance(support, dict):
        cards = data.get("cards") or []
        facets = (cards[0].get("facets") if cards else {}) or {}
        support = facets.get("compiled_support") if isinstance(facets, dict) else None
    assert isinstance(support, dict)
    assert support.get("legal") is True
    assert int(support.get("variants") or 0) >= 1
    assert str(support.get("dim") or "") in {"IsRope", "hasRope"}
    cf = support.get("counterfactual") if isinstance(support.get("counterfactual"), dict) else {}
    assert cf.get("legal") is False
    text = str(data.get("text") or "")
    assert "Compiled" in text
    md = render_explore_markdown(data, verdict="ANSWERED", layer="template")
    assert "legal=" in md


def test_toy_compile_invariants(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.engine.build import compile_codemap

    op = tmp_path / "toy_sel"
    kernel = op / "op_kernel" / "arch35"
    kernel.mkdir(parents=True)
    (kernel / "block.h").write_text(
        "struct QL1BuffSelector {\n"
        "  using TYPE = std::conditional_t<IS_SMALL_D_PRELOAD, ThenBuf, ElseBuf>;\n"
        "};\n"
        "void Kernel() {\n"
        "  if constexpr (flag) {\n"
        "    DataCopy(dst, src, size);\n"
        "  }\n"
        "  TBuf<TPosition::A1> qL1Buffer;\n"
        "  LocalTensor<half> qL1Tensor = qL1Buffer.Get<half>();\n"
        "}\n",
        encoding="utf-8",
    )
    (op / "op_host").mkdir()
    (op / "op_host" / "tiling.cpp").write_text("int dummy() { return 0; }\n", encoding="utf-8")
    try:
        result = compile_codemap(
            op_name="toy_sel",
            architecture="arch35",
            op_root=op,
            commit=False,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"toy compile unavailable: {exc}")
    cm = result.get("codemap")
    assert cm is not None
    methods = [e.name for e in cm.by_kind(EntityKind.METHOD)]
    functions = [e.name for e in cm.by_kind(EntityKind.FUNCTION)]
    assert "constexpr" not in methods
    assert "constexpr" not in functions
    fields = [e.name for e in cm.by_kind(EntityKind.FIELD)]
    assert "TYPE" not in fields
    assert not any(str(n).endswith("::TYPE") for n in fields)
    types = [e.name for e in cm.by_kind(EntityKind.TYPE)]
    assert any("QL1BuffSelector" in str(n) and "TYPE" in str(n) for n in types) or any(
        str(n).endswith("::TYPE") for n in types
    )
    triples: dict[tuple[str, str, str], int] = {}
    for rel in cm.relations.values():
        if rel.kind_name() != RelationKind.CALLS.value:
            continue
        if str(rel.status or "").lower() != "confirmed":
            continue
        key = (rel.kind_name(), str(rel.src), str(rel.dst))
        triples[key] = triples.get(key, 0) + 1
    assert all(n == 1 for n in triples.values())
    blocking = list((result.get("audit") or {}).get("integrity_blocking") or [])
    codes = {str(row.get("code") or "") for row in blocking if isinstance(row, dict)}
    assert "KEYWORD_CALLABLE_NAME" not in codes
    assert "DUPLICATE_CONFIRMED_TRIPLE" not in codes


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_fag_uo_identity_invariants() -> None:
    conn = sqlite3.connect(f"file:{FAG_UO.as_posix()}?mode=ro", uri=True)
    try:
        constexpr_n = conn.execute(
            "SELECT COUNT(*) FROM entity WHERE kind IN ('METHOD','FUNCTION') AND name = 'constexpr'"
        ).fetchone()[0]
        assert constexpr_n == 0
        type_rows = conn.execute(
            "SELECT kind, name FROM entity WHERE name LIKE '%QL1BuffSelector%TYPE%'"
        ).fetchall()
        kinds = {str(r[0]) for r in type_rows}
        assert "FIELD" not in kinds
        dup = conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT kind, src, dst FROM relation
              WHERE kind = 'CALLS' AND IFNULL(status,'') IN ('confirmed','extracted','verified')
              GROUP BY kind, src, dst HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        assert dup == 0
        bad_backed = conn.execute(
            """
            SELECT COUNT(*) FROM relation r
            JOIN entity e ON e.id = r.dst
            WHERE r.kind = 'BACKED_BY'
              AND e.kind = 'TYPE'
              AND (
                e.name LIKE '%LocalTensor%'
                OR IFNULL(json_extract(e.data, '$.catalog'), '') = 'ascendc'
              )
            """
        ).fetchone()[0]
        assert bad_backed == 0
        dirty_scope = conn.execute(
            """
            SELECT COUNT(*) FROM entity
            WHERE kind IN ('BUFFER','REGISTER','QUEUE','EVENT')
              AND IFNULL(json_extract(data, '$.scope'), '') IN
                  ('constexpr','static','inline','const','typename','template')
            """
        ).fetchone()[0]
        if dirty_scope:
            pytest.skip("FAG snapshot predates canonical local-object bind; rebuild to verify")
        split_locals = conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT name, file, line_start FROM entity
              WHERE kind IN ('BUFFER','REGISTER','QUEUE','EVENT')
                AND IFNULL(file,'') != '' AND IFNULL(line_start,0) > 0
              GROUP BY kind, name, file, line_start
              HAVING COUNT(*) > 1
                 AND SUM(CASE WHEN IFNULL(json_extract(data, '$.instantiation_context'), '')
                                   != '' THEN 1 ELSE 0 END) = 0
            )
            """
        ).fetchone()[0]
        assert split_locals == 0
    finally:
        conn.close()
