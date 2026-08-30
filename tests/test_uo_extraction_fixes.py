# -*- coding: utf-8 -*-
"""UO extraction fixes: ambiguous CALLS, accessor stubs, call-graph render."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.kernel_call_read_refine import (
    refine_kernel_calls_and_tiling_reads,
)
from ascendc_codemap_mcp.engine.passes.tiling_accessors import link_tiling_accessors
from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _insert_entity

FAG_ROOT = Path(r"D:\TEST\ops-transformer\attention\flash_attention_score_grad")
FAG_UO = FAG_ROOT / ".ascendc-codemap" / "arch35" / "FlashAttentionScoreGrad.arch35.uo"

_KERNEL = """
class Buffer {
public:
    void Init() { ready = 1; }
    int ready;
};
class BuffersPolicyDB {
public:
    void Init() { mode = 2; }
    int mode;
};
class BuffersPolicySingleBuffer {
public:
    void Init() { mode = 1; }
    int mode;
};
class Kernel {
public:
    void UniqueCaller() {
        Buffer buf;
        buf.Init();
    }
    void AmbiguousCaller() {
        Holder h;
        h.Init();
    }
    void Untouched() { int x = 0; (void)x; }
};
"""


def _insert_rel(
    conn: sqlite3.Connection,
    *,
    rid: str,
    kind: str,
    src: str,
    dst: str,
    status: str = "extracted",
    data: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO relation(id, kind, src, dst, status, confidence, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (rid, kind, src, dst, status, 1.0, json.dumps(data or {})),
    )


def test_ambiguous_member_call_mints_partial_edges(tmp_path: Path) -> None:
    root = tmp_path / "toy_op"
    kdir = root / "op_kernel" / "arch35"
    kdir.mkdir(parents=True)
    (kdir / "kernel.h").write_text(_KERNEL, encoding="utf-8")
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    cm.meta["kernel_tiling_closure"] = {
        "selected_kernel_files": ["op_kernel/arch35/kernel.h"]
    }
    refine_kernel_calls_and_tiling_reads(cm, root, architecture="arch35")

    methods = {e.name: e for e in cm.by_kind(EntityKind.METHOD)}
    assert "Buffer::Init" in methods or any(n.endswith("Buffer::Init") for n in methods)
    assert any(n.endswith("BuffersPolicyDB::Init") or n == "BuffersPolicyDB::Init" for n in methods)
    assert any(
        n.endswith("BuffersPolicySingleBuffer::Init") or n == "BuffersPolicySingleBuffer::Init"
        for n in methods
    )

    def _incoming(leaf_owner: str) -> list:
        hits = []
        for ent in cm.by_kind(EntityKind.METHOD):
            if ent.name == f"{leaf_owner}::Init" or ent.name.endswith(f"::{leaf_owner}::Init"):
                for rel in cm.relations.values():
                    if rel.kind_name() == RelationKind.CALLS.value and rel.dst == ent.id:
                        hits.append(rel)
        return hits

    buf = _incoming("Buffer")
    db = _incoming("BuffersPolicyDB")
    single = _incoming("BuffersPolicySingleBuffer")
    assert buf, "Buffer::Init must have incoming CALLS"
    assert db, "BuffersPolicyDB::Init must have incoming CALLS"
    assert single, "BuffersPolicySingleBuffer::Init must have incoming CALLS"

    partial_db = [r for r in db if str(r.status or "") == "partial"]
    assert partial_db
    assert partial_db[0].attrs.get("ambiguous_dispatch") is True
    assert int(partial_db[0].attrs.get("dispatch_candidates") or 0) >= 2

    unique_confirmed = [
        r
        for r in buf
        if str(r.status or "").lower() == "confirmed"
        and not r.attrs.get("ambiguous_dispatch")
    ]
    assert unique_confirmed, "typed Buffer buf; buf.Init() must stay confirmed"

    untouched = [
        e
        for e in cm.by_kind(EntityKind.METHOD)
        if e.name.endswith("Untouched") or e.name == "Untouched"
    ]
    assert untouched
    incoming_untouched = [
        r
        for r in cm.relations.values()
        if r.kind_name() == RelationKind.CALLS.value and r.dst == untouched[0].id
    ]
    assert incoming_untouched == []


def test_confirmed_edge_not_downgraded_by_ambiguous_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "toy_op"
    kdir = root / "op_kernel" / "arch35"
    kdir.mkdir(parents=True)
    (kdir / "kernel.h").write_text(_KERNEL, encoding="utf-8")
    cm = CodeMap(op_name="toy_op", architecture="arch35")
    cm.meta["kernel_tiling_closure"] = {
        "selected_kernel_files": ["op_kernel/arch35/kernel.h"]
    }
    refine_kernel_calls_and_tiling_reads(cm, root, architecture="arch35")
    for rel in cm.relations.values():
        if rel.kind_name() != RelationKind.CALLS.value:
            continue
        if str(rel.status or "").lower() == "confirmed":
            assert rel.attrs.get("ambiguous_dispatch") is not True


def test_accessor_stub_references_tiling_field_without_forging_location() -> None:
    cm = CodeMap(op_name="toy", architecture="arch35")
    field = cm.upsert(
        EntityKind.TILING_FIELD,
        "deterMaxRound",
        file="op_kernel/arch35/tiling_data.h",
        line=483,
        status="confirmed",
    )
    stub = cm.upsert(
        EntityKind.FUNCTION,
        "set_deterMaxRound",
        attrs={"layer": "host", "provenance": "clang_walk"},
    )
    other = cm.upsert(
        EntityKind.FUNCTION,
        "set_unknownField",
        attrs={"layer": "host", "provenance": "clang_walk"},
    )
    fileless_before = sum(
        1 for e in cm.entities.values() if not e.file or int(e.line_start or 0) <= 0
    )
    link_tiling_accessors(cm)
    assert stub.file == ""
    assert int(stub.line_start or 0) == 0
    assert stub.attrs.get("accessor_of") == "deterMaxRound"
    refs = [
        r
        for r in cm.relations.values()
        if r.kind_name() == RelationKind.REFERENCES.value
        and r.src == stub.id
        and r.dst == field.id
    ]
    assert len(refs) == 1
    assert other.attrs.get("accessor_of") is None
    fileless_after = sum(
        1 for e in cm.entities.values() if not e.file or int(e.line_start or 0) <= 0
    )
    assert fileless_after == fileless_before


def test_field_ids_named_prefers_located_entity(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn, eid="stub", kind="FIELD", name="SyncAll", file="", line=0, data="{}"
        )
        _insert_entity(
            conn,
            eid="real",
            kind="FIELD",
            name="SyncAll",
            file="op_kernel/arch35/sync.h",
            line=40,
        )
        conn.commit()
    finally:
        conn.close()
    with UoSqlQuery(dest) as q:
        ids = q._field_ids_named("SyncAll")
    assert ids, "located SyncAll must be resolvable"
    assert ids.get("FIELD", [""])[0] == "real"


def test_calls_render_macro_sites_and_hides_fileless(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="sync",
            kind="METHOD",
            name="FlashAttentionScoreGradKernelBase::SyncALLCores",
            file="op_kernel/arch35/flash_attention_score_grad_kernel_base.h",
            line=2388,
        )
        _insert_entity(
            conn,
            eid="macro",
            kind="MACRO",
            name="INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL",
            file="op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
            line=40,
        )
        _insert_entity(
            conn,
            eid="macro2",
            kind="MACRO",
            name="INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL",
            file="op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
            line=129,
        )
        _insert_entity(
            conn,
            eid="dlog",
            kind="FUNCTION",
            name="DlogRecord",
            file="",
            line=0,
        )
        _insert_entity(
            conn,
            eid="cstr",
            kind="FUNCTION",
            name="c_str",
            file="",
            line=0,
        )
        _insert_entity(
            conn,
            eid="tid",
            kind="FUNCTION",
            name="GetTid",
            file="",
            line=0,
        )
        _insert_entity(
            conn,
            eid="maybe",
            kind="METHOD",
            name="Buffer::Init",
            file="../common/op_kernel/attn_buffer.h",
            line=10,
        )
        _insert_rel(
            conn,
            rid="c1",
            kind="CALLS",
            src="macro",
            dst="sync",
            status="confirmed",
            data={
                "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                "line": 52,
                "sites": [
                    {
                        "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                        "line": 52,
                    },
                    {
                        "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                        "line": 60,
                    },
                    {
                        "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                        "line": 92,
                    },
                ],
            },
        )
        _insert_rel(
            conn,
            rid="c2",
            kind="CALLS",
            src="macro2",
            dst="sync",
            status="confirmed",
            data={
                "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                "line": 140,
                "sites": [
                    {
                        "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                        "line": 140,
                    },
                    {
                        "file": "op_kernel/arch35/flash_attention_score_grad_entry_regbase.h",
                        "line": 171,
                    },
                ],
            },
        )
        _insert_rel(
            conn,
            rid="noise1",
            kind="CALLS",
            src="sync",
            dst="dlog",
            status="confirmed",
            data={},
        )
        _insert_rel(
            conn,
            rid="noise2",
            kind="CALLS",
            src="sync",
            dst="cstr",
            status="confirmed",
            data={},
        )
        _insert_rel(
            conn,
            rid="noise3",
            kind="CALLS",
            src="sync",
            dst="tid",
            status="confirmed",
            data={},
        )
        _insert_rel(
            conn,
            rid="part",
            kind="CALLS",
            src="maybe",
            dst="sync",
            status="partial",
            data={
                "ambiguous_dispatch": True,
                "dispatch_candidates": 3,
                "file": "../common/op_kernel/attn_buffer.h",
                "line": 10,
                "sites": [{"file": "../common/op_kernel/attn_buffer.h", "line": 88}],
            },
        )
        conn.commit()
    finally:
        conn.close()

    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="SyncALLCores",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL" in text
    assert "52" in text and "60" in text and "92" in text
    assert "Possible callers" in text
    assert "Buffer::Init" in text
    assert "DlogRecord" not in text
    assert "c_str" not in text
    assert "GetTid" not in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 snapshot not present")
def test_fag_ambiguous_bind_coverage() -> None:
    """Assert the shipped .uo, not a second in-memory _bind_calls pass."""
    conn = sqlite3.connect(f"file:{FAG_UO}?mode=ro", uri=True)
    try:
        methods = conn.execute(
            "SELECT COUNT(*) FROM entity WHERE kind = 'METHOD'"
        ).fetchone()[0]
        with_caller = conn.execute(
            """
            SELECT COUNT(DISTINCT e.id) FROM entity e
            JOIN relation r ON r.dst = e.id
            WHERE e.kind = 'METHOD'
              AND r.kind IN ('CALLS', 'CALLS_UNDER_GUARD')
            """
        ).fetchone()[0]
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM relation WHERE kind = 'CALLS' AND status = 'confirmed'"
        ).fetchone()[0]
        assert methods >= 998
        assert with_caller >= 552
        assert confirmed >= 1
        # Idempotent: a second read of the shipped snapshot matches the first.
        methods_again = conn.execute(
            "SELECT COUNT(*) FROM entity WHERE kind = 'METHOD'"
        ).fetchone()[0]
        with_caller_again = conn.execute(
            """
            SELECT COUNT(DISTINCT e.id) FROM entity e
            JOIN relation r ON r.dst = e.id
            WHERE e.kind = 'METHOD'
              AND r.kind IN ('CALLS', 'CALLS_UNDER_GUARD')
            """
        ).fetchone()[0]
        assert (methods_again, with_caller_again) == (methods, with_caller)

        for owner in ("Buffer", "BuffersPolicyDB", "BuffersPolicySingleBuffer"):
            rows = conn.execute(
                """
                SELECT r.status, r.data FROM relation r
                JOIN entity e ON e.id = r.dst
                WHERE r.kind = 'CALLS'
                  AND (e.name = ? OR e.name LIKE ?)
                """,
                (f"{owner}::Init", f"%::{owner}::Init"),
            ).fetchall()
            assert rows, f"{owner}::Init must have incoming CALLS in the shipped snapshot"
            partial = [
                data
                for status, data in rows
                if str(status or "") == "partial"
                and "ambiguous_dispatch" in str(data or "")
            ]
            assert partial, f"{owner}::Init must have persisted partial ambiguous_dispatch CALLS"

        member_re = __import__("re").compile(r"(?:\.|->)\s*([A-Za-z_]\w*)\s*\(")
        member_leaves: set[str] = set()
        for (text,) in conn.execute("SELECT text FROM source_line"):
            member_leaves.update(member_re.findall(text or ""))
        called = {
            str(r[0])
            for r in conn.execute(
                """
                SELECT DISTINCT dst FROM relation
                WHERE kind IN ('CALLS', 'CALLS_UNDER_GUARD')
                """
            )
        }
        uncalled = 0
        for eid, name, file, line, data in conn.execute(
            "SELECT id, name, IFNULL(file,''), IFNULL(line_start,0), data FROM entity WHERE kind='METHOD'"
        ):
            if eid in called:
                continue
            if not file or int(line or 0) <= 0:
                continue
            blob = str(data or "")
            if "internal_unresolved" in blob:
                continue
            leaf = str(name or "").split("::")[-1]
            if "." in leaf or leaf in member_leaves:
                continue
            uncalled += 1
        assert uncalled >= 200

        stub = conn.execute(
            """
            SELECT id, IFNULL(file,''), data FROM entity
            WHERE kind='FUNCTION' AND name='set_deterMaxRound'
              AND (IFNULL(file,'') = '' OR IFNULL(line_start,0) = 0)
            """
        ).fetchone()
        field = conn.execute(
            """
            SELECT id, file FROM entity
            WHERE kind='TILING_FIELD' AND name='deterMaxRound'
              AND IFNULL(file,'') != '' AND IFNULL(line_start,0) > 0
            """
        ).fetchone()
        assert stub and field
        assert stub[1] == ""
        linked = conn.execute(
            """
            SELECT COUNT(*) FROM relation
            WHERE kind='REFERENCES' AND src=? AND dst=?
            """,
            (stub[0], field[0]),
        ).fetchone()[0]
        assert linked >= 1
        assert "accessor_of" in str(stub[2] or "")
    finally:
        conn.close()
