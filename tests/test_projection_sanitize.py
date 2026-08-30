# -*- coding: utf-8 -*-
"""Agent projection: one renderer, no noise, no duplicate facets."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel
from tests.test_resolve_site_facets import _HOST, _split_axis_fixture


def test_site_resolve_prints_used_by_once(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _split_axis_fixture(op)
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file=_HOST,
        line=45,
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert text.count("Used by") <= 1
    assert text.count("Controls") <= 1


def test_used_by_drops_validation_noise(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                ("op_host/t.cpp", 1, "void SetSplitAxis() {"),
                ("op_host/t.cpp", 2, "    OP_LOGD(CheckLogLevel());"),
                ("op_host/t.cpp", 3, "    GetWorkspaceSize();"),
                ("op_host/t.cpp", 4, "}"),
            ],
        )
        _insert_entity(
            conn,
            eid="fn",
            kind="FUNCTION",
            name="SetSplitAxis",
            file="op_host/t.cpp",
            line=1,
            line_end=4,
        )
        for i, name in enumerate(("CheckLogLevel", "DlogRecord", "GetTid", "GetWorkspaceSize"), start=1):
            _insert_entity(
                conn,
                eid=f"op{i}",
                kind="OPERATION",
                name=name,
                file="op_host/t.cpp",
                line=2,
            )
            _insert_rel(
                conn,
                rid=f"c{i}",
                kind="CALLS",
                src="fn",
                dst=f"op{i}",
                file="op_host/t.cpp",
                line=2,
            )
            conn.execute(f"UPDATE relation SET status='confirmed' WHERE id='c{i}'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="SetSplitAxis"
    )
    text = str((payload.get("data") or {}).get("text") or "")
    used = text.split("Used by", 1)[-1] if "Used by" in text else text
    assert "CheckLogLevel" not in used
    assert "DlogRecord" not in used
    assert "GetTid" not in used
    assert "GetWorkspaceSize" in used


def test_memory_facet_uses_proof_words(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fn1",
            kind="FUNCTION",
            name="IterateMmQK",
            file="op_kernel/k.h",
            line=1,
            line_end=80,
        )
        _insert_entity(conn, eid="ub1", kind="BUFFER", name="mmUb", file="op_kernel/k.h", line=20)
        _insert_entity(conn, eid="gm1", kind="BUFFER", name="mmGm", file="op_kernel/k.h", line=21)
        for rid, dst, data in (
            ("r1", "ub1", {"via": "MemoryTransfer", "src_space": "GM", "dst_space": "UB"}),
            ("r2", "gm1", {"via": "MemoryTransfer"}),
        ):
            _insert_rel(
                conn,
                rid=rid,
                kind="FLOWS_TO",
                src="fn1",
                dst=dst,
                file="op_kernel/k.h",
                line=20,
            )
            conn.execute(
                "UPDATE relation SET status='confirmed', data=? WHERE id=?",
                (json.dumps(data), rid),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="IterateMmQK"
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "1/2" in text or "resolved" in text.lower()
    assert "unresolved" in text.lower() or "exhaustive" in text.lower()
