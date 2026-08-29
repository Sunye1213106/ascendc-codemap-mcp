# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture


def _add_entity(
    conn: sqlite3.Connection,
    *,
    eid: str,
    kind: str,
    name: str,
    file: str = "",
    line: int = 1,
    data: str = "{}",
) -> None:
    conn.execute(
        "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
        "VALUES (?, ?, ?, 'verified', 1.0, ?, ?, ?, ?)",
        (eid, kind, name, file, line, line, data),
    )


def _add_rel(
    conn: sqlite3.Connection,
    *,
    rid: str,
    kind: str,
    src: str,
    dst: str,
    file: str = "",
    line: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO relation(id, kind, src, dst, status, confidence, data) "
        "VALUES (?, ?, ?, ?, 'verified', 1.0, ?)",
        (rid, kind, src, dst, f'{{"file": "{file}", "line": {line}}}'),
    )


def test_tiling_key_readers_include_binds(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_entity(conn, eid="host1", kind="METHOD", name="GetTilingKey", file="op_host/tiling.cpp", line=20)
        _add_entity(conn, eid="kern1", kind="KERNEL", name="KernelEntry", file="op_kernel/vec.h", line=30)
        _add_rel(conn, rid="r1", kind="WRITES", src="host1", dst="e1", file="op_host/tiling.cpp", line=20)
        _add_rel(conn, rid="r2", kind="BINDS", src="kern1", dst="e1", file="op_kernel/vec.h", line=30)
        conn.commit()
    finally:
        conn.close()

    with UoSqlQuery(dest) as query:
        impact = query.field_impact("IsPse")
    assert impact.get("ok") is True
    readers = impact.get("readers") or []
    assert any(row.get("name") == "KernelEntry" for row in readers)


def test_same_name_tiling_key_is_not_ambiguous(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_entity(conn, eid="e2", kind="FIELD", name="IsPse", file="op_host/tiling.cpp", line=11)
        conn.commit()
    finally:
        conn.close()

    with UoSqlQuery(dest) as query:
        payload = query.agent_query(pattern="IsPse")
    assert payload.get("completeness") != "AMBIGUOUS"
    cards = payload.get("cards") or []
    kinds = [c.get("kind") for c in cards]
    assert kinds.count("TILING_KEY") == 1
    assert len(cards) == 1
