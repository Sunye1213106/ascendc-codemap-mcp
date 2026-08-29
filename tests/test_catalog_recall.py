# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture


def test_tque_returns_queue_entities(tmp_path: Path) -> None:
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
    from ascendc_codemap_mcp.engine.store.accel import build_name_leaf

    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, 'verified', 1.0, ?, ?, ?, ?)",
            (
                "cat1",
                "TYPE",
                "TQue",
                "<cann>/queue.h",
                1,
                1,
                '{"catalog": "ascendc"}',
            ),
        )
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, 'verified', 1.0, ?, ?, ?, ?)",
            (
                "q1",
                "QUEUE",
                "pseInQue",
                "op_kernel/vec.h",
                40,
                42,
                "{}",
            ),
        )
        build_name_leaf(conn)
        conn.commit()
    finally:
        conn.close()

    with UoSqlQuery(dest) as query:
        payload = query.query_name_card("TQue")
    assert payload.get("empty_reason") != "no_substring_match"
    kinds = {str(c.get("kind") or "") for c in payload.get("cards") or []}
    names = {str(c.get("name") or "") for c in payload.get("cards") or []}
    assert "QUEUE" in kinds or "pseInQue" in names
    assert payload.get("count", 0) >= 1
