# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.engine.store.schema import SCHEMA_SQL
from ascendc_codemap_mcp.tools import query_codemap, status


def _write_fixture(op: Path, *, arch: str = "arch35") -> Path:
    dest = op / ".ascendc-codemap" / arch / f"{op.name}.{arch}.uo"
    dest.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(dest))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("architecture", arch))
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("op_name", op.name))
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "TILING_KEY",
                "IsPse",
                "verified",
                1.0,
                "op_host/tiling.cpp",
                10,
                12,
                "{}",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return dest


def test_query_name_card_from_fixture(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    st = status(project=str(op), architecture="arch35")
    assert st["indexed"] is True
    payload = query_codemap(project=str(op), architecture="arch35", pattern="IsPse")
    assert payload.get("ok") is True
    assert payload.get("shape") in {"name", "index"}
    assert int(payload.get("count") or 0) >= 1


def test_query_rejects_natural_language(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    payload = query_codemap(
        project=str(op),
        architecture="arch35",
        pattern="who writes IsPse",
    )
    assert payload.get("ok") is False
    assert payload.get("empty_reason") == "nl_or_multi_token"
