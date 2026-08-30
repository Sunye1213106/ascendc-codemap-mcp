# -*- coding: utf-8 -*-
"""search + kind greps entity-local fields. No entity_search table."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _insert_entity


def _ready(op: Path, entities: list[dict]) -> None:
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for row in entities:
            data = dict(row.get("data") or {})
            _insert_entity(
                conn,
                eid=row["eid"],
                kind=row["kind"],
                name=row["name"],
                file=row["file"],
                line=row["line"],
                data=json.dumps(data),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")


def test_search_kind_matches_local_attrs(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _ready(
        op,
        [
            {
                "eid": "buf_l1",
                "kind": "BUFFER",
                "name": "qL1Buffer",
                "file": "op_kernel/arch35/block.h",
                "line": 20,
                "data": {"memory_space": "L1", "physical_space": "L1"},
            },
            {
                "eid": "buf_ub",
                "kind": "BUFFER",
                "name": "vecInPing",
                "file": "op_kernel/arch35/block.h",
                "line": 40,
                "data": {"memory_space": "UB"},
            },
            {
                "eid": "op_init",
                "kind": "OPERATION",
                "name": "InitBuffer",
                "file": "op_kernel/arch35/block.h",
                "line": 80,
                "data": {"callee": "InitBuffer", "category": "alloc"},
            },
            {
                "eid": "ty_sel",
                "kind": "TYPE",
                "name": "QL1BuffSelector",
                "file": "op_kernel/arch35/sel.h",
                "line": 3,
                "data": {},
            },
            {
                "eid": "ev_hard",
                "kind": "EVENT",
                "name": "MTE3_S",
                "file": "op_kernel/arch35/sync.h",
                "line": 9,
                "data": {"mechanism": "hard_event", "event_type": "MTE3_S"},
            },
        ],
    )
    buf = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="L1",
        kind="BUFFER",
    )
    text = str((buf.get("data") or {}).get("text") or "")
    assert "qL1Buffer" in text
    assert "BUFFER" in text
    assert "vecInPing" not in text

    ops = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="InitBuffer",
        kind="OPERATION",
    )
    assert "InitBuffer" in str((ops.get("data") or {}).get("text") or "")

    types = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="BuffSelector",
        kind="TYPE",
    )
    assert "QL1BuffSelector" in str((types.get("data") or {}).get("text") or "")

    ev = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="hard_event",
        kind="EVENT",
    )
    assert "MTE3_S" in str((ev.get("data") or {}).get("text") or "")


def test_search_alias_is_symbol_not_source(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _ready(
        op,
        [
            {
                "eid": "ty_db",
                "kind": "TYPE",
                "name": "MutexBuffersPolicyDB",
                "file": "cube_api/mutex_buffers_policy.h",
                "line": 74,
                "data": {},
            }
        ],
    )
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="double buffer",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert text.startswith("0 matches") or "0 source matches" in text
    assert "Symbols" in text
    assert "MutexBuffersPolicyDB" in text
    assert "matched alias" in text
    assert "source matches:" not in text.lower() or "0 source" in text
