# -*- coding: utf-8 -*-
"""C6 dialect micro: agent phrases, not source-true names."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity


def _seed(op: Path) -> None:
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "op_kernel/arch35/sel.h",
                    55,
                    "using TYPE = MutexBuffersPolicy4buff<BufferType::L1>;",
                ),
                (
                    "op_kernel/arch35/sel.h",
                    57,
                    "MutexBuffersPolicy3buff<BufferType::L1>, MutexBuffersPolicyDB<BufferType::L1>",
                ),
                (
                    "op_kernel/arch35/block.h",
                    666,
                    "DataCopy(qL1Tensor, this->queryGm[offset], nd2NzParams);",
                ),
                (
                    "op_kernel/arch35/block.h",
                    670,
                    "DataCopyPad(mmUb, gmIn, padParams);",
                ),
                (
                    "op_host/tiling.cpp",
                    80,
                    "if (keepProb < 1.0f) { tiling.set_dropMask(dropMask); }",
                ),
                (
                    "op_kernel/arch35/sync.h",
                    9,
                    "SetFlag<HardEvent::MTE3_S>(eventId);",
                ),
            ],
        )
        _insert_entity(
            conn,
            eid="buf_l1",
            kind="BUFFER",
            name="qL1Buffer",
            file="op_kernel/arch35/block.h",
            line=20,
            data=json.dumps({"memory_space": "L1", "physical_space": "L1"}),
        )
        _insert_entity(
            conn,
            eid="buf_l1b",
            kind="BUFFER",
            name="kL1Buffer",
            file="op_kernel/arch35/block.h",
            line=22,
            data=json.dumps({"memory_space": "L1", "physical_space": "L1"}),
        )
        _insert_entity(
            conn,
            eid="buf_ub",
            kind="BUFFER",
            name="mmUb",
            file="op_kernel/arch35/block.h",
            line=24,
            data=json.dumps({"memory_space": "UB"}),
        )
        _insert_entity(
            conn,
            eid="ty_db",
            kind="TYPE",
            name="MutexBuffersPolicyDB",
            file="cube_api/mutex_buffers_policy.h",
            line=74,
        )
        _insert_entity(
            conn,
            eid="ev_hard",
            kind="EVENT",
            name="MTE3_S",
            file="op_kernel/arch35/sync.h",
            line=9,
            data=json.dumps({"mechanism": "hard_event"}),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")


def _search(op: Path, name: str, **kwargs) -> str:
    payload = query(
        project=str(op), architecture="arch35", operation="search", name=name, **kwargs
    )
    return str((payload.get("data") or {}).get("text") or "")


def test_dialect_micro_hits_without_true_names(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _seed(op)
    cases = [
        ("L1", (), "qL1"),
        ("4buff|3buff", (), "MutexBuffersPolicy"),
        ("DataCopy(Pad)?", (), "DataCopy"),
        ("keepProb|dropMask", (), "dropMask"),
        ("HardEvent", (), "SetFlag"),
    ]
    zero = 0
    useful = 0
    for name, extra, needle in cases:
        text = _search(op, name, **dict(extra))
        if needle.lower() not in text.lower() and "0 matches" in text:
            zero += 1
            continue
        assert needle.lower() in text.lower(), (name, text[:400])
        useful += 1
    assert zero == 0
    assert useful == len(cases)

    l1_bufs = _search(op, "L1", kind="BUFFER")
    assert "qL1Buffer" in l1_bufs
    assert "kL1Buffer" in l1_bufs
    assert "mmUb" not in l1_bufs

    db = _search(op, "double buffer")
    assert "0 matches" in db or "0 source matches" in db
    assert "Symbols" in db
    assert "MutexBuffersPolicyDB" in db
    assert "matched alias" in db

    ev = _search(op, "hard_event", kind="EVENT")
    assert "MTE3_S" in ev
    hard = _search(op, "hard event")
    assert "Symbols" in hard
    assert "MTE3_S" in hard
