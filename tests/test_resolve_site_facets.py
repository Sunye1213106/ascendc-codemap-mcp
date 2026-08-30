# -*- coding: utf-8 -*-
"""resolve(file,line) is source-centric; facets are confirmed-only."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines, _insert_entity, _insert_rel


def test_resolve_file_line_lists_site_entities(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "op_kernel/arch35/block.h",
                    40,
                    "if (flag) { DataCopy(dst, src, size); }",
                )
            ],
        )
        _insert_entity(
            conn, eid="fn1", kind="FUNCTION", name="Kernel", file="op_kernel/arch35/block.h", line=1, line_end=80
        )
        _insert_entity(
            conn, eid="br1", kind="BRANCH", name="flag", file="op_kernel/arch35/block.h", line=40
        )
        _insert_entity(
            conn,
            eid="op1",
            kind="OPERATION",
            name="DataCopy",
            file="op_kernel/arch35/block.h",
            line=40,
            data=json.dumps({"callee": "DataCopy"}),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file="op_kernel/arch35/block.h",
        line=40,
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "verdict:" not in text
    assert "completeness:" not in text
    assert "DataCopy" in text
    assert "BRANCH" in text or "flag" in text
    assert "At this site" in text or "DataCopy" in text


def test_storage_and_controls_facets(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="view1",
            kind="BUFFER",
            name="qL1Tensor",
            file="op_kernel/k.h",
            line=10,
            snippet="LocalTensor<half> qL1Tensor = qL1Buffer.Get<half>();",
        )
        _insert_entity(
            conn,
            eid="buf1",
            kind="BUFFER",
            name="qL1Buffer",
            file="op_kernel/k.h",
            line=8,
            data=json.dumps({"memory_space": "L1", "physical_space": "L1", "allocated": True}),
            snippet="TBuf<TPosition::A1> qL1Buffer;",
        )
        _insert_entity(
            conn,
            eid="ty1",
            kind="TYPE",
            name="AscendC::LocalTensor",
            file="cann.h",
            line=1,
        )
        _insert_entity(
            conn,
            eid="var1",
            kind="COMPILE_VAR",
            name="IS_SMALL_D_PRELOAD",
            file="op_kernel/k.h",
            line=2,
            snippet="constexpr static bool IS_SMALL_D_PRELOAD = true;",
        )
        _insert_entity(
            conn,
            eid="pol1",
            kind="TYPE",
            name="MutexBuffersPolicy4buff",
            file="op_kernel/k.h",
            line=50,
        )
        _insert_rel(
            conn,
            rid="r_back",
            kind="BACKED_BY",
            src="view1",
            dst="buf1",
            file="op_kernel/k.h",
            line=10,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='r_back'",
            (json.dumps({"file": "op_kernel/k.h", "line": 10}),),
        )
        _insert_rel(
            conn,
            rid="r_inst",
            kind="INSTANCE_OF",
            src="view1",
            dst="ty1",
            file="op_kernel/k.h",
            line=10,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='r_inst'")
        _insert_rel(
            conn,
            rid="r_ctrl",
            kind="CONTROLS",
            src="var1",
            dst="pol1",
            file="op_kernel/k.h",
            line=50,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='r_ctrl'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    stor = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="qL1Tensor"
    )
    text = str((stor.get("data") or {}).get("text") or "")
    assert "Storage" in text
    assert "qL1Buffer" in text
    assert "L1" in text
    assert "LocalTensor" in text
    assert "verdict:" not in text

    ctrl = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="IS_SMALL_D_PRELOAD",
    )
    ctext = str((ctrl.get("data") or {}).get("text") or "")
    assert "Controls" in ctext
    assert "MutexBuffersPolicy4buff" in ctext


def test_memory_facet_reports_coverage(tmp_path: Path) -> None:
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
        for i, (rid, dst, data) in enumerate(
            (
                ("r1", "ub1", {"via": "MemoryTransfer", "src_space": "GM", "dst_space": "UB"}),
                ("r2", "ub1", {"via": "MemoryTransfer", "src_space": "GM", "dst_space": "UB"}),
                ("r3", "gm1", {"via": "MemoryTransfer"}),
            ),
            start=1,
        ):
            _insert_rel(
                conn,
                rid=rid,
                kind="FLOWS_TO",
                src="fn1",
                dst=dst,
                file="op_kernel/k.h",
                line=20 + i,
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
    assert "Memory" in text
    assert "2/3 transfers resolved" in text
    assert "GM → UB" in text
    assert "1 unresolved endpoints" in text


def test_used_by_aggregates_callees(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn, eid="buf1", kind="BUFFER", name="qL1Buffer", file="op_kernel/k.h", line=8
        )
        for i in range(3):
            eid = f"op{i}"
            _insert_entity(
                conn,
                eid=eid,
                kind="OPERATION",
                name="DataCopy",
                file="op_kernel/k.h",
                line=30 + i,
                data=json.dumps({"callee": "DataCopy"}),
            )
            _insert_rel(
                conn,
                rid=f"rd{i}",
                kind="READS",
                src=eid,
                dst="buf1",
                file="op_kernel/k.h",
                line=30 + i,
            )
            conn.execute(f"UPDATE relation SET status='confirmed' WHERE id='rd{i}'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="qL1Buffer"
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "Used by" in text
    assert "DataCopy" in text
    assert "×3" in text
