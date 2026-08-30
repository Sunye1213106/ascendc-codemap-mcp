# -*- coding: utf-8 -*-
"""resolve(symbol) projects Assignments, Host→Kernel, and Compiled support."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _insert_entity, _insert_rel


def test_resolve_symbol_lists_assignments(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fld",
            kind="TILING_FIELD",
            name="isBn2MultiBlk",
            file="op_host/arch35/common_regbase.cpp",
            line=45,
        )
        _insert_entity(
            conn,
            eid="fn_split",
            kind="FUNCTION",
            name="SetSplitAxis",
            file="op_host/arch35/common_regbase.cpp",
            line=32,
            line_end=59,
        )
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="FUNCTION",
            name="DoSparse",
            file="op_host/arch35/normal_regbase.cpp",
            line=100,
            line_end=120,
        )
        _insert_rel(
            conn,
            rid="w1",
            kind="WRITES",
            src="fn_split",
            dst="fld",
            file="op_host/arch35/common_regbase.cpp",
            line=45,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w1'",
            (
                json.dumps(
                    {
                        "file": "op_host/arch35/common_regbase.cpp",
                        "line": 45,
                        "rhs": "bnSparseLimit && !hasRope",
                    }
                ),
            ),
        )
        _insert_rel(
            conn,
            rid="w2",
            kind="WRITES",
            src="fn_split",
            dst="fld",
            file="op_host/arch35/common_regbase.cpp",
            line=50,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w2'",
            (
                json.dumps(
                    {
                        "file": "op_host/arch35/common_regbase.cpp",
                        "line": 50,
                        "rhs": "false",
                    }
                ),
            ),
        )
        _insert_entity(
            conn,
            eid="br",
            kind="BRANCH",
            name="dropMaskOuter",
            file="op_host/arch35/common_regbase.cpp",
            line=50,
        )
        _insert_rel(
            conn,
            rid="g1",
            kind="GUARDED_BY",
            src="w2",
            dst="br",
            file="op_host/arch35/common_regbase.cpp",
            line=50,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='g1'")
        _insert_rel(
            conn,
            rid="w3",
            kind="WRITES",
            src="fn_sparse",
            dst="fld",
            file="op_host/arch35/normal_regbase.cpp",
            line=109,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w3'",
            (
                json.dumps(
                    {
                        "file": "op_host/arch35/normal_regbase.cpp",
                        "line": 109,
                        "rhs": "false",
                    }
                ),
            ),
        )
        _insert_entity(
            conn,
            eid="fn_ws",
            kind="FUNCTION",
            name="GetWorkspaceSize",
            file="op_host/arch35/normal_regbase.cpp",
            line=200,
        )
        _insert_rel(
            conn,
            rid="rd1",
            kind="READS",
            src="fn_ws",
            dst="fld",
            file="op_host/arch35/normal_regbase.cpp",
            line=210,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='rd1'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="isBn2MultiBlk",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "Assignments" in text
    assert "3/3" in text
    assert "exhaustive" in text.lower()
    assert "SetSplitAxis" in text
    assert "45" in text
    assert "bnSparseLimit" in text
    assert "dropMaskOuter" in text
    assert "DoSparse" in text
    assert "GetWorkspaceSize" in text


def test_resolve_symbol_host_kernel_coverage(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="fld",
            kind="TILING_FIELD",
            name="deterMaxRound",
            file="op_kernel/tiling_data.h",
            line=8,
        )
        _insert_entity(
            conn,
            eid="fn_cal",
            kind="FUNCTION",
            name="CalcleBandDeterParam",
            file="op_host/arch35/deter.cpp",
            line=20,
        )
        _insert_entity(
            conn,
            eid="fn_sched",
            kind="FUNCTION",
            name="DetermineBlockSchedule",
            file="op_host/arch35/deter.cpp",
            line=80,
        )
        _insert_entity(
            conn,
            eid="td",
            kind="TYPE",
            name="BaseDeterParam",
            file="op_kernel/tiling_data.h",
            line=1,
        )
        _insert_entity(
            conn,
            eid="kn",
            kind="METHOD",
            name="CalDeterMaxLoopNum",
            file="op_kernel/arch35/kernel_deter.h",
            line=40,
        )
        _insert_rel(
            conn,
            rid="w1",
            kind="WRITES",
            src="fn_cal",
            dst="fld",
            file="op_host/arch35/deter.cpp",
            line=24,
        )
        _insert_rel(
            conn,
            rid="w2",
            kind="WRITES",
            src="fn_sched",
            dst="fld",
            file="op_host/arch35/deter.cpp",
            line=90,
        )
        _insert_rel(
            conn,
            rid="rd1",
            kind="READS",
            src="kn",
            dst="fld",
            file="op_kernel/arch35/kernel_deter.h",
            line=44,
        )
        _insert_rel(
            conn,
            rid="b1",
            kind="BINDS",
            src="fld",
            dst="td",
            file="op_kernel/tiling_data.h",
            line=8,
        )
        for rid in ("w1", "w2", "rd1", "b1"):
            conn.execute(f"UPDATE relation SET status='confirmed' WHERE id='{rid}'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="deterMaxRound",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "Host producers" in text
    assert "CalcleBandDeterParam" in text
    assert "DetermineBlockSchedule" in text
    assert "Transport" in text
    assert "BaseDeterParam" in text
    assert "Kernel consumers" in text
    assert "CalDeterMaxLoopNum" in text
    assert "producers" in text.lower()
    assert "exhaustive" in text.lower()


def test_resolve_symbol_compiled_from_legal_keys(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="var",
            kind="COMPILE_VAR",
            name="hasRope",
            file="op_host/arch35/tiling.cpp",
            line=10,
        )
        _insert_entity(
            conn,
            eid="dim",
            kind="TILING_KEY",
            name="IsRope",
            file="op_host/arch35/tiling.cpp",
            line=12,
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (1, 'k1', '', '', 'ok')"
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (2, 'k2', '', '', 'ok')"
        )
        conn.execute("INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'IsRope', '1')")
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'DTemplate', '192')"
        )
        conn.execute("INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, 'IsRope', '0')")
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, 'DTemplate', '128')"
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="hasRope"
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "Compiled" in text
    assert "IsRope" in text or "hasRope" in text
    assert "legal" in text.lower()
    assert "192" in text or "1" in text
