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

_HOST = "op_host/arch35/common_regbase.cpp"


def _split_axis_fixture(op: Path) -> None:
    dest = write_uo_fixture(op)
    src = [
        (_HOST, 10, "bool SetSparseParams(FuzzyBaseInfoParamsRegbase &fBaseParams)"),
        (_HOST, 11, "{"),
        * [(_HOST, n, f"    // sparse body {n}") for n in range(12, 29)],
        (_HOST, 29, "    return false;"),
        (_HOST, 30, "}"),
        (_HOST, 32, "void SetSplitAxis(const gert::TilingContext *context_)"),
        (_HOST, 33, "{"),
        (_HOST, 34, "    bool hasRope = fBaseParams.hasRope;"),
        * [(_HOST, n, f"    // split body {n}") for n in range(35, 44)],
        (
            _HOST,
            45,
            "    fBaseParams.isBn2MultiBlk = bnSparseLimit && !hasRope;",
        ),
        * [(_HOST, n, f"    // split mid {n}") for n in range(46, 49)],
        (_HOST, 50, "    if (dropMaskOuter) { fBaseParams.isBn2MultiBlk = false; }"),
        * [(_HOST, n, f"    // split tail {n}") for n in range(51, 59)],
        (_HOST, 59, "}"),
    ]
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, src)
        _insert_entity(
            conn,
            eid="fn_sparse",
            kind="FUNCTION",
            name="SetSparseParams",
            file=_HOST,
            line=10,
            line_end=30,
        )
        _insert_entity(
            conn,
            eid="fn_split",
            kind="FUNCTION",
            name="SetSplitAxis",
            file=_HOST,
            line=32,
            line_end=59,
        )
        _insert_entity(
            conn,
            eid="fld_bn2",
            kind="TILING_FIELD",
            name="isBn2MultiBlk",
            file=_HOST,
            line=45,
        )
        _insert_entity(
            conn,
            eid="fld_rope",
            kind="FIELD",
            name="hasRope",
            file=_HOST,
            line=34,
        )
        _insert_rel(
            conn,
            rid="w_cand",
            kind="WRITES",
            src="fn_split",
            dst="fld_bn2",
            file=_HOST,
            line=45,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w_cand'",
            (
                json.dumps(
                    {
                        "file": _HOST,
                        "line": 45,
                        "rhs": "bnSparseLimit && !hasRope",
                    }
                ),
            ),
        )
        _insert_rel(
            conn,
            rid="w_drop",
            kind="WRITES",
            src="fn_split",
            dst="fld_bn2",
            file=_HOST,
            line=50,
        )
        conn.execute(
            "UPDATE relation SET status='confirmed', data=? WHERE id='w_drop'",
            (
                json.dumps(
                    {
                        "file": _HOST,
                        "line": 50,
                        "rhs": "false",
                    }
                ),
            ),
        )
        _insert_entity(
            conn,
            eid="br_drop",
            kind="BRANCH",
            name="dropMaskOuter",
            file=_HOST,
            line=50,
        )
        _insert_rel(
            conn,
            rid="g_drop",
            kind="GUARDED_BY",
            src="w_drop",
            dst="br_drop",
            file=_HOST,
            line=50,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='g_drop'")
        _insert_rel(
            conn,
            rid="c_rope",
            kind="CONTROLS",
            src="fld_rope",
            dst="fn_split",
            file=_HOST,
            line=34,
        )
        conn.execute("UPDATE relation SET status='confirmed' WHERE id='c_rope'")
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")


def test_resolve_file_line_returns_enclosing_function(tmp_path: Path) -> None:
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
    assert "SetSplitAxis" in text
    assert f"{_HOST}:32-59" in text or f"{_HOST}:32–59" in text
    assert "32|" in text
    assert "59|" in text
    assert "45|" in text
    assert "50|" in text
    assert "SetSparseParams" not in text.split("Source", 1)[0]
    assert "State changes" in text
    assert "isBn2MultiBlk" in text
    assert "1673" not in text
    assert "bnSparseLimit" in text
    assert "false" in text
    assert "dropMaskOuter" in text


def test_resolve_file_line_stays_in_enclosing_function(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _split_axis_fixture(op)
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        file=_HOST,
        line=20,
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "SetSparseParams" in text
    assert "10|" in text
    assert "30|" in text
    assert "void SetSplitAxis" not in text
    assert text.count("SetSplitAxis") == 0


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
