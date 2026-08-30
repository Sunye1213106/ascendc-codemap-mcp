# -*- coding: utf-8 -*-
"""search pattern= is regex over source_line. name= is a silent alias. FTS is an accelerator, not a dialect."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines


def _ready(op: Path, rows: list[tuple[str, int, str]]) -> None:
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(conn, rows)
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")


def test_search_short_literal_and_regex(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _ready(
        op,
        [
            ("op_kernel/arch35/block.h", 10, "TBuf<TPosition::A1> qL1Buffer;"),
            ("op_kernel/arch35/mutex.h", 74, "using MutexBuffersPolicy4buff = ...;"),
            ("op_kernel/arch35/mutex.h", 80, "using MutexBuffersPolicy3buff = ...;"),
            ("op_kernel/arch35/copy.h", 3, "DataCopy(dst, src, size);"),
            ("op_kernel/arch35/copy.h", 4, "DataCopyPad(dst, src, size);"),
        ],
    )
    short = query(project=str(op), architecture="arch35", operation="search", name="L1")
    text = str((short.get("data") or {}).get("text") or "")
    assert "0 matches" not in text.splitlines()[:1]
    assert "qL1Buffer" in text
    assert "verdict:" not in text
    assert "UNKNOWN" not in text
    assert "Graph search is not regex" not in text

    alts = query(
        project=str(op), architecture="arch35", operation="search", name="4buff|3buff"
    )
    atext = str((alts.get("data") or {}).get("text") or "")
    assert "MutexBuffersPolicy4buff" in atext
    assert "MutexBuffersPolicy3buff" in atext

    both = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="DataCopy(Pad)?",
    )
    btext = str((both.get("data") or {}).get("text") or "")
    assert "DataCopy(" in btext
    assert "DataCopyPad" in btext


def test_search_file_glob_and_zero_is_success(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _ready(
        op,
        [
            ("op_kernel/arch35/block.h", 10, "TBuf<TPosition::A1> qL1Buffer;"),
            ("op_host/tiling.cpp", 20, "TBuf<TPosition::A1> hostL1;"),
        ],
    )
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="L1",
        file="op_kernel/**/*.h",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "block.h:10" in text
    assert "tiling.cpp" not in text

    empty = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="NoSuchPhraseZZZ",
    )
    etext = str((empty.get("data") or {}).get("text") or "")
    assert etext.startswith("0 matches")
    assert "UNKNOWN" not in etext
    assert empty.get("ok") is True


def test_search_pages_same_result_set(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    rows = [
        (f"op_host/file_{i:02d}.cpp", i + 1, f"dropout_flag_{i:02d} = keepProb;")
        for i in range(25)
    ]
    _ready(op, rows)
    page1 = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="keepProb",
        limit=20,
    )
    data1 = page1.get("data") or {}
    text1 = str(data1.get("text") or "")
    assert "25 matches" in text1
    assert "showing 20/25" in text1
    assert "next_cursor=" in text1
    cursor = page1.get("next_cursor") or ""
    assert cursor
    page2 = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="keepProb",
        limit=20,
        cursor=cursor,
    )
    text2 = str((page2.get("data") or {}).get("text") or "")
    assert "25 matches" in text2
    assert "showing 5/25" in text2 or "5 matches" in text2
    assert "file_00.cpp" not in text2
    assert "file_24.cpp" in text2
    assert page2.get("next_cursor") in (None, "")
