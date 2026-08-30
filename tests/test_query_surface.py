# -*- coding: utf-8 -*-
"""Query surface: search / contract merge / cover matrix / OPERATION round-robin."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture
from ascendc_codemap_mcp.engine.store.accel import SOURCE_LINE_SQL, build_source_fts
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query


def _insert_entity(
    conn: sqlite3.Connection,
    *,
    eid: str,
    kind: str,
    name: str,
    file: str,
    line: int,
    line_end: int | None = None,
    data: str = "{}",
    snippet: str = "",
) -> None:
    end = line if line_end is None else line_end
    conn.execute(
        "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (eid, kind, name, "extracted", 1.0, file, line, end, data),
    )
    conn.execute(
        "INSERT INTO source_span(id, entity_id, file, line_start, line_end, snippet) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"span:{eid}", eid, file, line, end, snippet or name),
    )


def _insert_rel(
    conn: sqlite3.Connection,
    *,
    rid: str,
    kind: str,
    src: str,
    dst: str,
    file: str,
    line: int,
) -> None:
    conn.execute(
        "INSERT INTO relation(id, kind, src, dst, status, confidence, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            rid,
            kind,
            src,
            dst,
            "extracted",
            1.0,
            json.dumps({"file": file, "line": line}),
        ),
    )


def _add_source_lines(
    conn: sqlite3.Connection, rows: list[tuple[str, int, str]]
) -> None:
    conn.executescript(SOURCE_LINE_SQL)
    for path, line, text in rows:
        conn.execute(
            "INSERT INTO source_line(path, line, text) VALUES (?, ?, ?)",
            (path, line, text),
        )
    build_source_fts(conn)


def test_search_hits_source_line_not_entity_name(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "op_kernel/arch35/block_cube.h",
                    149,
                    "#define IS_SMALL_D_PRELOAD (!IS_DROP && QL1BufferNum == 4)",
                ),
                (
                    "op_kernel/arch35/block_cube.h",
                    150,
                    "int QL1BufferNum = 4;",
                ),
                (
                    "../common/op_kernel/attn_buffer.h",
                    10,
                    "class Buffer {};",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="BufferNum",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert payload.get("verdict") != "UNKNOWN"
    assert "block_cube.h:150" in text or "block_cube.h:149" in text
    assert "QL1BufferNum" in text or "BufferNum" in text
    assert "Top candidates:" not in text
    assert "FooBuffer" not in text

    miss = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        name="*BufferNum*",
    )
    mdata = miss.get("data") or {}
    assert miss.get("verdict") == "UNKNOWN" or (mdata.get("completeness") == "UNKNOWN")
    assert "search name=" in str(mdata.get("hint") or "")


def test_search_keepprob_and_zero_is_unknown(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "op_host/tiling_common_regbase.cpp",
                    1174,
                    "if (dropMask && keepProb >= 1.0) { return GRAPH_FAILED; }",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="search",
        name="keepProb",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "tiling_common_regbase.cpp:1174" in text
    assert "keepProb" in text

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


def test_contract_same_leaf_merges_host_and_kernel(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    host = op / "op_host"
    kernel = op / "op_kernel"
    host.mkdir(parents=True)
    kernel.mkdir()
    (host / "tiling.cpp").write_text(
        "void FuzzyForBestSplit() {\n    s1Inner = S1Template / 2;\n}\n",
        encoding="utf-8",
    )
    (kernel / "block_cube.h").write_text(
        "#pragma once\nCUBE_BASEM = s1Inner;\n",
        encoding="utf-8",
    )
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="tk_s1",
            kind="TILING_KEY",
            name="s1Inner",
            file="op_kernel/tiling_key.h",
            line=10,
            snippet="ASCENDC_TPL_UINT_DECL(s1Inner, 0, 1, 2),",
        )
        _insert_entity(
            conn,
            eid="tf_s1",
            kind="TILING_FIELD",
            name="s1Inner",
            file="op_host/occupancy.h",
            line=20,
            snippet="uint32_t s1Inner;",
        )
        _insert_entity(
            conn,
            eid="fn_fuzzy",
            kind="METHOD",
            name="FuzzyForBestSplit",
            file="op_host/tiling.cpp",
            line=1,
            line_end=3,
            snippet="s1Inner = S1Template / 2;",
        )
        _insert_entity(
            conn,
            eid="kn_cube",
            kind="KERNEL",
            name="flash_attention_score_grad",
            file="op_kernel/block_cube.h",
            line=2,
            snippet="CUBE_BASEM = s1Inner;",
        )
        _insert_rel(
            conn,
            rid="w1",
            kind="WRITES",
            src="fn_fuzzy",
            dst="tf_s1",
            file="op_host/tiling.cpp",
            line=2,
        )
        _insert_rel(
            conn,
            rid="r1",
            kind="READS",
            src="kn_cube",
            dst="tf_s1",
            file="op_kernel/block_cube.h",
            line=2,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="contract",
        symbol="s1Inner",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert data.get("completeness") != "AMBIGUOUS"
    assert "Host" in text
    assert "Kernel" in text
    assert "tiling.cpp" in text
    assert "block_cube.h" in text


def test_contract_kernel_getter_without_reads(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    host = op / "op_host"
    kernel = op / "op_kernel"
    host.mkdir(parents=True)
    kernel.mkdir()
    (host / "tiling.cpp").write_text(
        "void Pack() {\n    set_s1Inner(fBaseParams.s1Inner);\n}\n",
        encoding="utf-8",
    )
    (kernel / "tiling_data.h").write_text(
        "uint32_t s1Inner;\nuint32_t get_s1Inner() const { return s1Inner; }\n",
        encoding="utf-8",
    )
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="tf_s1",
            kind="TILING_FIELD",
            name="s1Inner",
            file="op_kernel/tiling_data.h",
            line=1,
            snippet="uint32_t s1Inner;",
        )
        _insert_entity(
            conn,
            eid="fn_pack",
            kind="METHOD",
            name="Pack",
            file="op_host/tiling.cpp",
            line=1,
            line_end=3,
            snippet="set_s1Inner(fBaseParams.s1Inner);",
        )
        _insert_entity(
            conn,
            eid="get_s1",
            kind="METHOD",
            name="get_s1Inner",
            file="op_kernel/tiling_data.h",
            line=2,
            snippet="uint32_t get_s1Inner() const { return s1Inner; }",
        )
        _insert_rel(
            conn,
            rid="w1",
            kind="WRITES",
            src="fn_pack",
            dst="tf_s1",
            file="op_host/tiling.cpp",
            line=2,
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="contract",
        symbol="s1Inner",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert data.get("completeness") != "AMBIGUOUS"
    assert "Host" in text
    assert "Kernel" in text
    assert "tiling.cpp" in text
    assert "tiling_data.h" in text
    assert "get_s1Inner" in text or ":2" in text


def test_resolve_compile_var_uses_statement_window(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    kernel = op / "op_kernel"
    kernel.mkdir(parents=True)
    (kernel / "block_cube.h").write_text(
        "\n" * 147
        + "#define IS_DROP 0\n"
        + "#define IS_SMALL_D_PRELOAD (!IS_DROP && (D_PRELOAD))\n"
        + "int dummy;\n",
        encoding="utf-8",
    )
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert_entity(
            conn,
            eid="cv1",
            kind="COMPILE_VAR",
            name="IS_SMALL_D_PRELOAD",
            file="op_kernel/block_cube.h",
            line=149,
            snippet="IS_SMALL_D_PRELOAD",
        )
        _add_source_lines(
            conn,
            [
                ("op_kernel/block_cube.h", 148, "#define IS_DROP 0"),
                (
                    "op_kernel/block_cube.h",
                    149,
                    "#define IS_SMALL_D_PRELOAD (!IS_DROP && (D_PRELOAD))",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="IS_SMALL_D_PRELOAD",
    )
    text = str((payload.get("data") or {}).get("text") or "")
    assert "!IS_DROP" in text or "D_PRELOAD" in text
    assert text.count("IS_SMALL_D_PRELOAD") >= 1


def test_cover_dim_only_counts_and_declared_zero(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op, symbol="InputDType")
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "UPDATE entity SET data = ? WHERE name = ?",
            (
                json.dumps({"value_domain": ["1", "2", "3", "4", "5", "6"]}),
                "InputDType",
            ),
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (1, 'a', '', '', 'ok')"
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (2, 'b', '', '', 'ok')"
        )
        conn.execute(
            "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (3, 'c', '', '', 'ok')"
        )
        for kid, dtype, swizzle in (
            (1, "1", "0"),
            (2, "2", "0"),
            (3, "3", "0"),
        ):
            conn.execute(
                "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (?, 'InputDType', ?)",
                (kid, dtype),
            )
            conn.execute(
                "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (?, 'IsTndSwizzle', ?)",
                (kid, swizzle),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    dim_only = query(
        project=str(op),
        architecture="arch35",
        dim="InputDType",
    )
    data = dim_only.get("data") or {}
    text = str(data.get("text") or "")
    assert int(data.get("legal_key_count") or 0) > 0
    assert "1:" in text or "1: " in text.replace(" ", "")
    assert "legal_key_count" in text

    zero = query(
        project=str(op),
        architecture="arch35",
        dim="InputDType",
        value="4",
    )
    zdata = zero.get("data") or {}
    ztext = str(zdata.get("text") or "") + " " + str(zdata.get("hint") or "")
    assert "legal_key=0" in ztext or int(zdata.get("legal_key_count") or 0) == 0
    assert "4" in ztext and "5" in ztext and "6" in ztext


def test_cover_cross_counts_include_zero(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op, symbol="IsTnd")
    conn = sqlite3.connect(str(dest))
    try:
        for kid in range(1, 4):
            conn.execute(
                "INSERT INTO legal_key(id, packed, hex, sel_group, status) VALUES (?, ?, '', '', 'ok')",
                (kid, f"k{kid}"),
            )
        # IsTnd=1 × swizzle=0 → 2 keys; IsTnd=1 × swizzle=1 → 0; IsTnd=0 × swizzle=1 → 1
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'IsTnd', '1')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'IsTndSwizzle', '0')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, 'IsTnd', '1')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, 'IsTndSwizzle', '0')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (1, 'DeterType', '3')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (2, 'DeterType', '3')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (3, 'DeterType', '3')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (3, 'IsTnd', '0')"
        )
        conn.execute(
            "INSERT INTO legal_key_dim(key_id, dim, value) VALUES (3, 'IsTndSwizzle', '1')"
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        dim="IsTnd",
        value="1",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    cross = data.get("cross_counts") or {}
    sw = cross.get("IsTndSwizzle") or {}
    assert str(sw.get("0") or sw.get(0) or "") == "2" or sw.get("0") == 2
    assert int(sw.get("1") or sw.get(1) or 0) == 0
    assert "IsTndSwizzle" in text
    assert "1: 0" in text or "1:0" in text.replace(" ", "")
    pair = data.get("cross_pair") or {}
    cells = pair.get("cells") or []
    deter3 = next(
        (c for c in cells if str(c.get("value")) == "3"),
        cells[0] if cells else {},
    )
    cmap = deter3.get("counts") if isinstance(deter3, dict) else {}
    if cmap:
        assert int(cmap.get("0") or cmap.get(0) or 0) == 2
        assert int(cmap.get("1") or cmap.get(1) or 0) == 0
    else:
        assert "DeterType × IsTndSwizzle" in text or "3 ×" in text


def test_find_operation_round_robin_by_file(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i in range(12):
            _insert_entity(
                conn,
                eid=f"op_a{i}",
                kind="OPERATION",
                name="SyncAll",
                file="op_kernel/block_vec.h",
                line=10 + i,
                data='{"callee":"SyncAll"}',
            )
        _insert_entity(
            conn,
            eid="op_b",
            kind="OPERATION",
            name="SyncAll",
            file="op_kernel/pre_regbase.h",
            line=40,
            data='{"callee":"SyncAll"}',
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        kind="OPERATION",
        callee="SyncAll",
        limit=8,
    )
    data = payload.get("data") or {}
    cards = [c for c in (data.get("cards") or []) if isinstance(c, dict)]
    files = {str(c.get("file") or "") for c in cards}
    assert len(files) >= 2
    assert any("block_vec.h" in f for f in files)
    assert any("pre_regbase.h" in f for f in files)
