# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture
from ascendc_codemap_mcp.engine.query.typed import InvalidQuery, validate_plan
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query


def test_validate_plan_rejects_nl_symbol() -> None:
    try:
        validate_plan(operation="resolve", symbol="quad buffer Q")
    except InvalidQuery as exc:
        assert "symbol" in str(exc)
        assert "Q" in exc.parsed_tokens or "quad" in exc.parsed_tokens
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_validate_plan_find_requires_kind() -> None:
    try:
        validate_plan(operation="find", callee="SyncAll")
    except InvalidQuery as exc:
        assert "kind" in str(exc).lower()
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_find_accepts_name_pattern_without_kind() -> None:
    plan = validate_plan(operation="find", name="*BuffSelector*")
    assert plan.name == "*BuffSelector*"
    assert plan.kind == ""


def test_search_requires_name() -> None:
    try:
        validate_plan(operation="search")
    except InvalidQuery as exc:
        assert "name" in str(exc).lower()
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_search_accepts_name_and_file() -> None:
    plan = validate_plan(operation="search", name="BufferNum", file="block_cube.h")
    assert plan.operation == "search"
    assert plan.name == "BufferNum"
    assert plan.file == "block_cube.h"


def test_illegal_filter_suggests_legal_rebinding() -> None:
    try:
        validate_plan(operation="find", kind="FUNCTION", symbol="SyncALLCores")
    except InvalidQuery as exc:
        calls = exc.did_you_mean
        assert calls, "expected a repaired call"
        assert all(c.get("operation") == "find" for c in calls)
        assert all(c.get("kind") == "FUNCTION" for c in calls)
        assert {"callee", "function", "name"} & set().union(*(set(c) for c in calls))
        assert all("symbol" not in c for c in calls)
        assert all("SyncALLCores" in c.values() for c in calls)
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_trace_with_one_endpoint_suggests_impact() -> None:
    try:
        validate_plan(operation="trace", from_symbol="SyncALLCores")
    except InvalidQuery as exc:
        ops = {c.get("operation") for c in exc.did_you_mean}
        assert "impact" in ops
        assert "find" in ops
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_find_returns_set_not_ambiguous(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i, line in enumerate((10, 20, 30), start=2):
            conn.execute(
                "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"op{i}",
                    "OPERATION",
                    "SyncAll",
                    "extracted",
                    1.0,
                    "op_kernel/block_vec.h" if i == 2 else "op_kernel/pre_regbase.h",
                    line,
                    line,
                    '{"callee":"SyncAll","layer":"kernel"}',
                ),
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
    assert payload.get("ok") is True
    data = payload.get("data") or {}
    cards = data.get("cards") or []
    assert len(cards) >= 2
    assert data.get("completeness") != "AMBIGUOUS"
    files = {str(c.get("file") or "") for c in cards if isinstance(c, dict)}
    assert any("block_vec.h" in f for f in files)

    resolve = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="SyncAll",
    )
    resolve_data = resolve.get("data") or {}
    assert resolve_data.get("completeness") == "AMBIGUOUS" or resolve.get("verdict") in {
        "PARTIAL",
        "UNKNOWN",
    }
    assert resolve_data.get("unresolved_reason") == "MULTIPLE_SEEDS" or len(
        resolve_data.get("cards") or []
    ) <= 1


def test_name_discovery_lists_idents_without_source(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i, name in enumerate(("QL1BuffSelector", "KL1BuffSelector"), start=2):
            conn.execute(
                "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"ty{i}", "TYPE", name, "extracted", 1.0, "op_kernel/cube.h", i * 10, i * 10, "{}"),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        name="*BuffSelector*",
        limit=10,
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert payload.get("verdict") == "ANSWERED"
    assert "QL1BuffSelector" in text and "KL1BuffSelector" in text
    assert "Matches:" in text
    assert len(text) < 2000


def test_include_guard_never_outranks_a_real_seed(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("g1", "BRANCH", "DROPOUT_H", "extracted", 1.0, "op_kernel/dropout.h", 14, 16, "{}"),
        )
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "g2",
                "TILING_FIELD",
                "dropoutIsDivisibleBy8",
                "extracted",
                1.0,
                "op_kernel/tiling.h",
                310,
                310,
                "{}",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    text = str(
        (
            query(project=str(op), architecture="arch35", operation="resolve", symbol="dropout").get(
                "data"
            )
            or {}
        ).get("text")
        or ""
    )
    assert "dropoutIsDivisibleBy8" in text
    assert "DROPOUT_H" not in text
    exact = query(
        project=str(op), architecture="arch35", operation="resolve", symbol="DROPOUT_H"
    )
    assert "DROPOUT_H" in str((exact.get("data") or {}).get("text") or "")


def test_site_list_merges_overlapping_windows(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    src = op / "op_kernel"
    src.mkdir(parents=True, exist_ok=True)
    (src / "block_vec.h").write_text(
        "\n".join(f"line {n} SyncAll();" for n in range(1, 60)) + "\n", encoding="utf-8"
    )
    conn = sqlite3.connect(str(dest))
    try:
        for i, line in enumerate((20, 21, 22, 23), start=2):
            conn.execute(
                "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"sy{i}",
                    "OPERATION",
                    "SyncAll",
                    "extracted",
                    1.0,
                    "op_kernel/block_vec.h",
                    line,
                    line,
                    '{"callee":"SyncAll","layer":"kernel"}',
                ),
            )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    text = str(
        (
            query(
                project=str(op),
                architecture="arch35",
                operation="find",
                kind="OPERATION",
                callee="SyncAll",
                limit=10,
            ).get("data")
            or {}
        ).get("text")
        or ""
    )
    numbered = [ln.split("|", 1)[0] for ln in text.splitlines() if "|" in ln[:6]]
    assert len(numbered) == len(set(numbered)), "a source line must be printed once"


def test_ast_filter_without_snapshot_ast_is_invalid(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        kind="BRANCH",
        referenced_value="TND",
        literal="0",
    )
    assert payload.get("error_code") == "INVALID_QUERY"
    assert payload.get("legal_filters") or (payload.get("data") or {}).get("legal_filters")


def test_index_and_miss_still_list_dims(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "UPDATE entity SET data = ? WHERE name = ?",
            ('{"source_declared": true}', "IsPse"),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    index = query(project=str(op), architecture="arch35")
    text = str((index.get("data") or {}).get("text") or "")
    assert "IsPse" in text
    miss = query(project=str(op), architecture="arch35", symbol="IsDrop")
    assert miss.get("verdict") == "UNKNOWN"


def test_resolve_name_glob_suggests_find() -> None:
    try:
        validate_plan(operation="resolve", name="*Buffer*")
    except InvalidQuery as exc:
        assert exc.did_you_mean
        assert exc.did_you_mean[0] == {"operation": "find", "name": "*Buffer*"}
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_name_discovery_ranks_token_match_above_long_setters(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "fn1",
                "FUNCTION",
                "set_inputBufferLen",
                "extracted",
                1.0,
                "op_host/tiling.cpp",
                10,
                10,
                "{}",
            ),
        )
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ty1",
                "TYPE",
                "QL1BuffSelector",
                "extracted",
                1.0,
                "op_kernel/cube.h",
                53,
                60,
                "{}",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    data = (
        query(
            project=str(op),
            architecture="arch35",
            operation="find",
            name="*buf*",
            limit=8,
        ).get("data")
        or {}
    )
    names = [str(c.get("name") or "") for c in (data.get("cards") or []) if isinstance(c, dict)]
    assert "QL1BuffSelector" in names
    assert names.index("QL1BuffSelector") < names.index("set_inputBufferLen")


def test_name_miss_does_not_dump_dims(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        name="BufferNum",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert "BufferNum" in str(data.get("hint") or "") or "no ident" in text.lower() or "no ident" in str(
        data.get("hint") or ""
    ).lower()
    assert "**Dims**" not in text
    assert not data.get("dim_names")


def test_fileless_api_does_not_make_located_ident_ambiguous(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("api1", "FUNCTION", "set_inputBufferLen", "extracted", 1.0, "", 0, 0, "{}"),
        )
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m1",
                "METHOD",
                "set_inputBufferLen",
                "extracted",
                1.0,
                "op_kernel/tiling_data.h",
                384,
                384,
                "{}",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    data = (
        query(
            project=str(op),
            architecture="arch35",
            operation="resolve",
            symbol="set_inputBufferLen",
        ).get("data")
        or {}
    )
    assert data.get("completeness") != "AMBIGUOUS" or not any(
        not c.get("file") for c in (data.get("candidates") or []) if isinstance(c, dict)
    )
    text = str(data.get("text") or "")
    assert "tiling_data.h" in text or "384" in text

