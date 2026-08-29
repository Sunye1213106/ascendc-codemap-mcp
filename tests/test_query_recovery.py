# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import write_uo_fixture
from ascendc_codemap_mcp.engine.query.sql import _ident_tokens, _recovery_tokens
from ascendc_codemap_mcp.engine.query.typed import InvalidQuery, validate_plan
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query


def test_recovery_tokens_split_camel_snake_digits_abbrev() -> None:
    assert "buffer" in _ident_tokens("BufferNum")
    assert "num" in _ident_tokens("BufferNum")
    assert "buffer" in _ident_tokens("buffer_num")
    assert "buffer" in _ident_tokens("GetQBufNum") or "buf" in _ident_tokens("GetQBufNum")
    assert "policy" in _ident_tokens("Policy4buff")
    assert "buffer" in _recovery_tokens("GetQBufNum")
    assert "buffer" in _recovery_tokens("Policy4buff")


def test_resolve_drops_extra_name_when_symbol_is_set() -> None:
    plan = validate_plan(operation="resolve", symbol="FooPolicy", name="FooPolicy")
    assert plan.symbol == "FooPolicy"
    assert plan.name == ""
    assert "name" in plan.dropped


def test_resolve_name_only_does_not_suggest_dim() -> None:
    try:
        validate_plan(operation="resolve", name="GetTilingKey")
    except InvalidQuery as exc:
        assert exc.did_you_mean
        assert all("dim" not in c for c in exc.did_you_mean)
        assert any(c.get("operation") == "find" and c.get("name") == "GetTilingKey" for c in exc.did_you_mean)
    else:
        raise AssertionError("expected INVALID_QUERY")


def test_entity_id_that_is_a_name_becomes_symbol() -> None:
    plan = validate_plan(operation="resolve", entity_id="FooPolicy")
    assert plan.symbol == "FooPolicy"
    assert plan.entity_id == ""


def _insert(
    conn: sqlite3.Connection,
    *,
    eid: str,
    kind: str,
    name: str,
    file: str,
    line: int,
    line_end: int,
    data: str = "{}",
    snippet: str = "",
) -> None:
    conn.execute(
        "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (eid, kind, name, "extracted", 1.0, file, line, line_end, data),
    )
    conn.execute(
        "INSERT INTO source_span(id, entity_id, file, line_start, line_end, snippet) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"span:{eid}", eid, file, line, line_end, snippet or name),
    )


def test_name_miss_points_to_search(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert(
            conn,
            eid="buf1",
            kind="TYPE",
            name="FooBuffer",
            file="op_kernel/buffer.h",
            line=10,
            line_end=12,
            snippet="class FooBuffer {};",
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")

    for needle in ("BufferNum", "buffer_num", "GetQBufNum"):
        payload = query(
            project=str(op),
            architecture="arch35",
            operation="find",
            name=needle,
        )
        data = payload.get("data") or {}
        text = str(data.get("text") or "")
        hint = str(data.get("hint") or "")
        assert "FooBuffer" not in text
        assert payload.get("verdict") == "UNKNOWN" or data.get("completeness") == "UNKNOWN"
        assert "search name=" in hint
        assert "showing buffer" not in hint.lower()
        assert "**Dims**" not in text


def test_type_definition_and_use_are_not_ambiguous(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    api = op / "cube_api"
    kernel = op / "op_kernel"
    api.mkdir()
    kernel.mkdir()
    (api / "api.h").write_text(
        "#pragma once\n"
        "class FooPolicy {\n"
        "public:\n"
        "    static constexpr int kBuf = 4;\n"
        "};\n",
        encoding="utf-8",
    )
    (kernel / "kernel.h").write_text(
        "#pragma once\n"
        "using Selected = std::conditional_t<true, FooPolicy, int>;\n",
        encoding="utf-8",
    )
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _insert(
            conn,
            eid="ty_def",
            kind="TYPE",
            name="FooPolicy",
            file="cube_api/api.h",
            line=2,
            line_end=5,
            data='{"cpp_kind":"class"}',
            snippet="class FooPolicy {\npublic:\n    static constexpr int kBuf = 4;\n};",
        )
        _insert(
            conn,
            eid="ty_use",
            kind="TYPE",
            name="FooPolicy",
            file="op_kernel/kernel.h",
            line=2,
            line_end=2,
            snippet="using Selected = std::conditional_t<true, FooPolicy, int>;",
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="FooPolicy",
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    assert data.get("completeness") != "AMBIGUOUS"
    assert data.get("unresolved_reason") != "MULTIPLE_SEEDS"
    assert "**References**" in text
    assert "kernel.h" in text

    filtered = query(
        project=str(op),
        architecture="arch35",
        operation="resolve",
        symbol="FooPolicy",
        file="op_kernel/kernel.h",
    )
    ftext = str((filtered.get("data") or {}).get("text") or "")
    assert "conditional_t" in ftext
    src_block = ftext.split("**References**")[0] if "**References**" in ftext else ftext
    assert "kernel.h" in src_block


def test_find_exact_operation_lists_call_sites(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        for i, (path, line) in enumerate(
            (("op_kernel/block_vec.h", 20), ("op_kernel/pre_regbase.h", 40)),
            start=2,
        ):
            _insert(
                conn,
                eid=f"op{i}",
                kind="OPERATION",
                name="SyncAll",
                file=path,
                line=line,
                line_end=line,
                data='{"callee":"SyncAll","layer":"kernel"}',
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
            name="SyncAll",
        ).get("data")
        or {}
    )
    text = str(data.get("text") or "")
    assert "**Call sites**" in text
    assert "block_vec.h:20" in text
    assert "pre_regbase.h:40" in text
    assert "already listed" in str(data.get("hint") or "").lower() or "Call sites" in text
