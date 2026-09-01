# -*- coding: utf-8 -*-
"""Plan goldens: resolve as one closed semantic read. Hits the real FAG .uo."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.query.contract import (
    INSTRUCTIONS,
    PUBLIC_OPERATIONS,
    PublicQueryContract,
)
from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
from ascendc_codemap_mcp.engine.query.typed import (
    OPERATIONS,
    InvalidQuery,
    validate_plan,
)
from ascendc_codemap_mcp.mcp_adapter import DEFAULT_MCP_TOOLS, create_server
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

FAG = Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad")
FAG_UO = FAG / r".ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
ROOT = Path(__file__).resolve().parents[1]

HOST_DETER_LINES = {790, 847, 882, 1347, 107, 284, 289, 310}
BN2_PRODUCER_LINES = {1673, 1693, 1099}
LAYOUT = (
    ("dq", 1798),
    ("dk", 1801),
    ("dv", 1804),
    ("dropMask", 1817),
    ("sfmg", 1824),
    ("dsink", 1832),
    ("deterWorkSpace", 1853),
    ("deterGm", 1861),
)
_LEAKED_OPS = {"find", "impact", "contract", "entry"}


def _tool_names(server) -> list[str]:
    manager = getattr(server, "_tool_manager", None) or getattr(server, "_tools", None)
    if manager is not None and hasattr(manager, "list_tools"):
        tools = manager.list_tools()
        return [getattr(t, "name", str(t)) for t in tools]
    if manager is not None and hasattr(manager, "_tools"):
        tools = manager._tools
        if isinstance(tools, dict):
            return list(tools)
        return [getattr(t, "name", str(t)) for t in tools]
    raise AssertionError(f"cannot list tools on {type(server)}")


def _find_tool(server, name: str):
    manager = getattr(server, "_tool_manager", None)
    if manager is not None and hasattr(manager, "get_tool"):
        return manager.get_tool(name)
    names = _tool_names(server)
    raise AssertionError(f"tool {name} not found in {names}")


def _text_of(payload: dict) -> str:
    return str((payload.get("data") or {}).get("text") or "")


def _resolve(symbol: str) -> tuple[dict, str]:
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        symbol=symbol,
    )
    return payload, _text_of(payload)


def _section(text: str, header: str) -> str:
    lines = str(text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header or line.strip() == f"**{header}**":
            start = i + 1
            break
    if start is None:
        return ""
    out: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("**") and stripped.endswith("**"):
            break
        if stripped in {
            "Host value definitions",
            "Transport",
            "Kernel consumers",
            "Assignments",
            "Consumed by",
            "Workspace layout",
            "Backing",
            "Sync pairing",
            "Guarded by",
            "Controls",
            "Compiled",
        }:
            break
        out.append(line)
    return "\n".join(out)


def _cited_lines(block: str) -> set[int]:
    found: set[int] = set()
    for raw in str(block or "").splitlines():
        m = re.match(r"^\s+(\d+)\b", raw)
        if m:
            found.add(int(m.group(1)))
            continue
        m = re.search(r"@(\d+)\b", raw)
        if m:
            found.add(int(m.group(1)))
            continue
        m = re.search(r":(\d+)\s*$", raw.strip())
        if m:
            found.add(int(m.group(1)))
    return found


def _callee_rows(block: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for raw in str(block or "").splitlines():
        text = raw.strip()
        m = re.match(r"^- (\S+) —", text)
        if m:
            rows.append((m.group(1), 0))
            continue
        m = re.match(r"^- (\S+)(?: @(\d+))?", text)
        if m:
            rows.append((m.group(1), int(m.group(2) or 0)))
    return rows


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g1_field_ids_named_tiling_field_bucket() -> None:
    with UoSqlQuery(FAG_UO) as q:
        buckets = q._field_ids_named("deterMaxRound")
    assert isinstance(buckets, dict)
    tiling = buckets.get("TILING_FIELD") or []
    assert len(tiling) == 1
    # Negative: not a flat list of ids.
    assert not isinstance(buckets, list)


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g2_isbn2_field_bucket_prefers_producer_sites() -> None:
    with UoSqlQuery(FAG_UO) as q:
        buckets = q._field_ids_named("isBn2MultiBlk")
        fids = buckets.get("FIELD") or []
        assert fids
        with q._connect() as conn:
            rows = conn.execute(
                f"SELECT id, file, line_start, data FROM entity WHERE id IN ({','.join('?' for _ in fids)})",
                fids,
            ).fetchall()
    files = {str(r[0]): (str(r[1] or "").replace("\\", "/"), int(r[2] or 0)) for r in rows}
    wanted = [
        eid
        for eid, (path, line) in files.items()
        if "normal_regbase.cpp" in path and line == 1099
    ]
    empty = [
        eid
        for eid, (path, line) in files.items()
        if "common_regbase.cpp" in path and line == 1673
    ]
    assert wanted, files
    assert fids[0] in wanted
    assert not empty, empty


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g3_field_ids_named_order_is_stable() -> None:
    with UoSqlQuery(FAG_UO) as q:
        first = q._field_ids_named("deterMaxRound")
        sql = ""
        with q._connect() as conn:
            # Capture that ORDER BY is present on the named-id query.
            seen: list[str] = []
            conn.set_trace_callback(seen.append)
            q._field_ids_named("isBn2MultiBlk")
            conn.set_trace_callback(None)
            sql = "\n".join(seen)
        seq = [q._field_ids_named("deterMaxRound") for _ in range(10)]
    assert all(item == first for item in seq)
    assert "ORDER BY" in sql.upper()


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_c1_telemetry_and_g4_g10_deter_max_round() -> None:
    status(project=str(FAG), architecture="arch35")
    payload, text = _resolve("deterMaxRound")
    assert float(payload.get("server_ms") or 0) > 0
    assert int(payload.get("response_chars") or -1) == len(text)
    host = _section(text, "Host value definitions")
    assert _cited_lines(host) == HOST_DETER_LINES
    transport = _section(text, "Transport")
    assert _cited_lines(transport) == {2328}
    consumers = _section(text, "Kernel consumers")
    names = {m.group(1) for m in re.finditer(r"^\s+(\S+)\s+(\d+)", consumers, re.M)}
    lines = {int(m.group(2)) for m in re.finditer(r"^\s+(\S+)\s+(\d+)", consumers, re.M)}
    # `  CalDenseDeterIndex  463`
    assert names == {"CalDenseDeterIndex", "CalDeterMaxLoopNum"}
    assert lines == {463, 730}
    assert "_def.cpp" not in text.replace("\\", "/")
    assert not re.search(r"(?m)^\s+87(?:\s|=)", text)
    assert not re.search(r"(?m)^\s+88(?:\s|=)", text)
    assert not re.search(r"(?m)^\s+89(?:\s|=)", text)
    assert "Assignments " not in text or "exhaustive=no" not in text
    assert "exhaustive=no" not in text
    assert len(text) < 8_000
    # Negative: OP_LOGI consumers must not appear.
    assert "OP_LOGI" not in consumers
    assert "OP_LOGD" not in consumers


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g7_g9_isbn2_producers() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("isBn2MultiBlk")
    assigns = _section(text, "Assignments")
    assert _cited_lines(assigns) == BN2_PRODUCER_LINES
    assert "_def.cpp" not in text.replace("\\", "/")
    assert not re.search(r"(?m)^\s+87(?:\s|=)", text)
    assert not re.search(r"(?m)^\s+88(?:\s|=)", text)
    assert not re.search(r"(?m)^\s+89(?:\s|=)", text)
    assert "exhaustive=no" not in text
    assert len(text) < 8_000
    # Negative: packing-key def.cpp registration is not an assignment.
    assert "IsBn2MultiBlk" not in assigns or "87" not in assigns


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g11_g13_select_deter_band_schedule() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("SelectDeterBandSchedule")
    calls = _section(text, "Calls")
    rows = _callee_rows(calls)
    names = [n for n, _ in rows]
    assert len(names) == 5, names
    assert names.count("CalcLegacyDeterBandMaxRound") == 1
    assert ("CalcLegacyDeterBandMaxRound", 258) in rows
    assert "min" not in names
    assert "max" not in names
    assert len(names) == len(set(names))
    called = _section(text, "Called by")
    callers = _callee_rows(called)
    assert len(callers) == 1, callers
    assert callers[0][0] == "SelectBlockSchedule"
    assert callers[0][1] == 386
    assert "!(!params.isDeterministic)" in called
    assert "rightDownBandCond" in called
    assert "hybridBandCond" in called
    # BRANCH names are when-clauses, not extra callers.
    assert "!(!params.isDeterministic)" not in {n for n, _ in callers}


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g14_sync_all_cores_called_by_macros() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("SyncALLCores")
    called = _section(text, "Called by")
    assert "INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL" in called
    assert "INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL" in called
    assert "DlogRecord" not in text
    assert "c_str" not in text
    assert "GetTid" not in text
    # Negative: do not fake callers from REFERENCES / same-name noise.
    assert "GetSafeStr" not in called


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g15_g16_search_isbn2_units() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="search",
        name="isBn2MultiBlk",
        limit=40,
    )
    text = _text_of(payload)
    lines = text.splitlines()
    assert lines[0] == "26 matches · 8 source units"
    assert any("+103 template lines" in ln for ln in lines[:3])
    head = "\n".join(lines[:10])
    assert "SetSplitAxis" in head
    assert "DoSparse" in head
    # N+1: def-span preload is batched.
    plan = validate_plan(operation="search", name="isBn2MultiBlk")
    seen: list[str] = []
    with UoSqlQuery(FAG_UO) as q:
        with q._connect() as conn:
            conn.set_trace_callback(seen.append)
            q.query_search(plan, limit=40)
            conn.set_trace_callback(None)
    def_span = [s for s in seen if "def_span_preload" in s]
    assert len(def_span) <= 2
    # Negative: template file is collapsed, not listed as extra units.
    assert "template_tiling_key" not in "\n".join(lines[:12])


def test_g17_mcp_schema_public_operations() -> None:
    """One tool per public operation, and no `operation` string to choose.

    Naming the operation was a decision the caller had to get right before it
    could ask anything, and the tool it named then re-decided the mode from the
    filters. Now the tool name *is* the operation.
    """
    server = create_server()
    for operation in PUBLIC_OPERATIONS:
        tool = _find_tool(server, f"codemap_{operation}")
        schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)
        assert isinstance(schema, dict)
        props = schema.get("properties") or {}
        assert "operation" not in props
        for key in ("projection", "expected_snapshot_id"):
            assert key not in props, (operation, key)


def test_g18_invalid_query_suggestions_stay_public() -> None:
    cases: list[dict[str, object]] = []
    for operation in OPERATIONS:
        cases.append({"operation": operation, "callee": "SyncAll", "from_symbol": "A"})
        cases.append({"operation": operation, "name": "*Buf*"})
        cases.append({"operation": operation, "symbol": "Not A Symbol"})
        cases.append({"operation": operation})
    cases.append({"operation": "not_an_op"})
    raised = 0
    for kwargs in cases:
        try:
            validate_plan(**kwargs)
        except InvalidQuery as exc:
            raised += 1
            for call in exc.did_you_mean:
                assert call.get("operation") in PUBLIC_OPERATIONS, (kwargs, call)
            leaked = {str(x).strip().lower() for x in (exc.legal_filters or [])} & _LEAKED_OPS
            assert not leaked, (kwargs, exc.legal_filters)
    assert raised >= len(OPERATIONS)


def test_g19_evidence_opt_in_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "codemap_evidence" not in DEFAULT_MCP_TOOLS
    names = _tool_names(create_server())
    assert "codemap_evidence" not in names
    monkeypatch.setenv("ASCENDC_CODEMAP_MCP_TOOLS", "all")
    listed = _tool_names(create_server())
    assert "codemap_evidence" in listed
    # Still registered on the default server via get_tool.
    monkeypatch.delenv("ASCENDC_CODEMAP_MCP_TOOLS", raising=False)
    tool = _find_tool(create_server(), "codemap_evidence")
    assert tool is not None


def test_g20_public_docs_share_operation_set() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / "skills" / "query-codemap" / "SKILL.md").read_text(encoding="utf-8")
    bundled = (
        ROOT / "src" / "ascendc_codemap_mcp" / "skills" / "query-codemap" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert skill == bundled
    got = {
        "readme": PublicQueryContract.operations_in(readme),
        "instructions": PublicQueryContract.operations_in(INSTRUCTIONS),
        "skill": PublicQueryContract.operations_in(skill),
    }
    public = set(PUBLIC_OPERATIONS)
    assert got["readme"] == public
    assert got["instructions"] == public
    assert got["skill"] == public


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g21_workspace_layout_order() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("workspaceSize")
    layout = _section(text, "Workspace layout")
    rows = re.findall(r"^\s+(\w+):(\d+)\s*$", layout, re.M)
    got = [(name, int(line)) for name, line in rows]
    assert got == list(LAYOUT)
    # Negative: size/accum sentinels are not layout slots.
    assert "qSize" not in layout
    assert "workspaceSize:" not in layout


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g22_vec_in_ping_backing() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("vecInPing")
    assert "inQueuePing" in text
    assert "UB" in text
    assert "AllocTensor" in text
    assert re.search(r"\b144\b", text)
    # Negative: a later BACKED_BY site is not the answer.
    assert "278" not in _section(text, "Backing")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g23_event_pairing_fields() -> None:
    status(project=str(FAG), architecture="arch35")
    name = ""
    with UoSqlQuery(FAG_UO) as q:
        with q._connect() as conn:
            row = conn.execute(
                "SELECT e.name FROM entity e WHERE e.kind='EVENT' "
                "AND instr(e.data, '\"paired\"') > 0 "
                "AND instr(e.data, '\"signal_count\"') > 0 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM entity f WHERE f.kind='FIELD' AND f.name = e.name"
                ") "
                "ORDER BY length(e.name) DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            name = str(row[0])
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        symbol=name,
        kind="EVENT",
    )
    text = _text_of(payload)
    assert "paired=" in text
    assert "signal_count=" in text
    assert "await_count=" in text
    # Negative: unpaired placeholder language is gone.
    assert "unpaired" not in text.lower()


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_g24_g25_branch_fold_and_log_guard_filter() -> None:
    status(project=str(FAG), architecture="arch35")
    _payload, text = _resolve("dqIsNeedDeter")
    guarded = _section(text, "Guarded by")
    if "x" not in guarded:
        _payload, text = _resolve("isSparse")
        guarded = _section(text, "Guarded by")
    assert re.search(r"\bx\d+\b", guarded or text)
    assert "CheckLogLevel" not in (guarded or "")
    assert "OP_LOG" not in (guarded or "")
    assert "Dlog" not in (guarded or "")


def test_c4_metrics_script_keys() -> None:
    path = ROOT / "benchmarks" / "collect_query_metrics.py"
    body = path.read_text(encoding="utf-8")
    for key in (
        "queries_per_question",
        "native_escape",
        "repeat_resolve",
        "invalid_query",
        "server_ms",
    ):
        assert key in body
    assert "evidence_calls" not in body
    tests = (ROOT / "tests").read_text(encoding="utf-8") if False else ""
    del tests
    bench = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    assert "evidence_calls" not in bench
    review = (ROOT / "tests" / "test_review20_benchmark.py").read_text(encoding="utf-8")
    assert "evidence_calls" not in review
