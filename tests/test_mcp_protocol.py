# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path

from ascendc_codemap_mcp.service.control import status, update_operator
from ascendc_codemap_mcp.service.identity import CodemapRef
from ascendc_codemap_mcp.service.query import query


EXPECTED_TOOLS = [
    "codemap_discover",
    "codemap_search",
    "codemap_trace",
    "codemap_source",
    "codemap_doctor",
    "codemap_index",
    "codemap_update",
]
HIDDEN_TOOLS = [
    "codemap_status",
    "codemap_overview",
    "codemap_symbol",
    "codemap_selection",
    "codemap_explore",
    "query_codemap",
    "index_operator",
    "update_operator",
]


def _tool_names(server) -> list[str]:
    manager = getattr(server, "_tool_manager", None) or getattr(server, "_tools", None)
    if hasattr(server, "list_tools") and manager is None:
        listed = server.list_tools()
        tools = getattr(listed, "tools", listed)
        return [getattr(t, "name", str(t)) for t in tools]
    if manager is not None and hasattr(manager, "list_tools"):
        tools = manager.list_tools()
        return [getattr(t, "name", str(t)) for t in tools]
    if manager is not None and hasattr(manager, "_tools"):
        tools = manager._tools
        if isinstance(tools, dict):
            return list(tools)
        return [getattr(t, "name", str(t)) for t in tools]
    if isinstance(manager, dict):
        return list(manager)
    raise AssertionError(f"cannot list tools on {type(server)}")


def _find_tool(server, name: str):
    manager = getattr(server, "_tool_manager", None)
    if manager is not None and hasattr(manager, "get_tool"):
        return manager.get_tool(name)
    if manager is not None and hasattr(manager, "_tools"):
        tools = manager._tools
        if isinstance(tools, dict) and name in tools:
            return tools[name]
        for tool in tools.values() if isinstance(tools, dict) else tools:
            if getattr(tool, "name", None) == name:
                return tool
    names = _tool_names(server)
    raise AssertionError(f"tool {name} not found in {names}")


def test_mcp_server_tool_names() -> None:
    from ascendc_codemap_mcp.mcp_adapter import create_server

    names = _tool_names(create_server())
    for name in EXPECTED_TOOLS:
        assert name in names
    for name in HIDDEN_TOOLS:
        assert name not in names
    assert names.index("codemap_search") < names.index("codemap_index")


def test_the_three_query_tools_do_not_share_a_seed() -> None:
    """No tool takes both a symbol and a location.

    A single tool that accepted them together had to guess, and it guessed
    location: a session that named `SetSplitAxis` forty times got a card about
    something else in twenty-four of them. Disjoint parameters remove the guess
    rather than documenting it.
    """
    from ascendc_codemap_mcp.mcp_adapter import create_server

    server = create_server()

    def _props(name: str) -> set[str]:
        tool = _find_tool(server, name)
        schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)
        assert isinstance(schema, dict)
        ann = getattr(tool, "annotations", None)
        assert ann is not None and ann.read_only_hint is True
        return set(schema.get("properties") or {})

    search, trace, source = _props("codemap_search"), _props("codemap_trace"), _props("codemap_source")
    assert "pattern" in search
    assert {"symbol", "line"} & search == set()
    assert {"symbol", "to_symbol", "dim", "value", "relation"} <= trace
    assert {"file", "line", "line_end"} & trace == set()
    assert {"file", "line", "line_end"} <= source
    assert {"symbol", "pattern"} & source == set()
    for props in (search, trace, source):
        assert {"projection", "expected_snapshot_id", "entity_id", "ctx"} & props == set()


def test_mcp_instructions_are_three_invariants() -> None:
    from ascendc_codemap_mcp.mcp_adapter import INSTRUCTIONS

    text = INSTRUCTIONS.strip()
    assert "Unknown name → search" in text
    assert "Known symbol → trace" in text
    assert "file:line → source" in text
    assert "Query reads snapshot only" in text
    assert "resolve" not in text
    assert "find" not in text
    assert "COMPLETE" not in text
    assert "UNKNOWN" not in text
    assert "FTS" not in text
    assert "evidence" not in text.lower()


def test_index_tool_is_not_read_only() -> None:
    from ascendc_codemap_mcp.mcp_adapter import create_server

    tool = _find_tool(create_server(), "codemap_index")
    ann = tool.annotations
    assert ann.read_only_hint is False
    assert ann.destructive_hint is False
    schema = tool.parameters or {}
    assert "ctx" not in (schema.get("properties") or {})
    assert "project" in (schema.get("required") or [])
    assert "architecture" in (schema.get("required") or [])


def test_query_missing_architecture_is_structured_error(tmp_path: Path) -> None:
    payload = query(project=str(tmp_path / "op"), symbol="IsPse")
    assert payload["ok"] is False
    text = f"{payload.get('error') or ''} {payload.get('error_code') or ''}".lower()
    assert "architecture" in text or payload.get("error_code") in {
        "ARCHITECTURE_MISSING_IN_RUN_STATE",
        "OPERATOR_DIR_NOT_FOUND",
        "PROJECT_REQUIRED",
    }


def test_status_not_indexed(tmp_path: Path) -> None:
    op = tmp_path / "op"
    op.mkdir()
    payload = status(project=str(op), architecture="arch35")
    assert payload["ok"] is True
    assert payload["indexed"] is False
    assert payload["codemap"]["alias"] == "op@arch35"
    assert payload["codemap"]["id"].endswith("::op@arch35")


def test_inmemory_client_lists_tools() -> None:
    from mcp import Client
    from ascendc_codemap_mcp.mcp_adapter import create_server

    async def _run() -> None:
        async with Client(create_server()) as client:
            listed = await client.list_tools()
            tools = getattr(listed, "tools", listed)
            names = [getattr(t, "name", str(t)) for t in tools]
            assert "codemap_trace" in names
            assert "codemap_discover" in names
            assert "codemap_symbol" not in names
            result = await client.call_tool(
                "codemap_trace", {"symbol": "IsPse"}
            )
            structured = getattr(result, "structured_content", None) or {}
            assert structured.get("ok") is False
            assert structured.get("error_code") in {
                "PROJECT_REQUIRED",
                "ARCHITECTURE_MISSING_IN_RUN_STATE",
                "CODEMAP_NOT_REGISTERED",
                "INVALID_CODEMAP_ID",
            }
            resources = await client.list_resources()
            uris = [str(r.uri) for r in getattr(resources, "resources", [])]
            assert "codemap://runtime" in uris
            templates = await client.list_resource_templates()
            t_uris = [
                str(t.uri_template)
                for t in getattr(templates, "resource_templates", [])
            ]
            assert any("{codemap_id}" in u for u in t_uris)
            prompts = await client.list_prompts()
            pnames = [p.name for p in getattr(prompts, "prompts", [])]
            assert "query_operator" in pnames
            assert "build_codemap" in pnames
            runtime_res = await client.read_resource("codemap://runtime")
            contents = getattr(runtime_res, "contents", None) or []
            assert contents
            assert "cache_size" in str(contents[0].text)

    asyncio.run(_run())


def test_map_resource_reads_canonical_id(tmp_path: Path) -> None:
    from mcp import Client
    from ascendc_codemap_mcp.mcp_adapter import create_server
    from tests.conftest import write_uo_fixture

    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)

    async def _run() -> None:
        async with Client(create_server()) as client:
            st = await client.call_tool(
                "codemap_status",
                {"project": str(op), "architecture": "arch35"},
            )
            structured = getattr(st, "structured_content", None) or {}
            cid = str((structured.get("codemap") or {}).get("id") or "")
            assert "::" in cid
            assert "/" not in cid
            mapped = await client.read_resource(f"codemap://map/{cid}")
            contents = getattr(mapped, "contents", None) or []
            assert contents
            text = str(contents[0].text)
            assert cid in text or "toy_op@arch35" in text
            assert '"ok"' in text or "ok" in text

    asyncio.run(_run())


def test_update_blocked_is_ok_but_not_updated(monkeypatch, tmp_path: Path) -> None:
    product = tmp_path / "toy.arch35.uo"
    product.write_bytes(b"")
    ref = CodemapRef(
        id="toy@arch35",
        project=tmp_path,
        architecture="arch35",
        op_name="toy",
        product=product,
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.control.resolve",
        lambda **kwargs: ref,
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.control._meta",
        lambda product: {
            "op_name": "toy",
            "architecture": "arch35",
            "schema": "codemap-uo/v3",
        },
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.control._freshness_for",
        lambda ref, meta: {
            "freshness": "blocked",
            "source_revision": "abc",
            "indexed_revision": "abc",
            "dirty": False,
            "changed_files": 0,
            "semantic_completeness": 0.9,
        },
    )

    def _apply(*args, **kwargs):
        return {
            "status": "blocked",
            "run_id": "r1",
            "plan": {
                "needs_scope_review": True,
                "mode": "rebuild",
                "affected_layers": [],
                "actions": [],
            },
            "change_set": {"scoped_change_count": 0},
            "receipt": {"message": "needs scope confirmation"},
        }

    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.update.update_operator",
        _apply,
    )
    payload = update_operator(codemap_id="toy@arch35")
    assert payload["ok"] is True
    assert payload["state"] == "needs_confirmation"
    assert payload["updated"] is False
    assert payload["error_code"] == "SCOPE_CONFIRMATION_REQUIRED"


def test_index_stops_between_steps(monkeypatch, tmp_path: Path) -> None:
    from ascendc_codemap_mcp.service.control import index_operator

    op = tmp_path / "toy_op"
    op.mkdir()
    calls: list[str] = []

    def _ok(*_a, **_k):
        return {"ok": True}

    def _prepare(*_a, **_k):
        calls.append("prepare")
        return {"ok": True}

    monkeypatch.setattr("ascendc_codemap_mcp.engine.codemap_engines.prepare", _prepare)
    monkeypatch.setattr("ascendc_codemap_mcp.engine.codemap_engines.extract", _ok)
    monkeypatch.setattr("ascendc_codemap_mcp.engine.codemap_engines.analyze", _ok)
    monkeypatch.setattr("ascendc_codemap_mcp.engine.codemap_engines.commit", _ok)

    payload = index_operator(
        project=str(op),
        architecture="arch35",
        should_stop=lambda: "prepare" in calls,
    )
    assert payload["ok"] is False
    assert payload["error_code"] == "CANCELLED"
    assert payload["updated"] is False
    assert calls == ["prepare"]


def test_cli_help_mentions_http() -> None:
    from ascendc_codemap_mcp.cli import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["-h"])
    assert code == 0
    text = buf.getvalue()
    assert "streamable-http" in text
    assert "serve" in text
    assert "cann-extract" in text


def test_cli_query_reaches_service_query(monkeypatch) -> None:
    # service/__init__ re-exports a `query` function that shadows the submodule;
    # attribute access on the package used to raise AttributeError here.
    from ascendc_codemap_mcp.cli import main
    from ascendc_codemap_mcp.service import query as query_mod
    import io
    from contextlib import redirect_stdout

    seen: dict[str, object] = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(query_mod, "query", fake_query)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(
            [
                "query",
                "--codemap-id",
                "p:1::Op@arch35",
                "--operation",
                "search",
                "--name",
                "SyncAll",
            ]
        )
    assert code == 0
    assert seen["operation"] == "search"
    assert seen["name"] == "SyncAll"


def test_cli_query_passes_symbol(monkeypatch) -> None:
    from ascendc_codemap_mcp.cli import main
    from ascendc_codemap_mcp.service import query as query_mod
    import io
    from contextlib import redirect_stdout

    seen: dict[str, object] = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(query_mod, "query", fake_query)
    with redirect_stdout(io.StringIO()):
        assert main(["query", "--codemap-id", "p:1::Op@arch35", "--symbol", "IsDrop"]) == 0
    assert seen["symbol"] == "IsDrop"


def test_query_operator_prompt_has_no_stale_protocol() -> None:
    from ascendc_codemap_mcp.mcp_adapter import query_operator

    text = query_operator(codemap_id="p:1::Op@arch35")
    assert "find needs kind" not in text
    assert "completeness=COMPLETE" not in text
    assert "resolve" not in text
    for name in ("codemap_search", "codemap_trace", "codemap_source"):
        assert name in text


def test_query_schema_documents_search_file_filter() -> None:
    from ascendc_codemap_mcp.mcp_adapter import create_server

    tool = _find_tool(create_server(), "codemap_search")
    schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None) or {}
    props = schema.get("properties") or {}
    file_prop = props.get("file") or {}
    desc = str(file_prop.get("description") or "") + " " + str(getattr(tool, "description", "") or "")
    assert "glob" in desc.lower() or "path filter" in desc.lower() or "filter" in desc.lower()


def test_evidence_description_is_debug_only() -> None:
    from ascendc_codemap_mcp.mcp_adapter import create_server, DEFAULT_MCP_TOOLS

    assert "codemap_evidence" not in DEFAULT_MCP_TOOLS
    tool = _find_tool(create_server(), "codemap_evidence")
    desc = str(getattr(tool, "description", "") or "")
    assert "debug" in desc.lower() or "explicit" in desc.lower()
    # It has to name the tool that reads source, or the reader reaches for this
    # one to do it: that was the whole reason ordinary reads landed here.
    assert "codemap_source" in desc.lower()
