# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from ascendc_codemap_mcp.constants import SERVER_NAME
from ascendc_codemap_mcp.server import handle


def test_initialize_echoes_protocol() -> None:
    reply = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert reply["result"]["protocolVersion"] == "2025-03-26"
    assert reply["result"]["serverInfo"]["name"] == SERVER_NAME


def test_tools_list_names() -> None:
    reply = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in reply["result"]["tools"]]
    assert names == [
        "codemap_doctor",
        "index_operator",
        "update_operator",
        "codemap_status",
        "query_codemap",
    ]


def test_unknown_tool_is_error() -> None:
    reply = handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "uo_query", "arguments": {}},
        }
    )
    assert reply["result"]["isError"] is True
    assert "unknown tool" in reply["result"]["content"][0]["text"]


def test_query_missing_architecture_is_error() -> None:
    reply = handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "query_codemap",
                "arguments": {"project": "/tmp/op"},
            },
        }
    )
    assert reply["result"]["isError"] is True
    assert "architecture" in reply["result"]["content"][0]["text"].lower()


def test_status_not_indexed(tmp_path) -> None:
    op = tmp_path / "op"
    op.mkdir()
    reply = handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "codemap_status",
                "arguments": {"project": str(op), "architecture": "arch35"},
            },
        }
    )
    body = json.loads(reply["result"]["content"][0]["text"])
    assert body["ok"] is True
    assert body["indexed"] is False
