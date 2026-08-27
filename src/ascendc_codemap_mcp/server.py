# -*- coding: utf-8 -*-
"""Stdio MCP server. NDJSON by default; also accepts Content-Length bodies."""
from __future__ import annotations

import json
import sys
from typing import Any, Callable

from ascendc_codemap_mcp.constants import PROTOCOL, SERVER_NAME, SERVER_VERSION
from ascendc_codemap_mcp import tools as tool_impl

ToolFn = Callable[..., dict[str, Any]]


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "codemap_doctor",
            "title": "codemap_doctor",
            "description": (
                "Check whether this machine can build an AscendC operator CodeMap: "
                "CANN headers, libclang, operator directory, and architecture."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Operator directory (absolute path).",
                    },
                    "architecture": {
                        "type": "string",
                        "description": "e.g. arch35. Required for index paths.",
                    },
                },
            },
        },
        {
            "name": "index_operator",
            "title": "index_operator",
            "description": (
                "Build or rebuild the operator CodeMap (prepare → extract → "
                "analyze → commit). Requires project + architecture. Do not call "
                "on MCP connect; only when the user asks to index or no .uo exists. "
                "If a .uo already exists, use update_operator instead."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["project", "architecture"],
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Operator directory (absolute path).",
                    },
                    "architecture": {
                        "type": "string",
                        "description": "e.g. arch35.",
                    },
                },
            },
        },
        {
            "name": "update_operator",
            "title": "update_operator",
            "description": (
                "Incrementally refresh an existing CodeMap after source changes "
                "(detect → plan → rebuild changed layers). Requires an existing .uo. "
                "Do not call on MCP connect. Use index_operator only when no .uo exists."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["project", "architecture"],
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Operator directory (absolute path).",
                    },
                    "architecture": {
                        "type": "string",
                        "description": "e.g. arch35.",
                    },
                    "confirm_scope": {
                        "type": "boolean",
                        "description": "Proceed when the plan asks for scope confirmation.",
                    },
                },
            },
        },
        {
            "name": "codemap_status",
            "title": "codemap_status",
            "description": (
                "Whether a committed .uo CodeMap exists for this operator + "
                "architecture, plus mtime and completeness."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["project", "architecture"],
                "properties": {
                    "project": {"type": "string"},
                    "architecture": {"type": "string"},
                },
            },
        },
        {
            "name": "query_codemap",
            "title": "query_codemap",
            "description": (
                "Read-only operator CodeMap query. Four shapes only: "
                "(1) no pattern = index, (2) identifier e.g. IsPse, "
                "(3) Dim=Name or Name=Value e.g. IsPse=1, "
                "(4) file + line copied from a previous card. "
                "Do not pass natural-language sentences."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Identifier, Dim=Name, or Name=Value. Omit for index.",
                    },
                    "file": {
                        "type": "string",
                        "description": "Relative path copied from a previous card.",
                    },
                    "line": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "project": {
                        "type": "string",
                        "description": "Operator directory.",
                    },
                    "architecture": {
                        "type": "string",
                        "description": "e.g. arch35. Required.",
                    },
                },
            },
        },
    ]


def _dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "codemap_doctor":
        return tool_impl.doctor(
            project=str(args.get("project") or ""),
            architecture=str(args.get("architecture") or ""),
        )
    if name == "index_operator":
        return tool_impl.index_operator(
            project=str(args.get("project") or ""),
            architecture=str(args.get("architecture") or ""),
        )
    if name == "update_operator":
        raw = args.get("confirm_scope")
        confirm = raw is True or str(raw or "").strip().lower() in {"1", "true", "yes"}
        return tool_impl.update_operator(
            project=str(args.get("project") or ""),
            architecture=str(args.get("architecture") or ""),
            confirm_scope=confirm,
        )
    if name == "codemap_status":
        return tool_impl.status(
            project=str(args.get("project") or ""),
            architecture=str(args.get("architecture") or ""),
        )
    if name == "query_codemap":
        return tool_impl.query_codemap(
            project=str(args.get("project") or ""),
            architecture=str(args.get("architecture") or ""),
            pattern=str(args.get("pattern") or ""),
            file=str(args.get("file") or ""),
            line=int(args.get("line") or 0),
            line_end=int(args.get("line_end") or 0),
        )
    raise KeyError(name)


def _read_message() -> dict[str, Any] | None:
    content_length: int | None = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if line.startswith(b"{") and content_length is None:
            return json.loads(line.decode("utf-8"))
        decoded = line.decode("utf-8", errors="replace").strip()
        if decoded.lower().startswith("content-length:"):
            content_length = int(decoded.split(":", 1)[1].strip())
    if content_length is None:
        return None
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(msg: dict[str, Any]) -> None:
    raw = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def _result(req_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": payload}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message[:500]},
    }


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = str(msg.get("method") or "")
    req_id = msg.get("id")
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    if method == "initialize":
        client_proto = str(params.get("protocolVersion") or PROTOCOL)
        return _result(
            req_id,
            {
                "protocolVersion": client_proto or PROTOCOL,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "notifications/initialized" or method.startswith("notifications/"):
        return None
    if method == "ping":
        return _result(req_id, {})
    if method in {"shutdown", "exit"}:
        return _result(req_id, {})
    if method == "tools/list":
        return _result(req_id, {"tools": _tool_schemas()})
    if method == "resources/list":
        return _result(req_id, {"resources": []})
    if method == "prompts/list":
        return _result(req_id, {"prompts": []})
    if method == "resources/templates/list":
        return _result(req_id, {"resourceTemplates": []})
    if method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            payload = _dispatch(name, args)
            text = json.dumps(payload, ensure_ascii=False, default=str)
            return _result(
                req_id,
                {"content": [{"type": "text", "text": text}], "isError": False},
            )
        except KeyError:
            return _result(
                req_id,
                {
                    "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                    "isError": True,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _result(
                req_id,
                {
                    "content": [{"type": "text", "text": str(exc)[:500]}],
                    "isError": True,
                },
            )
    if req_id is None:
        return None
    return _error(req_id, -32601, f"method not found: {method}")


def serve() -> int:
    while True:
        try:
            msg = _read_message()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[ascendc-codemap] read_error {exc}\n")
            continue
        if msg is None:
            return 0
        if not isinstance(msg, dict):
            continue
        reply = handle(msg)
        if reply is not None:
            _write_message(reply)
        if str(msg.get("method") or "") in {"shutdown", "exit"}:
            return 0
