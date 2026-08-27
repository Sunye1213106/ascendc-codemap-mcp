# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_NAME
from ascendc_codemap_mcp.install.jsonutil import command_is_ours, read_json, write_json


def config_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _entry(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    return {
        "command": command[0],
        "args": command[1:],
        "env": env,
    }


def install(command: list[str], env: dict[str, str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if dry_run:
        return {"ok": True, "client": "Cursor", "path": str(path), "detail": "mcp.json"}
    data = read_json(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers
    existing = servers.get(PRODUCT_NAME)
    if existing is not None and not command_is_ours(existing, command):
        return {
            "ok": False,
            "client": "Cursor",
            "path": str(path),
            "detail": "existing entry is not owned; left unchanged",
        }
    servers[PRODUCT_NAME] = _entry(command, env)
    write_json(path, data)
    return {"ok": True, "client": "Cursor", "path": str(path), "detail": "mcpServers"}


def uninstall(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {"ok": False, "client": "Cursor", "path": str(path), "detail": "missing"}
    if dry_run:
        return {"ok": True, "client": "Cursor", "path": str(path), "detail": "remove"}
    data = read_json(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or PRODUCT_NAME not in servers:
        return {"ok": False, "client": "Cursor", "path": str(path), "detail": "no entry"}
    if not command_is_ours(servers[PRODUCT_NAME], command):
        return {
            "ok": False,
            "client": "Cursor",
            "path": str(path),
            "detail": "existing entry is not owned; left unchanged",
        }
    servers.pop(PRODUCT_NAME, None)
    write_json(path, data)
    return {"ok": True, "client": "Cursor", "path": str(path), "detail": "removed"}
