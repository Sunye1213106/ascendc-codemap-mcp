# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_NAME
from ascendc_codemap_mcp.install.jsonutil import command_is_ours, read_json, write_json


def config_path() -> Path:
    custom = (os.environ.get("OPENCODE_CONFIG") or "").strip()
    if custom:
        return Path(custom).expanduser()
    base = Path.home() / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    if jsonc.is_file():
        return jsonc
    return base / "opencode.json"


def _entry(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "local",
        "command": command,
        "enabled": True,
        "environment": env,
    }


def install(command: list[str], env: dict[str, str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if dry_run:
        return {"ok": True, "client": "OpenCode", "path": str(path), "detail": "mcp.local"}
    data = read_json(path)
    bag = data.get("mcp")
    if not isinstance(bag, dict):
        bag = {}
        data["mcp"] = bag
    existing = bag.get(PRODUCT_NAME)
    if existing is not None and not command_is_ours(existing, command):
        return {
            "ok": False,
            "client": "OpenCode",
            "path": str(path),
            "detail": "existing entry is not owned; left unchanged",
        }
    bag[PRODUCT_NAME] = _entry(command, env)
    write_json(path, data)
    return {"ok": True, "client": "OpenCode", "path": str(path), "detail": "mcp"}


def uninstall(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {"ok": False, "client": "OpenCode", "path": str(path), "detail": "missing"}
    if dry_run:
        return {"ok": True, "client": "OpenCode", "path": str(path), "detail": "remove"}
    data = read_json(path)
    bag = data.get("mcp")
    if not isinstance(bag, dict) or PRODUCT_NAME not in bag:
        return {"ok": False, "client": "OpenCode", "path": str(path), "detail": "no entry"}
    if not command_is_ours(bag[PRODUCT_NAME], command):
        return {
            "ok": False,
            "client": "OpenCode",
            "path": str(path),
            "detail": "existing entry is not owned; left unchanged",
        }
    bag.pop(PRODUCT_NAME, None)
    write_json(path, data)
    return {"ok": True, "client": "OpenCode", "path": str(path), "detail": "removed"}
