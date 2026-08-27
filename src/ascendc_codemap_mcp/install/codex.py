# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import CODEX_ENV_VARS, MCP_MARK_BEGIN, MCP_MARK_END, PRODUCT_NAME


def config_path() -> Path:
    home = (os.environ.get("CODEX_HOME") or "").strip()
    if home:
        return Path(home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _block(command: list[str], env: dict[str, str]) -> str:
    exe = command[0].replace("\\", "\\\\").replace('"', '\\"')
    args = ", ".join('"' + a.replace('"', '\\"') + '"' for a in command[1:])
    env_list = ", ".join(f'"{name}"' for name in CODEX_ENV_VARS)
    return (
        f"{MCP_MARK_BEGIN}\n"
        f"[mcp_servers.{PRODUCT_NAME}]\n"
        f'command = "{exe}"\n'
        f"args = [{args}]\n"
        f"env_vars = [{env_list}]\n"
        f"{MCP_MARK_END}\n"
    )


def _upsert_block(text: str, block: str) -> str:
    start = text.find(MCP_MARK_BEGIN)
    end = text.find(MCP_MARK_END)
    if start >= 0 and end > start:
        end_at = end + len(MCP_MARK_END)
        while end_at < len(text) and text[end_at] in "\r\n":
            end_at += 1
        return text[:start] + block + text[end_at:]
    if text and not text.endswith("\n"):
        text += "\n"
    if text and not text.endswith("\n\n"):
        text += "\n"
    return text + block


def _remove_block(text: str) -> str:
    start = text.find(MCP_MARK_BEGIN)
    end = text.find(MCP_MARK_END)
    if start < 0 or end < start:
        return text
    end_at = end + len(MCP_MARK_END)
    while end_at < len(text) and text[end_at] in "\r\n":
        end_at += 1
    return text[:start] + text[end_at:]


def install(command: list[str], env: dict[str, str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if dry_run:
        return {"ok": True, "client": "Codex", "path": str(path), "detail": "config.toml"}
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_upsert_block(existing, _block(command, env)), encoding="utf-8")
    return {"ok": True, "client": "Codex", "path": str(path), "detail": "mcp_servers"}


def uninstall(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {"ok": False, "client": "Codex", "path": str(path), "detail": "missing"}
    if dry_run:
        return {"ok": True, "client": "Codex", "path": str(path), "detail": "remove"}
    text = path.read_text(encoding="utf-8")
    if MCP_MARK_BEGIN not in text:
        return {"ok": False, "client": "Codex", "path": str(path), "detail": "no entry"}
    path.write_text(_remove_block(text), encoding="utf-8")
    return {"ok": True, "client": "Codex", "path": str(path), "detail": "removed"}
