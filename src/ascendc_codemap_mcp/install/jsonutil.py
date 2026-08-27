# -*- coding: utf-8 -*-
"""JSON / JSONC helpers for MCP config upsert."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LINE_COMMENT = re.compile(r"(^|[^:])//.*?$", re.M)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def loads_jsonish(text: str) -> Any:
    raw = text.lstrip("\ufeff")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        stripped = _BLOCK_COMMENT.sub("", raw)
        stripped = _LINE_COMMENT.sub(r"\1", stripped)
        return json.loads(stripped)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = loads_jsonish(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def command_is_ours(entry: Any, command: list[str]) -> bool:
    if not isinstance(entry, dict):
        return False
    blob = json.dumps(entry, ensure_ascii=False)
    if "ascendc_codemap_mcp" in blob or "ascendc-codemap-mcp" in blob:
        return True
    cmd = entry.get("command")
    if isinstance(cmd, list) and cmd == command:
        return True
    if isinstance(cmd, str) and command and cmd == command[0]:
        args = entry.get("args")
        if isinstance(args, list) and args == command[1:]:
            return True
    return False
