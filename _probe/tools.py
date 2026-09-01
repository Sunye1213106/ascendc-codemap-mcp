# -*- coding: utf-8 -*-
"""List the advertised MCP tool surface and each tool's parameter names."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ascendc_codemap_mcp.mcp_adapter import create_server  # noqa: E402

server = create_server()
tools = server._tool_manager.list_tools()
for tool in tools:
    name = getattr(tool, "name", "?")
    schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None) or {}
    props = list((schema.get("properties") or {}).keys())
    print(f"{name}")
    print(f"    params: {', '.join(props)}")
