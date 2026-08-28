# -*- coding: utf-8 -*-
"""Stdio / Streamable HTTP MCP server via the official Python MCP SDK."""
from __future__ import annotations

from typing import Any


def serve(transport: str = "stdio", **kwargs: Any) -> int:
    from ascendc_codemap_mcp.mcp_adapter import create_server
    from ascendc_codemap_mcp.service import runtime

    kind = str(transport or "stdio").strip() or "stdio"
    try:
        create_server().run(transport=kind, **kwargs)  # type: ignore[arg-type]
    finally:
        runtime.shutdown()
    return 0
