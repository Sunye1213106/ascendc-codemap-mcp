# -*- coding: utf-8 -*-
"""CodeMap service API — identity, freshness, query, and control plane.

MCP and CLI are adapters over this package. The engine stays underneath.
"""
from __future__ import annotations

from ascendc_codemap_mcp.service.control import (
    doctor,
    index_operator,
    status,
    update_operator,
)
from ascendc_codemap_mcp.service.query import (
    evidence as query_evidence,
    overview,
    query_codemap,
    selection,
    symbol,
)
from ascendc_codemap_mcp.service.runtime import cache_stats, shutdown

__all__ = [
    "cache_stats",
    "doctor",
    "index_operator",
    "overview",
    "query_codemap",
    "query_evidence",
    "selection",
    "shutdown",
    "status",
    "symbol",
    "update_operator",
]
