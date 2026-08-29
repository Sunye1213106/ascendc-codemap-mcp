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
# Re-exported as run_query: a `query` attribute here would shadow the
# ascendc_codemap_mcp.service.query submodule on attribute access.
from ascendc_codemap_mcp.service.query import (
    evidence as query_evidence,
    query as run_query,
)
from ascendc_codemap_mcp.service.runtime import cache_stats, shutdown

__all__ = [
    "cache_stats",
    "doctor",
    "index_operator",
    "query_evidence",
    "run_query",
    "shutdown",
    "status",
    "update_operator",
]
