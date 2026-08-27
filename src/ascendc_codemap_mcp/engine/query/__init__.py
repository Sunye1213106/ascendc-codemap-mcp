# -*- coding: utf-8 -*-
"""CodeMap query API for agents (schema-agnostic)."""

from ascendc_codemap_mcp.engine.query.engine import CodeMapQuery, open_codemap_query
from ascendc_codemap_mcp.engine.query.slice import slice_backward, slice_forward

__all__ = [
    "CodeMapQuery",
    "open_codemap_query",
    "slice_backward",
    "slice_forward",
]
