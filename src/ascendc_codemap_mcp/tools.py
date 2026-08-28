# -*- coding: utf-8 -*-
"""CLI/test facade over the CodeMap service API."""
from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.service.control import (
    doctor,
    index_operator,
    status,
    update_operator,
)
from ascendc_codemap_mcp.service.query import query_codemap

__all__ = [
    "doctor",
    "index_operator",
    "query_codemap",
    "status",
    "update_operator",
]


def discover(*, project: str = "", architecture: str = "") -> dict[str, Any]:
    from ascendc_codemap_mcp.service.control import discover as impl

    return impl(project=project, architecture=architecture)
