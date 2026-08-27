# -*- coding: utf-8 -*-
"""Incremental KB update pipeline for the clang uo_init engine."""
from __future__ import annotations

from ascendc_codemap_mcp.engine.update.apply import update_operator
from ascendc_codemap_mcp.engine.update.artifacts import (
    load_change_set_if_fresh,
    load_update_plan_if_fresh,
)
from ascendc_codemap_mcp.engine.update.changes import detect_kb_changes
from ascendc_codemap_mcp.engine.update.diff import export_diff_product
from ascendc_codemap_mcp.engine.update.plan import plan_kb_update

__all__ = [
    "detect_kb_changes",
    "plan_kb_update",
    "update_operator",
    "export_diff_product",
    "load_change_set_if_fresh",
    "load_update_plan_if_fresh",
]
