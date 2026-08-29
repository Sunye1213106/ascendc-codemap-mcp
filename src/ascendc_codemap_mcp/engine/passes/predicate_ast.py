# -*- coding: utf-8 -*-
"""Stamp canonical predicate AST onto BRANCH and PREDICATE entities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.query.predicate_ast import annotate_attrs


def enrich_predicate_ast(
    codemap: CodeMap,
    operator_root: str | Path = "",
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    """Fill expr_ast / operators / literals / references on every BRANCH/PREDICATE."""
    del operator_root, architecture, host_ir, kernel_ir
    kinds = (EntityKind.BRANCH, EntityKind.PREDICATE)
    for kind in kinds:
        for ent in codemap.by_kind(kind):
            text = (
                str(ent.attrs.get("predicate") or "")
                or str(ent.attrs.get("condition") or "")
                or str(ent.name or "")
            )
            if not text.strip():
                continue
            if isinstance(ent.attrs.get("expr_ast"), dict) and ent.attrs.get("operators") is not None:
                continue
            ent.attrs.update(annotate_attrs(text))
    return codemap
