# -*- coding: utf-8 -*-
"""Unified AscendC CodeMap IR.

All Host / Kernel / Tiling / Macro / Template facts land in one graph.
Legacy IR modules adapt into :class:`CodeMap` rather than writing their own
persistent projections.

Kernel runtime root-trace facts (buffers / ops / WRAPS / ROOTED_AT) live on
the CodeMap directly via ``passes/kernel_root_trace.py``.
"""

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create
from ascendc_codemap_mcp.engine.ir.relation import Relation, RelationKind
from ascendc_codemap_mcp.engine.ir.evidence import (
    SOURCE_CLANG_AST,
    SOURCE_DSL,
    SOURCE_LEXICAL,
    TRUST_ADVISORY,
    TRUST_AUTHORITATIVE,
    TRUST_DERIVED,
    TRUST_LEGACY_UNKNOWN,
    build_context_id,
    stamp_attrs,
    summarize_trust,
    validate_trust_records,
)

__all__ = [
    "CodeMap",
    "Entity",
    "EntityKind",
    "Relation",
    "RelationKind",
    "SOURCE_CLANG_AST",
    "SOURCE_DSL",
    "SOURCE_LEXICAL",
    "TRUST_ADVISORY",
    "TRUST_AUTHORITATIVE",
    "TRUST_DERIVED",
    "TRUST_LEGACY_UNKNOWN",
    "bind_or_create",
    "build_context_id",
    "stamp_attrs",
    "summarize_trust",
    "validate_trust_records",
]
