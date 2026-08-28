# -*- coding: utf-8 -*-
"""Classify kernel/host consumers from AST parent / catalog call context.

Roles are written onto READS / CALLS / CALLS_UNDER_GUARD / ALLOCATES edges.
Function-name heuristics on project methods are forbidden; catalog API names
and statement-parent kinds are allowed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.semantics.ascendc_storage import (
    TENSOR_METHOD_BRIDGES,
    TPIPE_METHOD_BRIDGES,
    TQUE_METHOD_BRIDGES,
)

LOOP_BOUND = "LOOP_BOUND"
BRANCH_GUARD = "BRANCH_GUARD"
TILE_SHAPE = "TILE_SHAPE"
ADDRESSING = "ADDRESSING"
DATA_MOVE = "DATA_MOVE"
PRELOAD = "PRELOAD"
PIPELINE = "PIPELINE"
CORE_PARTITION = "CORE_PARTITION"
WORKSPACE = "WORKSPACE"
COMPUTE_MODE = "COMPUTE_MODE"
TEMPLATE_DISPATCH = "TEMPLATE_DISPATCH"
OUTPUT_BEHAVIOR = "OUTPUT_BEHAVIOR"
UNKNOWN_ROLE = "UNKNOWN_ROLE"

CONSUMER_ROLES = frozenset(
    {
        LOOP_BOUND,
        BRANCH_GUARD,
        TILE_SHAPE,
        ADDRESSING,
        DATA_MOVE,
        PRELOAD,
        PIPELINE,
        CORE_PARTITION,
        WORKSPACE,
        COMPUTE_MODE,
        TEMPLATE_DISPATCH,
        OUTPUT_BEHAVIOR,
        UNKNOWN_ROLE,
    }
)

_PARENT_ROLES = {
    "IF_STMT": BRANCH_GUARD,
    "CXX_IF_STMT": BRANCH_GUARD,
    "WHILE_STMT": BRANCH_GUARD,
    "SWITCH_STMT": BRANCH_GUARD,
    "CASE_STMT": BRANCH_GUARD,
    "FOR_STMT": LOOP_BOUND,
    "CXX_FOR_RANGE_STMT": LOOP_BOUND,
    "CONSTRUCTOR": PRELOAD,
    "CXX_CONSTRUCTOR": PRELOAD,
    "CXX_CTOR": PRELOAD,
}

_DATA_MOVE = frozenset(
    {
        "DataCopy",
        "DataCopyPad",
        "LoadData",
        "LoadAlign",
        "StoreAlign",
        "StoreUnAlign",
    }
)
_CORE = frozenset({"SetBlockDim", "GetBlockIdx", "GetBlockNum", "SetFullBlockDim"})
_OUTPUT = frozenset({"InitOutput"})
_COMPUTE = frozenset(
    {
        "SetSplitAxis",
        "SetMMLayout",
        "SetNormType",
        "SetFixpipe",
        "SetHF32",
        "SetAtomic",
    }
)
_IF_RE = re.compile(r"\b(if|else\s+if|while|switch)\s*\(")
_FOR_RE = re.compile(r"\bfor\s*\(")
_INDEX_RE = re.compile(r"\[[^\]]*\b[A-Za-z_]\w*\b[^\]]*\]")
_CALLEE_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

_CATALOG_CALLEES = (
    frozenset(TENSOR_METHOD_BRIDGES)
    | frozenset(TPIPE_METHOD_BRIDGES)
    | frozenset(TQUE_METHOD_BRIDGES)
    | _DATA_MOVE
    | _CORE
    | _OUTPUT
    | _COMPUTE
    | frozenset({"SetGlobalBuffer", "InitBuffer", "GetTPipePtr"})
)


def classify_statement(
    stmt: str,
    *,
    callee: str = "",
    parent_kind: str = "",
) -> str:
    """Return a consumer_role. Unsure → UNKNOWN_ROLE."""
    parent = str(parent_kind or "").upper()
    if parent in _PARENT_ROLES:
        return _PARENT_ROLES[parent]
    leaf = str(callee or "").split("::")[-1]
    if leaf == "InitBuffer":
        return PIPELINE
    if leaf == "SetGlobalBuffer":
        return ADDRESSING
    if leaf in _DATA_MOVE:
        return DATA_MOVE
    if leaf in _CORE:
        return CORE_PARTITION
    if leaf in _OUTPUT:
        return OUTPUT_BEHAVIOR
    if leaf in _COMPUTE:
        return COMPUTE_MODE
    text = str(stmt or "")
    if _IF_RE.search(text):
        return BRANCH_GUARD
    if _FOR_RE.search(text):
        return LOOP_BOUND
    if "workspace" in text.lower():
        return WORKSPACE
    if _INDEX_RE.search(text) and "constexpr" in text:
        return TILE_SHAPE
    call = _CALLEE_RE.search(text)
    if call:
        return classify_statement("", callee=call.group(1), parent_kind="")
    return UNKNOWN_ROLE


def _line_window(text: str, line: int, *, radius: int = 2) -> str:
    if line <= 0:
        return ""
    rows = text.splitlines()
    idx = line - 1
    if idx < 0 or idx >= len(rows):
        return ""
    start = max(0, idx - radius)
    end = min(len(rows), idx + radius + 1)
    return "\n".join(rows[start:end])


def _cache_text(root: Path, file: str, cache: dict[str, str]) -> str:
    key = str(file or "").replace("\\", "/")
    if key in cache:
        return cache[key]
    path = Path(file)
    if not path.is_file():
        path = root / key
    try:
        cache[key] = read_text(path)
    except OSError:
        cache[key] = ""
    return cache[key]


def enrich_consumer_roles(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    cache: dict[str, str] = {}
    kinds = {
        RelationKind.READS.value,
        RelationKind.CALLS.value,
        RelationKind.CALLS_UNDER_GUARD.value,
        RelationKind.ALLOCATES.value,
        RelationKind.BINDS.value,
    }
    for rel in list(codemap.relations.values()):
        if rel.kind_name() not in kinds:
            continue
        if rel.attrs.get("consumer_role"):
            continue
        if rel.kind_name() == RelationKind.BINDS.value:
            rel.attrs["consumer_role"] = TEMPLATE_DISPATCH
            continue
        callee = str(rel.attrs.get("callee") or "")
        parent_kind = str(rel.attrs.get("parent_kind") or "")
        file = str(rel.attrs.get("file") or "")
        line = int(rel.attrs.get("line") or 0)
        if not file or not line:
            src = codemap.entities.get(rel.src)
            if src is not None:
                file = file or str(src.file or "")
                line = line or int(src.line_start or 0)
                if not callee:
                    callee = str(src.name or "")
        stmt = ""
        if file and line:
            stmt = _line_window(_cache_text(root, file, cache), line)
        role = classify_statement(stmt, callee=callee, parent_kind=parent_kind)
        dst = codemap.entities.get(rel.dst)
        if role == UNKNOWN_ROLE and dst is not None:
            dname = str(dst.name or "").lower()
            if "workspace" in dname:
                role = WORKSPACE
            elif dst.kind_name() in {"TEMPLATE_ARG", "TILING_KEY"}:
                role = TILE_SHAPE
        rel.attrs["consumer_role"] = role
    return codemap
