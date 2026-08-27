# -*- coding: utf-8 -*-
"""Revalidate Host def-use producer sites against lexical source.

The historical assignment regex scans raw text and can see ``x = %ld`` inside a
log string as if it were C++ code.  This pass is a source-truth guard: a
``source_host_defuse`` definition survives only when the same ``lhs =`` exists
at that source line after comments and string/character literals are masked.
No token blacklist or operator-specific spelling is used.
"""
from __future__ import annotations

import re
from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.symbol_identity import normalize_symbol
from ascendc_codemap_mcp.engine.source_layout import selected_host_files

_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_ASSIGN_RE = re.compile(
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*"
    r"(?<![=!<>])=(?!=)",
)


def validate_host_defuse(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    host_files = selected_host_files(root, architecture)
    if not host_files:
        return codemap

    valid_sites: set[tuple[str, int, str]] = set()
    files: dict[str, str] = {}
    for path in host_files:
        raw = read_text(path)
        masked = _mask_non_code(raw)
        file = _rel(root, path)
        files[file] = raw
        for match in _ASSIGN_RE.finditer(masked):
            lhs = normalize_symbol(match.group("lhs"))
            valid_sites.add((file, _line(raw, match.start()), lhs))

    invalid_defs: set[str] = set()
    for ent in codemap.by_kind(EntityKind.PREDICATE):
        if str(ent.attrs.get("provenance") or "") != "source_host_defuse":
            continue
        lhs = normalize_symbol(str(ent.attrs.get("lhs") or ""))
        file = str(ent.file or "")
        line = int(ent.line_start or 0)
        if not lhs or (file, line, lhs) not in valid_sites:
            invalid_defs.add(ent.id)

    remove_rel = {
        rid for rid, rel in codemap.relations.items()
        if rel.src in invalid_defs or rel.dst in invalid_defs
    }
    for rid in remove_rel:
        codemap.relations.pop(rid, None)
    for eid in invalid_defs:
        codemap.entities.pop(eid, None)

    # Producer-site attributes are query-facing facts too; remove invalid sites
    # instead of leaving stale metadata after relation cleanup.
    cleaned_sites = 0
    for ent in codemap.entities.values():
        sites = [s for s in (ent.attrs.get("producer_sites") or []) if isinstance(s, dict)]
        if sites:
            kept = []
            for site in sites:
                file = str(site.get("file") or "")
                line = int(site.get("line") or 0)
                lhs = normalize_symbol(str(site.get("lhs") or ""))
                if not lhs or (file, line, lhs) in valid_sites:
                    kept.append(site)
                else:
                    cleaned_sites += 1
            ent.attrs["producer_sites"] = kept
            ent.attrs["producer_site_count"] = len(kept)

    # Remove unresolved dependency leaves made unreachable solely by invalid
    # string/comment definitions. Keep any node that still participates in a
    # real relation or is used elsewhere in the CodeMap.
    dangling: list[str] = []
    related: set[str] = set()
    for rel in codemap.relations.values():
        related.add(rel.src); related.add(rel.dst)
    for ent in list(codemap.entities.values()):
        if str(ent.attrs.get("provenance") or "") != "source_host_unresolved_dependency":
            continue
        if ent.id not in related:
            dangling.append(ent.id)
    for eid in dangling:
        codemap.entities.pop(eid, None)

    codemap.meta["host_defuse_validation"] = {
        "invalid_definition_nodes_removed": len(invalid_defs),
        "invalid_relations_removed": len(remove_rel),
        "producer_sites_removed": cleaned_sites,
        "dangling_dependency_nodes_removed": len(dangling),
        "policy": "lexically-masked-assignment-revalidation/v1",
    }
    return codemap


def _mask_non_code(text: str) -> str:
    """Mask comments and literals while preserving offsets and newlines."""
    out = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = "line"; continue
            if ch == "/" and nxt == "*":
                out[i] = out[i + 1] = " "; i += 2; state = "block"; continue
            if ch in {'\"', "'"}:
                quote = ch; out[i] = " "; i += 1; state = "literal"; continue
            i += 1; continue
        if state == "line":
            if ch == "\n": state = "code"
            else: out[i] = " "
            i += 1; continue
        if state == "block":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = "code"
            else:
                if ch != "\n": out[i] = " "
                i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            out[i] = " "
            if text[i + 1] != "\n": out[i + 1] = " "
            i += 2; continue
        if ch == quote:
            out[i] = " "; i += 1; state = "code"
        else:
            if ch != "\n": out[i] = " "
            i += 1
    return "".join(out)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1
