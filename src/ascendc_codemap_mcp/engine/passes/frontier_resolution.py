# -*- coding: utf-8 -*-
"""Resolve class-declaration frontier gaps via out-of-class method definitions.

Historical extractors often anchor a frontier request on a class declaration
while the actual templated method bodies appear later in the same header.  This
pass follows the candidate symbol to ``Class<...>::method`` definitions, records
source-backed branch sites, and resolves the gap only when such bodies exist.
"""
from __future__ import annotations

import re
from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create, is_forbidden_callable_name
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

_BRANCH_RE = re.compile(r"\b(if\s+constexpr|if|while|for|switch)\s*\(")
_PP_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif)\b(.*)$", re.M)


def resolve_class_frontiers(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    del architecture  # candidate sources already carry the selected arch path
    root = Path(operator_root).expanduser().resolve()
    resolved = 0
    branch_count = 0

    # Enrichment adds METHOD/BRANCH entities, so iterate over the pre-pass
    # snapshot rather than the live dict.
    unresolved = list(codemap.entities.values())
    for gap in unresolved:
        if str(gap.status).lower() != "unresolved":
            continue
        if str(gap.attrs.get("reason") or "") != "frontier_sites":
            continue
        candidates = [x for x in (gap.attrs.get("candidate_sources") or []) if isinstance(x, dict)]
        gap_branches = 0
        covered = False
        for candidate in candidates:
            if str(candidate.get("anchor_kind") or "") != "class_declaration":
                continue
            symbol = str(candidate.get("symbol") or "").strip().split("::")[-1]
            if not symbol:
                continue
            path = _resolve_source(root, str(candidate.get("file") or ""))
            if path is None:
                continue
            text = read_text(path)
            for method, body_start, body_end, start_line in _method_bodies(text, symbol):
                body = text[body_start:body_end]
            if is_forbidden_callable_name(method):
                continue
            owner = bind_or_create(
                codemap,
                EntityKind.METHOD,
                method,
                file=_rel(root, path),
                line=start_line,
                owner=symbol,
                attrs={
                    "owner": symbol,
                    "provenance": "source_class_frontier",
                    "architecture_scope": "current",
                },
                status="confirmed",
            )
            if owner is None:
                continue
                local = 0
                for match in _BRANCH_RE.finditer(body):
                    absolute = body_start + match.start()
                    line = _line(text, absolute)
                    kind = match.group(1).replace(" ", "_")
                    branch = codemap.upsert(
                        EntityKind.BRANCH,
                        f"{symbol}::{method}:{kind}@{line}",
                        eid=f"SRCFRONTIER::{_rel(root, path)}::{line}::{kind}",
                        attrs={
                            "branch_kind": kind,
                            "owner": symbol,
                            "method": method,
                            "provenance": "source_class_frontier",
                        },
                        file=_rel(root, path),
                        line=line,
                        status="confirmed",
                    )
                    codemap.link(
                        RelationKind.CONTROLS,
                        branch.id,
                        owner.id,
                        attrs={"provenance": "source_class_frontier"},
                        status="confirmed",
                    )
                    local += 1
                for match in _PP_RE.finditer(body):
                    absolute = body_start + match.start()
                    line = _line(text, absolute)
                    kind = f"pp_{match.group(1)}"
                    branch = codemap.upsert(
                        EntityKind.BRANCH,
                        f"{symbol}::{method}:{kind}@{line}",
                        eid=f"SRCFRONTIER::{_rel(root, path)}::{line}::{kind}",
                        attrs={
                            "branch_kind": kind,
                            "condition": match.group(2).strip(),
                            "owner": symbol,
                            "method": method,
                            "provenance": "source_class_frontier",
                        },
                        file=_rel(root, path),
                        line=line,
                        status="confirmed",
                    )
                    codemap.link(
                        RelationKind.CONTROLS,
                        branch.id,
                        owner.id,
                        attrs={"provenance": "source_class_frontier"},
                        status="confirmed",
                    )
                    local += 1
                if local:
                    covered = True
                    gap_branches += local
        if covered:
            gap.status = "resolved"
            gap.confidence = 1.0
            gap.attrs["resolved_by"] = "source_class_frontier"
            gap.attrs["resolved_from_current_source"] = True
            gap.attrs["resolved_frontier_branch_count"] = gap_branches
            resolved += 1
            branch_count += gap_branches

    codemap.meta["source_class_frontier_resolved"] = resolved
    codemap.meta["source_class_frontier_branches"] = branch_count
    return codemap


def _resolve_source(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)


def _method_bodies(text: str, symbol: str):
    pattern = re.compile(
        rf"\b{re.escape(symbol)}\s*<[^;{{}}]*>\s*::\s*([A-Za-z_]\w*)\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{",
        re.S,
    )
    for match in pattern.finditer(text):
        open_pos = text.find("{", match.start(), match.end())
        close_pos = _matching_brace(text, open_pos)
        if close_pos < 0:
            continue
        yield match.group(1), open_pos + 1, close_pos, _line(text, match.start())


def _matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    quote = ""
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()
