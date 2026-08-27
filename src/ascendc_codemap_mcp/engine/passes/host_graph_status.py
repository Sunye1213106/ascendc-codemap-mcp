# -*- coding: utf-8 -*-
"""Host graph-engine catalog roots, isomorphic to AscendC kernel roots.

CANN Host compilation exits through ``ge::graphStatus``. A failure spelling
means that path never produces a later Host packing / TilingKey write. Roots
are catalog TYPEs keyed by status spelling — not operator lemmas.
"""

from __future__ import annotations

import re
from typing import Any

from ascendc_codemap_mcp.engine.ids import branch_id, make_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

GRAPH_STATUS_CATALOG = "ge.graphStatus"
HOST_GRAPH_PROVENANCE = "host_graph_status"

# Spellings from clang_walk._ERROR_EXIT_RE plus the success code used to
# classify forwarded ``return ret`` after ``ret != GRAPH_SUCCESS``.
_STATUS_ROLES: tuple[tuple[str, str], ...] = (
    ("GRAPH_FAILED", "host_refuse"),
    ("PARAM_INVALID", "host_refuse"),
    ("GRAPH_PARAM", "host_refuse"),
    ("GRAPH_SUCCESS", "host_accept"),
)
_STATUS_BY_LEN = tuple(sorted(_STATUS_ROLES, key=lambda item: len(item[0]), reverse=True))
_SUCCESS_NE_RE = re.compile(r"!=\s*(?:\w+\s*::\s*)?\w*SUCCESS\b|\w*SUCCESS\s*!=")
_LOGE_RE = re.compile(r"\w*LOGE\b")
_FAILED_WORD_RE = re.compile(r"\bFAILED\b")


def _ensure_graph_status_root(codemap: CodeMap, spelling: str) -> str:
    role = "host_accept" if spelling == "GRAPH_SUCCESS" else "host_refuse"
    for name, named_role in _STATUS_ROLES:
        if name == spelling:
            role = named_role
            break
    eid = make_id("Type", GRAPH_STATUS_CATALOG, spelling)
    codemap.upsert(
        EntityKind.TYPE,
        spelling,
        eid=eid,
        attrs={
            "catalog": GRAPH_STATUS_CATALOG,
            "spelling": spelling,
            "role": role,
            "root": f"ge::graphStatus::{spelling}",
            "root_kind": "GRAPH_STATUS",
            # One of four synthesised catalog nodes, not a source fact.
            "provenance": "catalog_root",
        },
        status="extracted",
        confidence=1.0,
    )
    return eid


def status_spelling_from_text(text: str) -> str | None:
    """Map already-extracted Host text onto a catalog spelling. No source scrape."""
    blob = str(text or "")
    for spelling, _role in _STATUS_BY_LEN:
        if spelling in blob:
            if spelling == "GRAPH_SUCCESS" and _SUCCESS_NE_RE.search(blob):
                return "GRAPH_FAILED"
            return spelling
    if _SUCCESS_NE_RE.search(blob):
        return "GRAPH_FAILED"
    if _FAILED_WORD_RE.search(blob) or _LOGE_RE.search(blob):
        return "GRAPH_FAILED"
    return None


def _link_site_to_root(
    codemap: CodeMap,
    site_id: str,
    spelling: str,
    *,
    file: str = "",
    line: int = 0,
    function: str = "",
    predicate: str = "",
) -> None:
    if not site_id or not spelling:
        return
    root_id = _ensure_graph_status_root(codemap, spelling)
    payload = {
        "provenance": HOST_GRAPH_PROVENANCE,
        "catalog": GRAPH_STATUS_CATALOG,
        "spelling": spelling,
        "file": file,
        "line": line,
        "function": function,
        "predicate": predicate,
    }
    codemap.link(RelationKind.RETURNS, site_id, root_id, attrs=payload, status="confirmed")
    codemap.link(RelationKind.ROOTED_AT, site_id, root_id, attrs=payload, status="confirmed")


def _branch_at(codemap: CodeMap, file: str, line: int):
    path = str(file or "").replace("\\", "/")
    loc = int(line or 0)
    if not path or loc <= 0:
        return None
    for ent in codemap.by_kind(EntityKind.BRANCH):
        if str(ent.file or "").replace("\\", "/") == path and int(ent.line_start or 0) == loc:
            return ent
    return None


def _function_named(codemap: CodeMap, name: str):
    needle = str(name or "").strip()
    if not needle:
        return None
    hits = list(codemap.by_name(needle, kind=EntityKind.FUNCTION))
    return hits[0] if hits else None


def _mint_bailout_branch(
    codemap: CodeMap,
    *,
    file: str,
    line: int,
    function: str,
    predicate: str,
    ordinals: dict[tuple[str, str, str], int],
):
    existing = _branch_at(codemap, file, line)
    if existing is not None:
        return existing
    guard = (predicate or "host_refuse")[:120]
    okey = (file, function or "_refuse", guard)
    ordinal = ordinals.get(okey, 0)
    ordinals[okey] = ordinal + 1
    eid = branch_id(
        side="host",
        file=file,
        function=function or "_refuse",
        guard=guard,
        ordinal=ordinal,
    )
    return codemap.upsert(
        EntityKind.BRANCH,
        guard,
        eid=eid,
        attrs={
            "layer": "host",
            "predicate": guard,
            "branch_kind": "host_refuse",
            "function": function,
            "universe": "VALIDATION_ONLY",
            "provenance": HOST_GRAPH_PROVENANCE,
        },
        file=file,
        line=line,
        status="confirmed",
    )


def enrich_host_graph_status(codemap: CodeMap, host_ir: Any) -> CodeMap:
    """Attach Host bailout / return sites to ``ge.graphStatus`` catalog roots."""
    ordinals: dict[tuple[str, str, str], int] = {}

    premises = []
    if host_ir is not None and hasattr(host_ir, "legality_premises"):
        try:
            premises = list(host_ir.legality_premises() or [])
        except Exception:  # noqa: BLE001
            premises = []
    for item in premises:
        if not isinstance(item, (tuple, list)) or len(item) < 4:
            continue
        text, fn_name, file, line = item[0], item[1], item[2], item[3]
        spelling = status_spelling_from_text(str(text)) or "GRAPH_FAILED"
        br = _mint_bailout_branch(
            codemap,
            file=str(file or ""),
            line=int(line or 0),
            function=str(fn_name or ""),
            predicate=str(text or ""),
            ordinals=ordinals,
        )
        if br is not None:
            _link_site_to_root(
                codemap,
                br.id,
                spelling,
                file=str(file or ""),
                line=int(line or 0),
                function=str(fn_name or ""),
                predicate=str(text or ""),
            )
        fn = _function_named(codemap, str(fn_name or ""))
        if fn is not None:
            _link_site_to_root(
                codemap,
                fn.id,
                spelling,
                file=str(file or ""),
                line=int(line or 0),
                function=str(fn_name or ""),
                predicate=str(text or ""),
            )

    for name, summary in (getattr(host_ir, "summaries", None) or {}).items():
        fn = _function_named(codemap, str(name))
        if fn is None:
            continue
        for text in getattr(summary, "returns", None) or []:
            spelling = status_spelling_from_text(str(text))
            if not spelling:
                continue
            _link_site_to_root(
                codemap,
                fn.id,
                spelling,
                file=str(getattr(summary, "file", "") or ""),
                line=int(getattr(summary, "line", 0) or 0),
                function=str(name),
                predicate=str(text),
            )

    for node in getattr(host_ir, "controls", None) or []:
        universe = str(getattr(node, "universe", "") or "")
        cond = str(getattr(node, "condition", "") or "")
        snippet = str(getattr(node, "snippet", "") or "")
        hay = f"{snippet} {cond}"
        if universe != "VALIDATION_ONLY" and not status_spelling_from_text(hay):
            continue
        spelling = status_spelling_from_text(hay) or "GRAPH_FAILED"
        file = str(getattr(node, "file", "") or "")
        line = int(getattr(node, "line", 0) or 0)
        fn_name = str(getattr(node, "function", "") or "")
        br = _mint_bailout_branch(
            codemap,
            file=file,
            line=line,
            function=fn_name,
            predicate=cond or snippet,
            ordinals=ordinals,
        )
        if br is not None:
            _link_site_to_root(
                codemap,
                br.id,
                spelling,
                file=file,
                line=line,
                function=fn_name,
                predicate=cond or snippet,
            )

    return codemap
