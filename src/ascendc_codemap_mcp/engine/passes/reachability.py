# -*- coding: utf-8 -*-
"""Pass 0 — ReachabilityPass (input-/entry-rooted semantic slice).

Formalises the closure already implemented by
``uo_init.clang_walk.reachable_function_names``: light symbol/call indexing,
then reachable function closure, then deep AST walk consumers.
"""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.clang_walk import reachable_function_names
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind


def compute_reachable(
    functions: dict[str, Any],
    call_sites: list[Any],
    *,
    frame_files: frozenset[str] | set[str] = frozenset(),
    needle: str = "",
    op_root: str = "",
    scope: Any = None,
) -> frozenset[str]:
    """Return the reachable function-name closure (Pass 0 contract)."""
    return reachable_function_names(
        functions,
        call_sites,
        frame_files=frozenset(frame_files),
        needle=needle,
        op_root=op_root,
        scope=scope,
    )


def apply_reachability_to_codemap(
    codemap: CodeMap,
    reachable: frozenset[str] | set[str],
) -> CodeMap:
    """Tag FUNCTION entities with reachable/unreachable attrs."""
    reach = {str(n) for n in reachable}
    for ent in codemap.by_kind(EntityKind.FUNCTION):
        ent.attrs["reachable"] = ent.name in reach
    codemap.meta["reachable_functions"] = sorted(reach)
    codemap.meta["reachability_pass"] = "v1"
    return codemap


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    """Pass entry: annotate CodeMap from context host_ir call graph if present."""
    ctx = context or {}
    host_ir = ctx.get("host_ir")
    if host_ir is None:
        return codemap
    functions = getattr(host_ir, "summaries", None) or {}
    # Adapt FuncSummary → minimal FuncRecord-like for reachable_function_names.
    from ascendc_codemap_mcp.engine.clang_walk import CallSite, FuncRecord

    func_records: dict[str, FuncRecord] = {}
    for name, summary in functions.items():
        file = ""
        # Prefer first write/call site file if summaries lack file.
        func_records[str(name)] = FuncRecord(
            name=str(name),
            file=file,
            line=0,
            params=list(getattr(summary, "params", None) or []),
        )
    call_sites = list(getattr(host_ir, "call_sites", None) or [])
    # If summaries lack files, seed from call sites.
    for site in call_sites:
        if not isinstance(site, CallSite):
            continue
        for nm in (site.caller, site.callee):
            rec = func_records.get(nm)
            if rec is not None and not rec.file and site.file:
                func_records[nm] = FuncRecord(
                    name=nm,
                    file=site.file,
                    line=site.line,
                    params=list(rec.params),
                )
    frame_files = frozenset(str(x) for x in (ctx.get("frame_files") or ()))
    reach = compute_reachable(
        func_records,
        call_sites,
        frame_files=frame_files,
        needle=str(ctx.get("needle") or ""),
        op_root=str(ctx.get("op_root") or ""),
        scope=ctx.get("scope"),
    )
    apply_reachability_to_codemap(codemap, reach)
    # Ensure CALLS edges exist for reachable pair.
    for site in call_sites:
        caller = str(getattr(site, "caller", "") or "")
        callee = str(getattr(site, "callee", "") or "")
        if caller not in reach or callee not in reach:
            continue
        # The call site is a clang fact; reachability only decides whether to
        # keep it, so the edge is as good as the walk that produced it.
        attrs = {"layer": "host", "provenance": "clang_walk"}
        src = codemap.upsert(EntityKind.FUNCTION, caller, attrs=dict(attrs))
        dst = codemap.upsert(EntityKind.FUNCTION, callee, attrs=dict(attrs))
        codemap.link(
            RelationKind.CALLS, src.id, dst.id, attrs={"provenance": "clang_walk"}
        )
    return codemap
