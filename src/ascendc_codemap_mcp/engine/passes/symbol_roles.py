# -*- coding: utf-8 -*-
"""Project semantic roles onto symbols from already-walked IR.

Glob/path/stem only recall files. FunctionDecl / CXXRecordDecl / KERNEL carry
roles. FILE entities keep ``file_role_summary.contains`` and must not expand
scope. No extra Clang walk: HostIR / KernelIR / CodeMap only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create, is_forbidden_callable_name
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.evidence import (
    SOURCE_CLANG_AST,
    SOURCE_DSL,
    TRUST_AUTHORITATIVE,
    TRUST_DERIVED,
    is_classified_source,
    is_classified_trust,
    mint_payload,
)

ROLE_HOST_TILING_ENTRY = "host_tiling_entry"
ROLE_HOST_TILING_HELPER = "host_tiling_helper"
ROLE_KERNEL_ENTRY = "kernel_entry"
ROLE_KERNEL_SEMANTIC_ROOT = "kernel_semantic_root"
ROLE_OP_DEFINITION = "op_definition"

_SINK_CALLEES = frozenset(
    {
        "SetTilingKey",
        "GET_TPL_TILING_KEY",
        "GET_TILING_KEY",
        "GET_TILINGKEY",
    }
)


def _short(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    return text.split("::")[-1].split("(")[0].strip()


def _norm_file(path: str) -> str:
    return str(path or "").replace("\\", "/")


def _add_role(ent: Entity, role: str, *, provenance: str, source: str, trust: str) -> None:
    roles = [str(r) for r in (ent.attrs.get("symbol_roles") or []) if str(r)]
    if role not in roles:
        roles.append(role)
    ent.attrs["symbol_roles"] = roles
    payload = mint_payload(provenance=provenance, source=source, trust=trust)
    for key in ("evidence_source", "semantic_state", "trust", "build_context_id"):
        if key not in payload:
            continue
        # `upsert` already left an unclassified trust/source behind, so keying
        # on mere presence would drop every classification this pass makes.
        current = str(ent.attrs.get(key) or "")
        if key == "trust" and not is_classified_trust(current):
            ent.attrs[key] = payload[key]
        elif key == "evidence_source" and not is_classified_source(current):
            ent.attrs[key] = payload[key]
        elif key not in ent.attrs:
            ent.attrs[key] = payload[key]
    if provenance and not ent.attrs.get("role_provenance"):
        ent.attrs["role_provenance"] = provenance
    # Keep provenance and trust telling the same story: a node this pass is the
    # first to classify would otherwise carry a trust with nothing to justify it.
    if provenance and not str(ent.attrs.get("provenance") or ""):
        ent.attrs["provenance"] = provenance


def _upsert_fn(codemap: CodeMap, name: str, *, file: str = "", line: int = 0, layer: str) -> Entity | None:
    if is_forbidden_callable_name(str(name).split("::")[-1]):
        return None
    for kind in (EntityKind.FUNCTION, EntityKind.METHOD):
        hits = codemap.by_name(str(name), kind=kind)
        if hits:
            ent = hits[0]
            if file and not ent.file:
                ent.file = _norm_file(file)
                ent.line_start = int(line or 0)
            return ent
    kind = EntityKind.METHOD if "::" in str(name) else EntityKind.FUNCTION
    return bind_or_create(
        codemap,
        kind,
        str(name).split("::")[-1],
        file=_norm_file(file),
        line=int(line or 0),
        owner=str(name).rsplit("::", 1)[0] if "::" in str(name) else "",
        attrs={"layer": layer},
    )


def _is_tiling_sink(callee: str) -> bool:
    return _short(callee) in _SINK_CALLEES


def _project_host(codemap: CodeMap, host_ir: Any) -> None:
    if host_ir is None:
        return
    entries: set[str] = set()
    for site in getattr(host_ir, "call_sites", ()) or ():
        if not _is_tiling_sink(getattr(site, "callee", "")):
            continue
        caller = str(getattr(site, "caller", "") or "")
        if not caller:
            continue
        entries.add(caller)
        ent = _upsert_fn(
            codemap,
            caller,
            file=str(getattr(site, "file", "") or ""),
            line=int(getattr(site, "line", 0) or 0),
            layer="host",
        )
        if ent is None:
            continue
        _add_role(
            ent,
            ROLE_HOST_TILING_ENTRY,
            provenance="symbol_role_host_sink",
            source=SOURCE_CLANG_AST,
            trust=TRUST_AUTHORITATIVE,
        )
    summaries = getattr(host_ir, "summaries", None) or {}
    for name in entries:
        summary = summaries.get(name)
        if summary is None:
            continue
        for callee, _args in getattr(summary, "calls", ()) or ():
            callee_s = str(callee or "")
            if not callee_s or callee_s in entries or _is_tiling_sink(callee_s):
                continue
            helper = summaries.get(callee_s)
            if helper is None:
                continue
            helper_file = str(getattr(helper, "file", "") or "")
            helper_line = int(getattr(helper, "line", 0) or 0)
            ent = _upsert_fn(
                codemap, callee_s, file=helper_file, line=helper_line, layer="host"
            )
            if ent is None:
                continue
            _add_role(
                ent,
                ROLE_HOST_TILING_HELPER,
                provenance="symbol_role_host_helper",
                source=SOURCE_CLANG_AST,
                trust=TRUST_DERIVED,
            )
    for base in getattr(host_ir, "base_decls", ()) or ():
        if _short(getattr(base, "base_name", "")) != "OpDef":
            continue
        derived = str(getattr(base, "derived_name", "") or "")
        if not derived:
            continue
        ent = codemap.upsert(
            EntityKind.TYPE,
            derived,
            attrs={"layer": "host"},
            file=_norm_file(getattr(base, "file", "") or ""),
            line=int(getattr(base, "line", 0) or 0),
        )
        _add_role(
            ent,
            ROLE_OP_DEFINITION,
            provenance="symbol_role_opdef_base",
            source=SOURCE_CLANG_AST,
            trust=TRUST_AUTHORITATIVE,
        )


def _project_kernel(codemap: CodeMap, kernel_ir: Any) -> None:
    kernel_names = {e.name for e in codemap.by_kind(EntityKind.KERNEL)}
    for kernel in codemap.by_kind(EntityKind.KERNEL):
        _add_role(
            kernel,
            ROLE_KERNEL_ENTRY,
            provenance="symbol_role_kernel_entity",
            source=SOURCE_DSL,
            trust=TRUST_DERIVED,
        )
        fn = _upsert_fn(
            codemap,
            kernel.name,
            file=str(kernel.file or ""),
            line=int(kernel.line_start or 0),
            layer="kernel",
        )
        if fn is None:
            continue
        _add_role(
            fn,
            ROLE_KERNEL_ENTRY,
            provenance="symbol_role_kernel_entry_fn",
            source=SOURCE_DSL,
            trust=TRUST_DERIVED,
        )
        codemap.link(
            RelationKind.DECLARES,
            kernel.id,
            fn.id,
            attrs={"provenance": "symbol_role_kernel_entry_fn"},
        )
    functions = getattr(kernel_ir, "functions", None) or {} if kernel_ir is not None else {}
    for name, rec in functions.items():
        if name not in kernel_names and _short(name) not in kernel_names:
            continue
        ent = _upsert_fn(
            codemap,
            str(name),
            file=str((rec or {}).get("file") or ""),
            line=int((rec or {}).get("line") or 0),
            layer="kernel",
        )
        if ent is None:
            continue
        _add_role(
            ent,
            ROLE_KERNEL_ENTRY,
            provenance="symbol_role_kernel_walk",
            source=SOURCE_CLANG_AST,
            trust=TRUST_AUTHORITATIVE,
        )
        for callee in (rec or {}).get("calls") or []:
            callee_s = str(callee or "")
            if not callee_s or callee_s == name or _short(callee_s) in kernel_names:
                continue
            callee_rec = functions.get(callee_s)
            if callee_rec is None:
                short = _short(callee_s)
                callee_rec = functions.get(short)
                callee_s = short if callee_rec is not None else callee_s
            if callee_rec is None:
                continue
            root = _upsert_fn(
                codemap,
                callee_s,
                file=str(callee_rec.get("file") or rec.get("file") or ""),
                line=int(callee_rec.get("line") or 0),
                layer="kernel",
            )
            if root is None:
                continue
            _add_role(
                root,
                ROLE_KERNEL_SEMANTIC_ROOT,
                provenance="symbol_role_kernel_callee",
                # Projected from a called-from relation, not read off the decl,
                # which is what this label already says it is.
                source=SOURCE_DSL,
                trust=TRUST_DERIVED,
            )


def _match_file_entity(codemap: CodeMap, file_path: str) -> Entity | None:
    needle = _norm_file(file_path).lower()
    if not needle:
        return None
    base = Path(needle).name
    best: Entity | None = None
    for ent in codemap.by_kind(EntityKind.FILE):
        rel = _norm_file(ent.name or ent.file).lower()
        if rel == needle or rel.endswith("/" + needle) or needle.endswith("/" + rel):
            return ent
        if Path(rel).name == base:
            best = ent
    return best


def _summarize_files(codemap: CodeMap) -> dict[str, list[str]]:
    by_file: dict[str, set[str]] = {}
    for kind in (
        EntityKind.FUNCTION,
        EntityKind.METHOD,
        EntityKind.TYPE,
        EntityKind.KERNEL,
    ):
        for ent in codemap.by_kind(kind):
            roles = [str(r) for r in (ent.attrs.get("symbol_roles") or []) if str(r)]
            if not roles:
                continue
            file_path = _norm_file(ent.file or "")
            if not file_path:
                continue
            by_file.setdefault(file_path, set()).update(roles)
            file_ent = _match_file_entity(codemap, file_path)
            if file_ent is None:
                continue
            summary = dict(file_ent.attrs.get("file_role_summary") or {})
            contains = {str(x) for x in (summary.get("contains") or []) if str(x)}
            contains.update(roles)
            file_ent.attrs["file_role_summary"] = {"contains": sorted(contains)}
            layout = str(file_ent.attrs.get("role") or file_ent.attrs.get("layout_role") or "")
            if layout:
                file_ent.attrs["layout_role"] = layout
    return {path: sorted(roles) for path, roles in sorted(by_file.items())}


def project_symbol_roles(
    codemap: CodeMap,
    operator_root: str | Path | None = None,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    """Stamp symbol_roles / file_role_summary. Does not add files to scope."""
    del operator_root, architecture
    _project_host(codemap, host_ir)
    _project_kernel(codemap, kernel_ir)
    by_file = _summarize_files(codemap)
    symbols = [
        e
        for kind in (EntityKind.FUNCTION, EntityKind.METHOD, EntityKind.TYPE, EntityKind.KERNEL)
        for e in codemap.by_kind(kind)
    ]
    resolved = sum(1 for e in symbols if e.attrs.get("symbol_roles"))
    codemap.meta["symbol_roles"] = {
        "resolved_symbol_count": resolved,
        "candidate_symbol_count": len(symbols),
        "file_summaries": by_file,
    }
    return codemap
