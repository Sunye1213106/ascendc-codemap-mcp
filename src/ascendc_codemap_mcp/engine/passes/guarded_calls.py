# -*- coding: utf-8 -*-
"""Mint CALLS_UNDER_GUARD from clang CallSite.path_conditions.

HostIR and kernel walk-cache already record the guard stack on every call.
Graph ingest used to drop it, so impact stopped at the callee and never reached
sibling calls under the same predicate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ids import branch_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create, is_forbidden_callable_name
from ascendc_codemap_mcp.engine.ir.evidence import SOURCE_CLANG_AST
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.query.predicate_ast import annotate_attrs
from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file, is_other_arch_path

_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_SKIP_IDENTS = frozenset(
    {
        "true",
        "false",
        "bool",
        "auto",
        "const",
        "static_cast",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int",
        "int32_t",
        "int64_t",
        "void",
        "static",
        "nullptr",
        "this",
        "if",
        "else",
        "return",
        "unlikely",
        "likely",
        "sizeof",
        "NULL",
    }
)
_OPERAND_KINDS = (
    EntityKind.INPUT,
    EntityKind.OUTPUT,
    EntityKind.TILING_FIELD,
    EntityKind.TILING_KEY,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
    EntityKind.COMPILE_VAR,
    EntityKind.MACRO,
    EntityKind.TEMPLATE_ARG,
)
_CALLABLE_KINDS = (
    EntityKind.METHOD,
    EntityKind.FUNCTION,
    EntityKind.KERNEL,
    EntityKind.OPERATION,
)


def _pc_rows(site: Any) -> list[Any]:
    raw = getattr(site, "path_conditions", None)
    if raw is None and isinstance(site, dict):
        raw = site.get("path_conditions")
    if not raw:
        return []
    return list(raw)


def _pc_field(pc: Any, name: str, default: Any = "") -> Any:
    if isinstance(pc, dict):
        return pc.get(name, default)
    return getattr(pc, name, default)


def _pc_pretty(pc: Any) -> str:
    text = str(_pc_field(pc, "text", "") or "").strip()
    if not text:
        return ""
    if bool(_pc_field(pc, "negated", False)):
        return f"!({text})"
    pretty = getattr(pc, "pretty", None)
    if callable(pretty):
        return str(pretty())
    return text


def _pc_skip(pc: Any) -> bool:
    kind = str(_pc_field(pc, "kind", "if") or "if")
    text = str(_pc_field(pc, "text", "") or "").strip()
    if not text:
        return True
    if kind == "bailout":
        return True
    is_opaque = getattr(pc, "is_opaque", None)
    if callable(is_opaque):
        return bool(is_opaque)
    if bool(getattr(pc, "is_opaque", False)):
        return True
    return False


def _site_field(site: Any, name: str, default: Any = "") -> Any:
    if isinstance(site, dict):
        return site.get(name, default)
    return getattr(site, name, default)


def _resolve_callable(codemap: CodeMap, name: str, *, layer: str) -> Entity | None:
    leaf = str(name or "").split("::")[-1]
    if is_forbidden_callable_name(leaf):
        return None
    hits: list[Entity] = []
    for kind in _CALLABLE_KINDS:
        hits.extend(codemap.by_name(leaf, kind=kind))
        if leaf != name:
            hits.extend(codemap.by_name(name, kind=kind))
    uniq: dict[str, Entity] = {e.id: e for e in hits}
    if len(uniq) == 1:
        return next(iter(uniq.values()))
    layered = [e for e in uniq.values() if str(e.attrs.get("layer") or "") == layer]
    if len(layered) == 1:
        return layered[0]
    kind = EntityKind.FUNCTION if layer == "host" else EntityKind.METHOD
    return bind_or_create(
        codemap,
        kind,
        leaf or name,
        attrs={"layer": layer, "provenance": "clang_walk"},
    )


def _unique_operand(codemap: CodeMap, ident: str) -> Entity | None:
    ident = str(ident or "").strip()
    if not ident or ident in _SKIP_IDENTS:
        return None
    hits: dict[str, Entity] = {}
    for kind in _OPERAND_KINDS:
        for ent in codemap.by_name(ident, kind=kind):
            hits[ent.id] = ent
        leaf = ident.replace("::", ".").rsplit(".", 1)[-1]
        if leaf != ident:
            for ent in codemap.by_name(leaf, kind=kind):
                hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    return None


def ingest_guarded_calls(
    codemap: CodeMap,
    sites: Iterable[Any],
    *,
    side: str,
    architecture: str = "",
    ordinals: dict[tuple[str, str, str], int] | None = None,
) -> int:
    """Mint BRANCH + CALLS_UNDER_GUARD + CONTROLS for guarded call sites."""
    layer = "host" if side == "host" else "kernel"
    counts = 0
    used = ordinals if ordinals is not None else {}
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    for site in sites or []:
        caller_name = str(_site_field(site, "caller", "") or "")
        callee_name = str(_site_field(site, "callee", "") or "").split("::")[-1]
        site_file = str(_site_field(site, "file", "") or "")
        site_line = int(_site_field(site, "line", 0) or 0)
        if not caller_name or not callee_name or not site_file:
            continue
        if layer == "host":
            if not host_ir_keeps_file(site_file, arch):
                continue
        elif arch and is_other_arch_path(Path(site_file), arch):
            continue
        conds = [_pc for _pc in _pc_rows(site) if not _pc_skip(_pc)]
        if not conds:
            continue
        caller = _resolve_callable(codemap, caller_name, layer=layer)
        callee = _resolve_callable(codemap, callee_name, layer=layer)
        if caller is None or callee is None:
            continue
        for pc in conds:
            gtext = _pc_pretty(pc)
            if not gtext:
                continue
            fn_name = caller_name.split("::")[-1]
            okey = (site_file, fn_name, gtext)
            ordinal = used.get(okey, 0)
            used[okey] = ordinal + 1
            guard_file = str(_pc_field(pc, "file", "") or site_file)
            guard_line = int(_pc_field(pc, "line", 0) or site_line)
            # Without the branch extent the guard can only ever be matched by a
            # site sitting on the condition line itself, which no read or call
            # under the guard ever does. The range is the guarded body, not the
            # statement: a negated guard's body is the `else`, and stretching it
            # back to the condition would cover the `then` block it excludes.
            # 0 means clang gave no usable body.
            guard_body = int(_pc_field(pc, "body_start", 0) or 0)
            guard_end = int(_pc_field(pc, "line_end", 0) or 0)
            if guard_body <= 0 or guard_end < guard_body:
                guard_body = guard_end = 0
            eid = branch_id(
                side=side,
                file=guard_file,
                function=fn_name,
                guard=gtext,
                ordinal=ordinal,
            )
            br = codemap.upsert(
                EntityKind.BRANCH,
                gtext[:120],
                eid=eid,
                attrs={
                    "layer": layer,
                    "predicate": gtext,
                    "branch_kind": str(_pc_field(pc, "kind", "if") or "if"),
                    "function": fn_name,
                    "provenance": "clang_call_guard",
                    **({"guard_body_start": guard_body} if guard_body else {}),
                    **annotate_attrs(gtext),
                },
                file=guard_file,
                line=guard_line,
                line_end=guard_end or None,
                status="confirmed",
            )
            extra = {
                "file": site_file,
                "line": site_line,
                "branch_id": br.id,
                "guard": gtext[:200],
                "callee": callee_name,
            }
            codemap.mint_semantic_relation(
                RelationKind.CALLS_UNDER_GUARD,
                br.id,
                callee.id,
                provenance="clang_call_guard",
                source=SOURCE_CLANG_AST,
                extra=extra,
            )
            codemap.mint_semantic_relation(
                RelationKind.CONTROLS,
                br.id,
                callee.id,
                provenance="clang_call_guard",
                source=SOURCE_CLANG_AST,
                extra={"file": site_file, "line": site_line},
            )
            codemap.mint_semantic_relation(
                RelationKind.GUARDED_BY,
                callee.id,
                br.id,
                provenance="clang_call_guard",
                source=SOURCE_CLANG_AST,
                extra={"file": site_file, "line": site_line},
            )
            # Caller still has unguarded CALLS; this records which caller made
            # the guarded invocation without collapsing BRANCH identity.
            codemap.mint_semantic_relation(
                RelationKind.CALLS_UNDER_GUARD,
                caller.id,
                callee.id,
                provenance="clang_call_guard",
                source=SOURCE_CLANG_AST,
                extra={**extra, "via_branch": br.id},
            )
            for ident in _IDENT_RE.findall(gtext):
                operand = _unique_operand(codemap, ident)
                if operand is None or operand.id == br.id:
                    continue
                codemap.link(
                    RelationKind.READS,
                    br.id,
                    operand.id,
                    attrs={"provenance": "clang_call_guard", "symbol": ident},
                    status="confirmed",
                )
            counts += 1
    return counts


def enrich_guarded_calls(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    """Host sites from HostIR; kernel sites from walk cache (no re-parse)."""
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    if host_ir is not None:
        ingest_guarded_calls(
            codemap,
            getattr(host_ir, "call_sites", None) or [],
            side="host",
            architecture=arch,
        )
    root = Path(operator_root).expanduser().resolve()
    try:
        from ascendc_codemap_mcp.engine.passes.kernel_scan import collect_call_sites_from_walks
        import time

        calls, _decls, _controls, _prov = collect_call_sites_from_walks(
            root,
            architecture=arch,
            reachable=set(),
            filter_strict=False,
            deadline=time.perf_counter() + 30.0,
        )
    except Exception:  # noqa: BLE001
        calls = []
    if calls:
        ingest_guarded_calls(codemap, calls, side="kernel", architecture=arch)
    return codemap
