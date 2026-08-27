# -*- coding: utf-8 -*-
"""Bind source-declared TilingKey fields to Host ``GET_TPL_TILING_KEY`` args.

The positional packed-key call is a source contract. This pass records that
contract and identifies the concrete Host symbol used for every argument. It
must not derive the full value function and must not guess across ambiguous
short names.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.cpp_expr import parse_expr
from ascendc_codemap_mcp.engine.expr_ir import Bin, Call, Const, Expr, Ite, Ref, Select, Un, Unknown
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.tiling_binding import KEY_CHAIN_PACKING, KEY_CHAIN_PRODUCER
from ascendc_codemap_mcp.engine.passes.source_contract import iter_bitpack_dims, iter_packing_helper_calls
from ascendc_codemap_mcp.engine.passes.host_defuse import _compile_symbols, _is_compile_reference
from ascendc_codemap_mcp.engine.passes.symbol_identity import normalize_symbol, short_symbol
from ascendc_codemap_mcp.engine.passes.source_text_cache import mask_cached, read_text
from ascendc_codemap_mcp.engine.source_layout import (
    _path_is_under,
    quoted_include_basenames,
    selected_host_files,
)
from ascendc_codemap_mcp.engine.tpl_dsl import expand_tpl_source, load_quoted_include_texts
from ascendc_codemap_mcp.engine.cpp_lex import containing_function, iter_function_defs, line_at, line_index

_CALL_TOKEN = "GET_TPL_TILING_KEY"
_SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_FUNC_CACHE: dict[int, list] = {}
_LINE_CACHE: dict[int, list[int]] = {}
_RUNTIME_KINDS = {EntityKind.VARIABLE.value, EntityKind.FIELD.value}
_DIRECT_ROOT_KINDS = {
    EntityKind.INPUT.value,
    EntityKind.COMPILE_VAR.value,
    EntityKind.MACRO.value,
    EntityKind.BUILD_VARIANT.value,
    EntityKind.ARCH.value,
}
_PACKING_SOURCE_KINDS = _RUNTIME_KINDS | _DIRECT_ROOT_KINDS
_IF_COND_RE = re.compile(r"\bif\s*\((.+?)\)", re.S)
# SetTilingKey(GetTilingKey()) / return tilingKey_ write the already-computed
# packed value. They are launch/pass-through sites, not a new packing formula.
_PASSTHROUGH_KEY_RE = re.compile(
    r"^(?:(?:this->|context_->)?"
    r"GetTilingKey\s*\(\s*\)"
    r"|(?:this->)?(?:tilingKey_|tiling_key_))$"
)


def _is_passthrough_key_expr(expr: str) -> bool:
    return bool(_PASSTHROUGH_KEY_RE.match((expr or "").strip()))


def bind_host_tiling_key_expressions(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    architecture = str(architecture or getattr(codemap, "architecture", "") or "")
    root = Path(operator_root).expanduser().resolve()
    all_declared = sorted(
        (
            e
            for e in codemap.by_kind(EntityKind.TILING_KEY)
            if e.attrs.get("source_declared")
        ),
        key=lambda e: (int(e.attrs.get("decl_order") or 0), e.name),
    )
    meta_names = [str(n) for n in (codemap.meta.get("source_declared_tiling_keys") or []) if n]
    if meta_names:
        by_name = {e.name: e for e in all_declared}
        keys = [by_name[name] for name in meta_names if name in by_name]
    else:
        tpl_keys = [
            e
            for e in all_declared
            if str(e.attrs.get("provenance") or "") == "source_tpl_args_decl"
        ]
        keys = tpl_keys or all_declared
    if not keys:
        keys = sorted(codemap.by_kind(EntityKind.TILING_KEY), key=lambda e: e.name)
    if not keys:
        codemap.meta["host_tiling_key_packing"] = {"calls": 0, "fields_bound": 0, "declared": 0}
        return codemap

    exact_index, short_index = _symbol_indexes(codemap)
    calls = 0
    bound_names: set[str] = set()
    mismatches: list[dict[str, Any]] = []
    ambiguous_sources: list[dict[str, Any]] = []
    _FUNC_CACHE.clear()
    _LINE_CACHE.clear()

    host_files: list[tuple[Path, str]] = []
    for path in selected_host_files(root, architecture):
        raw = read_text(path)
        extras = load_quoted_include_texts(path)
        host_files.append((path, expand_tpl_source(raw, extras)))

    compile_symbols = _compile_symbols(codemap, host_files)
    schema_headers = {
        Path(str(key.file or "")).name.lower()
        for key in keys
        if key.file
    }
    local_host = [(path, text) for path, text in host_files if _path_is_under(path, root)]
    other_host = [(path, text) for path, text in host_files if not _path_is_under(path, root)]

    def _bind_tpl_from_files(files: list[tuple[Path, str]]) -> None:
        nonlocal calls
        for path, text in files:
            file = _rel(root, path)
            included = quoted_include_basenames(path)
            other_tpl = {
                name
                for name in included
                if ("tiling_key" in name or name.endswith("tilingkey.h"))
                and schema_headers
                and name not in schema_headers
            }
            if other_tpl and not (included & schema_headers) and _CALL_TOKEN not in text:
                continue
            key_provs = {str(e.attrs.get("provenance") or "") for e in keys}
            for start, _end, args, helper_name in iter_packing_helper_calls(text):
                if "source_tpl_args_decl" in key_provs and helper_name != _CALL_TOKEN:
                    continue
                calls += 1
                line = _line(text, start)
                function = _containing_function(text, start)
                provenance = (
                    "source_get_tpl_tiling_key"
                    if helper_name == _CALL_TOKEN
                    else "source_packing_helper"
                )
                if len(args) != len(keys):
                    mismatches.append(
                        {
                            "file": file,
                            "line": line,
                            "argument_count": len(args),
                            "declared_key_count": len(keys),
                        }
                    )
                    continue
                for index, (key, expr) in enumerate(zip(keys, args)):
                    expr = expr.strip()
                    node = codemap.upsert(
                        EntityKind.PREDICATE,
                        expr,
                        eid=f"HOSTKEYEXPR::{file}::{line}::{index}",
                        attrs={
                            "predicate_role": "host_tiling_key_argument",
                            "tiling_key": key.name,
                            "argument_index": index,
                            "expression": expr,
                            "function": function,
                            "provenance": provenance,
                        },
                        file=file,
                        line=line,
                        status="confirmed",
                    )
                    codemap.link(
                        RelationKind.DERIVES,
                        node.id,
                        key.id,
                        attrs={
                            "provenance": provenance,
                            "argument_index": index,
                            "expression": expr,
                            "file": file,
                            "line": line,
                            "function": function,
                            "key_chain_role": KEY_CHAIN_PACKING,
                        },
                        status="confirmed",
                    )
                    key.attrs.setdefault("host_packing_expressions", [])
                    if expr not in key.attrs["host_packing_expressions"]:
                        key.attrs["host_packing_expressions"].append(expr)
                    site = {
                        "file": file,
                        "line": line,
                        "expression": expr[:300],
                        "function": function,
                    }
                    key.attrs.setdefault("packing_value_sites", [])
                    if site not in key.attrs["packing_value_sites"]:
                        key.attrs["packing_value_sites"].append(site)
                    bound_names.add(key.name)
                    ambiguity = _link_expression_sources(
                        codemap,
                        node,
                        expr,
                        exact_index=exact_index,
                        short_index=short_index,
                        compile_symbols=compile_symbols,
                        file=file,
                        line=line,
                        function=function,
                        key_name=key.name,
                    )
                    if ambiguity:
                        ambiguous_sources.append(ambiguity)

    def _bind_non_tpl_from_files(files: list[tuple[Path, str]]) -> None:
        nonlocal calls
        if len(bound_names) >= len(keys) or not files:
            return
        extra_calls, extra_bound = _bind_non_tpl_packing(
            codemap,
            root,
            files,
            keys,
            bound_names,
            exact_index=exact_index,
            short_index=short_index,
            compile_symbols=compile_symbols,
        )
        calls += extra_calls
        bound_names.update(extra_bound)

    # This operator's host first. Sibling / 3rd-party GET_TPL often has a
    # coincidental arity and must not zip onto a TILING_KEY_IS catalog.
    _bind_tpl_from_files(local_host)
    _bind_non_tpl_from_files(local_host)
    if len(bound_names) < len(keys):
        _bind_tpl_from_files(other_host)
        _bind_non_tpl_from_files(other_host)

    codemap.meta["host_tiling_key_packing"] = {
        "calls": calls,
        "fields_bound": len(bound_names),
        "declared": len(keys),
        "bound_keys": sorted(bound_names),
        "argument_count_mismatches": mismatches,
        "ambiguous_source_count": len(ambiguous_sources),
        "ambiguous_sources": ambiguous_sources[:50],
    }
    return codemap


_RETURN_RE = re.compile(r"\breturn\s+([^;]+);")
_GET_TILING_KEY_NAME_RE = re.compile(r"GetTilingKey\s*$")
_ASSIGN_TILING_KEY_RE = re.compile(
    r"\b(?P<lhs>tilingKey_|tiling_key_|tilingKey)\s*(?:\+=|=)\s*(?P<rhs>[^;]+);"
)
_CATALOG_KEY_PROVENANCE = {
    "source_tiling_key_is",
    "source_tiling_key_define",
    "source_tiling_key_constexpr",
    "source_set_tiling_key",
}
_DIM_KEY_PROVENANCE = {
    "source_tpl_args_decl",
    "source_packing_helper_arg",
    "source_bitpack_dim",
}


def _catalog_keys(keys: list[Entity]) -> list[Entity]:
    """Integer / TILING_KEY_IS catalog — not packing dimension fields."""
    hits = [
        key
        for key in keys
        if str(key.attrs.get("provenance") or "") in _CATALOG_KEY_PROVENANCE
    ]
    if hits:
        return hits
    if any(str(key.attrs.get("provenance") or "") in _DIM_KEY_PROVENANCE for key in keys):
        return []
    return list(keys)


def _bind_non_tpl_packing(
    codemap: CodeMap,
    root: Path,
    host_files: list[tuple[Path, str]],
    keys: list[Entity],
    bound_names: set[str],
    *,
    exact_index: dict[str, list[Entity]],
    short_index: dict[str, list[Entity]],
    compile_symbols: dict[str, Any],
) -> tuple[int, set[str]]:
    """Pack integer/macro keys from GetTilingKey / SetTilingKey when there is no TPL call."""
    key_by_name = {k.name: k for k in keys}
    calls = 0
    extra: set[str] = set()
    int_defs: dict[str, int] = {}
    for _path, text in host_files:
        int_defs.update(_int_literals(text))
    key_values: dict[str, int] = {}
    for key in keys:
        if key.name in int_defs:
            key_values[key.name] = int_defs[key.name]
        elif str(key.attrs.get("value") or "").isdigit():
            key_values[key.name] = int(key.attrs["value"])
        elif key.name.isdigit():
            key_values[key.name] = int(key.name)

    def keys_for_expr(expr: str) -> list[Entity]:
        hits: list[Entity] = []
        seen: set[str] = set()
        for key in keys:
            if re.search(rf"\b{re.escape(key.name)}\b", expr) and key.id not in seen:
                hits.append(key)
                seen.add(key.id)
        ident = expr.split("::")[-1].strip()
        hit = key_by_name.get(ident)
        if hit is not None and hit.id not in seen:
            hits.append(hit)
            seen.add(hit.id)
        value: int | None = int_defs.get(ident)
        if value is None and ident.isdigit():
            value = int(ident)
        if value is not None:
            for key in keys:
                if key_values.get(key.name) == value and key.id not in seen:
                    hits.append(key)
                    seen.add(key.id)
        return hits

    def bind(key: Entity, expr: str, file: str, line: int, function: str, provenance: str) -> Entity | None:
        expr = expr.strip()
        if not expr:
            return None
        node = codemap.upsert(
            EntityKind.PREDICATE,
            expr,
            eid=f"HOSTKEYEXPR::{file}::{line}::{key.name}",
            attrs={
                "predicate_role": "host_tiling_key_argument",
                "tiling_key": key.name,
                "expression": expr,
                "function": function,
                "provenance": provenance,
            },
            file=file,
            line=line,
            status="confirmed",
        )
        codemap.link(
            RelationKind.DERIVES,
            node.id,
            key.id,
            attrs={
                "provenance": provenance,
                "expression": expr,
                "file": file,
                "line": line,
                "function": function,
                "key_chain_role": KEY_CHAIN_PACKING,
            },
            status="confirmed",
        )
        key.attrs.setdefault("host_packing_expressions", [])
        if expr not in key.attrs["host_packing_expressions"]:
            key.attrs["host_packing_expressions"].append(expr)
        extra.add(key.name)
        site = {
            "file": file,
            "line": line,
            "expression": expr[:300],
            "function": function,
        }
        key.attrs.setdefault("packing_value_sites", [])
        if site not in key.attrs["packing_value_sites"]:
            key.attrs["packing_value_sites"].append(site)
        _link_expression_sources(
            codemap,
            node,
            expr,
            exact_index=exact_index,
            short_index=short_index,
            compile_symbols=compile_symbols,
            file=file,
            line=line,
            function=function,
            key_name=key.name,
        )
        return node

    for path, text in host_files:
        file = _rel(root, path)
        for dim in iter_bitpack_dims(text):
            key = key_by_name.get(dim["name"])
            if key is None:
                continue
            if str(key.attrs.get("provenance") or "") != "source_bitpack_dim":
                continue
            expr = str(dim.get("expr") or "").strip()
            if not expr:
                continue
            line = _line(text, int(dim["offset"]))
            function = _containing_function(text, int(dim["offset"]))
            calls += 1
            bind(key, expr, file, line, function, "source_bitpack_dim")

    for path, text in host_files:
        file = _rel(root, path)
        for match in _ASSIGN_TILING_KEY_RE.finditer(text):
            rhs = match.group("rhs").strip()
            line = _line(text, match.start())
            function = _containing_function(text, match.start())
            calls += 1
            hits = keys_for_expr(rhs)
            if hits:
                for key in hits:
                    bind(key, rhs, file, line, function, "source_assign_tiling_key")
            else:
                catalog = _catalog_keys(keys)
                if catalog:
                    for key in catalog:
                        bind(key, rhs, file, line, function, "source_assign_tiling_key")
                elif len(keys) == 1:
                    bind(keys[0], rhs, file, line, function, "source_assign_tiling_key")
        for start, _end, args_text in _calls(text, "SetTilingKey"):
            args = [a.strip() for a in _split_args(args_text) if a.strip()]
            if not args:
                continue
            expr = args[0]
            if _is_passthrough_key_expr(expr):
                continue
            line = _line(text, start)
            function = _containing_function(text, start)
            calls += 1
            hits = keys_for_expr(expr)
            if hits:
                for key in hits:
                    bind(key, expr, file, line, function, "source_set_tiling_key")
            elif re.search(r"\bGetTilingKey\s*\(", expr):
                catalog = _catalog_keys(keys)
                for key in catalog:
                    bind(key, expr, file, line, function, "source_set_tiling_key")
            else:
                catalog = _catalog_keys(keys)
                if catalog:
                    for key in catalog:
                        bind(key, expr, file, line, function, "source_set_tiling_key")
                elif len(keys) == 1:
                    bind(keys[0], expr, file, line, function, "source_set_tiling_key")
        for hit in _funcs(text):
            name = hit.name
            if not _GET_TILING_KEY_NAME_RE.search(name.split("::")[-1] if "::" in name else name):
                if not name.endswith("GetTilingKey"):
                    continue
            open_brace = hit.open_brace
            close = hit.close_brace
            if close < 0:
                continue
            body = text[open_brace : close + 1]
            function = name
            conditions = [m.group(1).strip() for m in _IF_COND_RE.finditer(body) if m.group(1).strip()]
            packed_nodes: list[tuple[Entity, Entity]] = []
            for ret in _RETURN_RE.finditer(body):
                expr = ret.group(1).strip()
                if _is_passthrough_key_expr(expr):
                    continue
                line = _line(text, open_brace + ret.start())
                calls += 1
                hits = keys_for_expr(expr)
                if hits:
                    for key in hits:
                        node = bind(key, expr, file, line, function, "source_get_tiling_key")
                        if node is not None:
                            packed_nodes.append((key, node))
                elif re.fullmatch(r"tilingKey_|tiling_key_", expr.strip()):
                    catalog = _catalog_keys(keys)
                    for key in catalog:
                        node = bind(key, expr, file, line, function, "source_get_tiling_key")
                        if node is not None:
                            packed_nodes.append((key, node))
                else:
                    catalog = _catalog_keys(keys)
                    if catalog:
                        for key in catalog:
                            node = bind(key, expr, file, line, function, "source_get_tiling_key")
                            if node is not None:
                                packed_nodes.append((key, node))
                    elif len(keys) == 1:
                        node = bind(keys[0], expr, file, line, function, "source_get_tiling_key")
                        if node is not None:
                            packed_nodes.append((keys[0], node))
            for cond in conditions:
                for key, node in packed_nodes:
                    _link_expression_sources(
                        codemap,
                        node,
                        cond,
                        exact_index=exact_index,
                        short_index=short_index,
                        compile_symbols=compile_symbols,
                        file=file,
                        line=_line(text, open_brace),
                        function=function,
                        key_name=key.name,
                    )
    return calls, extra


def _entity_spellings(ent: Entity) -> set[str]:
    candidates = {ent.name}
    norm = ((ent.attrs.get("identity") or {}).get("normalized") or {})
    for key in ("source_name", "qualified_name"):
        value = norm.get(key) if isinstance(norm, dict) else None
        if value:
            candidates.add(str(value))
        value = ent.attrs.get(key)
        if value:
            candidates.add(str(value))
    owner = str(ent.attrs.get("owner") or "")
    if owner and ent.name:
        candidates.add(f"{owner}.{ent.name}")
        candidates.add(f"{owner}::{ent.name}")
    return {normalize_symbol(value) for value in candidates if normalize_symbol(value)}


def _symbol_indexes(codemap: CodeMap) -> tuple[dict[str, list[Entity]], dict[str, list[Entity]]]:
    """Build separate canonical and short-name indexes.

    A previous implementation inserted short aliases into the same map as exact
    identities, so looking up bare ``x`` incorrectly looked "exact" when the
    only entities were ``foo.x`` and ``bar.x``. Keeping the namespaces separate
    makes the fallback policy explicit and auditable.
    """
    exact: dict[str, list[Entity]] = {}
    short: dict[str, list[Entity]] = {}
    for ent in codemap.entities.values():
        if ent.kind_name() not in _PACKING_SOURCE_KINDS:
            continue
        for spelling in _entity_spellings(ent):
            exact.setdefault(spelling, []).append(ent)
            short.setdefault(short_symbol(spelling), []).append(ent)
    return exact, short


def _dedupe_entities(values: list[Entity]) -> list[Entity]:
    seen: set[str] = set()
    out: list[Entity] = []
    for ent in values:
        if ent.id not in seen:
            seen.add(ent.id)
            out.append(ent)
    return out


def _source_matches(
    exact_index: dict[str, list[Entity]],
    short_index: dict[str, list[Entity]],
    token: str,
) -> tuple[list[Entity], bool]:
    """Prefer canonical exact identity; short-name fallback must be unique."""
    normalized = normalize_symbol(token)
    exact = _dedupe_entities(list(exact_index.get(normalized) or []))
    if exact:
        runtime = [e for e in exact if e.kind_name() in _RUNTIME_KINDS]
        if runtime:
            preferred = sorted(
                runtime,
                key=lambda e: (
                    0 if ("." in normalized and e.kind_name() == EntityKind.FIELD.value) else 1,
                    0 if ("." not in normalized and e.kind_name() == EntityKind.VARIABLE.value) else 1,
                    e.id,
                ),
            )
            return [preferred[0]], False
        return exact, False

    # Member paths must not collapse to a unique short name (API attr
    # ``inputLayout`` is not ``tilingKeyInfo_.inputLayout``).
    if "." in normalized:
        return [], False

    short = _dedupe_entities(list(short_index.get(short_symbol(normalized)) or []))
    # A short fallback is safe only when all hits describe one canonical symbol.
    canonical = {
        spelling
        for ent in short
        for spelling in _entity_spellings(ent)
        if short_symbol(spelling) == short_symbol(normalized)
    }
    if len(short) == 1 and len(canonical) == 1:
        return short, False
    return [], bool(short)


def _mark_host_key_source(
    source: Entity,
    key_name: str,
    *,
    file: str,
    line: int,
    function: str,
) -> None:
    if source.kind_name() not in _RUNTIME_KINDS:
        return
    source.attrs["host_key_argument"] = True
    source.attrs["canonical_symbol"] = normalize_symbol(source.name)
    keys = source.attrs.setdefault("host_key_argument_keys", [])
    if key_name not in keys:
        keys.append(key_name)
    sites = source.attrs.setdefault("host_key_use_sites", [])
    site = {"file": file, "line": line, "function": function, "key": key_name}
    if site not in sites:
        sites.append(site)


def _link_expression_sources(
    codemap: CodeMap,
    expression: Entity,
    expr: str,
    *,
    exact_index: dict[str, list[Entity]],
    short_index: dict[str, list[Entity]],
    compile_symbols: set[str],
    file: str,
    line: int,
    function: str,
    key_name: str,
) -> dict[str, Any] | None:
    tree = parse_expr(expr)
    linked: set[str] = set()
    ambiguous_tokens: list[str] = []
    residual = False

    def _link_entity(source: Entity, token: str, provenance: str) -> None:
        if source.id == expression.id or source.id in linked:
            return
        if source.kind_name() not in _PACKING_SOURCE_KINDS:
            return
        _mark_host_key_source(source, key_name, file=file, line=line, function=function)
        codemap.link(
            RelationKind.DERIVES,
            source.id,
            expression.id,
            attrs={
                "provenance": provenance,
                "symbol": token,
                "canonical_symbol": normalize_symbol(token),
                "file": file,
                "line": line,
                "function": function,
                "key_chain_role": KEY_CHAIN_PRODUCER,
            },
            status="confirmed",
        )
        linked.add(source.id)

    def _link_compile(token: str, value: Any = None) -> None:
        eid = f"HOSTKEYCONST::{key_name}::{token}"
        attrs = {
            "compile_root": True,
            "provenance": "source_get_tpl_tiling_key_literal" if value is not None else "source_get_tpl_tiling_key_symbol",
        }
        if value is not None:
            attrs["value"] = value
        root = codemap.upsert(
            EntityKind.COMPILE_VAR,
            f"{key_name}={token}" if value is not None else token,
            eid=eid if value is None else f"HOSTKEYCONST::{key_name}::{value}",
            attrs=attrs,
            file=file,
            line=line,
            status="confirmed",
        )
        if root.id in linked:
            return
        provenance = str(attrs["provenance"])
        codemap.link(
            RelationKind.DERIVES,
            root.id,
            expression.id,
            attrs={"provenance": provenance, "symbol": token, "file": file, "line": line, "function": function},
            status="confirmed",
        )
        linked.add(root.id)

    def _link_runtime(token: str, *, prefer_field: bool) -> None:
        matches, ambiguous = _source_matches(exact_index, short_index, token)
        if ambiguous:
            ambiguous_tokens.append(normalize_symbol(token))
        linked_before = len(linked)
        for source in matches:
            _link_entity(source, token, "source_get_tpl_tiling_key_symbol")
        if len(linked) > linked_before:
            return
        kind = EntityKind.FIELD if prefer_field or "." in token or "->" in token else EntityKind.VARIABLE
        local = codemap.upsert(
            kind,
            normalize_symbol(token),
            eid=f"HOSTKEYVAR::{file}::{line}::{key_name}::{normalize_symbol(token)}",
            attrs={
                "source_name": short_symbol(token),
                "canonical_symbol": normalize_symbol(token),
                "host_key_argument": True,
                "host_key_argument_keys": [key_name],
                "host_key_use_sites": [{"file": file, "line": line, "function": function, "key": key_name}],
                "provenance": "source_get_tpl_tiling_key_symbol",
            },
            file=file,
            line=line,
            status="confirmed",
        )
        _mark_host_key_source(local, key_name, file=file, line=line, function=function)
        if local.id in linked:
            return
        codemap.link(
            RelationKind.DERIVES,
            local.id,
            expression.id,
            attrs={
                "provenance": "source_get_tpl_tiling_key_symbol",
                "canonical_symbol": normalize_symbol(token),
                "file": file,
                "line": line,
                "function": function,
                "key_chain_role": KEY_CHAIN_PRODUCER,
            },
            status="confirmed",
        )
        linked.add(local.id)

    def _walk(node: Expr) -> None:
        nonlocal residual
        if isinstance(node, Unknown):
            residual = True
            return
        if isinstance(node, Const):
            _link_compile(repr(node.value), value=node.value)
            return
        if isinstance(node, Ref):
            symbol = normalize_symbol(node.symbol)
            if _is_compile_reference(symbol, compile_symbols):
                _link_compile(symbol)
                return
            input_hits = codemap.by_name(node.symbol, kind=EntityKind.INPUT) or codemap.by_name(
                short_symbol(node.symbol), kind=EntityKind.INPUT
            )
            if input_hits:
                _link_entity(input_hits[0], node.symbol, "source_get_tpl_tiling_key_symbol")
                return
            _link_runtime(node.symbol, prefer_field=False)
            return
        if isinstance(node, Call) and node.func.startswith("field:"):
            path = _member_path(node)
            if path:
                _link_runtime(path, prefer_field=True)
                return
        children: list[Expr] = []
        if isinstance(node, Un):
            children = [node.arg]
        elif isinstance(node, Bin):
            children = [node.left, node.right]
        elif isinstance(node, Ite):
            children = [node.cond, node.then, node.else_]
        elif isinstance(node, Call):
            children = list(node.args)
        elif isinstance(node, Select):
            children = [node.array, node.index]
        else:
            residual = True
            return
        for child in children:
            _walk(child)

    _walk(tree)
    if residual and not linked:
        return {"key": key_name, "file": file, "line": line, "tokens": ["<unresolved-expr>"]}
    if ambiguous_tokens:
        return {"key": key_name, "file": file, "line": line, "tokens": sorted(set(ambiguous_tokens))}
    return None


def _member_path(expr: Expr) -> str | None:
    parts: list[str] = []
    cur: Expr | None = expr
    while isinstance(cur, Call) and cur.func.startswith("field:"):
        parts.append(cur.func[len("field:") :])
        cur = cur.args[0] if cur.args else None
    if isinstance(cur, Ref):
        parts.append(cur.symbol)
    elif cur is not None:
        return None
    return ".".join(reversed(parts)) if len(parts) >= 2 else None


def _calls(text: str, token: str):
    pattern = re.compile(rf"\b{re.escape(token)}\s*\(")
    for match in pattern.finditer(text):
        open_pos = text.find("(", match.start(), match.end())
        close_pos = _matching_paren(text, open_pos)
        if close_pos >= 0:
            yield match.start(), close_pos + 1, text[open_pos + 1 : close_pos]


def _matching_paren(text: str, open_pos: int) -> int:
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
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


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


def _funcs(text: str):
    key = id(text)
    hits = _FUNC_CACHE.get(key)
    if hits is None:
        hits = iter_function_defs(mask_cached(text))
        _FUNC_CACHE[key] = hits
    return hits


def _containing_function(text: str, offset: int) -> str:
    return containing_function(_funcs(text), offset)


def _split_args(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    escape = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closes = set(pairs.values())
    for ch in text:
        if quote:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
            buf.append(ch)
        elif ch in pairs:
            depth += 1
            buf.append(ch)
        elif ch in closes:
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


_INT_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([-+]?0[xX][0-9A-Fa-f]+|[-+]?\d+)[uUlL]*\s*$",
    re.M,
)
_INT_CONST_RE = re.compile(
    r"\b(?:const\s+static|static\s+const|static\s+constexpr|constexpr\s+static|"
    r"constexpr)\s+(?:const\s+)?"
    r"(?:u?int(?:32|64)_t|int)\s+([A-Za-z_]\w*)\s*=\s*"
    r"([-+]?0[xX][0-9A-Fa-f]+|[-+]?\d+)[uUlL]*\s*;"
)


def _int_literals(text: str) -> dict[str, int]:
    """Integer macros / constexpr used as packed TilingKey values."""
    out: dict[str, int] = {}
    for name, raw in _INT_DEFINE_RE.findall(text or ""):
        try:
            out[name] = int(raw, 0)
        except ValueError:
            continue
    for name, raw in _INT_CONST_RE.findall(text or ""):
        try:
            out[name] = int(raw, 0)
        except ValueError:
            continue
    return out


def _line(text: str, offset: int) -> int:
    key = id(text)
    idx = _LINE_CACHE.get(key)
    if idx is None:
        idx = line_index(text)
        _LINE_CACHE[key] = idx
    return line_at(idx, offset)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()
