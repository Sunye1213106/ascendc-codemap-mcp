# -*- coding: utf-8 -*-
"""Refine selected-architecture Kernel calls and TilingData reads.

The broad lexical inventory is useful for discovery but not sufficient for a
sound execution graph.  This pass rescans only the architecture-selected source
files, discovers real function/method bodies from brace syntax, binds local
calls to definitions where static source identity is known, and resolves
TilingData reads against qualified owner types.  Anything ambiguous remains a
partial reference and never contributes to closure.
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ascendc_codemap_mcp.engine.cpp_lex import iter_function_defs, line_at, line_index, method_identity
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import bind_or_create, is_forbidden_callable_name
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.tiling_gaps import record_unresolved_tiling
from ascendc_codemap_mcp.engine.source_layout import includes_architecture, keep_lexical_kernel_path

_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_CONTROL = {
    "if", "else", "constexpr", "while", "for", "switch", "catch", "return",
    "sizeof", "alignof", "decltype", "static_cast", "reinterpret_cast",
    "const_cast", "dynamic_cast", "likely", "unlikely", "do",
}
_CLASS_RE = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)[^;{]*\{", re.S)
_ALIAS_RE = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.S)
_TILING_ACCESS_RE = re.compile(
    r"\b(?P<base>[A-Za-z_]\w*)\s*->\s*(?P<outer>[A-Za-z_]\w*)"
    r"(?:\s*\.\s*(?P<inner>[A-Za-z_]\w*))?"
)
_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_CALL_RE = re.compile(
    r"(?:(?P<receiver>[A-Za-z_]\w*)\s*(?:\.|->)\s*)?"
    r"(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*"
    r"(?:<[^;{}()]{0,1200}>)?\s*\("
)
_DEFINE_HEAD_RE = re.compile(r"\s*#\s*define\s+([A-Za-z_]\w*)")
_CALL_PROV = {"source_kernel_call_bound_v2", "source_kernel_macro_call_bound_v2"}
_CONFIRMED_STATUS = {"confirmed", "extracted", "verified"}
_LINE_CACHE: dict[int, list[int]] = {}
_LINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Class:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class _Macro:
    name: str
    start: int
    end: int
    line: int


@dataclass
class _Scope:
    entity: Entity
    name: str
    owner: str
    file: str
    raw: str
    masked: str
    body_start: int
    body_end: int
    params: str
    kind: str

    @property
    def param_count(self) -> int:
        return _arg_count(self.params)


def refine_kernel_calls_and_tiling_reads(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    selected = list((codemap.meta.get("kernel_tiling_closure") or {}).get("selected_kernel_files") or [])
    texts = _load_selected(root, selected, architecture)
    _LINE_CACHE.clear()
    _purge_previous_refinement(codemap)

    scopes, class_names = _discover_scopes(codemap, root, architecture, texts)
    call_stats = _bind_calls(codemap, scopes, class_names)
    read_stats = _bind_tiling_reads(codemap, scopes)
    reachable = _reachable_scopes(codemap)

    reachable_read_sites: set[str] = set()
    reachable_fields: set[str] = set()
    reachable_unresolved_reads = 0
    reachable_unresolved_calls = 0
    for rel in codemap.relations.values():
        provenance = str(rel.attrs.get("provenance") or "")
        if provenance == "source_tilingdata_read_qualified_v2":
            is_reachable = rel.src in reachable
            rel.attrs["entry_reachable"] = is_reachable
            if is_reachable:
                reachable_fields.add(rel.dst)
                for site in rel.attrs.get("sites") or []:
                    reachable_read_sites.add(f"{site.get('file')}:{site.get('line')}:{rel.src}:{rel.dst}")
                if not rel.attrs.get("sites"):
                    reachable_read_sites.add(rel.id)
        elif provenance == "source_tilingdata_read_unresolved_v2" and rel.src in reachable:
            reachable_unresolved_reads += 1
        elif provenance == "source_kernel_call_unresolved_v2" and rel.src in reachable:
            reachable_unresolved_calls += 1

    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure.update(
        {
            "kernel_scopes": len(scopes),
            "kernel_bound_call_sites": call_stats["bound_sites"],
            "kernel_bound_call_edges": call_stats["bound_edges"],
            "kernel_external_call_sites": call_stats["external"],
            "kernel_unresolved_internal_call_sites": call_stats["unresolved_internal"],
            "kernel_reachable_unresolved_internal_call_sites": reachable_unresolved_calls,
            "kernel_reachable_scopes": len(reachable),
            "tiling_read_sites": read_stats["sites"],
            "tiling_resolved_read_sites": read_stats["resolved"],
            "tiling_ambiguous_read_sites": read_stats["ambiguous"],
            "tiling_entry_reachable_read_sites": len(reachable_read_sites),
            "tiling_entry_reachable_fields": len(reachable_fields),
            "tiling_entry_reachable_unresolved_read_sites": reachable_unresolved_reads,
            "call_read_policy": "verified-body-qualified-owner/v2",
        }
    )
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _load_selected(
    root: Path, selected: list[str], architecture: str
) -> dict[Path, tuple[str, str]]:
    from ascendc_codemap_mcp.engine.parallel import map_files
    from ascendc_codemap_mcp.engine.passes.source_text_cache import masked_text, read_text

    paths: set[Path] = set()
    for raw in selected:
        path = _resolve_file(root, raw)
        if path is not None:
            paths.add(path.resolve())
    if not paths:
        arch_dir = root / "op_kernel" / architecture
        if arch_dir.is_dir():
            paths.update(
                p.resolve() for p in arch_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _SUFFIXES
            )
        kernel_root = root / "op_kernel"
        if kernel_root.is_dir():
            for path in kernel_root.iterdir():
                if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
                    continue
                raw = read_text(path)
                if includes_architecture(raw, architecture):
                    paths.add(path.resolve())
    kept = sorted(p for p in paths if keep_lexical_kernel_path(p, architecture))

    def _one(path: Path) -> tuple[Path, tuple[str, str]]:
        return path, (read_text(path), masked_text(path))

    return dict(map_files(kept, _one))


def _purge_previous_refinement(codemap: CodeMap) -> None:
    relation_prov = {
        "source_kernel_call_bound", "source_kernel_macro_call_bound", "source_kernel_call_unresolved",
        "source_kernel_frontier_bound", "source_tilingdata_read_qualified", "source_tilingdata_read_unresolved",
        "source_kernel_call_bound_v2", "source_kernel_macro_call_bound_v2", "source_kernel_call_unresolved_v2",
        "source_tilingdata_read_qualified_v2", "source_tilingdata_read_unresolved_v2",
    }
    entity_prov = {
        "source_kernel_definition", "source_kernel_macro_definition", "source_kernel_call_unresolved",
        "source_kernel_frontier_bound", "source_tilingdata_read_unresolved",
        "source_kernel_definition_v2", "source_kernel_macro_definition_v2", "source_kernel_call_unresolved_v2",
        "source_tilingdata_read_unresolved_v2",
    }
    remove_rel = {
        rid for rid, rel in codemap.relations.items()
        if str(rel.attrs.get("provenance") or "") in relation_prov
    }
    remove_ent = {
        eid for eid, ent in codemap.entities.items()
        if str(ent.attrs.get("provenance") or "") in entity_prov
    }
    for rid, rel in list(codemap.relations.items()):
        if rel.src in remove_ent or rel.dst in remove_ent:
            remove_rel.add(rid)
    for rid in remove_rel:
        codemap.relations.pop(rid, None)
    for eid in remove_ent:
        codemap.entities.pop(eid, None)


def _discover_scopes(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    texts: dict[Path, tuple[str, str]],
) -> tuple[list[_Scope], set[str]]:
    from ascendc_codemap_mcp.engine.parallel import map_files

    items = list(texts.items())

    def _parse(item: tuple[Path, tuple[str, str]]) -> tuple[Path, list[_Class], list[_Macro], list]:
        path, (raw, masked) = item
        classes = _class_scopes(masked)
        macros = _macro_spans(masked)
        hits = []
        for hit in iter_function_defs(masked):
            if any(m.start <= hit.open_brace < m.end for m in macros):
                continue
            hits.append(hit)
        return path, classes, macros, hits

    parsed = map_files(items, _parse)

    scopes: list[_Scope] = []
    class_names: set[str] = set()
    for path, classes, macros, hits in parsed:
        raw, masked = texts[path]
        file = _rel(root, path)
        class_names.update(c.name for c in classes)
        for macro in macros:
            ent = _existing_macro(codemap, macro.name, file)
            if ent is None:
                ent = codemap.upsert(
                    EntityKind.MACRO,
                    macro.name,
                    eid=f"SRCKMACROV2::{file}::{macro.line}::{macro.name}",
                    attrs={
                        "layer": "kernel", "source_definition": True,
                        "architecture": architecture, "provenance": "source_kernel_macro_definition_v2",
                    },
                    file=file, line=macro.line, status="confirmed",
                )
            scopes.append(_Scope(ent, macro.name, "", file, raw, masked, macro.start, macro.end, "", "macro"))

        newlines = line_index(raw)
        for hit in hits:
            qualified = hit.name
            params = hit.params
            open_paren = hit.open_paren
            short, owner, signature = method_identity(qualified)
            if short in _CONTROL:
                continue
            brace = hit.open_brace
            close = hit.close_brace
            if close < 0:
                continue
            containing = [c for c in classes if c.start <= brace <= c.end]
            if not owner:
                owner = (
                    min(containing, key=lambda c: c.end - c.start).name if containing else ""
                )
            owner = _base_type(owner) or owner
            line = line_at(newlines, max(0, open_paren - len(qualified) - 8))
            end_line = line_at(newlines, close)
            kernel_hits = codemap.by_name(short, kind=EntityKind.KERNEL)
            if kernel_hits and "__global__" in masked[max(0, open_paren - 400):brace]:
                ent = kernel_hits[0]
                if end_line > int(ent.line_end or 0):
                    ent.line_end = end_line
            else:
                kind = EntityKind.METHOD if owner else EntityKind.FUNCTION
                if is_forbidden_callable_name(short):
                    continue
                ent = bind_or_create(
                    codemap,
                    kind,
                    short,
                    file=file,
                    line=line,
                    owner=owner,
                    architecture=architecture,
                    attrs={
                        "owner": owner, "source_definition": True,
                        "architecture": architecture, "provenance": "source_kernel_definition_v2",
                        "signature": signature,
                    },
                    status="confirmed",
                )
                if ent is None:
                    continue
                if end_line > int(ent.line_end or 0):
                    ent.line_end = end_line
            scopes.append(_Scope(ent, short, owner, file, raw, masked, brace + 1, close, params, "method" if owner else "function"))
    # one entity/scope per physical source definition
    unique: dict[tuple[str, int, int], _Scope] = {}
    for scope in scopes:
        unique[(scope.file, scope.body_start, scope.body_end)] = scope
    return list(unique.values()), class_names


def _function_before_brace(text: str, brace: int) -> tuple[str, str, int] | None:
    window_start = max(0, brace - 2400)
    prefix = text[window_start:brace]
    close_rel = prefix.rfind(")")
    if close_rel < 0:
        return None
    close = window_start + close_rel
    tail = text[close + 1:brace].strip()
    if tail and not re.fullmatch(
        r"(?:(?:const|override|final)\s*)*(?:noexcept(?:\s*\([^)]*\))?\s*)?",
        tail,
    ):
        return None
    open_paren = _matching_backward(text, close, "(", ")")
    if open_paren < 0:
        return None
    before = text[max(window_start, open_paren - 1400):open_paren].rstrip()
    cleaned = _remove_angle_groups(before)
    match = re.search(r"((?:[A-Za-z_~]\w*\s*::\s*)*[A-Za-z_~]\w*)\s*$", cleaned)
    if not match:
        return None
    qualified = re.sub(r"\s+", "", match.group(1))
    if qualified.split("::")[-1] in _CONTROL:
        return None
    params = text[open_paren + 1:close]
    return qualified, params, open_paren


def _bind_calls(codemap: CodeMap, scopes: list[_Scope], class_names: set[str]) -> dict[str, int]:
    from ascendc_codemap_mcp.engine.parallel import map_files

    macros: dict[str, list[_Scope]] = defaultdict(list)
    funcs: dict[str, list[_Scope]] = defaultdict(list)
    methods: dict[tuple[str, str], list[_Scope]] = defaultdict(list)
    any_defs: set[str] = set()
    for scope in scopes:
        any_defs.add(scope.name)
        if scope.kind == "macro":
            macros[scope.name].append(scope)
        elif scope.owner:
            owner_key = _base_type(scope.owner) or scope.owner
            methods[(owner_key, scope.name)].append(scope)
            if owner_key != scope.owner:
                methods[(scope.owner, scope.name)].append(scope)
        else:
            funcs[scope.name].append(scope)

    # `setdefault(k, f())` evaluates `f()` on every pass, so the per-file cache
    # this dict is here to be never took effect: `_aliases` re-scanned the whole
    # masked file once per scope and 890 of the 949 results were discarded. Keep
    # the membership test explicit. `scope.masked` is the whole file's masked
    # text -- body_start/body_end index into it -- so one entry per file is the
    # same answer the first scope produced.
    aliases_by_file: dict[str, dict[str, set[str]]] = {}
    for scope in scopes:
        if scope.file not in aliases_by_file:
            aliases_by_file[scope.file] = _aliases(scope.masked, class_names)

    def _scan(scope: _Scope) -> tuple[_Scope, dict[str, set[str]], list[tuple[str, str, int, int]]]:
        aliases = aliases_by_file[scope.file]
        var_types = _variable_types(scope, class_names, aliases)
        sites = list(_call_sites(scope.masked, scope.body_start, scope.body_end))
        return scope, var_types, sites

    scanned = map_files(scopes, _scan)

    bound_sites = bound_edges = external = unresolved = 0
    for scope, var_types, sites in scanned:
        for receiver, name, open_abs, close_abs in sites:
            if name in _CONTROL or name == scope.name:
                continue
            argc = _arg_count(scope.masked[open_abs + 1:close_abs])
            candidates: list[_Scope] = []
            prov = "source_kernel_call_bound_v2"
            internal_hint = name in any_defs or name in macros
            owner_types: set[str] = set()
            if receiver:
                owner_types.update(var_types.get(receiver) or ())
                if receiver == "this" and scope.owner:
                    owner_types.add(scope.owner)
                if owner_types:
                    internal_hint = True
                    for owner in owner_types:
                        for key in {owner, _base_type(owner)}:
                            if key:
                                candidates.extend(methods.get((key, name), ()))
            else:
                if name in macros:
                    candidates.extend(macros[name])
                    prov = "source_kernel_macro_call_bound_v2"
                if scope.owner:
                    for key in {scope.owner, _base_type(scope.owner)}:
                        if key:
                            candidates.extend(methods.get((key, name), ()))
                candidates.extend(funcs.get(name, ()))

            candidates = _dedupe(candidates)
            exact = [c for c in candidates if c.param_count == argc]
            if exact:
                candidates = exact
            line = _line(scope.raw, open_abs)
            site = {"file": scope.file, "line": line, "receiver": receiver, "call": name, "argc": argc}
            if len(candidates) == 1:
                _link_site(codemap, scope.entity, candidates[0].entity, prov, site)
                bound_sites += 1
                bound_edges += 1
            elif len(candidates) > 1 and receiver and owner_types and {c.owner for c in candidates if c.owner} <= owner_types:
                for target in candidates:
                    _link_site(codemap, scope.entity, target.entity, prov, {**site, "conditional_dispatch": True})
                    bound_edges += 1
                bound_sites += 1
            else:
                if not candidates and receiver and name in any_defs:
                    candidates = _same_name_scopes(methods, funcs, name)
                if candidates:
                    n_cand = len(candidates)
                    amb = {
                        **site,
                        "ambiguous_dispatch": True,
                        "dispatch_candidates": n_cand,
                    }
                    for target in candidates:
                        _link_site(
                            codemap,
                            scope.entity,
                            target.entity,
                            prov,
                            amb,
                            status="partial",
                        )
                        bound_edges += 1
                    _call_unresolved(codemap, scope, site, candidates)
                    unresolved += 1
                elif internal_hint:
                    _call_unresolved(codemap, scope, site, [])
                    unresolved += 1
                else:
                    external += 1
    return {"bound_sites": bound_sites, "bound_edges": bound_edges, "external": external, "unresolved_internal": unresolved}


def _bind_tiling_reads(codemap: CodeMap, scopes: list[_Scope]) -> dict[str, int]:
    types = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    fields: dict[tuple[str, str], Entity] = {}
    by_name: dict[str, list[Entity]] = defaultdict(list)
    nested: dict[str, set[str]] = defaultdict(set)
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        owner = str(field.attrs.get("owner") or "")
        fields[(owner, field.name)] = field
        by_name[field.name].append(field)
        child = _base_type(str(field.attrs.get("cpp_type") or ""))
        if child in types:
            nested[field.name].add(child)

    aliases_by_file: dict[str, dict[str, set[str]]] = {}
    sites = resolved = ambiguous = 0
    known = set(types)
    for scope in scopes:
        # Same eager-default trap as in `_bind_calls`: the cache only works if
        # the miss is tested for rather than defaulted into.
        aliases = aliases_by_file.get(scope.file)
        if aliases is None:
            aliases = aliases_by_file[scope.file] = _aliases(scope.masked, known)
        var_types = _variable_types(scope, known, aliases)
        body = scope.masked[scope.body_start:scope.body_end]
        for match in _TILING_ACCESS_RE.finditer(body):
            base, outer, inner = match.group("base"), match.group("outer"), match.group("inner")
            # Keep this pass scoped to variables that are source-evidently tiling
            # values, or to nested TilingData member names.
            base_types = set(var_types.get(base) or ())
            looks_tiling = bool(base_types) or base.lower().endswith("tilingdata") or "tiling" in base.lower()
            if not looks_tiling and outer not in nested:
                continue
            sites += 1
            candidates: list[Entity] = []
            leaf = inner or outer
            if inner:
                if inner.startswith("get_"):
                    leaf = inner[4:]
                child_types = set(nested.get(outer) or ())
                candidates.extend(fields.get((owner, leaf)) for owner in child_types)
            else:
                for owner in base_types:
                    candidates.append(fields.get((owner, leaf)))
                if not candidates and len(by_name.get(leaf) or ()) == 1:
                    # Unique declaration owner is a source fact, not a short-name
                    # guess across multiple structs.
                    candidates.extend(by_name[leaf])
            candidates = _unique_entities(candidates)
            absolute = scope.body_start + match.start()
            line = _line(scope.raw, absolute)
            expression = scope.raw[absolute:scope.body_start + match.end()].replace("\n", " ").strip()
            if len(candidates) == 1:
                rel = codemap.link(
                    RelationKind.READS,
                    scope.entity.id,
                    candidates[0].id,
                    attrs={
                        "provenance": "source_tilingdata_read_qualified_v2",
                        "file": scope.file,
                        "line": line,
                        "expression": expression,
                        "field_owner": candidates[0].attrs.get("owner"),
                    },
                    status="confirmed",
                )
                rel.attrs["provenance"] = "source_tilingdata_read_qualified_v2"
                rel.attrs.setdefault("sites", []).append({"file": scope.file, "line": line, "expression": expression})
                resolved += 1
            else:
                record_unresolved_tiling(
                    codemap,
                    scope.entity,
                    role="tilingdata_read_unresolved",
                    file=scope.file,
                    line=line,
                    expression=expression,
                    extra={
                        "reason": "field_owner_ambiguous" if candidates else "field_owner_unknown",
                        "candidate_fields": [f.attrs.get("qualified_name") for f in candidates],
                        "provenance": "source_tilingdata_read_unresolved_v2",
                    },
                )
                ambiguous += 1
    return {"sites": sites, "resolved": resolved, "ambiguous": ambiguous}


def _reachable_scopes(codemap: CodeMap) -> set[str]:
    starts = {e.id for e in codemap.by_kind(EntityKind.KERNEL) if e.attrs.get("source_signature")}
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if (
            rel.kind_name() == RelationKind.CALLS.value
            and str(rel.attrs.get("provenance") or "") in _CALL_PROV
            and str(rel.status or "").lower() in _CONFIRMED_STATUS
        ):
            adj[rel.src].add(rel.dst)
    seen = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _class_scopes(masked: str) -> list[_Class]:
    out: list[_Class] = []
    for match in _CLASS_RE.finditer(masked):
        open_pos = masked.find("{", match.start(), match.end())
        close = _matching_forward(masked, open_pos, "{", "}")
        if close >= 0:
            out.append(_Class(match.group(1), open_pos + 1, close))
    return out


def _macro_spans(masked: str) -> list[_Macro]:
    lines = masked.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    out: list[_Macro] = []
    i = 0
    while i < len(lines):
        match = _DEFINE_HEAD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        start_i = i
        rest = lines[i][match.end():]
        function_like = rest.startswith("(")
        while i < len(lines) - 1 and lines[i].rstrip("\r\n").rstrip().endswith("\\"):
            i += 1
        end_i = i
        if not function_like:
            parts = [lines[start_i][match.end():]]
            parts.extend(lines[j] for j in range(start_i + 1, end_i + 1))
            body = "".join(parts).replace("\\\r\n", " ").replace("\\\n", " ")
            if not body.strip():
                i += 1
                continue
        out.append(_Macro(match.group(1), offsets[start_i], offsets[end_i] + len(lines[end_i]), start_i + 1))
        i += 1
    return out


def _existing_macro(codemap: CodeMap, name: str, file: str) -> Entity | None:
    want = str(file or "").replace("\\", "/")
    hits = [
        ent
        for ent in codemap.by_name(name, kind=EntityKind.MACRO)
        if str(ent.file or "").replace("\\", "/") == want
    ]
    return hits[0] if len(hits) == 1 else None


def _call_sites(text: str, start: int, end: int) -> Iterable[tuple[str, str, int, int]]:
    region = text[start:end]
    for match in _CALL_RE.finditer(region):
        receiver = str(match.group("receiver") or "")
        name = match.group("name").split("::")[-1]
        open_abs = start + match.end() - 1
        close = _matching_forward(text, open_abs, "(", ")")
        if close < 0 or close >= end:
            continue
        yield receiver, name, open_abs, close


def _aliases(text: str, known: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    raw = {m.group(1): m.group(2) for m in _ALIAS_RE.finditer(text)}
    for _ in range(3):
        for alias, expr in raw.items():
            tokens = set(_WORD_RE.findall(expr))
            candidates = tokens & known
            for token in tokens:
                candidates.update(out.get(token) or ())
            if candidates:
                out[alias] = candidates
    return out


def _variable_types(scope: _Scope, known: set[str], aliases: dict[str, set[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for param in _split_args(scope.params):
        _consume_declaration(param, known, aliases, out)
    body = scope.masked[scope.body_start:scope.body_end].replace("\\\n", " ")
    for stmt in body.split(";"):
        _consume_declaration(stmt, known, aliases, out)
    if scope.owner:
        out["this"].add(scope.owner)
    return out


def _consume_declaration(
    fragment: str,
    known: set[str],
    aliases: dict[str, set[str]],
    out: dict[str, set[str]],
) -> None:
    text = fragment.strip()
    if not text:
        return
    left = text.split("=", 1)[0].strip()
    match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", left)
    if not match:
        return
    var = match.group(1)
    prefix = left[:match.start(1)]
    tokens = set(_WORD_RE.findall(prefix))
    types = tokens & known
    for token in tokens:
        types.update(aliases.get(token) or ())
    if types and var not in known and var not in aliases:
        out[var].update(types)


def _link_site(
    codemap: CodeMap,
    src: Entity,
    dst: Entity,
    provenance: str,
    site: dict,
    *,
    status: str = "confirmed",
) -> None:
    from ascendc_codemap_mcp.engine.ir.codemap import relation_id

    rid = relation_id(RelationKind.CALLS.value, src.id, dst.id)
    existing = codemap.relations.get(rid)
    if (
        existing is not None
        and status == "partial"
        and str(existing.status or "").lower() in _CONFIRMED_STATUS
    ):
        return
    rel = codemap.mint_candidate_relation(
        RelationKind.CALLS,
        src.id,
        dst.id,
        provenance=provenance,
        extra=dict(site),
        status=status,
    )
    if status == "confirmed":
        rel.status = "confirmed"
    rel.attrs["provenance"] = provenance
    sites = rel.attrs.setdefault("sites", [])
    if site not in sites:
        sites.append(site)


def _call_unresolved(codemap: CodeMap, scope: _Scope, site: dict, candidates: list[_Scope]) -> None:
    name = f"{site.get('receiver') + '.' if site.get('receiver') else ''}{site.get('call')}"
    ref = codemap.upsert(
        EntityKind.METHOD,
        name,
        eid=f"SRCKCALLV2::{scope.file}::{site['line']}::{scope.entity.id}::{name}",
        attrs={
            "call_target": site.get("call"), "receiver": site.get("receiver"),
            "candidate_definitions": [c.entity.id for c in candidates],
            "internal_unresolved": True, "provenance": "source_kernel_call_unresolved_v2",
        },
        file=scope.file, line=int(site["line"]), status="partial", confidence=0.5,
    )
    codemap.link(
        RelationKind.CALLS,
        scope.entity.id,
        ref.id,
        attrs={"provenance": "source_kernel_call_unresolved_v2", **site},
        status="partial", confidence=0.5,
    )


def _remove_angle_groups(text: str) -> str:
    out = list(text)
    depth = 0
    for idx in range(len(text) - 1, -1, -1):
        ch = text[idx]
        if ch == ">":
            depth += 1
            out[idx] = " "
        elif ch == "<" and depth:
            depth -= 1
            out[idx] = " "
        elif depth:
            if ch != "\n":
                out[idx] = " "
    return "".join(out)


def _matching_forward(text: str, pos: int, opener: str, closer: str) -> int:
    if pos < 0 or pos >= len(text) or text[pos] != opener:
        return -1
    depth = 0
    for idx in range(pos, len(text)):
        if text[idx] == opener:
            depth += 1
        elif text[idx] == closer:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _matching_backward(text: str, pos: int, opener: str, closer: str) -> int:
    if pos < 0 or pos >= len(text) or text[pos] != closer:
        return -1
    depth = 0
    for idx in range(pos, -1, -1):
        if text[idx] == closer:
            depth += 1
        elif text[idx] == opener:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _arg_count(text: str) -> int:
    stripped = text.strip()
    if not stripped or stripped == "void":
        return 0
    return len(_split_args(stripped))


def _split_args(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "(<[{":
            depth += 1
            buf.append(ch)
        elif ch in ")>]}":
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


def _same_name_scopes(
    methods: dict[tuple[str, str], list[_Scope]],
    funcs: dict[str, list[_Scope]],
    name: str,
) -> list[_Scope]:
    found: list[_Scope] = []
    for (owner, mname), scopes in methods.items():
        del owner
        if mname == name:
            found.extend(scopes)
    found.extend(funcs.get(name) or ())
    return _dedupe(found)


def _dedupe(scopes: Iterable[_Scope]) -> list[_Scope]:
    out: list[_Scope] = []
    seen: set[str] = set()
    for scope in scopes:
        if scope.entity.id in seen:
            continue
        seen.add(scope.entity.id)
        out.append(scope)
    return out


def _unique_entities(items: Iterable[Entity | None]) -> list[Entity]:
    out: list[Entity] = []
    seen: set[str] = set()
    for item in items:
        if item is None or item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    return out


def _base_type(raw: str) -> str:
    text = re.sub(r"\b(?:const|volatile|typename|class|struct)\b", " ", raw or "")
    text = text.replace("*", " ").replace("&", " ").strip()
    text = re.sub(r"<.*>", "", text).strip()
    return text.split("::")[-1].strip().split()[-1] if text else ""


def _mask_non_code(text: str) -> str:
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
                quote = ch; out[i] = " "; i += 1; state = "string"; continue
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
        if state == "string":
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


def _resolve_file(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _line(text: str, offset: int) -> int:
    key = id(text)
    with _LINE_LOCK:
        idx = _LINE_CACHE.get(key)
        if idx is None:
            idx = line_index(text)
            _LINE_CACHE[key] = idx
    return line_at(idx, offset)
