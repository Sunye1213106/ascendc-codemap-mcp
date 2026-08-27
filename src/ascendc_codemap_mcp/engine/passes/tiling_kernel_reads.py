# -*- coding: utf-8 -*-
"""Build source-sound Kernel reads of TilingData fields.

Only pointer/member chains that are structurally TilingData-like are accepted;
ordinary ``this->member`` accesses are never treated as TilingData reads.  The
pass supports both ``tilingData->field`` and ``this->tilingData->nested.field``
forms and resolves nested fields through their declared TilingData owner type,
including ``std::conditional`` member types.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.type_identity import macro_type_aliases, merge_unique_macro_aliases
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.tiling_gaps import record_unresolved_tiling

_ACCESS_RE = re.compile(
    r"(?:(?:this\s*(?:->|\.)\s*)?(?P<base>[A-Za-z_]\w*))\s*(?:->|\.)\s*"
    r"(?P<outer>[A-Za-z_]\w*)"
    r"(?:\s*(?:\.|->)\s*(?P<inner>[A-Za-z_]\w*))?"
)
_GET_TILING_WITH_RE = re.compile(
    r"GET_TILING_DATA_WITH_STRUCT\s*\(\s*(?P<type>[A-Za-z_:]\w*(?:::\w+)*)\s*,\s*(?P<var>[A-Za-z_]\w*)"
)
_GET_TILING_BARE_VARS_RE = re.compile(
    r"\bGET_TILING_DATA\s*\(\s*(?P<var>[A-Za-z_]\w*)\s*,"
)
_GET_TILING_MEMBER_RE = re.compile(
    r"GET_TILING_DATA_MEMBER\s*\(\s*(?P<type>[A-Za-z_:]\w*(?:::\w+)*)\s*,\s*"
    r"(?P<member>[A-Za-z_]\w*)\s*,\s*(?P<var>[A-Za-z_]\w*)"
)
# Pointer/reference declarators often insert ``const`` / ``__restrict`` between
# ``*`` and the name (``Type *__restrict tilingData``). Those tokens are
# qualifiers, not the variable.
_DECL_AFTER_TYPE = (
    r"(?:\s*(?:const|volatile|mutable|__restrict(?:__)?|restrict|[*&]))*"
)
_DECL_QUAL_NAMES = frozenset(
    {"const", "volatile", "mutable", "restrict", "__restrict", "__restrict__"}
)
_DEFAULT_REG_RE = re.compile(r"\bREGISTER_TILING_DEFAULT\s*\(\s*([A-Za-z_:][A-Za-z0-9_:]*)\s*\)")
_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_ALIAS_RE = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.S)
_BOUND_CALLS = {
    "source_kernel_call_bound_v2", "source_kernel_macro_call_bound_v2",
    "source_kernel_call_bound_v3", "source_kernel_call_dispatch_set_v3",
}


def rebuild_verified_tiling_reads(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    selected = list((codemap.meta.get("kernel_tiling_closure") or {}).get("selected_kernel_files") or [])
    texts = _load(root, selected)
    _purge_old_reads(codemap)

    types = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    known_types = set(types)
    fields: dict[tuple[str, str], Entity] = {}
    nested: dict[str, set[str]] = defaultdict(set)
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        owner = str(field.attrs.get("owner") or "")
        fields[(owner, field.name)] = field
        nested[field.name].update(_referenced_types(str(field.attrs.get("cpp_type") or ""), known_types))

    scopes = [
        e
        for e in codemap.entities.values()
        if str(e.attrs.get("provenance") or "") in {
            "source_kernel_definition_v2",
            "source_kernel_macro_definition_v2",
            "source_kernel_definition",
            "source_kernel_macro_definition",
        }
    ]
    aliases_by_file = {file: _aliases(raw, known_types) for file, raw in texts.items()}
    global_macros = merge_unique_macro_aliases(*texts.values(), known=known_types)
    if global_macros:
        for aliases in aliases_by_file.values():
            for name, types in global_macros.items():
                aliases[name].update(types)
    variable_types_by_file = {
        file: _declared_variable_types(raw, known_types, aliases_by_file[file], fields)
        for file, raw in texts.items()
    }
    # Function parameters named ``tilingData`` (empty-tensor Init) must not
    # poison unique member typing of the same name on another class.
    inherited = _unique_cross_file_var_types(
        {
            file: _declared_variable_types(
                raw,
                known_types,
                aliases_by_file[file],
                fields,
                statement_only=True,
            )
            for file, raw in texts.items()
        }
    )
    if inherited:
        for var_types in variable_types_by_file.values():
            for name, types in inherited.items():
                var_types.setdefault(name, set()).update(types)

    sites = resolved = unresolved = 0
    for scope in scopes:
        file = str(scope.file or "")
        raw = texts.get(file)
        if raw is None or not scope.line_start:
            continue
        body = _scope_body(raw, int(scope.line_start), scope.name.split("::")[-1])
        if body is None:
            continue
        body_start, body_end = body
        body_text = raw[body_start:body_end]
        var_types = dict(variable_types_by_file.get(file) or {})
        for match in _ACCESS_RE.finditer(body_text):
            base, outer, inner = match.group("base"), match.group("outer"), match.group("inner")
            # Nested TilingData field *names* also appear on ordinary locals
            # (``info.s2Size``). Only typed TilingData values or tiling-named
            # pointers are structurally TilingData-like.
            looks_tiling = bool(var_types.get(base))
            if not looks_tiling:
                continue
            sites += 1
            leaf = inner or outer
            if leaf.startswith("get_"):
                leaf = leaf[4:]
            candidates: list[Entity] = []
            if inner:
                child_types: set[str] = set()
                for owner in var_types.get(base) or ():
                    parent = fields.get((owner, outer))
                    if parent is None:
                        continue
                    child_types.update(
                        _referenced_types(
                            str(parent.attrs.get("cpp_type") or ""), known_types
                        )
                    )
                if not child_types:
                    child_types = set(nested.get(outer) or ())
                for child_type in child_types:
                    candidates.append(fields.get((child_type, leaf)))
            else:
                for owner in var_types.get(base) or ():
                    candidates.append(fields.get((owner, leaf)))
                candidates = _unique(candidates)
            candidates = _unique(candidates)
            absolute = body_start + match.start()
            line = _line(raw, absolute)
            expression = raw[absolute:body_start + match.end()].replace("\n", " ").strip()
            if len(candidates) == 1:
                field = candidates[0]
                rel = codemap.mint_candidate_relation(
                    RelationKind.READS,
                    scope.id,
                    field.id,
                    provenance="source_tilingdata_read_verified",
                    extra={
                        "file": file,
                        "line": line,
                        "expression": expression,
                        "field_owner": field.attrs.get("owner"),
                    },
                    status="confirmed",
                )
                rel.attrs["provenance"] = "source_tilingdata_read_verified"
                site = {"file": file, "line": line, "expression": expression}
                if site not in rel.attrs.setdefault("sites", []):
                    rel.attrs["sites"].append(site)
                resolved += 1
            elif len(candidates) > 1:
                record_unresolved_tiling(
                    codemap,
                    scope,
                    role="tilingdata_read_unresolved",
                    file=file,
                    line=line,
                    expression=expression,
                    extra={
                        "reason": "field_owner_ambiguous",
                        "candidate_fields": [f.attrs.get("qualified_name") for f in candidates],
                        "provenance": "source_tilingdata_read_unresolved_verified",
                    },
                )
                unresolved += 1

    reachable = _reachable(codemap)
    reachable_fields: set[str] = set()
    reachable_sites: set[str] = set()
    reachable_unresolved = 0
    for rel in codemap.relations.values():
        prov = str(rel.attrs.get("provenance") or "")
        if prov == "source_tilingdata_read_verified":
            is_reachable = rel.src in reachable
            rel.attrs["entry_reachable"] = is_reachable
            if is_reachable:
                reachable_fields.add(rel.dst)
                for site in rel.attrs.get("sites") or []:
                    reachable_sites.add(f"{site.get('file')}:{site.get('line')}:{rel.src}:{rel.dst}")
        elif prov == "source_tilingdata_read_unresolved_verified" and rel.src in reachable:
            reachable_unresolved += 1

    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure.update(
        {
            "tiling_read_sites": sites,
            "tiling_resolved_read_sites": resolved,
            "tiling_ambiguous_read_sites": unresolved,
            "tiling_entry_reachable_read_sites": len(reachable_sites),
            "tiling_entry_reachable_fields": len(reachable_fields),
            "tiling_entry_reachable_unresolved_read_sites": reachable_unresolved,
            "tiling_read_policy": "tiling-pointer-chain-conditional/v2",
        }
    )
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _purge_old_reads(codemap: CodeMap) -> None:
    rel_prov = {
        "source_tilingdata_read_qualified_v2", "source_tilingdata_read_unresolved_v2",
        "source_tilingdata_read_verified", "source_tilingdata_read_unresolved_verified",
    }
    ent_prov = {
        "source_tilingdata_read_unresolved_v2", "source_tilingdata_read_unresolved_verified",
    }
    remove_rel = {rid for rid, rel in codemap.relations.items() if str(rel.attrs.get("provenance") or "") in rel_prov}
    remove_ent = {eid for eid, ent in codemap.entities.items() if str(ent.attrs.get("provenance") or "") in ent_prov}
    for rid, rel in list(codemap.relations.items()):
        if rel.src in remove_ent or rel.dst in remove_ent:
            remove_rel.add(rid)
    for rid in remove_rel:
        codemap.relations.pop(rid, None)
    for eid in remove_ent:
        codemap.entities.pop(eid, None)


def _scope_body(raw: str, start_line: int, short_name: str) -> tuple[int, int] | None:
    lines = raw.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos); pos += len(line)
    start_idx = max(0, start_line - 4)
    start_pos = offsets[start_idx] if start_idx < len(offsets) else 0
    limit_idx = min(len(offsets) - 1, start_line + 16) if offsets else 0
    limit_pos = offsets[limit_idx] if offsets else len(raw)
    name_match = re.search(rf"\b{re.escape(short_name)}\s*(?:<[^;{{}}]*>)?\s*\(", raw[start_pos:limit_pos], re.S)
    if not name_match:
        return None
    open_brace = raw.find("{", start_pos + name_match.end(), min(len(raw), limit_pos + 1200))
    if open_brace < 0:
        return None
    close_brace = _matching_brace(raw, open_brace)
    if close_brace < 0:
        return None
    return open_brace + 1, close_brace


def _is_stmt_declarator(raw: str, name_end: int) -> bool:
    """True for ``Type name;`` / ``Type name = …;`` / ``Type name[N];``, not params."""
    i = name_end
    n = len(raw)
    while i < n and raw[i] in " \t\r\n":
        i += 1
    return i < n and raw[i] in ";=["


def _declared_variable_types(
    raw: str,
    known: set[str],
    aliases: dict[str, set[str]],
    fields: dict[tuple[str, str], Entity] | None = None,
    *,
    statement_only: bool = False,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    field_index = fields or {}
    symbols = sorted(known | set(aliases), key=len, reverse=True)
    if symbols:
        alt = "|".join(re.escape(x) for x in symbols)
        pattern = re.compile(
            rf"\b(?P<type>{alt})\b(?:\s*<[^;{{}}]*>)?{_DECL_AFTER_TYPE}\s*(?P<name>[A-Za-z_]\w*)\b"
        )
        for match in pattern.finditer(raw):
            name = match.group("name")
            if name in _DECL_QUAL_NAMES or name in known or name in aliases:
                continue
            if statement_only and not _is_stmt_declarator(raw, match.end()):
                continue
            token = match.group("type")
            if token in known:
                out[name].add(token)
            out[name].update(aliases.get(token) or ())
    for match in _GET_TILING_WITH_RE.finditer(raw):
        type_name = match.group("type").split("::")[-1]
        if type_name in known:
            out[match.group("var")].add(type_name)
        out[match.group("var")].update(aliases.get(type_name) or ())
    for match in _GET_TILING_MEMBER_RE.finditer(raw):
        type_name = match.group("type").split("::")[-1]
        member = match.group("member")
        var = match.group("var")
        parent = field_index.get((type_name, member))
        child_types = _referenced_types(
            str((parent.attrs.get("cpp_type") if parent else "") or ""), known
        )
        if child_types:
            out[var].update(child_types)
        elif type_name in known:
            out[var].add(type_name)
        out[var].update(aliases.get(type_name) or ())
    defaults = {_DEFAULT_REG_RE.search(raw).group(1).split("::")[-1]} if _DEFAULT_REG_RE.search(raw) else set()
    defaults &= known
    if len(known) == 1:
        defaults |= known
    if defaults:
        for match in _GET_TILING_BARE_VARS_RE.finditer(raw):
            out[match.group("var")].update(defaults)
    return out


def _aliases(raw: str, known: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    matches = [(m.group(1), m.group(2)) for m in _ALIAS_RE.finditer(raw)]
    for _ in range(3):
        for alias, expr in matches:
            tokens = set(_WORD_RE.findall(expr))
            out[alias].update(tokens & known)
            for token in tokens:
                out[alias].update(out.get(token) or ())
    for alias, types in macro_type_aliases(raw, known).items():
        out[alias].update(types)
    return out


_TILING_PTR_NAME_RE = re.compile(r"(?i)tiling")


def _unique_cross_file_var_types(
    variable_types_by_file: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Promote uniquely typed tiling pointers declared in one TU into others.

    FAG declares ``FagTilingType tilingData`` on the base class and reads
    ``this->tilingData->nested.field`` in derived headers. Only names that
    look like tiling pointers are inherited, and only when every *statement*
    declaration of that name maps to one TilingData type. Init parameters
    that reuse the same name (empty-tensor ``tilingData``) are excluded from
    the inherit source so they cannot drop the member type. Ordinary locals
    (``info``) never qualify.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for var_types in variable_types_by_file.values():
        for name, types in var_types.items():
            if not _TILING_PTR_NAME_RE.search(name):
                continue
            grouped[name].update(types)
    return {name: set(types) for name, types in grouped.items() if len(types) == 1}


def _referenced_types(raw: str, known: set[str]) -> set[str]:
    return set(_WORD_RE.findall(raw or "")) & known


def _reachable(codemap: CodeMap) -> set[str]:
    starts = {e.id for e in codemap.by_kind(EntityKind.KERNEL) if e.attrs.get("source_signature")}
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if rel.kind_name() == RelationKind.CALLS.value and str(rel.attrs.get("provenance") or "") in _BOUND_CALLS:
            adj[rel.src].add(rel.dst)
    seen = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt); q.append(nxt)
    return seen


def _load(root: Path, selected: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in selected:
        path = _resolve_file(root, raw)
        if path is not None:
            out[raw.replace("\\", "/").lstrip("./")] = read_text(path)
    return out


def _unique(items: list[Entity | None]) -> list[Entity]:
    out: list[Entity] = []
    seen: set[str] = set()
    for item in items:
        if item is None or item.id in seen:
            continue
        seen.add(item.id); out.append(item)
    return out


def _matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    quote = ""
    escape = False
    for idx in range(open_pos, len(text)):
        ch = text[idx]
        if quote:
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == quote: quote = ""
            continue
        if ch in {'\"', "'"}: quote = ch
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return idx
    return -1


def _resolve_file(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1
