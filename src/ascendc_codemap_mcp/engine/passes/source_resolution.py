# -*- coding: utf-8 -*-
"""Resolve historical UO gaps from operator-agnostic current-source facts.

The pass inventories arch-scoped kernel functions, macro expansions, direct call
sites, compile-time constants, control-flow frontier sites, TilingData reads and
hardware-resource members.  Historical gaps are upgraded only when their own
candidate source span is covered by corresponding machine-verifiable facts.

A complete C++ call graph remains a compiler responsibility; this fallback does
not claim completeness for template/macro call resolution merely from regex.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import (
    bind_or_create,
    find_declaration,
    is_alias_not_field,
    is_forbidden_callable_name,
)
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.semantics.const_expr import occupancy_overlap, worth_sharing
from ascendc_codemap_mcp.engine.source_layout import iter_cpp, selected_host_files, selected_kernel_files

_CPP_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_CONSTEXPR_RE = re.compile(
    r"\b(?:static\s+constexpr|constexpr\s+static|constexpr)\s+"
    r"(?:const\s+)?[A-Za-z_:][\w:<>,\s*&]*?\s+([A-Za-z_]\w*)(?:\[[^\]]+\])?\s*=\s*([^;]+);"
)
_DEFINE_OBJECT_RE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+([^\n\\]+)\s*$", re.M)
_ENUM_RE = re.compile(r"enum(?:\s+class)?\s+([A-Za-z_]\w*)[^\{;]*\{(.*?)\};", re.S)
_STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_]\w*)[^\{;]*\{", re.S)
_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)[^\{;]*\{", re.S)
_MEMBER_RE = re.compile(
    r"^\s*([A-Za-z_][\w:\s<>,*&]*?)\s+([A-Za-z_]\w*)(?:\[[^\]]+\])?\s*(?:=[^;]+)?;\s*$"
)
_CONTINUATION_NAME_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*;\s*$")
_CLASS_METHOD_RE = re.compile(
    r"(?:template\s*<[^;{}]{0,400}>\s*)?"
    r"(?:__aicore__\s+)?(?:static\s+)?(?:inline\s+)?"
    r"(?:[\w:<>,\s*&]{0,200}?\s+)?"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]{0,800}\)"
    r"(?:\s*const)?"
    r"(?:\s*=\s*delete)?"
    r"\s*\{",
    re.S,
)
_SKIP_MEMBER_PREFIXES = ("using ", "typedef ", "template ", "static_assert")
_TILING_READ_RE = re.compile(r"\btilingData\s*->\s*([A-Za-z_]\w*)(?:\s*\.\s*([A-Za-z_]\w*))?")
_FIELD_TILING_ASSIGN_RE = re.compile(
    r"\b(?:[A-Za-z_]\w*\s*(?:\.|->)\s*)*([A-Za-z_]\w*)\s*=\s*[^=\n]{0,240}?\btilingData\b"
)
_CALL_RE = re.compile(
    r"(?:(?P<receiver>[A-Za-z_]\w*)\s*(?:\.|->)\s*)?"
    r"(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*"
    r"(?:<[^;{}()]{0,600}>)?\s*\("
)
_BRANCH_RE = re.compile(r"\b(if\s+constexpr|if|while|for|switch)\s*\(")
_PP_BRANCH_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif)\b(.*)$", re.M)
_TYPE_ALIAS_RE = re.compile(
    r"\busing\s+([A-Za-z_]\w*)\s*=\s*(?:typename\s+)?([A-Za-z_:][A-Za-z0-9_:]*)\s*<",
    re.S,
)
_RESOURCE_TYPES = (
    "TBuf",
    "TQue",
    "GlobalTensor",
    "LocalTensor",
    "MutexBufferManager",
    "TPipe",
    "TEventID",
)
_CALL_SKIP = {
    "if", "while", "for", "switch", "sizeof", "alignof", "decltype", "static_cast",
    "reinterpret_cast", "const_cast", "dynamic_cast", "return", "likely", "unlikely",
    "constexpr", "consteval", "constinit",
}
_METHOD_NAME_SKIP = _CALL_SKIP | {
    "else",
    "do",
    "catch",
    "try",
    "constexpr",
    "public",
    "private",
    "protected",
}


@dataclass(frozen=True)
class _Scope:
    name: str
    file: str
    start: int
    end: int
    body_start: int
    body_end: int
    kind: str


@dataclass(frozen=True)
class _KernelSource:
    path: Path
    text: str
    file: str
    newlines: list[int]
    functions: list[_Scope]


def resolve_source_gaps(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    sources = _load_kernel_sources(root, architecture)
    host_sources = _load_text_sources(
        root, selected_host_files(root, architecture), parse_functions=False
    )
    stats: dict[str, Any] = {}
    stats.update(_extract_calls_macros_and_frontiers(codemap, sources))
    stats.update(_resolve_tiling_reads(codemap, sources))
    stats.update(_extract_compile_facts(codemap, sources + host_sources, architecture))
    # Host param structs are what most review questions are about, and they were
    # never turned into types: the host sources were loaded for compile facts
    # only, so a line inside `FuzzyBaseInfoParamsRegbase` belonged to nothing.
    stats.update(
        _extract_runtime_structs_and_resources(
            codemap, sources + host_sources, architecture
        )
    )
    stats["enum_membership_edges"] = link_enum_membership(codemap)
    arch_dir = root / "op_kernel" / str(architecture or "")
    assign_sources = sources
    if arch_dir.is_dir():
        assign_sources = sources + _load_text_sources(
            root, list(iter_cpp(arch_dir)), parse_functions=False
        )
    stats.update(_extract_field_tiling_assigns(codemap, assign_sources))
    stats.update(_resolve_gap_records(codemap, stats))
    codemap.meta["source_resolution"] = "ascendc-source-resolution/v2"
    codemap.meta["source_resolution_stats"] = stats
    return codemap


def _files(path: Path, *, recursive: bool = True) -> list[Path]:
    if not path.is_dir():
        return []
    it = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in _CPP_SUFFIXES)


def _kernel_files(root: Path, architecture: str) -> list[Path]:
    from ascendc_codemap_mcp.engine.source_layout import keep_lexical_kernel_path

    return [
        path
        for path in selected_kernel_files(root, architecture)
        if keep_lexical_kernel_path(path, architecture)
    ]


def _read(path: Path) -> str:
    from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

    return read_text(path)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _line_index(text: str) -> list[int]:
    """Newline offsets for O(log n) line lookup via bisect."""
    return [i for i, ch in enumerate(text) if ch == "\n"]


def _line_at(newlines: list[int], offset: int) -> int:
    return bisect.bisect_right(newlines, max(0, offset)) + 1


def _load_kernel_sources(root: Path, architecture: str) -> list[_KernelSource]:
    return _load_text_sources(
        root, _kernel_files(root, architecture), parse_functions=True
    )


def _load_text_sources(
    root: Path, paths: Iterable[Path], *, parse_functions: bool
) -> list[_KernelSource]:
    from ascendc_codemap_mcp.engine.parallel import map_files

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    def _one(path: Path) -> _KernelSource:
        text = _read(path)
        file = _rel(root, path)
        newlines = _line_index(text)
        return _KernelSource(
            path=path,
            text=text,
            file=file,
            newlines=newlines,
            functions=_function_scopes(text, file, newlines=newlines)
            if parse_functions
            else [],
        )

    return map_files(unique, _one)


def _iter_brace_body_lines(
    text: str, open_pos: int, close_pos: int, newlines: list[int]
) -> Iterable[tuple[int, str]]:
    """Physical 1-based line of each row in ``text[open_pos+1:close_pos]``."""
    body = text[open_pos + 1 : close_pos]
    abs_off = 0
    for raw in body.splitlines(keepends=True):
        yield _line_at(newlines, open_pos + 1 + abs_off), raw.rstrip("\r\n")
        abs_off += len(raw)


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


def _find_kernel(
    codemap: CodeMap,
    source_name: str,
    cache: dict[str, Entity | None] | None = None,
) -> Entity | None:
    short = source_name.split("::")[-1]
    if cache is not None and short in cache:
        return cache[short]
    exact = codemap.by_name(source_name, kind=EntityKind.KERNEL)
    if exact:
        hit = exact[0]
        if cache is not None:
            cache[short] = hit
        return hit
    hit = None
    for ent in codemap.by_kind(EntityKind.KERNEL):
        if ent.name.split("::")[-1] == short:
            hit = ent
            break
    if cache is not None:
        cache[short] = hit
    return hit


def _function_scopes(
    text: str, file: str, *, newlines: list[int] | None = None
) -> list[_Scope]:
    from ascendc_codemap_mcp.engine.cpp_lex import iter_function_defs
    from ascendc_codemap_mcp.engine.passes.source_text_cache import mask_cached

    out: list[_Scope] = []
    line_of = (lambda off: _line_at(newlines, off)) if newlines is not None else (
        lambda off: _line(text, off)
    )
    for hit in iter_function_defs(mask_cached(text)):
        name = hit.name
        short = name.split("::")[-1].split("<", 1)[0].strip()
        if short in _CALL_SKIP:
            continue
        out.append(
            _Scope(
                name=name,
                file=file,
                start=line_of(hit.start),
                end=line_of(hit.close_brace),
                body_start=hit.open_brace + 1,
                body_end=hit.close_brace,
                kind="function",
            )
        )
    return out


def _macro_scopes(text: str, file: str) -> list[_Scope]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for raw in lines:
        offsets.append(pos)
        pos += len(raw)
    out: list[_Scope] = []
    i = 0
    while i < len(lines):
        match = re.match(r"\s*#define\s+([A-Za-z_]\w*)\s*(?:\([^\n]*?\))?(.*)$", lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1)
        start_i = i
        while i < len(lines) - 1 and lines[i].rstrip().endswith("\\"):
            i += 1
        end_i = i
        start_off = offsets[start_i]
        end_off = offsets[end_i] + len(lines[end_i])
        out.append(
            _Scope(
                name=name,
                file=file,
                start=start_i + 1,
                end=end_i + 1,
                body_start=start_off,
                body_end=end_off,
                kind="macro",
            )
        )
        i += 1
    return out


def _scope_entity(codemap: CodeMap, scope: _Scope) -> Entity | None:
    if scope.kind == "function":
        kernel = _find_kernel(codemap, scope.name)
        if kernel is not None:
            kernel.attrs.setdefault("source_definition", True)
            return kernel
        kind = EntityKind.METHOD if "::" in scope.name else EntityKind.FUNCTION
        if is_forbidden_callable_name(scope.name):
            return None
        bound = bind_or_create(
            codemap,
            kind,
            scope.name.split("::")[-1],
            file=scope.file,
            line=scope.start,
            owner=scope.name.rsplit("::", 1)[0] if "::" in scope.name else "",
            architecture=str(getattr(codemap, "architecture", "") or ""),
            attrs={"layer": "kernel", "source_scope": True, "provenance": "source_scope"},
            status="confirmed",
        )
        return bound
    else:
        kind = EntityKind.MACRO
    return codemap.upsert(
        kind,
        scope.name,
        eid=f"SRCSCOPE::{scope.kind}::{scope.file}::{scope.start}::{scope.name}",
        attrs={"layer": "kernel", "source_scope": True, "provenance": "source_scope"},
        file=scope.file,
        line=scope.start,
        status="confirmed",
    )


def _containing_scope(scopes: Iterable[_Scope], offset: int) -> _Scope | None:
    matches = [s for s in scopes if s.body_start <= offset <= s.body_end]
    if not matches:
        return None
    return min(matches, key=lambda s: s.body_end - s.body_start)


def _extract_calls_macros_and_frontiers(
    codemap: CodeMap, sources: list[_KernelSource]
) -> dict[str, int]:
    direct_kernel_calls = 0
    call_edges = 0
    type_dispatch_edges = 0
    branch_sites = 0
    macro_scopes_count = 0
    kernel_cache: dict[str, Entity | None] = {}
    kernel_by_short: dict[str, list] = {}
    for kernel in codemap.by_kind(EntityKind.KERNEL):
        short = kernel.name.split("::")[-1]
        if short:
            kernel_by_short.setdefault(short, []).append(kernel)
    kernel_type_re = None
    if kernel_by_short:
        shorts = sorted(kernel_by_short, key=len, reverse=True)
        kernel_type_re = re.compile(
            r"\b(" + "|".join(re.escape(s) for s in shorts) + r")\s*<"
        )

    for src in sources:
        text = src.text
        file = src.file
        newlines = src.newlines
        functions = src.functions
        macros = _macro_scopes(text, file)
        macro_scopes_count += len(macros)
        all_scopes = functions + macros
        scope_entities = {scope: _scope_entity(codemap, scope) for scope in all_scopes}

        # Macro references from functions are explicit source expansion edges.
        macro_by_name = {m.name: m for m in macros}
        macro_re = None
        if macro_by_name:
            # Longest-first so prefixes do not steal longer macro names.
            names = sorted(macro_by_name, key=len, reverse=True)
            macro_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\s*\(")
        for function in functions:
            body = text[function.body_start:function.body_end]
            caller = scope_entities[function]
            if caller is None:
                continue
            if macro_re is None:
                continue
            seen_macros: set[str] = set()
            for hit in macro_re.finditer(body):
                mname = hit.group(1)
                if mname in seen_macros:
                    continue
                seen_macros.add(mname)
                macro = macro_by_name[mname]
                macro_ent = scope_entities[macro]
                if macro_ent is None:
                    continue
                codemap.link(
                    RelationKind.CALLS,
                    caller.id,
                    macro_ent.id,
                    attrs={"provenance": "source_macro_invocation", "file": file},
                    status="confirmed",
                )
                call_edges += 1

        for scope in all_scopes:
            caller = scope_entities[scope]
            if caller is None:
                continue
            body = text[scope.body_start:scope.body_end]
            body_abs = scope.body_start

            # Direct function/method call sites. Existing KERNEL names receive a
            # real call edge; unknown callees are retained as method call targets.
            for match in _CALL_RE.finditer(body):
                target_name = match.group("name")
                if (
                    target_name in _CALL_SKIP
                    or is_forbidden_callable_name(target_name)
                    or target_name == scope.name.split("::")[-1]
                ):
                    continue
                absolute = body_abs + match.start()
                line = _line_at(newlines, absolute)
                target_kernel = _find_kernel(codemap, target_name, kernel_cache)
                if target_kernel is not None and target_kernel.id != caller.id:
                    target = target_kernel
                    direct_kernel_calls += 1
                else:
                    receiver = str(match.group("receiver") or "").strip()
                    target = bind_or_create(
                        codemap,
                        EntityKind.METHOD,
                        target_name,
                        file=file,
                        line=line,
                        owner=receiver,
                        architecture=str(getattr(codemap, "architecture", "") or ""),
                        attrs={
                            "call_target": target_name,
                            "receiver": receiver,
                            "provenance": "source_call_site",
                        },
                        status="confirmed",
                    )
                    if target is None:
                        continue
                codemap.link(
                    RelationKind.CALLS,
                    caller.id,
                    target.id,
                    attrs={
                        "provenance": "source_call_site",
                        "file": file,
                        "line": line,
                    },
                    status="confirmed",
                )
                call_edges += 1

            # Template/class types named in a scope can choose an existing
            # kernel implementation even when the invocation is indirect via an
            # object or std::conditional. Record only textual type references.
            if kernel_type_re is not None:
                seen_shorts: set[str] = set()
                for hit in kernel_type_re.finditer(body):
                    short = hit.group(1)
                    if short in seen_shorts:
                        continue
                    seen_shorts.add(short)
                    for kernel in kernel_by_short.get(short, []):
                        if kernel.id == caller.id:
                            continue
                        codemap.link(
                            RelationKind.CONTROLS,
                            caller.id,
                            kernel.id,
                            attrs={"provenance": "source_kernel_type_reference", "file": file},
                            status="confirmed",
                        )
                        type_dispatch_edges += 1

            # Control-flow frontier inventory.
            for branch in _BRANCH_RE.finditer(body):
                absolute = body_abs + branch.start()
                kind = branch.group(1).replace(" ", "_")
                line = _line_at(newlines, absolute)
                node = codemap.upsert(
                    EntityKind.BRANCH,
                    f"{scope.name}:{kind}@{line}",
                    eid=f"SRCBRANCH::{file}::{line}::{kind}",
                    attrs={"branch_kind": kind, "provenance": "source_frontier"},
                    file=file,
                    line=line,
                    status="confirmed",
                )
                codemap.link(
                    RelationKind.CONTROLS,
                    node.id,
                    caller.id,
                    attrs={"provenance": "source_frontier"},
                    status="confirmed",
                )
                branch_sites += 1

        for branch in _PP_BRANCH_RE.finditer(text):
            line = _line_at(newlines, branch.start())
            node = codemap.upsert(
                EntityKind.BRANCH,
                f"pp_{branch.group(1)}@{line}",
                eid=f"SRCPPBRANCH::{file}::{line}::{branch.group(1)}",
                attrs={
                    "branch_kind": f"pp_{branch.group(1)}",
                    "condition": branch.group(2).strip(),
                    "provenance": "source_frontier",
                },
                file=file,
                line=line,
                status="confirmed",
            )
            owner = _containing_scope(all_scopes, branch.start())
            if owner is not None:
                codemap.link(
                    RelationKind.CONTROLS,
                    node.id,
                    scope_entities[owner].id,
                    attrs={"provenance": "source_frontier"},
                    status="confirmed",
                )
            branch_sites += 1

    return {
        "source_call_edges": call_edges,
        "source_direct_kernel_calls": direct_kernel_calls,
        "source_kernel_type_dispatch_edges": type_dispatch_edges,
        "source_frontier_sites": branch_sites,
        "source_macro_scopes": macro_scopes_count,
    }


def _field_index(codemap: CodeMap) -> dict[str, list[Entity]]:
    out: dict[str, list[Entity]] = {}
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        out.setdefault(field.name, []).append(field)
    return out


def _resolve_tiling_reads(codemap: CodeMap, sources: list[_KernelSource]) -> dict[str, int]:
    fields = _field_index(codemap)
    reads = 0
    for src in sources:
        text = src.text
        file = src.file
        scopes = src.functions
        for match in _TILING_READ_RE.finditer(text):
            outer, inner = match.groups()
            name = inner or outer
            candidates = fields.get(name) or []
            if not candidates:
                continue
            scope = _containing_scope(scopes, match.start())
            if scope is not None:
                owner = _scope_entity(codemap, scope)
            else:
                owner = None
            if owner is None:
                continue
            for field in candidates:
                if inner and field.name != inner:
                    continue
                codemap.link(
                    RelationKind.READS,
                    owner.id,
                    field.id,
                    attrs={
                        "provenance": "source_tilingdata_read",
                        "file": file,
                        "line": _line_at(src.newlines, match.start()),
                        "container": outer,
                    },
                    status="confirmed",
                )
                reads += 1
    return {"tilingdata_read_edges": reads}


def _extract_compile_facts(
    codemap: CodeMap, sources: list[_KernelSource], architecture: str
) -> dict[str, int]:
    macros = 0
    compile_vars = 0
    type_aliases = 0
    archs = codemap.by_name(architecture, kind=EntityKind.ARCH)
    arch_ent = archs[0] if archs else None
    for src in sources:
        text = src.text
        file = src.file
        newlines = src.newlines
        for m in _DEFINE_OBJECT_RE.finditer(text):
            name, value = m.groups()
            ent = codemap.upsert(
                EntityKind.MACRO,
                name,
                eid=f"SRCMACRO::{file}::{name}",
                attrs={"value": value.strip(), "provenance": "source_define", "architecture": architecture},
                file=file,
                line=_line_at(newlines, m.start()),
                status="confirmed",
            )
            if arch_ent:
                codemap.link(RelationKind.ACTIVE_UNDER, ent.id, arch_ent.id, attrs={"provenance": "source_arch_file"}, status="confirmed")
            macros += 1
        for m in _CONSTEXPR_RE.finditer(text):
            name, value = m.groups()
            ent = codemap.upsert(
                EntityKind.COMPILE_VAR,
                name,
                eid=f"SRCCONST::{file}::{name}",
                attrs={"value_expr": value.strip(), "provenance": "source_constexpr", "architecture": architecture},
                file=file,
                line=_line_at(newlines, m.start()),
                status="confirmed",
            )
            if arch_ent:
                codemap.link(RelationKind.ACTIVE_UNDER, ent.id, arch_ent.id, attrs={"provenance": "source_arch_file"}, status="confirmed")
            compile_vars += 1
        for em in _ENUM_RE.finditer(text):
            enum_name, body = em.groups()
            value = -1
            for raw in body.split(","):
                item = re.sub(r"//.*", "", raw).strip()
                if not item:
                    continue
                parts = item.split("=", 1)
                member = parts[0].strip()
                if not re.match(r"^[A-Za-z_]\w*$", member):
                    continue
                if len(parts) == 2:
                    try:
                        value = int(parts[1].strip(), 0)
                    except ValueError:
                        value = -1
                else:
                    value += 1
                codemap.upsert(
                    EntityKind.COMPILE_VAR,
                    f"{enum_name}::{member}",
                    eid=f"SRCENUM::{file}::{enum_name}::{member}",
                    attrs={"value": value if value >= 0 else None, "enum": enum_name, "provenance": "source_enum"},
                    file=file,
                    line=_line_at(newlines, em.start()),
                    status="confirmed",
                )
                compile_vars += 1
        for m in _TYPE_ALIAS_RE.finditer(text):
            alias, target = m.groups()
            if is_alias_not_field(alias, f"using {alias} = {target}"):
                continue
            bound = bind_or_create(
                codemap,
                EntityKind.TYPE,
                alias,
                file=file,
                line=_line_at(newlines, m.start()),
                architecture=architecture,
                attrs={
                    "role": "type_alias",
                    "alias_of": target,
                    "provenance": "source_type_alias",
                    "architecture": architecture,
                },
                status="confirmed",
            )
            if bound is None:
                continue
            type_aliases += 1
    shared = _link_shared_compile_values(codemap)
    return {
        "source_macros": macros,
        "source_compile_vars": compile_vars,
        "source_type_aliases": type_aliases,
        "shared_compile_values": shared,
    }


def link_enum_membership(codemap: CodeMap) -> int:
    """CONTAINS from an enum TYPE to each COMPILE_VAR that recorded that enum.

    The member already carries ``attrs['enum']`` from the declaration. This
    join is that declared name, not a scan of source text. A TYPE is minted
    only when the enum name is present and no TYPE of that name exists yet.
    """
    linked = 0
    for cv in list(codemap.by_kind(EntityKind.COMPILE_VAR)):
        enum_name = str((cv.attrs or {}).get("enum") or "").strip()
        if not enum_name:
            continue
        types = list(codemap.by_name(enum_name, kind=EntityKind.TYPE))
        if not types:
            types = [
                codemap.upsert(
                    EntityKind.TYPE,
                    enum_name,
                    eid=f"SRCENUMTYPE::{cv.file or ''}::{enum_name}",
                    attrs={
                        "cpp_kind": "enum",
                        "provenance": "source_enum",
                    },
                    file=str(cv.file or ""),
                    line=int(cv.line_start or 0) or None,
                    status="confirmed",
                )
            ]
        for typ in types:
            if typ.id == cv.id:
                continue
            codemap.link(
                RelationKind.CONTAINS,
                typ.id,
                cv.id,
                attrs={"provenance": "source_enum", "enum": enum_name},
                status="confirmed",
            )
            linked += 1
    return linked


def _compile_value_text(ent: Entity) -> str:
    attrs = ent.attrs or {}
    for key in ("value_expr", "value", "definition"):
        text = str(attrs.get(key) or "").strip()
        if text:
            return text
    return ""


def _link_shared_compile_values(codemap: CodeMap) -> int:
    """ALIASES among same-file COMPILE_VAR/MACRO whose integer sets overlap."""
    by_file: dict[str, list[Entity]] = {}
    for kind in (EntityKind.COMPILE_VAR, EntityKind.MACRO):
        for ent in codemap.by_kind(kind):
            path = str(ent.file or "").replace("\\", "/")
            if not path:
                continue
            by_file.setdefault(path, []).append(ent)
    linked = 0
    for _path, rows in by_file.items():
        if len(rows) < 2:
            continue
        payloads = [(ent, _compile_value_text(ent)) for ent in rows]
        for i, (left, ltext) in enumerate(payloads):
            if not ltext:
                continue
            neighbors = 0
            for right, rtext in payloads[i + 1 :]:
                if neighbors >= 8:
                    break
                if left.id == right.id or not rtext:
                    continue
                left_enum = str((left.attrs or {}).get("enum") or "")
                right_enum = str((right.attrs or {}).get("enum") or "")
                if left_enum != right_enum:
                    continue
                overlap = occupancy_overlap(ltext, rtext)
                if not worth_sharing(overlap, ltext, rtext):
                    continue
                attrs = {
                    "via": "shares_value",
                    "overlap": sorted(overlap)[:8],
                    "provenance": "source_compile_occupancy",
                }
                codemap.link(RelationKind.ALIASES, left.id, right.id, attrs=attrs, status="extracted")
                linked += 1
                neighbors += 1
    return linked


def _extract_field_tiling_assigns(
    codemap: CodeMap, sources: list[_KernelSource]
) -> dict[str, int]:
    """Record ``lhs = … tilingData …`` as definition sites on known FIELD names."""
    fields: dict[str, list[Entity]] = {}
    for field in codemap.by_kind(EntityKind.FIELD):
        leaf = str(field.name or "").rsplit(".", 1)[-1]
        if leaf:
            fields.setdefault(leaf, []).append(field)
    if not fields:
        return {"field_tiling_assign_sites": 0}
    assigned = 0
    for src in sources:
        for match in _FIELD_TILING_ASSIGN_RE.finditer(src.text):
            name = match.group(1)
            candidates = fields.get(name) or []
            if not candidates:
                continue
            line = _line_at(src.newlines, match.start())
            site = {
                "file": src.file,
                "line": line,
                "line_start": line,
                "kind": "tiling_assign",
                "provenance": "source_field_tiling_assign",
            }
            for field in candidates:
                sites = list(field.attrs.get("definition_sites") or [])
                if any(
                    isinstance(item, dict)
                    and str(item.get("file") or "") == src.file
                    and int(item.get("line") or item.get("line_start") or 0) == line
                    for item in sites
                ):
                    continue
                sites.append(site)
                field.attrs["definition_sites"] = sites
                assigned += 1
    return {"field_tiling_assign_sites": assigned}


def _extract_runtime_structs_and_resources(
    codemap: CodeMap, sources: list[_KernelSource], architecture: str
) -> dict[str, int]:
    structs = 0
    resources = 0
    methods = 0
    for src in sources:
        text = src.text
        file = src.file
        newlines = src.newlines
        for kind_re, kind_name in ((_STRUCT_RE, "struct"), (_CLASS_RE, "class")):
            for m in kind_re.finditer(text):
                owner = m.group(1)
                open_pos = text.find("{", m.start(), m.end())
                close_pos = _matching_brace(text, open_pos)
                if close_pos < 0:
                    continue
                owner_ent = bind_or_create(
                    codemap,
                    EntityKind.TYPE,
                    owner,
                    file=file,
                    line=_line_at(newlines, m.start()),
                    architecture=architecture,
                    attrs={"cpp_kind": kind_name, "architecture": architecture, "provenance": "source_runtime_type"},
                    status="confirmed",
                )
                if owner_ent is None:
                    continue
                # The closing brace is already known here. Leaving it off made
                # every struct a one-line entity, so nothing could be resolved
                # as being *inside* a type and a card for one showed only its
                # declaration line.
                end_line = _line_at(newlines, close_pos)
                if end_line > int(owner_ent.line_start or 0):
                    owner_ent.line_end = end_line
                structs += 1
                pending: str | None = None
                pending_line = 0
                depth = 0
                for line_no, raw_line in _iter_brace_body_lines(text, open_pos, close_pos, newlines):
                    stripped = re.sub(r"//.*", "", raw_line).strip()
                    if depth > 0:
                        depth += raw_line.count("{") - raw_line.count("}")
                        depth = max(0, depth)
                        continue
                    if pending is not None:
                        combined = f"{pending} {stripped}"
                        emit_type = ""
                        emit_name = ""
                        nm = _CONTINUATION_NAME_RE.match(stripped)
                        if nm:
                            emit_type = pending
                            emit_name = nm.group("name")
                        elif ";" in stripped:
                            mm = _MEMBER_RE.match(re.sub(r"\s+", " ", combined).strip())
                            if mm:
                                emit_type = " ".join(mm.group(1).split())
                                emit_name = mm.group(2)
                        if emit_name:
                            if _mint_runtime_field(
                                codemap,
                                owner_ent,
                                owner,
                                emit_name,
                                emit_type,
                                file,
                                line_no,
                            ):
                                resources += 1
                            pending = None
                        else:
                            pending = combined
                    elif stripped.startswith(_SKIP_MEMBER_PREFIXES):
                        if not stripped.startswith("template ") and ";" not in stripped:
                            pending = stripped
                            pending_line = line_no
                    elif ";" not in stripped and (
                        stripped.endswith("::type")
                        or stripped.endswith(",")
                        or (
                            (
                                "MutexBuffer" in stripped
                                or "conditional" in stripped
                                or "Tensor" in stripped
                            )
                            and not re.search(r"\b[A-Za-z_]\w*\s*;\s*$", stripped)
                            and "(" not in stripped
                        )
                    ):
                        pending = stripped
                        pending_line = line_no
                    elif (
                        stripped
                        and "(" not in stripped
                        and not stripped.endswith(":")
                    ):
                        mm = _MEMBER_RE.match(stripped)
                        if mm:
                            if _mint_runtime_field(
                                codemap,
                                owner_ent,
                                owner,
                                mm.group(2),
                                " ".join(mm.group(1).split()),
                                file,
                                line_no,
                            ):
                                resources += 1
                    depth += raw_line.count("{") - raw_line.count("}")
                    depth = max(0, depth)
                body = text[open_pos + 1 : close_pos]
                for hit in _CLASS_METHOD_RE.finditer(body):
                    name = hit.group("name")
                    if not name or name in _METHOD_NAME_SKIP or is_forbidden_callable_name(name):
                        continue
                    line = _line_at(newlines, open_pos + 1 + hit.start("name"))
                    method = bind_or_create(
                        codemap,
                        EntityKind.METHOD,
                        name,
                        file=file,
                        line=line,
                        owner=owner,
                        architecture=architecture,
                        attrs={
                            "owner": owner,
                            "source_definition": True,
                            "architecture": architecture,
                            "provenance": "source_runtime_method",
                            "qualified_name": f"{owner}::{name}",
                        },
                        status="confirmed",
                    )
                    if method is None:
                        continue
                    codemap.link(
                        RelationKind.DECLARES,
                        owner_ent.id,
                        method.id,
                        attrs={"provenance": "source_runtime_type", "via": "class_method"},
                        status="confirmed",
                    )
                    methods += 1
    return {
        "runtime_types": structs,
        "hardware_resources": resources,
        "runtime_methods": methods,
    }


def _mint_runtime_field(
    codemap: CodeMap,
    owner_ent: Entity,
    owner: str,
    name: str,
    ctype: str,
    file: str,
    line: int,
) -> bool:
    if not name or name in {"public", "private", "protected"}:
        return False
    if not ctype or ctype.startswith("#"):
        return False
    if is_alias_not_field(name, ctype):
        return False
    existing_type = find_declaration(
        codemap, EntityKind.TYPE, symbol=name, owner=owner, file=file
    )
    if existing_type is not None:
        return False
    field = bind_or_create(
        codemap,
        EntityKind.FIELD,
        name,
        file=file,
        line=line,
        owner=owner,
        architecture=str(getattr(codemap, "architecture", "") or ""),
        attrs={"owner": owner, "cpp_type": ctype, "provenance": "source_runtime_member"},
        status="confirmed",
    )
    if field is None:
        return False
    codemap.link(
        RelationKind.DECLARES,
        owner_ent.id,
        field.id,
        attrs={"provenance": "source_runtime_type"},
        status="confirmed",
    )
    if any(token in ctype for token in _RESOURCE_TYPES):
        field.attrs["hardware_resource"] = True
        field.attrs["resource_type"] = next(
            (x for x in _RESOURCE_TYPES if x in ctype), ctype
        )
        return True
    return False


def _candidate_spans(ent: Entity) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for src in ent.attrs.get("candidate_sources") or []:
        if not isinstance(src, dict) or not src.get("file"):
            continue
        span = src.get("span") or {}
        start = int(span.get("start_line") or 0)
        end = int(span.get("end_line") or start or 0)
        out.append((str(src.get("file") or "").replace("\\", "/"), start, end))
    return out


def _facts_cover_candidates(codemap: CodeMap, ent: Entity, *, kinds: set[str], provenances: set[str]) -> bool:
    candidates = _candidate_spans(ent)
    if not candidates:
        return False
    facts = [
        fact for fact in codemap.entities.values()
        if fact.kind_name() in kinds and str(fact.attrs.get("provenance") or "") in provenances
    ]
    for file, start, end in candidates:
        matched = False
        for fact in facts:
            fact_file = str(fact.file or "").replace("\\", "/")
            if not (fact_file.endswith(file) or file.endswith(fact_file)):
                continue
            line = int(fact.line_start or 0)
            if not start or not end or start <= line <= end:
                matched = True
                break
        if not matched:
            return False
    return True


def _resolve_gap_records(codemap: CodeMap, stats: dict[str, Any]) -> dict[str, int]:
    contract = codemap.meta.get("source_contract_stats") or {}
    resolved = 0
    reason_counts: dict[str, int] = {}
    for ent in codemap.entities.values():
        if str(ent.status).lower() != "unresolved" or ent.attrs.get("role") != "unresolved":
            continue
        reason = str(ent.attrs.get("reason") or "")
        ok = False
        evidence = ""
        if reason == "entry_call_relation" and (
            int(stats.get("source_direct_kernel_calls") or 0) > 0
            or int(stats.get("source_kernel_type_dispatch_edges") or 0) > 0
        ):
            ok, evidence = True, "source_dispatch_inventory"
        elif reason == "kernel_parameters" and int(contract.get("source_template_args_bound") or 0) > 0 and int(contract.get("source_kernel_abi_links") or 0) > 0:
            ok, evidence = True, "source_kernel_signature"
        elif reason == "tilingdata_structs" and int(contract.get("source_tiling_data_classes") or 0) > 0 and int(contract.get("source_tiling_data_fields") or 0) > 0:
            ok, evidence = True, "source_tiling_data_class"
        elif reason == "tilingdata_read_sites" and int(stats.get("tilingdata_read_edges") or 0) > 0:
            ok, evidence = True, "source_tilingdata_read"
        elif reason == "compile_info" and (int(stats.get("source_compile_vars") or 0) + int(stats.get("source_macros") or 0)) > 0:
            ok, evidence = True, "source_compile_facts"
        elif reason == "kernel_runtime_structs" and _facts_cover_candidates(
            codemap,
            ent,
            kinds={EntityKind.TYPE.value},
            provenances={"source_runtime_type"},
        ):
            ok, evidence = True, "source_runtime_type"
        elif reason == "global_resources" and int(stats.get("hardware_resources") or 0) > 0:
            ok, evidence = True, "source_hardware_resources"
        elif reason == "frontier_sites" and _facts_cover_candidates(
            codemap,
            ent,
            kinds={EntityKind.BRANCH.value},
            provenances={"source_frontier"},
        ):
            ok, evidence = True, "source_frontier_inventory"
        # kernel_call_edges intentionally remains unresolved: syntax-level call
        # sites are useful CodeMap facts but do not prove a complete C++ graph.
        if not ok:
            continue
        ent.status = "resolved"
        ent.confidence = 1.0
        ent.attrs["resolved_by"] = evidence
        ent.attrs["resolved_from_current_source"] = True
        resolved += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    codemap.meta["resolved_archive_gaps"] = reason_counts
    return {"resolved_archive_gap_count": resolved, **{f"resolved_{k}": v for k, v in reason_counts.items()}}
