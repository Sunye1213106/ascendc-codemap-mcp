# -*- coding: utf-8 -*-
"""Source-backed Host producer/def-use graph for packed TilingKey arguments.

This pass deliberately stops at a dependency skeleton. It finds the current
source producer sites for every Host value passed to ``GET_TPL_TILING_KEY`` and
links API/compile/runtime dependencies without deriving a closed-form key
formula. Member identity is canonical (``this.foo.x == foo.x``); ambiguous
short names are never silently merged.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.cpp_lex import iter_function_defs, mask_non_code, matching_brace
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.symbol_identity import is_member_symbol, normalize_symbol, short_symbol
from ascendc_codemap_mcp.engine.source_layout import selected_host_files

_SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_ENUM_INDEX_RE = re.compile(r"enum\s+class\s+(InputIndex|AttrIndex)\s*:[^{]+\{(.*?)\};", re.S)
_ENUM_RE = re.compile(r"enum\s+(?P<scoped>class\s+)?(?P<name>[A-Za-z_]\w*)[^\{;]*\{(?P<body>.*?)\};", re.S)
_CONST_INT_RE = re.compile(
    r"\bconstexpr\s+(?:static\s+)?(?:const\s+)?[A-Za-z_:][\w:<>,\s*&]*?\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[-+]?0[xX][0-9A-Fa-f]+|[-+]?\d+)\s*;"
)
_CONST_ANY_RE = re.compile(
    r"\bconstexpr\s+(?:static\s+)?(?:const\s+)?[A-Za-z_:][\w:<>,\s*&]*?\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);"
)
_DEFINE_RE = re.compile(r"^\s*#define\s+(?P<name>[A-Za-z_]\w*)\s+(?P<value>[^\n\\]+)\s*$", re.M)
_CONST_STATIC_INT_RE = re.compile(
    r"\b(?:const\s+static|static\s+const|static\s+constexpr|constexpr\s+static)\s+"
    r"(?:const\s+)?(?:u?int(?:32|64)_t|int)\s+(?P<name>[A-Za-z_]\w*)\s*="
)
_ASSIGN_RE = re.compile(
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*(?<![=!<>])=(?!=)\s*(?P<rhs>[^;]+);",
    re.S,
)
_IF_RE = re.compile(r"\b(?:if|else\s+if)\s*\((?P<cond>[^{};]*)\)\s*\{", re.S)
_SWITCH_RE = re.compile(r"\bswitch\s*\((?P<cond>[^{};]*)\)\s*\{", re.S)
_IDENT_RE = re.compile(r"(?<![0-9A-Za-z_])[A-Za-z_]\w*(?:\s*(?:\.|->|::)\s*[A-Za-z_]\w*)*")
_API_TOKEN_RE = re.compile(r"\b(InputIndex|AttrIndex)::([A-Za-z_]\w*)")
_INPUT_ACCESS_RE = re.compile(
    r"\bGet(?:Optional|Dynamic)?Input(?:Shape|Desc|Tensor|DataType)?\s*\(\s*"
    r"(?P<arg>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?|[-+]?\d+)"
)
_TILING_PACK_TOKEN_RE = re.compile(
    r"\b(?:GET_TPL_TILING_KEY|GET_TILINGKEY|[A-Z][A-Z0-9_]*GET_TILING_?KEY|"
    r"SetTilingKey|GetTilingKey|DoOpTiling|GenTilingKey)\b"
)
_ATTR_ACCESS_RE = re.compile(
    r"\bGetAttrPointer(?:\s*<[^>]+>)?\s*\(\s*(?P<arg>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?|[-+]?\d+)"
)
_CAST_RE = re.compile(r"\b(?:static_cast|reinterpret_cast|const_cast|dynamic_cast)\s*<[^<>]*>")
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_CALL_SUFFIX_RE = re.compile(r"\s*(?:<[^;{}()]*>)?\s*\(")
_CONTROL_NAMES = {"if", "else", "for", "while", "switch", "catch", "return", "sizeof", "alignof"}
_IGNORED = {
    "auto", "const", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "true", "false",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
    "int64_t", "size_t", "bool", "int", "unsigned", "long", "short", "float", "double", "char", "void",
    "return", "nullptr", "std", "ge", "this", "typename", "decltype",
}
_RUNTIME_KINDS = {EntityKind.VARIABLE.value, EntityKind.FIELD.value}
_ROOT_RELATIONS = {RelationKind.DERIVES.value, RelationKind.FLOWS_TO.value}


@dataclass(frozen=True)
class _Scope:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class _Record:
    lhs: str
    rhs: str
    guards: tuple[str, ...]
    file: str
    line: int
    function: str


def trace_host_key_roots(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    architecture = str(architecture or getattr(codemap, "architecture", "") or "")
    root = Path(operator_root).expanduser().resolve()
    host_files = selected_host_files(root, architecture)
    if not host_files:
        return codemap

    texts: list[tuple[Path, str]] = []
    records: list[_Record] = []
    clang_records = _records_from_host_ir(root, host_ir)
    clang_keys = {(r.file, r.line, r.lhs) for r in clang_records}
    records.extend(clang_records)

    from ascendc_codemap_mcp.engine.parallel import map_files

    def _load_one(path: Path) -> tuple[Path, str, list[_Record]]:
        text = read_text(path)
        if clang_records:
            return path, text, []
        recs = [
            rec
            for rec in _assignments(root, path, text)
            if (rec.file, rec.line, rec.lhs) not in clang_keys
        ]
        return path, text, recs

    for path, text, recs in map_files(host_files, _load_one):
        texts.append((path, text))
        records.extend(recs)

    by_exact: dict[str, list[_Record]] = defaultdict(list)
    by_short: dict[str, list[_Record]] = defaultdict(list)
    for record in records:
        by_exact[record.lhs].append(record)
        by_short[short_symbol(record.lhs)].append(record)

    api_maps = _api_maps(codemap, texts)
    compile_symbols = _compile_symbols(codemap, texts)
    symbol_nodes: dict[str, Entity] = {}
    for kind in (EntityKind.VARIABLE, EntityKind.FIELD):
        for ent in codemap.by_kind(kind):
            if not ent.attrs.get("host_key_argument"):
                continue
            canonical = normalize_symbol(str(ent.attrs.get("canonical_symbol") or ent.name))
            symbol_nodes[canonical] = ent
            # A member must never be aliased by its bare leaf name. Otherwise
            # ``fBaseParams.isNzOut`` can accidentally absorb an unrelated
            # declaration initializer ``isNzOut = false``.
            source_name = str(ent.attrs.get("source_name") or "")
            if source_name and not is_member_symbol(canonical):
                symbol_nodes.setdefault(normalize_symbol(source_name), ent)

    targets = [
        e
        for kind in (EntityKind.VARIABLE, EntityKind.FIELD)
        for e in codemap.by_kind(kind)
        if e.attrs.get("host_key_argument")
    ]
    visiting: set[str] = set()
    visited: set[str] = set()
    for target in targets:
        use_file, use_function = _target_scope(target)
        _resolve_symbol(
            codemap,
            target,
            str(target.attrs.get("canonical_symbol") or target.name),
            by_exact=by_exact,
            by_short=by_short,
            api_maps=api_maps,
            compile_symbols=compile_symbols,
            symbol_nodes=symbol_nodes,
            visiting=visiting,
            visited=visited,
            use_file=use_file,
            use_function=use_function,
        )

    rooted = _source_rooted_entities(codemap)
    rooted_targets = 0
    producer_targets = 0
    for target in targets:
        producer_count = int(target.attrs.get("producer_site_count") or 0)
        if producer_count:
            producer_targets += 1
        if producer_count and target.id in rooted:
            target.attrs["upstream_unresolved"] = False
            target.attrs["rooted_by_current_source"] = True
            target.status = "confirmed"
            target.confidence = 1.0
            rooted_targets += 1
        elif producer_count:
            target.attrs["rooted_by_current_source"] = False
            target.attrs["upstream_unresolved"] = True

    codemap.meta["host_key_root_trace"] = {
        "target_variables": len(targets),
        "producer_variables": producer_targets,
        "rooted_variables": rooted_targets,
        "assignment_records": len(records),
        "policy": "canonical-source-producer/v3",
    }
    _link_tiling_host_inputs(codemap, root, texts, api_maps)
    return codemap


def _function_scopes(masked: str) -> list[_Scope]:
    out: list[_Scope] = []
    for hit in iter_function_defs(masked):
        if hit.close_brace > hit.open_brace:
            out.append(_Scope(hit.name, hit.open_brace + 1, hit.close_brace))
    return out


def _containing_scope(scopes: list[_Scope], offset: int) -> str:
    matches = [s for s in scopes if s.start <= offset <= s.end]
    if not matches:
        return ""
    return min(matches, key=lambda s: s.end - s.start).name


def _assignments(root: Path, path: Path, text: str) -> list[_Record]:
    masked = mask_non_code(text)
    scopes = _function_scopes(masked)
    guard_scopes: list[tuple[int, int, str]] = []
    for match in _IF_RE.finditer(masked):
        open_pos = masked.find("{", match.start(), match.end())
        close_pos = matching_brace(masked, open_pos)
        if close_pos >= 0:
            guard_scopes.append((open_pos + 1, close_pos, match.group("cond").strip()))
    for match in _SWITCH_RE.finditer(masked):
        open_pos = masked.find("{", match.start(), match.end())
        close_pos = matching_brace(masked, open_pos)
        if close_pos >= 0:
            guard_scopes.append((open_pos + 1, close_pos, match.group("cond").strip()))
    out: list[_Record] = []
    for match in _ASSIGN_RE.finditer(masked):
        lhs = normalize_symbol(match.group("lhs"))
        rhs = text[match.start("rhs"):match.end("rhs")].strip()
        guards = tuple(cond for start, end, cond in guard_scopes if start <= match.start() <= end)
        out.append(
            _Record(
                lhs=lhs,
                rhs=rhs,
                guards=guards,
                file=_rel(root, path),
                line=_line(text, match.start()),
                function=_containing_scope(scopes, match.start()),
            )
        )
    return out


def _records_from_host_ir(root: Path, host_ir: Any) -> list[_Record]:
    """Prefer Clang SSA writes when a HostIR walk is available."""
    if host_ir is None:
        return []
    out: list[_Record] = []
    seen: set[tuple[str, int, str, str]] = set()
    events = list(getattr(host_ir, "writes", None) or [])
    events.extend(getattr(host_ir, "local_writes", None) or [])
    for ev in events:
        kind = str(getattr(ev, "kind", "assign") or "assign")
        if kind not in {"assign", "replace", ""}:
            continue
        lhs = normalize_symbol(str(getattr(ev, "path", "") or "").replace("->", "."))
        rhs = str(getattr(ev, "rhs", "") or "").strip()
        if not lhs or not rhs:
            continue
        file = str(getattr(ev, "file", "") or "").replace("\\", "/")
        try:
            file = _rel(root, Path(file)) if file else file
        except Exception:
            pass
        line = int(getattr(ev, "line", 0) or 0)
        function = str(getattr(ev, "function", "") or "")
        key = (file, line, lhs, rhs)
        if key in seen:
            continue
        seen.add(key)
        guards = tuple(str(g) for g in (ev.guards() if hasattr(ev, "guards") else []) if g)
        out.append(
            _Record(
                lhs=lhs,
                rhs=rhs,
                guards=guards,
                file=file,
                line=line,
                function=function,
            )
        )
    return out


def _api_maps(codemap: CodeMap, texts: list[tuple[Path, str]]) -> dict[str, Any]:
    tensor_inputs = sorted(
        (e for e in codemap.by_kind(EntityKind.INPUT) if e.attrs.get("api_kind") == "tensor"),
        key=lambda e: int(e.attrs.get("api_index") or 0),
    )
    attrs = sorted(
        (e for e in codemap.by_kind(EntityKind.INPUT) if e.attrs.get("api_kind") == "attribute"),
        key=lambda e: int(e.attrs.get("api_attr_index") or 0),
    )
    tokens: dict[str, list[str]] = {"InputIndex": [], "AttrIndex": []}
    constants: dict[str, int] = {}
    for _path, text in texts:
        for match in _ENUM_INDEX_RE.finditer(text):
            tokens[match.group(1)] = _enum_names(match.group(2))
        for match in _CONST_INT_RE.finditer(text):
            try:
                constants[match.group("name")] = int(match.group("value"), 0)
            except ValueError:
                pass
        for match in _DEFINE_RE.finditer(text):
            raw = match.group("value").strip()
            try:
                constants[match.group("name")] = int(raw, 0)
            except ValueError:
                pass
    return {
        "InputIndex": {name: tensor_inputs[i] for i, name in enumerate(tokens["InputIndex"]) if i < len(tensor_inputs)},
        "AttrIndex": {name: attrs[i] for i, name in enumerate(tokens["AttrIndex"]) if i < len(attrs)},
        "input_by_position": {i: ent for i, ent in enumerate(tensor_inputs)},
        "attr_by_position": {i: ent for i, ent in enumerate(attrs)},
        "constants": constants,
        "input_spellings": _InputSpellings(codemap),
    }


class _InputSpellings:
    """Every spelling of every API input, mapped to the inputs that answer to it.

    `_input_by_name` rebuilt this over the whole INPUT set on each of its ~7k
    calls, which is where 2.4M of the stage's 2.45M `normalize_symbol` calls came
    from -- the cost was quadratic in inputs x lookups to answer a question whose
    inputs do not change while one trace runs. Buckets keep API order, so a
    lookup still resolves to the same input the scan would have found first.
    """

    __slots__ = ("_buckets",)

    def __init__(self, codemap: CodeMap) -> None:
        buckets: dict[str, list[tuple[int, Entity]]] = {}
        for pos, ent in enumerate(codemap.by_kind(EntityKind.INPUT)):
            for spelling in self._spellings(ent):
                if spelling:
                    buckets.setdefault(spelling, []).append((pos, ent))
        self._buckets = buckets

    @staticmethod
    def _spellings(ent: Entity) -> set[str]:
        out = {normalize_symbol(ent.name), short_symbol(ent.name)}
        src = str(ent.attrs.get("source_name") or "")
        if src:
            out.add(normalize_symbol(src))
            out.add(short_symbol(src))
        return out

    def first(self, *spellings: str) -> Entity | None:
        """The earliest input in API order answering to any of `spellings`."""
        best: tuple[int, Entity] | None = None
        for spelling in spellings:
            bucket = self._buckets.get(spelling)
            if bucket and (best is None or bucket[0][0] < best[0]):
                best = bucket[0]
        return None if best is None else best[1]

    def sole(self, spelling: str) -> Entity | None:
        """The one input answering to `spelling`, or None if it is ambiguous."""
        bucket = self._buckets.get(spelling) or ()
        return bucket[0][1] if len(bucket) == 1 else None


def _compile_symbols(codemap: CodeMap, texts: list[tuple[Path, str]]) -> set[str]:
    symbols: set[str] = set()
    for _path, text in texts:
        for match in _CONST_ANY_RE.finditer(text):
            symbols.add(normalize_symbol(match.group("name")))
        for match in _DEFINE_RE.finditer(text):
            symbols.add(normalize_symbol(match.group("name")))
        for match in _CONST_STATIC_INT_RE.finditer(text):
            symbols.add(normalize_symbol(match.group("name")))
        for match in _ENUM_RE.finditer(text):
            enum_name = match.group("name")
            scoped = bool(match.group("scoped"))
            for member in _enum_names(match.group("body")):
                symbols.add(f"{enum_name}::{member}")
                if not scoped:
                    symbols.add(member)
    for kind in (EntityKind.COMPILE_VAR, EntityKind.MACRO):
        for ent in codemap.by_kind(kind):
            if _trusted_compile_root(ent):
                symbols.add(normalize_symbol(ent.name))
    return symbols


def _target_scope(target: Entity) -> tuple[str, str]:
    sites = [s for s in (target.attrs.get("host_key_use_sites") or []) if isinstance(s, dict)]
    if not sites:
        return "", ""
    files = {str(s.get("file") or "") for s in sites}
    functions = {str(s.get("function") or "") for s in sites}
    return (next(iter(files)) if len(files) == 1 else "", next(iter(functions)) if len(functions) == 1 else "")


def _select_records(
    symbol: str,
    *,
    by_exact: dict[str, list[_Record]],
    by_short: dict[str, list[_Record]],
    use_file: str = "",
    use_function: str = "",
) -> tuple[list[_Record], bool]:
    normalized = normalize_symbol(symbol)
    records = list(by_exact.get(normalized) or [])
    if records:
        if not is_member_symbol(normalized) and use_function:
            scoped = [r for r in records if r.function == use_function and (not use_file or r.file == use_file)]
            if scoped:
                records = scoped
        return records, False

    candidates = list(by_short.get(short_symbol(normalized)) or [])
    if not candidates:
        return [], False
    if not is_member_symbol(normalized) and use_function:
        scoped = [r for r in candidates if r.function == use_function and (not use_file or r.file == use_file)]
        if scoped:
            candidates = scoped
    spellings = {r.lhs for r in candidates}
    if len(spellings) == 1:
        return candidates, False
    return [], True


def _resolve_symbol(
    codemap: CodeMap,
    target: Entity,
    symbol: str,
    *,
    by_exact: dict[str, list[_Record]],
    by_short: dict[str, list[_Record]],
    api_maps: dict[str, Any],
    compile_symbols: set[str],
    symbol_nodes: dict[str, Entity],
    visiting: set[str],
    visited: set[str],
    use_file: str = "",
    use_function: str = "",
) -> None:
    normalized = normalize_symbol(symbol)
    state = f"{target.id}:{normalized}:{use_file}:{use_function}"
    if state in visited or state in visiting:
        return
    visiting.add(state)
    records, ambiguous = _select_records(
        normalized,
        by_exact=by_exact,
        by_short=by_short,
        use_file=use_file,
        use_function=use_function,
    )
    if ambiguous:
        target.attrs["producer_lookup_ambiguous"] = True
        target.attrs.setdefault("producer_lookup_symbols", []).append(normalized)

    producer_sites: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        site = {"file": record.file, "line": record.line, "function": record.function, "lhs": record.lhs}
        if site not in producer_sites:
            producer_sites.append(site)
        expr = codemap.upsert(
            EntityKind.PREDICATE,
            record.rhs,
            eid=f"HOSTDEF::{record.file}::{record.line}::{short_symbol(record.lhs)}::{index}",
            attrs={
                "predicate_role": "host_definition",
                "lhs": record.lhs,
                "expression": record.rhs,
                "guards": list(record.guards),
                "function": record.function,
                "provenance": "source_host_defuse",
            },
            file=record.file,
            line=record.line,
            status="confirmed",
        )
        codemap.link(
            RelationKind.DERIVES,
            expr.id,
            target.id,
            attrs={
                "provenance": "source_host_defuse",
                "lhs": record.lhs,
                "file": record.file,
                "line": record.line,
                "function": record.function,
            },
            status="confirmed",
        )
        resolved_any = False
        for text in [record.rhs, *record.guards]:
            if _link_api_accesses(codemap, expr, text, api_maps, file=record.file, line=record.line):
                resolved_any = True
            for ref, origin in _identifier_refs(text):
                ref_norm = normalize_symbol(ref)
                if ref_norm == normalized:
                    continue
                if _is_compile_reference(ref_norm, compile_symbols):
                    compile_root = codemap.upsert(
                        EntityKind.COMPILE_VAR,
                        ref_norm,
                        eid=f"HOSTCONST::{ref_norm}",
                        attrs={
                            "compile_root": True,
                            "provenance": "source_host_compile_symbol" if ref_norm in compile_symbols else "source_host_qualified_constant",
                        },
                        file=record.file,
                        line=record.line,
                        status="confirmed",
                    )
                    codemap.link(RelationKind.DERIVES, compile_root.id, expr.id, attrs={"provenance": compile_root.attrs["provenance"]}, status="confirmed")
                    resolved_any = True
                    continue

                api = _input_by_name(codemap, ref_norm, api_maps["input_spellings"])
                if api is not None:
                    codemap.link(
                        RelationKind.DERIVES,
                        api.id,
                        expr.id,
                        attrs={"provenance": "source_host_api_name", "symbol": ref_norm},
                        status="confirmed",
                    )
                    resolved_any = True
                    continue

                upstream_records, upstream_ambiguous = _select_records(
                    ref_norm,
                    by_exact=by_exact,
                    by_short=by_short,
                    use_file=record.file,
                    use_function=record.function,
                )
                if upstream_records:
                    upstream = symbol_nodes.get(ref_norm)
                    if upstream is None:
                        kind = EntityKind.FIELD if is_member_symbol(ref_norm) else EntityKind.VARIABLE
                        upstream = codemap.upsert(
                            kind,
                            ref_norm,
                            eid=f"HOSTDEFVAR::{kind.value}::{ref_norm}",
                            attrs={
                                "source_name": short_symbol(ref_norm),
                                "canonical_symbol": ref_norm,
                                "provenance": "source_host_defuse",
                            },
                            file=upstream_records[0].file,
                            line=upstream_records[0].line,
                            status="confirmed",
                        )
                        symbol_nodes[ref_norm] = upstream
                    codemap.link(RelationKind.DERIVES, upstream.id, expr.id, attrs={"provenance": "source_host_defuse_dependency", "symbol": ref_norm}, status="confirmed")
                    resolved_any = True
                    _resolve_symbol(
                        codemap,
                        upstream,
                        ref_norm,
                        by_exact=by_exact,
                        by_short=by_short,
                        api_maps=api_maps,
                        compile_symbols=compile_symbols,
                        symbol_nodes=symbol_nodes,
                        visiting=visiting,
                        visited=visited,
                        use_file=record.file,
                        use_function=record.function,
                    )
                    continue

                # A method-call receiver is a lookup key, not a value. Keep it
                # when it names an INPUT or has a producer; otherwise do not
                # invent a runtime leaf that pollutes TilingKey completeness.
                if origin == "call_receiver":
                    continue
                if _looks_like_runtime_reference(ref_norm):
                    unresolved = codemap.upsert(
                        EntityKind.FIELD if is_member_symbol(ref_norm) else EntityKind.VARIABLE,
                        ref_norm,
                        eid=f"HOSTUNRESOLVED::{ref_norm}",
                        attrs={
                            "source_name": short_symbol(ref_norm),
                            "canonical_symbol": ref_norm,
                            "dependency_unresolved": True,
                            "producer_lookup_ambiguous": upstream_ambiguous,
                            "provenance": "source_host_unresolved_dependency",
                        },
                        file=record.file,
                        line=record.line,
                        status="partial",
                    )
                    codemap.link(RelationKind.DERIVES, unresolved.id, expr.id, attrs={"provenance": "source_host_unresolved_dependency", "symbol": ref_norm}, status="partial")

        if not resolved_any and not _value_identifiers(record.rhs) and not _link_api_accesses(codemap, expr, record.rhs, api_maps, file=record.file, line=record.line):
            root = codemap.upsert(
                EntityKind.COMPILE_VAR,
                f"host-expr:{record.file}:{record.line}",
                attrs={"value_expr": record.rhs, "compile_root": True, "provenance": "source_host_constant_expr"},
                file=record.file,
                line=record.line,
                status="confirmed",
            )
            codemap.link(RelationKind.DERIVES, root.id, expr.id, attrs={"provenance": "source_host_constant_expr"}, status="confirmed")

    if producer_sites:
        existing = [s for s in (target.attrs.get("producer_sites") or []) if isinstance(s, dict)]
        for site in producer_sites:
            if site not in existing:
                existing.append(site)
        target.attrs["producer_sites"] = existing
        target.attrs["producer_site_count"] = len(existing)
        target.attrs["producer_provenance"] = "source_host_defuse"

    visiting.discard(state)
    visited.add(state)


def _link_api_accesses(
    codemap: CodeMap,
    expression: Entity,
    text: str,
    api_maps: dict[str, Any],
    *,
    file: str,
    line: int,
) -> bool:
    linked = False
    for kind, token in _API_TOKEN_RE.findall(text):
        api = api_maps.get(kind, {}).get(token)
        if api is not None:
            codemap.link(RelationKind.DERIVES, api.id, expression.id, attrs={"provenance": "source_host_api_index", "token": f"{kind}::{token}"}, status="confirmed")
            linked = True
    for match in _INPUT_ACCESS_RE.finditer(text):
        api = _api_from_index(match.group("arg"), "input", api_maps)
        if api is not None:
            codemap.link(RelationKind.DERIVES, api.id, expression.id, attrs={"provenance": "source_host_api_accessor", "accessor": match.group(0)}, status="confirmed")
            linked = True
    for match in _ATTR_ACCESS_RE.finditer(text):
        api = _api_from_index(match.group("arg"), "attr", api_maps)
        if api is not None:
            codemap.link(RelationKind.DERIVES, api.id, expression.id, attrs={"provenance": "source_host_api_accessor", "accessor": match.group(0)}, status="confirmed")
            linked = True
    if "GetDeterministic(" in text:
        runtime = codemap.upsert(
            EntityKind.INPUT,
            "__context__.deterministic",
            eid="HOST_CONTEXT::deterministic",
            attrs={"api_kind": "runtime_context", "source_accessor": "GetDeterministic", "provenance": "source_host_runtime_context"},
            file=file,
            line=line,
            status="confirmed",
        )
        codemap.link(RelationKind.DERIVES, runtime.id, expression.id, attrs={"provenance": "source_host_runtime_context"}, status="confirmed")
        linked = True
    return linked


def _link_tiling_host_inputs(
    codemap: CodeMap,
    root: Path,
    texts: list[tuple[Path, str]],
    api_maps: dict[str, Any],
) -> None:
    """Host tiling that reads INPUT and packs keys is the selection path.

    Catalog / mixed-literal TPL formulas can be compile-rooted without an
    INPUT→KEY edge. When the same tiling file both accesses GetInput* and
    packs keys, that access is the source-backed path.
    """
    packed = [
        key
        for key in codemap.by_kind(EntityKind.TILING_KEY)
        if key.attrs.get("source_declared")
        and (key.attrs.get("host_packing_expressions") or key.attrs.get("packing_value_sites"))
    ]
    packing_sources = [
        e
        for e in list(codemap.by_kind(EntityKind.VARIABLE)) + list(codemap.by_kind(EntityKind.FIELD))
        if e.attrs.get("host_key_argument")
    ]
    if not packed:
        return
    tensor_inputs = [
        e for e in codemap.by_kind(EntityKind.INPUT) if e.attrs.get("api_kind") == "tensor"
    ]
    if not tensor_inputs:
        tensor_inputs = list(codemap.by_kind(EntityKind.INPUT))
    if not tensor_inputs:
        return
    linked = 0
    for path, text in texts:
        if not _TILING_PACK_TOKEN_RE.search(text):
            continue
        if not _INPUT_ACCESS_RE.search(text) and not _ATTR_ACCESS_RE.search(text):
            continue
        accessed: list[Entity] = []
        seen: set[str] = set()
        for match in _INPUT_ACCESS_RE.finditer(text):
            api = _api_from_index(match.group("arg"), "input", api_maps)
            if api is not None and api.id not in seen:
                seen.add(api.id)
                accessed.append(api)
        for match in _ATTR_ACCESS_RE.finditer(text):
            api = _api_from_index(match.group("arg"), "attr", api_maps)
            if api is not None and api.id not in seen:
                seen.add(api.id)
                accessed.append(api)
        if not accessed:
            accessed = tensor_inputs
        file = _rel(root, path)
        for inp in accessed:
            for dest in (*packed, *packing_sources):
                codemap.link(
                    RelationKind.DERIVES,
                    inp.id,
                    dest.id,
                    attrs={
                        "provenance": "source_host_tiling_input",
                        "file": file,
                    },
                    status="confirmed",
                )
                linked += 1
    trace = codemap.meta.setdefault("host_key_root_trace", {})
    if isinstance(trace, dict):
        trace["tiling_input_key_links"] = linked


def _api_from_index(raw: str, kind: str, api_maps: dict[str, Any]) -> Entity | None:
    token = normalize_symbol(raw).strip()
    enum_kind = "InputIndex" if kind == "input" else "AttrIndex"
    if token.startswith(enum_kind + "::"):
        return api_maps.get(enum_kind, {}).get(token.split("::", 1)[1])
    try:
        position = int(token, 0)
    except ValueError:
        position = api_maps.get("constants", {}).get(token)
    if position is None:
        return None
    table = api_maps.get("input_by_position" if kind == "input" else "attr_by_position", {})
    return table.get(int(position))


def _input_by_name(codemap: CodeMap, name: str, spellings: "_InputSpellings") -> Entity | None:
    needle = normalize_symbol(name)
    if not needle:
        return None
    hits = codemap.by_name(name, kind=EntityKind.INPUT) or codemap.by_name(short_symbol(name), kind=EntityKind.INPUT)
    if hits:
        return hits[0]
    found = spellings.first(needle, short_symbol(needle))
    if found is not None:
        return found
    # Member paths such as ``ifaContext.query.desc`` still name the API tensor.
    for part in needle.replace("::", ".").split("."):
        if not part:
            continue
        sole = spellings.sole(part)
        if sole is not None:
            return sole
    return None


def _is_compile_reference(value: str, compile_symbols: set[str]) -> bool:
    if value in compile_symbols:
        return True
    if re.fullmatch(r"(?:T(?:I)?LING_KEY_|[A-Z0-9_]*TILING_KEY)[A-Z0-9_]*", value or ""):
        return True
    if "::" not in value:
        return False
    tail = value.rsplit("::", 1)[-1]
    # External CANN enum values are accepted only when the value token itself
    # is constant-like; a qualified type or method name is not a root.
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*|NUM\d+|DT_[A-Z0-9_]+", tail))


def _looks_like_runtime_reference(value: str) -> bool:
    if not value:
        return False
    head = value.split(".")[0].split("::")[0]
    return head not in _IGNORED and not value.isdigit()


def _code_only(text: str) -> str:
    """Remove lexical regions that cannot contain value dependencies."""
    cleaned = _BLOCK_COMMENT_RE.sub(" ", text or "")
    cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)
    cleaned = _STRING_RE.sub(" ", cleaned)
    cleaned = _CAST_RE.sub("static_cast", cleaned)
    return cleaned


def _identifier_refs(text: str) -> list[tuple[str, str]]:
    """Scan value names and method-call receivers.

    A trailing ``(...)`` is a call, not a value. Nested receivers such as
    ``context.query.desc->GetDataType()`` still name an API tensor and are
    kept for INPUT lookup. A bare receiver such as ``context_->GetInputShape()``
    is kept only as a lookup key (producer / INPUT); it must not become an
    unresolved runtime leaf. Qualified ``ns::Type(...)`` construction is
    dropped entirely.
    """
    cleaned = _code_only(text)
    out: list[tuple[str, str]] = []
    for match in _IDENT_RE.finditer(cleaned):
        token = normalize_symbol(match.group(0))
        head = token.split(".")[0].split("::")[0]
        if head in _IGNORED or token in _IGNORED or token.isdigit():
            continue
        rest = cleaned[match.end():]
        origin = "value"
        if _CALL_SUFFIX_RE.match(rest):
            if "." not in token:
                # Free function or qualified ctor/call: ``Foo()``, ``ns::Type()``.
                continue
            token = token.rsplit(".", 1)[0]
            if not token or token.split(".")[0].split("::")[0] in _IGNORED:
                continue
            origin = "call_receiver"
        out.append((token, origin))
    return out


def _value_identifiers(text: str) -> list[str]:
    return [name for name, kind in _identifier_refs(text) if kind == "value"]


def _identifiers(text: str) -> list[str]:
    return [name for name, _kind in _identifier_refs(text)]


def _source_rooted_entities(codemap: CodeMap) -> set[str]:
    roots = {e.id for e in codemap.entities.values() if _trusted_root(e)}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() in _ROOT_RELATIONS:
            adjacency[rel.src].append(rel.dst)
    seen = set(roots)
    queue = deque(roots)
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _trusted_root(entity: Entity) -> bool:
    kind = entity.kind_name()
    if kind in {EntityKind.INPUT.value, EntityKind.BUILD_VARIANT.value, EntityKind.ARCH.value}:
        return True
    if kind in {EntityKind.COMPILE_VAR.value, EntityKind.MACRO.value}:
        return _trusted_compile_root(entity)
    return False


def _trusted_compile_root(entity: Entity) -> bool:
    provenance = str(entity.attrs.get("provenance") or "")
    origin = str(entity.attrs.get("origin") or "")
    return bool(entity.attrs.get("compile_root") or provenance.startswith("source_") or provenance.startswith("source_host_") or origin == "constexpr_or_define")


def _enum_names(body: str) -> list[str]:
    out: list[str] = []
    for raw in body.split(","):
        item = re.sub(r"//.*", "", raw).strip()
        if not item:
            continue
        name = item.split("=", 1)[0].strip()
        if re.match(r"^[A-Za-z_]\w*$", name):
            out.append(name)
    return out


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
