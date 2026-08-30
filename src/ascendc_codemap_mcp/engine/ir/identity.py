# -*- coding: utf-8 -*-
"""Canonical declaration identity: bind existing Clang facts, never mint a second leaf."""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind

#: C++ keywords / specifiers that are never a FUNCTION or METHOD name.
CXX_SPECIFIERS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "try",
        "catch",
        "throw",
        "sizeof",
        "alignof",
        "decltype",
        "static_assert",
        "constexpr",
        "consteval",
        "constinit",
        "const",
        "volatile",
        "static",
        "extern",
        "inline",
        "virtual",
        "explicit",
        "friend",
        "typedef",
        "using",
        "typename",
        "template",
        "namespace",
        "class",
        "struct",
        "enum",
        "union",
        "public",
        "private",
        "protected",
        "new",
        "delete",
        "this",
        "true",
        "false",
        "nullptr",
        "operator",
        "noexcept",
        "co_await",
        "co_return",
        "co_yield",
        "concept",
        "requires",
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "likely",
        "unlikely",
    }
)

_DECLARATION_KINDS = frozenset(
    {
        EntityKind.TYPE.value,
        EntityKind.FIELD.value,
        EntityKind.METHOD.value,
        EntityKind.FUNCTION.value,
    }
)
_ALIAS_MEMBER_NAMES = frozenset({"TYPE", "type", "value_type", "type_t"})


def _norm_file(path: str) -> str:
    return str(path or "").replace("\\", "/")


def _leaf(name: str) -> str:
    return str(name or "").replace(".", "::").split("::")[-1].strip()


def qualified_name(symbol: str, *, owner: str = "") -> str:
    leaf = _leaf(symbol)
    owner_s = str(owner or "").strip()
    if owner_s and leaf and not leaf.startswith(owner_s + "::"):
        return f"{owner_s}::{leaf}"
    return leaf or str(symbol or "")


def is_forbidden_callable_name(name: str) -> bool:
    """True when *name* cannot be a C++ function or method identifier."""
    leaf = _leaf(name)
    if not leaf or not leaf.isidentifier():
        return True
    return leaf in CXX_SPECIFIERS


def is_untrusted_scope(scope: str) -> bool:
    """True when *scope* is a C++ specifier and must not mint a local identity."""
    leaf = _leaf(scope)
    return bool(leaf) and leaf in CXX_SPECIFIERS


def trusted_scope_name(scope: str) -> str:
    leaf = _leaf(scope)
    if not leaf or leaf in CXX_SPECIFIERS:
        return ""
    return leaf


def is_alias_not_field(name: str, ctype: str = "") -> bool:
    """True when a member scan is a ``using`` / typedef alias, not a FIELD."""
    leaf = _leaf(name)
    text = str(ctype or "").strip()
    compact = " ".join(text.split())
    if compact.startswith(("using ", "typedef ", "typename ")) or compact in {
        "using",
        "typedef",
        "typename",
    }:
        return True
    if leaf in _ALIAS_MEMBER_NAMES and (
        compact.startswith(("using", "typedef"))
        or "conditional" in compact
        or "typename" in compact
    ):
        return True
    return False


def declaration_id(
    *,
    kind: EntityKind | str,
    architecture: str,
    file: str,
    owner: str,
    symbol: str,
) -> str:
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    return "::".join(
        [
            kind_name,
            str(architecture or "").strip() or "_",
            _norm_file(file) or "_",
            str(owner or "").strip() or "_",
            _leaf(symbol) or str(symbol or "_"),
        ]
    )


def _owner_of(ent: Entity) -> str:
    return str(
        ent.attrs.get("lexical_owner")
        or ent.attrs.get("owner")
        or ""
    ).strip()


def find_declaration(
    codemap: CodeMap,
    kind: EntityKind | str,
    *,
    symbol: str,
    owner: str = "",
    file: str = "",
) -> Entity | None:
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    leaf = _leaf(symbol)
    want_q = qualified_name(symbol, owner=owner)
    want_file = _norm_file(file)
    hits: list[Entity] = []
    for ent in codemap.by_kind(kind_name):
        name = str(ent.name or "")
        ent_leaf = _leaf(name)
        if ent_leaf != leaf:
            continue
        ent_owner = _owner_of(ent)
        ent_file = _norm_file(ent.file)
        if owner:
            if not (
                ent_owner == owner
                or name == want_q
                or name.startswith(owner + "::")
            ):
                continue
            if want_file and ent_file and ent_file != want_file and not ent_file.endswith(
                "/" + want_file
            ) and not want_file.endswith("/" + ent_file):
                continue
            hits.append(ent)
            continue
        if want_file and (ent_file == want_file or ent_file.endswith("/" + want_file)):
            if name == leaf or name == want_q:
                hits.append(ent)
    if len(hits) == 1:
        return hits[0]
    exact = [e for e in hits if str(e.name or "") == want_q]
    if len(exact) == 1:
        return exact[0]
    return None


def bind_or_create(
    codemap: CodeMap,
    kind: EntityKind | str,
    symbol: str,
    *,
    file: str = "",
    line: int = 0,
    owner: str = "",
    architecture: str = "",
    attrs: dict[str, Any] | None = None,
    status: str = "extracted",
) -> Entity | None:
    """Reuse a Clang (or earlier) declaration; mint owner-aware id only if absent.

    TYPE / FIELD / METHOD / FUNCTION C++ declarations enter the graph here.
    Returns ``None`` when *symbol* is a C++ keyword/specifier for METHOD/FUNCTION.
    """
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    if kind_name in {EntityKind.METHOD.value, EntityKind.FUNCTION.value}:
        if is_forbidden_callable_name(symbol):
            return None
    if kind_name == EntityKind.FIELD.value and is_alias_not_field(symbol, str((attrs or {}).get("cpp_type") or "")):
        return None
    existing = find_declaration(
        codemap, kind, symbol=symbol, owner=owner, file=file
    )
    payload = dict(attrs or {})
    if owner:
        payload.setdefault("lexical_owner", owner)
        payload.setdefault("owner", owner)
    if architecture:
        payload.setdefault("architecture", architecture)
    name = qualified_name(symbol, owner=owner)
    if existing is not None:
        existing.attrs.update({k: v for k, v in payload.items() if v not in (None, "")})
        if file and not existing.file:
            existing.file = file
            existing.line_start = int(line or existing.line_start or 0)
        return existing
    eid = declaration_id(
        kind=kind,
        architecture=architecture or str(getattr(codemap, "architecture", "") or ""),
        file=file,
        owner=owner,
        symbol=symbol,
    )
    return codemap.upsert(
        kind,
        name,
        eid=eid,
        attrs=payload,
        file=file,
        line=line,
        status=status,
    )


def declaration_key(ent: Entity) -> tuple[str, str, str, str]:
    """Canonical (kind, file, owner, leaf) for duplicate-declaration probes."""
    return (
        ent.kind_name(),
        _norm_file(ent.file),
        _owner_of(ent),
        _leaf(ent.name),
    )


def is_declaration_kind(kind: EntityKind | str) -> bool:
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    return kind_name in _DECLARATION_KINDS


_LOCAL_OBJECT_KINDS = frozenset(
    {
        EntityKind.BUFFER.value,
        EntityKind.REGISTER.value,
        EntityKind.QUEUE.value,
        EntityKind.EVENT.value,
        EntityKind.PIPE.value,
    }
)

_LOCAL_ID_KIND = {
    EntityKind.BUFFER.value: "Buffer",
    EntityKind.REGISTER.value: "Register",
    EntityKind.QUEUE.value: "Queue",
    EntityKind.EVENT.value: "Event",
    EntityKind.PIPE.value: "Pipe",
}


def is_local_object_kind(kind: EntityKind | str) -> bool:
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    return kind_name in _LOCAL_OBJECT_KINDS


def local_object_key(
    ent: Entity,
) -> tuple[str, str, int, int, str, str]:
    """(kind, file, line, column, name, instantiation) for duplicate-site probes."""
    inst = str(ent.attrs.get("instantiation_context") or "").strip()
    col = int(ent.attrs.get("column") or 0)
    return (
        ent.kind_name(),
        _norm_file(ent.file),
        int(ent.line_start or 0),
        col,
        _leaf(ent.name),
        inst,
    )


def find_local_object(
    codemap: CodeMap,
    kind: EntityKind | str,
    *,
    name: str,
    file: str,
    line: int,
    column: int = 0,
    instantiation_context: str = "",
) -> Entity | None:
    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    leaf = _leaf(name)
    want_file = _norm_file(file)
    want_line = int(line or 0)
    want_col = int(column or 0)
    want_inst = str(instantiation_context or "").strip()
    if not leaf or not want_file or want_line <= 0:
        return None
    hits: list[Entity] = []
    for ent in codemap.by_kind(kind_name):
        if _leaf(ent.name) != leaf:
            continue
        if _norm_file(ent.file) != want_file:
            continue
        if int(ent.line_start or 0) != want_line:
            continue
        if int(ent.attrs.get("column") or 0) != want_col:
            continue
        if str(ent.attrs.get("instantiation_context") or "").strip() != want_inst:
            continue
        hits.append(ent)
    if len(hits) == 1:
        return hits[0]
    return hits[0] if hits else None


def bind_local_object(
    codemap: CodeMap,
    kind: EntityKind | str,
    name: str,
    *,
    file: str = "",
    line: int = 0,
    column: int = 0,
    scope: str = "",
    instantiation_context: str = "",
    root: str = "",
    attrs: dict[str, Any] | None = None,
    status: str = "extracted",
    confidence: float = 1.0,
) -> Entity | None:
    """Bind a BUFFER/REGISTER/QUEUE/EVENT/PIPE to one source declaration.

    Untrusted scope never participates in identity. Only an explicit
    ``instantiation_context`` may split the same ``file:line:name``.
    """
    from ascendc_codemap_mcp.engine.ids import local_object_id

    kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
    if kind_name not in _LOCAL_OBJECT_KINDS:
        return None
    leaf = _leaf(name)
    if not leaf or not str(file or "").strip() or int(line or 0) <= 0:
        return None
    inst = str(instantiation_context or "").strip()
    existing = find_local_object(
        codemap,
        kind,
        name=leaf,
        file=file,
        line=line,
        column=column,
        instantiation_context=inst,
    )
    payload = dict(attrs or {})
    trusted = trusted_scope_name(scope)
    if trusted:
        payload.setdefault("scope", trusted)
    elif "scope" in payload and is_untrusted_scope(str(payload.get("scope") or "")):
        payload.pop("scope", None)
    if inst:
        payload["instantiation_context"] = inst
    if column:
        payload["column"] = int(column)
    if existing is not None:
        existing.attrs.update({k: v for k, v in payload.items() if v not in (None, "")})
        if file and not existing.file:
            existing.file = file
            existing.line_start = int(line or existing.line_start or 0)
        return existing
    eid = local_object_id(
        kind=kind_name,
        file=file,
        line=line,
        name=leaf,
        column=column,
        instantiation_context=inst,
        root=root,
    )
    return codemap.upsert(
        kind,
        leaf,
        eid=eid,
        attrs=payload,
        file=file,
        line=line,
        status=status,
        confidence=confidence,
    )
