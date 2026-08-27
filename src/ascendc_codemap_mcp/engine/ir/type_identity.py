# -*- coding: utf-8 -*-
"""Unique Clang/macro type identity for uo-init analyze.

Regex may name a type; it must uniquely hit Clang ``type_decls`` / ``field_decls``
or a ``BEGIN_TILING_DATA_DEF`` / REGISTER contract. Short names never spray.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Iterator

_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def short_type_name(name: str) -> str:
    return str(name or "").split("::")[-1]


def type_tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text or ""))


_DEFINE_HEAD_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\b(.*)$"
)


def iter_object_macro_bodies(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``#define Name replacement`` bodies, joining ``\\`` continuations.

    Function-like macros (``#define NAME(``) are skipped. Replacement may
    uniquely name a TilingData type, as in FAG's ``FagTilingType``.
    """
    if not text:
        return
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        match = _DEFINE_HEAD_RE.match(lines[i])
        i += 1
        if match is None:
            continue
        name, rest = match.group(1), match.group(2)
        if rest.lstrip().startswith("("):
            continue
        parts = [rest]
        while parts and parts[-1].rstrip().endswith("\\"):
            parts[-1] = parts[-1].rstrip()[:-1]
            if i >= n:
                break
            parts.append(lines[i])
            i += 1
        yield name, " ".join(p.strip() for p in parts)


def macro_type_aliases(text: str, known: Iterable[str]) -> dict[str, set[str]]:
    """Map object-macro names onto a unique known TilingData type in the body."""
    known_set = {str(item) for item in known if item}
    out: dict[str, set[str]] = {}
    if not known_set:
        return out
    for name, body in iter_object_macro_bodies(text):
        hits = type_tokens(body) & known_set
        if len(hits) == 1:
            out[name] = set(hits)
    return out


def merge_unique_macro_aliases(*texts: str, known: Iterable[str]) -> dict[str, set[str]]:
    """Union object-macro aliases across files; drop names that map to two types."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        for name, types in macro_type_aliases(text, known).items():
            grouped[name].update(types)
    return {name: types for name, types in grouped.items() if len(types) == 1}


def clang_type_short_counts(*irs: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ir in irs:
        if ir is None:
            continue
        for td in getattr(ir, "type_decls", None) or ():
            if isinstance(td, dict):
                name = short_type_name(str(td.get("name") or td.get("qualified_name") or ""))
            else:
                name = short_type_name(
                    str(getattr(td, "name", "") or getattr(td, "qualified_name", "") or "")
                )
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def named_type_is_unique(
    name: str,
    *,
    clang_counts: dict[str, int] | None = None,
    macro_names: Iterable[str] = (),
) -> bool:
    """Accept a macro argument type when Clang is unambiguous or the macro list names it."""
    short = short_type_name(name)
    if not short:
        return False
    macros = {short_type_name(n) for n in macro_names}
    if short in macros:
        return True
    n = (clang_counts or {}).get(short, 0)
    return n <= 1


def _merge_type_text(left: str, right: str) -> str:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a:
        return b
    if not b or b == a or b in a:
        return a
    if a in b:
        return b
    return f"{a} | {b}"


def iter_field_decl_rows(*irs: Any) -> Iterator[tuple[str, str, str, str, int]]:
    """Yield (owner_short, member, type_text, file, line) from HostIR/KernelIR field_decls."""
    collected: dict[tuple[str, str, str], tuple[str, str, str, str, int]] = {}
    for ir in irs:
        if ir is None:
            continue
        decls = getattr(ir, "field_decls", None) or {}
        items = decls.items() if isinstance(decls, dict) else enumerate(decls)
        for key, decl in items:
            owner = ""
            member = ""
            if isinstance(key, tuple) and len(key) >= 2:
                owner = short_type_name(str(key[0]))
                member = str(key[1])
            if isinstance(decl, dict):
                owner = owner or short_type_name(str(decl.get("host") or decl.get("owner") or ""))
                member = member or str(decl.get("name") or decl.get("member") or "")
                type_text = str(decl.get("type_text") or decl.get("canonical_type") or "")
                file = str(decl.get("file") or "")
                line = int(decl.get("line") or 0)
            else:
                owner = owner or short_type_name(
                    str(getattr(decl, "host", "") or getattr(decl, "owner_qualified", "") or "")
                )
                member = member or str(getattr(decl, "name", "") or "")
                type_text = str(
                    getattr(decl, "type_text", "") or getattr(decl, "canonical_type", "") or ""
                )
                file = str(getattr(decl, "file", "") or "")
                line = int(getattr(decl, "line", 0) or 0)
            if not owner or not member:
                continue
            dedupe = (owner, member, file)
            prev = collected.get(dedupe)
            if prev is not None:
                collected[dedupe] = (prev[0], prev[1], _merge_type_text(prev[2], type_text), prev[3], prev[4])
                continue
            collected[dedupe] = (owner, member, type_text, file, line)
    yield from collected.values()


def iter_unique_field_decls(*irs: Any) -> Iterator[tuple[str, str, str, str, int]]:
    """One row per (owner, member). Divergent types stay as guarded alternatives."""
    grouped: dict[tuple[str, str], list[tuple[str, str, str, str, int]]] = defaultdict(list)
    for row in iter_field_decl_rows(*irs):
        grouped[(row[0], row[1])].append(row)
    for rows in grouped.values():
        if len(rows) == 1:
            yield rows[0]
            continue
        texts = {str(row[2] or "").strip() for row in rows if str(row[2] or "").strip()}
        if len(texts) <= 1:
            continue
        type_text = rows[0][2]
        for extra in rows[1:]:
            type_text = _merge_type_text(type_text, extra[2])
        first = rows[0]
        yield (first[0], first[1], type_text, first[3], first[4])


def ir_var_types(host_ir: Any, known: set[str]) -> dict[str, set[str]]:
    """Map locals / members onto uniquely known TilingData type names."""
    out: dict[str, set[str]] = defaultdict(set)
    if host_ir is None or not known:
        return out
    for decl in getattr(host_ir, "local_decls", None) or ():
        if isinstance(decl, dict):
            name = str(decl.get("name") or "")
            type_text = str(decl.get("type_text") or decl.get("canonical_type") or "")
        else:
            name = str(getattr(decl, "name", "") or "")
            type_text = str(
                getattr(decl, "type_text", "") or getattr(decl, "canonical_type", "") or ""
            )
        hits = type_tokens(type_text) & known
        if name and hits:
            out[name].update(hits)
    for owner, member, type_text, _file, _line in iter_unique_field_decls(host_ir):
        if owner in known:
            hits = type_tokens(type_text) & known
            if hits:
                out[member].update(hits)
                if not member.endswith("_"):
                    out[member + "_"].update(hits)
    return out
