# -*- coding: utf-8 -*-
"""Host virtual call families: empty base vs override.

Clang keys host functions by short name, so
``virtual Foo() {}`` and ``Foo() override { ... }`` collapse to whichever
definition has the wider span. The call graph then looks like a direct call
to the override, which is how Q4 lost
``DoSparse:1110 GetSparseUnpadBlockInfo()`` → empty virtual at the base header
plus the varlen override. This module reconstructs that family from the
entities and declaration text that survived.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

_VIRTUAL_RE = re.compile(r"\bvirtual\b")
_OVERRIDE_RE = re.compile(r"\boverride\b")
_EMPTY_BODY_RE = re.compile(r"\{\s*\}")
_PURE_RE = re.compile(r"=\s*0\b")
_OWNER_RE = re.compile(r"^([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)::[A-Za-z_]\w*$")


def leaf_name(name: str) -> str:
    return str(name or "").replace(".", "::").split("::")[-1].strip()


def owner_of(name: str, attrs: dict[str, Any] | None = None) -> str:
    data = attrs if isinstance(attrs, dict) else {}
    owner = str(data.get("owner") or data.get("lexical_owner") or "").strip()
    if owner:
        return owner.split("::")[-1]
    match = _OWNER_RE.match(str(name or "").replace(".", "::").strip())
    if match:
        return match.group(1).split("::")[-1]
    qual = str(data.get("qualified_name") or "")
    match = _OWNER_RE.match(qual.replace(".", "::").strip())
    return match.group(1).split("::")[-1] if match else ""


def classify_decl(text: str, *, line_start: int = 0, line_end: int = 0) -> dict[str, bool]:
    """Flags from a declaration window. Empty ``virtual Foo() {}`` is the Q4 base."""
    blob = str(text or "")
    virtual = bool(_VIRTUAL_RE.search(blob))
    override = bool(_OVERRIDE_RE.search(blob))
    empty_src = bool(_EMPTY_BODY_RE.search(blob) or _PURE_RE.search(blob))
    start = int(line_start or 0)
    end = int(line_end or 0)
    empty_span = bool(start) and end <= start
    # A one-line `override { return 0; }` is a body, not an empty virtual.
    empty = bool(empty_src or (empty_span and virtual and not override))
    has_body = (not empty) and (
        end > start or override or ("{" in blob and not empty_src)
    )
    return {
        "virtual": virtual,
        "override": override,
        "empty": empty,
        "has_body": has_body,
    }


def _file_stem(path: str) -> str:
    base = Path(str(path or "").replace("\\", "/")).name
    return base.rsplit(".", 1)[0].lower()


def _base_name(path: str) -> str:
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1]


def _same_site(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        _file_stem(str(a.get("file") or "")) == _file_stem(str(b.get("file") or ""))
        and int(a.get("line") or 0) == int(b.get("line") or 0)
    )


def _member(
    raw: dict[str, Any],
    *,
    text: str = "",
    owner_hint: str = "",
) -> dict[str, Any] | None:
    name = str(raw.get("name") or "")
    file = str(raw.get("file") or "").replace("\\", "/")
    line = int(raw.get("line") or raw.get("line_start") or 0)
    if not name or not file or line <= 0:
        return None
    flags = classify_decl(
        text or str(raw.get("text") or ""),
        line_start=line,
        line_end=int(raw.get("line_end") or 0),
    )
    owner = owner_of(name, raw.get("attrs") if isinstance(raw.get("attrs"), dict) else raw)
    if not owner:
        owner = str(owner_hint or "").strip()
    out = {
        "id": str(raw.get("id") or ""),
        "kind": str(raw.get("kind") or ""),
        "name": name,
        "leaf": leaf_name(name),
        "file": file,
        "line": line,
        "line_end": int(raw.get("line_end") or 0),
        "owner": owner,
        **flags,
    }
    return out


def build_family(
    members: Iterable[dict[str, Any]],
    *,
    texts: dict[str, str] | None = None,
    owners_by_file: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """A virtual family is an empty/virtual base plus a different-type override.

    Same-class header declaration + cpp definition is not a family: those share
    an owner or a file stem. The Q4 pair does not
    (``NormalRegbase`` header stub vs ``VarlenRegbase`` override body).
    """
    texts = texts or {}
    owners_by_file = owners_by_file or {}
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in members or []:
        if not isinstance(raw, dict):
            continue
        file = str(raw.get("file") or "").replace("\\", "/")
        key = f"{file}:{int(raw.get('line') or raw.get('line_start') or 0)}"
        item = _member(
            raw,
            text=str(raw.get("text") or texts.get(key) or texts.get(file) or ""),
            owner_hint=str(owners_by_file.get(file) or ""),
        )
        if item is None:
            continue
        site = (_file_stem(item["file"]), int(item["line"]))
        prev = seen.get(site)
        if prev is None:
            seen[site] = item
            continue
        # Prefer the member that already knows its owner / virtual flags.
        if (not prev.get("owner") and item.get("owner")) or (
            not prev.get("virtual") and item.get("virtual")
        ):
            seen[site] = item
    rows = list(seen.values())
    if len(rows) < 2:
        return None
    leaves = {str(r.get("leaf") or "") for r in rows}
    if len(leaves) != 1 or not next(iter(leaves)):
        return None
    bases = [r for r in rows if r.get("virtual") and r.get("empty")]
    if not bases:
        bases = [r for r in rows if r.get("virtual")]
    overrides = [r for r in rows if r.get("override") and r.get("has_body")]
    if not overrides:
        overrides = [
            r
            for r in rows
            if r.get("has_body")
            and not r.get("empty")
            and any(not _same_type(r, base) for base in bases)
        ]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for base in bases:
        for over in overrides:
            if _same_site(base, over) or _same_type(base, over):
                continue
            pairs.append((base, over))
    if not pairs:
        return None
    # One leaf can have several overrides; keep every distinct target.
    base_rows: list[dict[str, Any]] = []
    over_rows: list[dict[str, Any]] = []
    for base, over in pairs:
        if not any(_same_site(base, existing) for existing in base_rows):
            base_rows.append(base)
        if not any(_same_site(over, existing) for existing in over_rows):
            over_rows.append(over)
    return {
        "leaf": next(iter(leaves)),
        "base": [_public_member(r) for r in base_rows],
        "overrides": [_public_member(r) for r in over_rows],
        "members": [_public_member(r) for r in rows if r in base_rows or r in over_rows],
    }


def _same_type(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ao = str(a.get("owner") or "")
    bo = str(b.get("owner") or "")
    if ao and bo:
        return ao == bo
    return _file_stem(str(a.get("file") or "")) == _file_stem(str(b.get("file") or ""))


def _public_member(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "kind": str(row.get("kind") or ""),
        "name": str(row.get("name") or ""),
        "leaf": str(row.get("leaf") or ""),
        "file": str(row.get("file") or ""),
        "line": int(row.get("line") or 0),
        "owner": str(row.get("owner") or ""),
        "virtual": bool(row.get("virtual")),
        "override": bool(row.get("override")),
        "empty": bool(row.get("empty")),
        "has_body": bool(row.get("has_body")),
    }


def family_sites(family: dict[str, Any] | None) -> set[tuple[str, int]]:
    if not isinstance(family, dict):
        return set()
    out: set[tuple[str, int]] = set()
    for row in list(family.get("base") or []) + list(family.get("overrides") or []):
        if not isinstance(row, dict):
            continue
        file = str(row.get("file") or "")
        line = int(row.get("line") or 0)
        if file and line:
            out.add((file.replace("\\", "/"), line))
    return out


def _loc(row: dict[str, Any], *, home: str = "") -> str:
    file = str(row.get("file") or "")
    line = int(row.get("line") or 0)
    base = _base_name(file)
    home_base = _base_name(home)
    where = f"{base}:{line}" if base and base != home_base else (f"{line}" if line else "")
    owner = str(row.get("owner") or "")
    if owner and where:
        return f"{owner} — {where}"
    if owner:
        return owner
    return where or file


def render_virtual_dispatch(
    family: dict[str, Any] | None,
    *,
    home: str = "",
    indent: str = "",
    heading: bool = False,
) -> list[str]:
    """The Q4 sentence, as card lines.

    ``DoSparse:1110 GetSparseUnpadBlockInfo() → varlen override / empty base``.
    """
    if not isinstance(family, dict):
        return []
    bases = [r for r in (family.get("base") or []) if isinstance(r, dict)]
    overrides = [r for r in (family.get("overrides") or []) if isinstance(r, dict)]
    if not bases and not overrides:
        return []
    lines: list[str] = []
    if heading:
        lines.append("**Virtual dispatch**")
    for row in overrides:
        lines.append(f"{indent}- override {_loc(row, home=home)}")
    for row in bases:
        label = "empty virtual" if row.get("empty") else "virtual"
        lines.append(f"{indent}- {label} {_loc(row, home=home)}")
    if heading:
        lines.append("")
    return lines


def annotate_call_line(line: str) -> str:
    """Mark a Calls row as virtual without disturbing its ``when`` clause."""
    text = str(line or "")
    if " virtual" in text:
        return text
    if "  when " in text:
        return text.replace("  when ", "  virtual  when ", 1)
    return text + "  virtual"
