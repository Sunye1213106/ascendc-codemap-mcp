# -*- coding: utf-8 -*-
"""Content-stable identifiers for every KB entity.

Node ids must survive edits that do not change semantics. `file:line:col` does
not: inserting a line above a branch renames it, which breaks incremental
update and any historical comparison CE wants to do. So ids hash the *meaning*
of an entity (file, enclosing function, normalized guard, ordinal) and line
numbers live in evidence records, where drift is expected and harmless.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

HASH_LEN = 12

# kind -> id prefix. Mirrors the prefixes TG already validates against, so a
# node id minted here is directly usable downstream without translation.
KIND_PREFIX: dict[str, str] = {
    "Input": "IN",
    "OptionalInput": "OPT",
    "Output": "OUT",
    "Attribute": "ATTR",
    "Variable": "VAR",
    "TilingKeyDim": "KEY",
    "TilingDataField": "TDF",
    "HostBranch": "HBR",
    "KernelBranch": "KBR",
    "TemplateBinding": "KTPL",
    "KernelPath": "KPATH",
    "Predicate": "CON",
    "Family": "FAM",
    "ApiContract": "API",
    "Evidence": "EV",
    "Function": "FN",
    "Method": "MTH",
    "Operation": "OP",
    "Buffer": "BUF",
    "Register": "REG",
    "Pipe": "PIPE",
    "Event": "EVENT",
    "Queue": "QUEUE",
    "Type": "TYPE",
    "Root": "ROOT",
}

PREFIX_KIND = {v: k for k, v in KIND_PREFIX.items()}

_ID_RE = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]*)_(?P<body>[A-Za-z0-9_]+)$")
_NON_ID_CHARS = re.compile(r"[^A-Za-z0-9]+")
_WS = re.compile(r"\s+")


def hash12(*parts: Any) -> str:
    """Stable short digest over the parts, joined by a separator that cannot
    appear inside a normalized part.

    Upper case because downstream id validation (TG ``STABLE_ID_RE``) accepts
    ``[A-Z0-9_]`` only.
    """
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LEN].upper()


def content_hash(payload: Any) -> str:
    """Digest of an arbitrary JSON-serializable structure, key order independent."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(text: str, *, upper: bool = True) -> str:
    """`query_padding.size` -> `QUERY_PADDING_SIZE`, safe inside an id."""
    out = _NON_ID_CHARS.sub("_", str(text)).strip("_")
    return out.upper() if upper else out


def rel_posix(path: str, root: str = "") -> str:
    """Normalize a source path so ids do not depend on where the repo is checked out."""
    p = str(path).replace("\\", "/")
    if root:
        r = str(root).replace("\\", "/").rstrip("/")
        low_p, low_r = p.lower(), r.lower()
        if low_p.startswith(low_r + "/"):
            p = p[len(r) + 1 :]
    return p


def normalize_guard(text: str) -> str:
    """Canonical form of a guard, used as id material.

    Only whitespace is collapsed here. Callers that have a normalized predicate
    should pass its canonical finite-predicate expression instead, which additionally makes
    the id immune to formatting and spelling of the same condition.
    """
    return _WS.sub(" ", str(text or "").strip())


def make_id(kind: str, *parts: Any) -> str:
    """`make_id("HostBranch", file, func, guard, ordinal)` -> `HBR_a1b2c3d4e5f6`."""
    prefix = KIND_PREFIX.get(kind)
    if prefix is None:
        raise ValueError(f"unknown node kind: {kind}")
    return f"{prefix}_{hash12(kind, *parts)}"


def named_id(kind: str, name: str) -> str:
    """Id for entities whose name is already stable and human-meaningful.

    Inputs, attributes and TilingKey dims are named in the operator definition,
    so hashing them would only make the KB harder to read.
    """
    prefix = KIND_PREFIX.get(kind)
    if prefix is None:
        raise ValueError(f"unknown node kind: {kind}")
    return f"{prefix}_{slug(name)}"


def branch_id(
    *,
    side: str,
    file: str,
    function: str,
    guard: str,
    ordinal: int,
    root: str = "",
) -> str:
    """Content-stable id for a control node.

    `ordinal` disambiguates repeated identical guards inside one function (a
    loop body testing the same flag twice). It is scoped to the function, not
    to the file, so edits elsewhere in the file do not renumber it.
    """
    kind = "KernelBranch" if side == "kernel" else "HostBranch"
    return make_id(
        kind,
        rel_posix(file, root),
        function,
        normalize_guard(guard),
        ordinal,
    )


def predicate_id(owner_id: str, polarity: bool, canonical: str) -> str:
    return make_id("Predicate", owner_id, "1" if polarity else "0", canonical)


def evidence_id(file: str, line_start: int, line_end: int, root: str = "") -> str:
    return make_id("Evidence", rel_posix(file, root), line_start, line_end)


def operation_site_id(
    *,
    file: str,
    line: int,
    column: int,
    callee: str,
    ordinal: int = 0,
    root: str = "",
) -> str:
    """Source-location id for one execution operation site.

    Template instantiations of the same ``file:line:callee`` share one node.
    ``column`` joins the id only when it is a real source column (``>0``), so
    two different calls on one line stay distinct. ``ordinal`` is ignored —
    kept in the signature so older call sites still type-check.
    """
    del ordinal
    parts: list[Any] = [rel_posix(file, root), int(line), str(callee or "")]
    col = int(column or 0)
    if col > 0:
        parts.append(col)
    return make_id("Operation", *parts)


def buffer_site_id(
    *,
    file: str,
    line: int,
    scope: str,
    name: str,
    root: str = "",
) -> str:
    return make_id("Buffer", rel_posix(file, root), int(line), str(scope or ""), str(name or ""))


def register_site_id(
    *,
    file: str,
    line: int,
    scope: str,
    name: str,
    root: str = "",
) -> str:
    return make_id("Register", rel_posix(file, root), int(line), str(scope or ""), str(name or ""))


def buffer_view_id(
    *,
    buffer_id: str,
    name: str,
    file: str = "",
    line: int = 0,
    root: str = "",
) -> str:
    return make_id("BufferView", buffer_id, str(name or ""), rel_posix(file, root), int(line))


def sync_event_id(
    *,
    file: str,
    line: int,
    column: int,
    kind: str,
    ordinal: int = 0,
    root: str = "",
) -> str:
    return make_id(
        "SyncEvent",
        rel_posix(file, root),
        int(line),
        int(column),
        str(kind or ""),
        int(ordinal),
    )


def exec_region_id(
    *,
    kind: str,
    file: str,
    line: int,
    function: str,
    ordinal: int = 0,
    root: str = "",
) -> str:
    return make_id(
        "ExecRegion",
        str(kind or ""),
        rel_posix(file, root),
        int(line),
        str(function or ""),
        int(ordinal),
    )


def edge_id(kind: str, src: str, dst: str) -> str:
    return f"REL_{hash12(kind, src, dst)}"


def parse_kind(node_id: str) -> str | None:
    """Recover the entity kind from an id prefix, for schema checks."""
    m = _ID_RE.match(str(node_id or ""))
    if not m:
        return None
    return PREFIX_KIND.get(m.group("prefix"))


def is_valid_id(node_id: str) -> bool:
    return parse_kind(node_id) is not None
