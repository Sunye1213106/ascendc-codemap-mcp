# -*- coding: utf-8 -*-
"""Public query contract: search + trace + source. Single source of truth.

The three operations are disjoint on purpose. One tool that accepted a symbol
and a file:line together had to guess which the caller meant, and it guessed
location: a session that asked for `SetSplitAxis` forty times got the symbol it
named back as the subject of the card fifteen times. Separating them removes the
guess instead of documenting it.
"""
from __future__ import annotations

import re
from typing import Iterable

PUBLIC_OPERATIONS: tuple[str, ...] = ("search", "trace", "source")

#: Accepted on the wire but not advertised: one tool that took an ``operation``
#: string. Sessions that already handshook it keep working.
COMPAT_OPERATIONS: tuple[str, ...] = ("resolve",)

# search: `pattern` is the canonical slot (regex over indexed source lines).
# `name` is a silent alias so existing agent calls keep working.
SEARCH_FILTERS: tuple[str, ...] = ("pattern", "file", "kind")
SEARCH_PATTERN_ALIASES: tuple[str, ...] = ("name",)

#: The only relation vocabulary a caller has to hold. The graph carries
#: thirty-two kinds; asking an agent to pick among them by name is asking it to
#: learn the schema, and a wrong pick reads as an empty answer rather than as a
#: mistake. Four families cover what questions actually divide along, and
#: omitting the filter walks all of them, so the shortest call is also the
#: complete one. A raw kind is still accepted for a caller that knows the graph.
RELATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "call": ("CALLS", "CALLS_UNDER_GUARD", "LAUNCHES", "EXPANDS_TO", "WRAPS"),
    "data": (
        "READS",
        "WRITES",
        "FLOWS_TO",
        "DERIVES",
        "MATERIALIZES_AS",
        "BACKED_BY",
        "ALLOCATES",
        "RETURNS",
        "BINDS",
        "ALIASES",
    ),
    "control": (
        "GUARDED_BY",
        "CONTROLS",
        "ACTIVE_UNDER",
        "SELECTS",
        "PRECEDES",
        "SIGNALS",
        "AWAITS",
    ),
    "compile": (
        "INSTANTIATES",
        "SPECIALIZES",
        "INSTANCE_OF",
        "AVAILABLE_ON",
        "DECLARES",
        "DEFINES",
    ),
}

RELATION_FAMILY_NAMES: tuple[str, ...] = tuple(RELATION_FAMILIES)


def expand_relation(value: str) -> tuple[frozenset[str], frozenset[str]]:
    """Split a ``relation=`` value into (raw kinds, family names).

    Accepts ``data``, ``data,control``, or a bare ``WRITES``. An unrecognised
    token returns empty on both sides so the caller can reject it by name
    rather than silently walking nothing.
    """
    tokens = [t.strip() for t in str(value or "").replace("|", ",").split(",")]
    kinds: set[str] = set()
    families: set[str] = set()
    for token in tokens:
        if not token:
            continue
        low = token.lower()
        if low in RELATION_FAMILIES:
            families.add(low)
            kinds.update(RELATION_FAMILIES[low])
        elif low == "all":
            families.update(RELATION_FAMILIES)
            for group in RELATION_FAMILIES.values():
                kinds.update(group)
        else:
            kinds.add(token.upper())
    return frozenset(kinds), frozenset(families)


INSTRUCTIONS = (
    "Unknown name → search\n"
    "Known symbol → trace\n"
    "file:line → source\n"
    "Compiled key space beats reading dispatch code\n"
    "Query reads snapshot only\n"
)

_OP_TOKEN_RE = re.compile(
    r"\b(search|resolve|find|trace|contract|impact|entry|source)\b",
    re.IGNORECASE,
)


class PublicQueryContract:
    """Closed public surface. Internal typed ops may still exist; they must not leak."""

    operations: tuple[str, ...] = PUBLIC_OPERATIONS
    instructions: str = INSTRUCTIONS

    @classmethod
    def is_public(cls, operation: str) -> bool:
        return str(operation or "").strip().lower() in cls.operations

    @classmethod
    def operations_in(cls, text: str) -> set[str]:
        """Operation names mentioned in a public-facing document."""
        return {m.group(1).lower() for m in _OP_TOKEN_RE.finditer(str(text or ""))}

    @classmethod
    def public_operations_in(cls, text: str) -> set[str]:
        return cls.operations_in(text) & set(cls.operations)

    @classmethod
    def assert_public_only(cls, items: Iterable[str]) -> None:
        extra = {str(x).strip().lower() for x in items if str(x).strip()} - set(cls.operations)
        if extra:
            raise AssertionError(f"non-public operations leaked: {sorted(extra)}")


README_OPERATIONS_SENTENCE = (
    "三个工具互斥：`codemap_search`（正则扫源码行，找名字）/ "
    "`codemap_trace`（已知符号的语义闭合读；给第二个符号则求两点间路径；"
    "给 dim/value 则查编译期合法键空间）/ `codemap_source`（按 file+line 读快照源码）。"
)
