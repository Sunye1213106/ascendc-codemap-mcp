# -*- coding: utf-8 -*-
"""Public query contract: search + resolve. Single source of truth."""
from __future__ import annotations

import re
from typing import Iterable

PUBLIC_OPERATIONS: tuple[str, ...] = ("search", "resolve")

# search: `pattern` is the canonical slot (regex over indexed source lines).
# `name` is a silent alias so existing agent calls keep working.
SEARCH_FILTERS: tuple[str, ...] = ("pattern", "file", "kind")
SEARCH_PATTERN_ALIASES: tuple[str, ...] = ("name",)

INSTRUCTIONS = (
    "Unknown → search\n"
    "Known or file:line → resolve\n"
    "Query reads snapshot only\n"
)

_OP_TOKEN_RE = re.compile(
    r"\b(search|resolve|find|trace|contract|impact|entry)\b",
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
    "`codemap_query` 的 `operation`：`search`（源码行）/ `resolve`（缺省，一次闭合的语义读）。"
)
