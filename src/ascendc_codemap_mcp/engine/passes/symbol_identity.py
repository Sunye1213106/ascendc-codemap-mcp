# -*- coding: utf-8 -*-
"""Canonical source-symbol identity helpers used by Host structural passes.

The CodeMap must not treat ``this.foo.bar`` and ``foo.bar`` as different
members, nor collapse unrelated ``Other.bar`` symbols merely because their
short name is the same.  These helpers intentionally perform only lexical
normalization; type resolution remains a compiler/frontend responsibility.
"""
from __future__ import annotations

import re
from functools import lru_cache

_MEMBER_SEP_RE = re.compile(r"\s*(?:->|\.)\s*")
_PAREN_THIS_RE = re.compile(r"^\(\s*\*\s*this\s*\)\.")

#: The Host passes ask for the identity of the same few thousand tokens over and
#: over -- one operator's analyze made 2.45M calls, the single largest cost in
#: the stage. The answer depends on nothing but the token, so it is cached.
#: Sized to hold every distinct symbol in an operator rather than to evict.
_CACHE = 1 << 17


@lru_cache(maxsize=_CACHE)
def _normalize(text: str) -> str:
    text = _MEMBER_SEP_RE.sub(".", text)
    while text.startswith("this."):
        text = text[5:]
    # Some frontend spellings retain an explicit parenthesized this receiver.
    text = _PAREN_THIS_RE.sub("", text)
    if "." not in text:
        return text[:-1] if _strips(text) else text
    parts: list[str] = []
    for part in text.split("."):
        # Class members are often spelled ``foo_``; strip one trailing
        # underscore so ``tilingKeyInfo_.x`` and ``tilingKeyInfo.x`` match.
        if _strips(part):
            part = part[:-1]
        parts.append(part)
    return ".".join(parts)


def _strips(part: str) -> bool:
    return len(part) > 1 and part.endswith("_") and not part.endswith("__")


def normalize_symbol(value: str) -> str:
    """Return a stable lexical identity for a C/C++ variable/member token."""
    # Coercion stays outside the cache: callers pass None and non-str values,
    # which an lru_cache key would reject.
    return _normalize(str(value or "").strip())


@lru_cache(maxsize=_CACHE)
def _short(text: str) -> str:
    return text.split(".")[-1].split("::")[-1]


def short_symbol(value: str) -> str:
    return _short(normalize_symbol(value))


def is_member_symbol(value: str) -> bool:
    return "." in normalize_symbol(value)


def canonical_candidates(values: list[str]) -> set[str]:
    out = {normalize_symbol(value) for value in values}
    out.discard("")
    return out
