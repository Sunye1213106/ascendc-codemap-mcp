# -*- coding: utf-8 -*-
"""Query-pattern diagnostics so empty hits are not silent absences."""

from __future__ import annotations

import re
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REGEX_MARKERS_RE = re.compile(r"(\\\||\||\.\*)")


def identifier_tokens(pattern: str) -> list[str]:
    """Extract C/C++-like identifiers from a free-text pattern."""
    return _TOKEN_RE.findall(str(pattern or ""))


def looks_like_regex(pattern: str) -> bool:
    text = str(pattern or "")
    return bool(_REGEX_MARKERS_RE.search(text))


def is_multi_token(pattern: str) -> bool:
    text = str(pattern or "").strip()
    if not text or "=" in text:
        return False
    tokens = identifier_tokens(text)
    if len(tokens) <= 1:
        return False
    # Qualified C++ names (result.mode, ns::Foo) are still one identifier query.
    rest = _TOKEN_RE.sub("", text)
    rest = rest.replace("::", "").replace(".", "").strip()
    return bool(rest)


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def looks_like_nl_or_multi_token(pattern: str) -> bool:
    """Whole-sentence NL or multi-token queries that are not Dim=V cover."""
    text = str(pattern or "").strip()
    if not text or "=" in text:
        return False
    if is_multi_token(text):
        return True
    if _CJK_RE.search(text):
        return True
    return bool(re.search(r"\s", text))


def nl_or_multi_token_payload(pattern: str) -> dict[str, Any]:
    tokens = identifier_tokens(pattern)
    return {
        "ok": False,
        "empty_reason": "nl_or_multi_token",
        "pattern": str(pattern or "").strip(),
        "cards": [],
        "count": 0,
        "hint": "Whole-sentence / multi-token queries are rejected. "
        "Use search pattern= (name= is an alias), resolve symbol=, find kind=, or file+line.",
        "pattern_tokens": tokens,
    }


def search_needles(pattern: str) -> list[str]:
    """Needles to OR for locate/search. Structured Dim=V stays a single string."""
    text = str(pattern or "").strip()
    if not text:
        return []
    if "=" in text and "," in text:
        return [text]
    if "=" in text and identifier_tokens(text) and not looks_like_regex(text):
        # Single Dim=V belongs to legal_key / template_match, not search OR.
        return [text]
    tokens = identifier_tokens(text)
    if looks_like_regex(text) or len(tokens) > 1:
        return tokens or [text]
    return [text]


def absent_ident_hint(pattern: str, dim_names: Iterable[str] | None = None) -> str:
    tokens = identifier_tokens(pattern)
    name = tokens[0] if tokens else (str(pattern or "").strip() or "this identifier")
    dims = [str(d).strip() for d in (dim_names or []) if str(d).strip()]
    if name and dims:
        prefix = "".join(ch for ch in name if ch.isalpha())[:2].lower()
        if prefix:
            near = [d for d in dims if d[:2].lower() == prefix]
            rest = [d for d in dims if d not in near]
            dims = near + rest
    if dims:
        shown = ", ".join(dims[:12])
        extra = f" (+{len(dims) - 12})" if len(dims) > 12 else ""
        sample = dims[0]
        return (
            f"{name} is not a compiled dim on this operator. Dims: {shown}{extra}. "
            f"trace dim={sample} lists that dim's built values. "
            f"trace dim=* lists every dim."
        )
    return (
        f"{name} is not a compiled dim on this operator, and this snapshot "
        f"has no compiled dims to list."
    )


def attach_query_hints(
    payload: dict[str, Any],
    pattern: str,
    *,
    count: int,
    indexed: bool | None = None,
    kinds: Iterable[str] | None = None,
    mode: str = "",
) -> dict[str, Any]:
    """Annotate empty / regex / multi-token queries. Does not change hit rows."""
    del kinds
    text = str(pattern or "").strip()
    tokens = identifier_tokens(text)
    regex = looks_like_regex(text)
    multi = is_multi_token(text)
    if regex:
        payload.setdefault("empty_reason", "pattern_looks_like_regex")
        payload["hint"] = (
            "Use search pattern= (name= is an alias) for a regex over snapshot source lines."
        )
        payload["pattern_tokens"] = tokens
    elif count == 0 and str(mode or "") == "around":
        if str(payload.get("snippet") or "").strip():
            payload["ok"] = True
            payload.pop("empty_reason", None)
        else:
            payload["ok"] = False
            payload["empty_reason"] = "no_entity_at_line"
            payload["hint"] = (
                "No CodeMap span covers this line (format-only hunks are expected empty). "
                "This is not proof the file is unindexed. Query Added identifiers instead."
            )
    elif count == 0 and multi:
        payload["empty_reason"] = "no_substring_match"
        payload["hint"] = (
            "Retry one shorter identifier, or dim=/value= for coverage, "
            "or file= line= from a previous card."
        )
        payload["pattern_tokens"] = tokens
    elif count == 0 and str(mode or "") == "index":
        payload["empty_reason"] = payload.get("empty_reason") or "no_substring_match"
    elif count == 0:
        payload["empty_reason"] = payload.get("empty_reason") or "no_substring_match"
        dims = payload.get("dim_names") if isinstance(payload.get("dim_names"), list) else []
        payload["hint"] = absent_ident_hint(text, dims)
    if indexed is False:
        extra = "Template coverage prefers dim= / value=; free-text is unindexed."
        prev = str(payload.get("hint") or "").strip()
        payload["hint"] = f"{prev} {extra}".strip() if prev else extra
    payload.pop("suggested_retries", None)
    return payload
