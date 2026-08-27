# -*- coding: utf-8 -*-
"""Query-pattern diagnostics so empty hits are not silent absences."""

from __future__ import annotations

import re
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REGEX_MARKERS_RE = re.compile(r"(\\\||\||\.\*)")
_PIPE_EMPTY_RE = re.compile(
    r"(PRE_CORE_POST|三相|\bPIPE\b|\bTPipe\b|Pre/Main/Post|\bPre\b|\bPost\b)",
    re.I,
)


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
_FOUR_FORMS_HINT = (
    "Use one of four forms: (1) no-arg index, (2) one identifier, "
    "(3) Dim=<dimName> for one coverage list or Name=Value combo filter, "
    "(4) --file PATH --line N copied from a previous card."
)


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
        "hint": "Whole-sentence / multi-token queries are rejected. " + _FOUR_FORMS_HINT,
        "suggested_retries": tokens[:4],
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
    text = str(pattern or "").strip()
    tokens = identifier_tokens(text)
    regex = looks_like_regex(text)
    multi = is_multi_token(text)
    kinds_u = {str(k).upper() for k in (kinds or ())}
    pipe_empty = int(count or 0) == 0 and (
        "PIPE" in kinds_u
        or str(mode or "") == "kernel_launch"
        or bool(_PIPE_EMPTY_RE.search(text))
    )
    if pipe_empty:
        payload["empty_reason"] = payload.get("empty_reason") or "no_substring_match"
        payload["hint"] = (
            "Omit the identifier and call acp uo-query --project <operator-abs> "
            "for the operator index (launch phases). PRE_CORE_POST is not a graph token."
        )
        payload["suggested_retries"] = [
            "TPipe",
            "InitBuffer",
            "PopStackBuffer",
            "InitShareBufStart",
        ]
    elif regex:
        payload.setdefault("empty_reason", "pattern_looks_like_regex")
        payload["hint"] = (
            "Graph search is not regex; query one identifier. "
            "For template coverage use Dim=<dimName> or Name=Value combo filters."
        )
        payload["suggested_retries"] = tokens[:4]
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
            "Retry one shorter identifier, or Dim=<dimName> / Name=Value for coverage, "
            "or --file --line from a previous card."
        )
        payload["suggested_retries"] = tokens[:4]
        payload["pattern_tokens"] = tokens
    elif count == 0:
        payload["empty_reason"] = payload.get("empty_reason") or "no_substring_match"
        payload.setdefault(
            "hint",
            "Retry a shorter identifier, or Dim=<dimName> / Name=Value for coverage, "
            "or --file --line from a previous card. "
            "Empty is not proof the symbol is absent.",
        )
        if tokens:
            payload["suggested_retries"] = tokens[:4]
    if indexed is False:
        extra = "Template coverage prefers Dim=<dimName> or Name=Value (Dim=V); free-text is unindexed."
        prev = str(payload.get("hint") or "").strip()
        payload["hint"] = f"{prev} {extra}".strip() if prev else extra
    return payload
