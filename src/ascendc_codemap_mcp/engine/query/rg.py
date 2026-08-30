# -*- coding: utf-8 -*-
"""Regex search over persisted source_line. FTS is an accelerator, not a dialect."""
from __future__ import annotations

import fnmatch
import re
from typing import Any

_UNESCAPED_META = re.compile(r"(?<!\\)[.^*+?{}\[\]|()$]")


class InvalidRegex(ValueError):
    error_code = "INVALID_REGEX"


def compile_search(pattern: str) -> re.Pattern[str]:
    text = str(pattern or "")
    try:
        return re.compile(text)
    except re.error as exc:
        raise InvalidRegex(str(exc)) from exc


def is_pure_literal(pattern: str) -> bool:
    """True when the pattern is a fixed substring (regex-equivalent to itself)."""
    text = str(pattern or "")
    if not text or _UNESCAPED_META.search(text):
        return False
    return "\\" not in text


def path_matches(path: str, glob: str) -> bool:
    if not glob:
        return True
    norm = str(path or "").replace("\\", "/")
    pat = str(glob or "").replace("\\", "/").strip()
    if not pat:
        return True
    if "**" in pat:
        rx = re.escape(pat).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
        return re.search(rx, norm) is not None
    if any(ch in pat for ch in "*?["):
        return fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(norm.rsplit("/", 1)[-1], pat)
    return pat in norm or norm.endswith("/" + pat) or norm.endswith(pat)


def rank_path(path: str, architecture: str = "") -> tuple[int, str, int]:
    norm = str(path or "").replace("\\", "/")
    if "/op_host/" in norm or "/op_kernel/" in norm:
        bucket = 0
    elif architecture and f"/{architecture}/" in norm:
        bucket = 1
    elif "/common/" in norm:
        bucket = 3
    else:
        bucket = 2
    return (bucket, norm, 0)


def line_matches(cre: re.Pattern[str], text: str) -> bool:
    return cre.search(str(text or "")) is not None
