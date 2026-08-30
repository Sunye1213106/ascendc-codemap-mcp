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
    flags = 0
    # Keep IGNORECASE on pure literals so agent case (DT_HIFLOAT8) still hits.
    if text and not _UNESCAPED_META.search(text) and "\\" not in text:
        flags = re.IGNORECASE
    try:
        return re.compile(text, flags)
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
        # `**` matches zero or more directories: `op_host/**/*.cpp` includes
        # `op_host/foo.cpp`. fullmatch avoids accidental substring hits.
        rx = re.escape(pat)
        rx = rx.replace(r"\*\*/", r"(?:.*/)?")
        rx = rx.replace(r"/\*\*", r"(?:/.*)?")
        rx = rx.replace(r"\*\*", r".*")
        rx = rx.replace(r"\*", "[^/]*").replace(r"\?", ".")
        if re.fullmatch(rx, norm):
            return True
        # Basename fallback only for globs with no concrete directory prefix
        # (`**/*.cpp`), so `op_host/**/*.cpp` cannot match `op_kernel/foo.cpp`.
        if pat.startswith("**/"):
            base_pat = pat.rsplit("/", 1)[-1]
            if any(ch in base_pat for ch in "*?"):
                return fnmatch.fnmatch(norm.rsplit("/", 1)[-1], base_pat)
        return False
    if any(ch in pat for ch in "*?["):
        return fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(norm.rsplit("/", 1)[-1], pat)
    return pat in norm or norm.endswith("/" + pat) or norm.endswith(pat)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TPL_RE = re.compile(r"ASCENDC_TPL_|TPL_BOOL_SEL|TPL_ARGS_SEL|TPL_SEL\b")
_LOG_RE = re.compile(r"\bOP_LOG[A-Z]*\b")
_CTRL_RE = re.compile(r"^(?:else\s+)?(?:if|for|while|switch|return|else)\b")
_MEMBER_ASSIGN_RE = re.compile(r"(?:\.|->)\s*[A-Za-z_]\w*\s*=(?!=)")
_SET_CALL_RE = re.compile(r"\bset_[A-Za-z_]\w*\s*\(")
_BARE_ASSIGN_RE = re.compile(r"\b[A-Za-z_]\w*\s*=(?!=)")
_FN_DEF_RE = re.compile(
    r"^(?:template\b.{0,240})?(?:__\w+__\s+|inline\s+|static\s+|virtual\s+|"
    r"constexpr\s+|explicit\s+)*[\w:<>,\s\*&]+?\s+[A-Za-z_]\w*\s*\("
)
_TYPE_DECL_RE = re.compile(
    r"^(?:const(?:expr)?\s+|static\s+|unsigned\s+|signed\s+|volatile\s+)*"
    r"[\w:<>,\s\*&]+?\s+[A-Za-z_]\w*\s*(?:=\s*[^;]+)?;?\s*$"
)
_TYPE_LEAD_RE = re.compile(
    r"^(?:const(?:expr)?\s+|static\s+|unsigned\s+|signed\s+|volatile\s+)*"
    r"(?:bool|int|uint\d*_t|int\d+_t|size_t|float|double|void|auto|char|"
    r"unsigned|long|short|class|struct|enum|using|typedef)\b"
)

ROLE_DEF = 0
ROLE_ASSIGN = 1
ROLE_CTRL = 2
ROLE_OTHER = 3
ROLE_NOISE = 4
ROLE_TPL = 5


def path_layer(path: str, architecture: str = "") -> int:
    """Own host, own kernel, shared common, then everything else."""
    del architecture
    norm = str(path or "").replace("\\", "/")
    blob = f"/{norm.strip('/')}/"
    common = "/common/" in blob or norm.startswith("../common/")
    host = "/op_host/" in blob or norm.startswith("op_host/")
    kernel = "/op_kernel/" in blob or norm.startswith("op_kernel/")
    if common:
        return 2
    if host:
        return 0
    if kernel:
        return 1
    return 3


def line_role(text: str) -> int:
    """Prefer definitions and writes over logs and tiling-key macros."""
    raw = str(text or "")
    stripped = raw.strip()
    if _TPL_RE.search(raw):
        return ROLE_TPL
    if stripped.startswith(("//", "/*", "*")):
        return ROLE_NOISE
    if _LOG_RE.search(raw):
        return ROLE_NOISE
    if _CTRL_RE.match(stripped):
        return ROLE_CTRL
    # Keep set_X(...) / member-assign as ROLE_DEF so writes rank with defs.
    if _MEMBER_ASSIGN_RE.search(raw) or _SET_CALL_RE.search(raw):
        return ROLE_DEF
    if _FN_DEF_RE.match(stripped) and not _CTRL_RE.match(stripped):
        return ROLE_DEF
    if stripped.startswith(("class ", "struct ", "enum ", "using ", "typedef ")):
        return ROLE_DEF
    if _TYPE_DECL_RE.match(stripped) and _TYPE_LEAD_RE.match(stripped):
        return ROLE_DEF
    if _BARE_ASSIGN_RE.search(raw):
        return ROLE_ASSIGN
    return ROLE_OTHER


def match_tightness(pattern: str, text: str) -> int:
    """0 when a pattern identifier appears as a whole word in the line."""
    idents = _IDENT_RE.findall(str(pattern or ""))
    blob = str(text or "")
    if not idents:
        return 1
    for ident in idents:
        if re.search(rf"\b{re.escape(ident)}\b", blob):
            return 0
    return 1


def rank_hit(
    path: str,
    line: int,
    text: str,
    pattern: str = "",
    architecture: str = "",
) -> tuple[int, int, int, str, int]:
    return (
        line_role(text),
        match_tightness(pattern, text),
        path_layer(path, architecture),
        str(path or "").replace("\\", "/"),
        int(line or 0),
    )


def rank_path(path: str, architecture: str = "") -> tuple[int, str, int]:
    norm = str(path or "").replace("\\", "/")
    return (path_layer(norm, architecture), norm, 0)


def line_matches(cre: re.Pattern[str], text: str) -> bool:
    return cre.search(str(text or "")) is not None
