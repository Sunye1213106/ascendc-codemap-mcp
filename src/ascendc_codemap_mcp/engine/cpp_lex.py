# -*- coding: utf-8 -*-
"""Linear C++ lex helpers. Avoid DOTALL ``.*?`` scans on multi-MB kernel TUs."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

_FUNC_BODY_START_RE = re.compile(
    r"\)\s*(?:const\s*)?(?:noexcept(?:\s*\([^;{}()]*\))?\s*)?(?:override\s*)?\{"
)
_FUNC_NAME_TAIL_RE = re.compile(
    r"(?P<name>(?:[A-Za-z_]\w*(?:\s*<[^;{}()]{0,200}>)?\s*::\s*)*[A-Za-z_~]\w*)\s*$"
)
_CONTROL = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "catch",
        "return",
        "sizeof",
        "alignof",
        "decltype",
    }
)
_DECL_KW_RE = re.compile(
    r"\b(?:__aicore__|__global__|__host__|__device__|inline|static|constexpr|"
    r"virtual|explicit|const|volatile|typename|struct|class|enum|friend|"
    r"extern|register|mutable|restrict|__restrict__)\b"
)
_QUALIFIED_TAIL_RE = re.compile(
    r"((?:[A-Za-z_]\w*(?:\s*<[^;{}()]*>)?\s*::\s*)*[A-Za-z_~]\w*)\s*$"
)


@dataclass(frozen=True)
class FuncHit:
    start: int
    open_brace: int
    close_brace: int
    name: str
    params: str
    open_paren: int


def line_index(text: str) -> list[int]:
    """Offsets of every newline, ascending.

    ``str.find`` walks the buffer in C and returns once per line; comparing
    every character from Python returned once per *character*, which on these
    multi-hundred-KB TUs is the same answer for roughly thirty times the work.
    """
    out: list[int] = []
    start = 0
    while True:
        pos = text.find("\n", start)
        if pos < 0:
            return out
        out.append(pos)
        start = pos + 1


def line_at(newlines: list[int], offset: int) -> int:
    return bisect.bisect_right(newlines, max(0, offset)) + 1


def _strip_leading_templates(text: str) -> str:
    s = str(text or "").lstrip()
    while s.startswith("template"):
        lt = s.find("<")
        if lt < 0:
            break
        depth = 0
        end = -1
        for idx, ch in enumerate(s[lt:], lt):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end < 0:
            break
        s = s[end + 1 :].lstrip()
    return s


def _strip_angle_args(text: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def method_identity(qualified: str) -> tuple[str, str, str]:
    """Return ``(short_name, owner_class, signature)``.

    Kernel declarators often include ``template<>`` / ``__aicore__`` / a return
    type. Those belong in ``signature``, not in ``name`` or ``owner``. Owner is
    the class ident without template arguments so ``this``-typed calls bind.
    """
    signature = str(qualified or "").strip()
    text = _DECL_KW_RE.sub(" ", _strip_leading_templates(signature))
    text = re.sub(r"\s+", " ", text).strip()
    match = _QUALIFIED_TAIL_RE.search(text)
    qname = _strip_angle_args(match.group(1) if match else text)
    qname = re.sub(r"\s+", "", qname)
    parts = [p for p in qname.split("::") if p]
    short = parts[-1] if parts else ""
    owner = parts[-2] if len(parts) >= 2 else ""
    return short, owner, signature


def mask_non_code(text: str) -> str:
    out = list(text)
    i = 0
    n = len(text)
    state = "code"
    quote = ""
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "line"
                continue
            if ch == "/" and nxt == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = "block"
                continue
            if ch in {'"', "'"}:
                quote = ch
                out[i] = " "
                i += 1
                state = "string"
                continue
            i += 1
            continue
        if state == "line":
            if ch == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
            else:
                if ch != "\n":
                    out[i] = " "
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out[i] = " "
            if text[i + 1] != "\n":
                out[i + 1] = " "
            i += 2
            continue
        if ch == quote:
            out[i] = " "
            i += 1
            state = "code"
        else:
            if ch != "\n":
                out[i] = " "
            i += 1
    return "".join(out)


def matching_brace(text: str, open_pos: int) -> int:
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != "{":
        return -1
    depth = 0
    quote = ""
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _matching_paren(text: str, close_pos: int) -> int:
    if close_pos < 0:
        return -1
    depth = 0
    quote = ""
    escape = False
    i = close_pos
    while i >= 0:
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            i -= 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            i -= 1
            continue
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return -1


def iter_function_defs(masked: str) -> list[FuncHit]:
    """Linear ``name(params) ... {`` definitions. Skips if/for/while/switch."""
    out: list[FuncHit] = []
    accepted_end = -1
    for body in _FUNC_BODY_START_RE.finditer(masked):
        close_paren = body.start()
        open_paren = _matching_paren(masked, close_paren)
        if open_paren < 0:
            continue
        window_lo = max(0, open_paren - 400)
        prefix = masked[window_lo:open_paren]
        name_match = _FUNC_NAME_TAIL_RE.search(prefix)
        if name_match is None:
            continue
        name = name_match.group("name")
        short = name.split("::")[-1].split("<", 1)[0].strip()
        if short in _CONTROL or not short:
            continue
        start = window_lo + name_match.start("name")
        if start <= accepted_end:
            continue
        params = masked[open_paren + 1 : close_paren]
        if "{" in params or "}" in params:
            continue
        open_brace = body.end() - 1
        close_brace = matching_brace(masked, open_brace)
        if close_brace < 0:
            continue
        accepted_end = close_brace
        out.append(
            FuncHit(
                start=start,
                open_brace=open_brace,
                close_brace=close_brace,
                name=name,
                params=params,
                open_paren=open_paren,
            )
        )
    return out


def containing_function(hits: list[FuncHit], offset: int) -> str:
    """Innermost function name covering ``offset``, or empty."""
    best = ""
    best_span = 10**18
    for hit in hits:
        if hit.open_brace <= offset <= hit.close_brace:
            span = hit.close_brace - hit.open_brace
            if span < best_span:
                best_span = span
                best = hit.name
    return best
