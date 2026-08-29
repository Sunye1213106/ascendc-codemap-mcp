# -*- coding: utf-8 -*-
"""Canonical predicate AST for C++ guard / branch expressions.

``layout == TND && s == 0``
  → AND(EQ(REF layout, ENUM TND), EQ(REF s, INT 0))

``coreNum / 2``
  → DIV(REF coreNum, INT 2)

No synonym tables. Identifiers stay as spelled. Parse failure yields RAW.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

OPERATORS = (
    "AND",
    "OR",
    "NOT",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
    "BITAND",
    "BITOR",
    "BITXOR",
    "SHL",
    "SHR",
    "CALL",
    "REF",
    "INT",
    "FLOAT",
    "ENUM",
    "STR",
    "BOOL",
    "RAW",
)

_CASTS = frozenset({"static_cast", "const_cast", "reinterpret_cast", "dynamic_cast"})
_BOOLS = {"true": True, "false": False}
_ENUM_PREFIX = (
    "INPUT_FORMAT_",
    "LAYOUT_",
    "DT_",
    "PIPE_",
    "TPL_",
    "FORMAT_",
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM = re.compile(
    r"(?:0[xX][0-9A-Fa-f]+(?:[uUlL]*)|0[bB][01]+|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?[fFlLuU]*)"
)
_TWOCHAR = (
    "&&",
    "||",
    "==",
    "!=",
    "<=",
    ">=",
    "<<",
    ">>",
    "->",
    "::",
)
_SKIP_WS = re.compile(r"\s+")
_CPP_KW = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "return",
        "sizeof",
        "sizeof...",
        "this",
        "nullptr",
        "const",
        "constexpr",
        "auto",
        "void",
        "int",
        "bool",
        "char",
        "float",
        "double",
        "unsigned",
        "signed",
        "long",
        "short",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "size_t",
        "static",
        "struct",
        "class",
        "enum",
        "typename",
        "template",
        "new",
        "delete",
        "using",
        "namespace",
        "public",
        "private",
        "protected",
    }
)


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"op": self.op}
        if self.value != "":
            out["value"] = self.value
        if self.args:
            out["args"] = [a.to_dict() for a in self.args]
        return out

    def walk(self) -> Iterable["Expr"]:
        yield self
        for child in self.args:
            yield from child.walk()

    def operators(self) -> list[str]:
        seen: list[str] = []
        for node in self.walk():
            if node.op in {"REF", "INT", "FLOAT", "ENUM", "STR", "BOOL", "RAW"}:
                continue
            if node.op not in seen:
                seen.append(node.op)
        return seen

    def literals(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for node in self.walk():
            if node.op in {"INT", "FLOAT", "BOOL", "STR"} and node.value != "":
                key = node.value
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def references(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for node in self.walk():
            if node.op == "REF" and node.value:
                leaf = node.value.replace("::", ".").rsplit(".", 1)[-1]
                if leaf not in seen:
                    seen.add(leaf)
                    out.append(leaf)
        return out

    def enum_values(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        def add(token: str) -> None:
            tok = str(token or "").strip()
            if not tok or tok in seen:
                return
            seen.add(tok)
            out.append(tok)

        for node in self.walk():
            if node.op == "ENUM" and node.value:
                add(node.value)
                for part in node.value.split("_"):
                    if len(part) >= 2 and not part.isdigit():
                        add(part)
            elif node.op == "STR" and node.value and _is_enum_ident(node.value):
                add(node.value)
        return out


def from_dict(data: dict[str, Any] | None) -> Expr | None:
    if not isinstance(data, dict) or not data.get("op"):
        return None
    args = tuple(from_dict(a) or Expr("RAW") for a in (data.get("args") or []) if isinstance(a, dict))
    return Expr(str(data.get("op") or "RAW"), args, str(data.get("value") or ""))


def _is_enum_ident(name: str) -> bool:
    if not name or name in _CPP_KW or name in _BOOLS:
        return False
    if any(name.startswith(p) for p in _ENUM_PREFIX):
        return True
    letters = [ch for ch in name if ch.isalpha()]
    if len(letters) >= 2 and all(ch.isupper() for ch in letters) and "_" in name:
        return True
    if name.isupper() and len(name) >= 2 and name not in _CASTS:
        return True
    return False


class _Tok:
    __slots__ = ("kind", "text")

    def __init__(self, kind: str, text: str) -> None:
        self.kind = kind
        self.text = text


def _tokenize(text: str) -> list[_Tok]:
    src = str(text or "")
    i = 0
    n = len(src)
    out: list[_Tok] = []
    while i < n:
        ch = src[i]
        if ch.isspace():
            m = _SKIP_WS.match(src, i)
            i = m.end() if m else i + 1
            continue
        if src.startswith("//", i):
            break
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        two = src[i : i + 2]
        if two in _TWOCHAR:
            kind = {
                "&&": "AND",
                "||": "OR",
                "==": "EQ",
                "!=": "NE",
                "<=": "LE",
                ">=": "GE",
                "<<": "SHL",
                ">>": "SHR",
                "->": "ARROW",
                "::": "SCOPE",
            }[two]
            out.append(_Tok(kind, two))
            i += 2
            continue
        if ch == ".":
            nxt = src[i + 1 : i + 2]
            if nxt.isdigit():
                m = _NUM.match(src, i)
                if m:
                    out.append(_Tok("FLOAT", m.group(0)))
                    i = m.end()
                    continue
            out.append(_Tok("DOT", "."))
            i += 1
            continue
        if ch in "()[],?:.":
            out.append(_Tok(ch, ch))
            i += 1
            continue
        if ch in "!+-*/%&|^~<>":
            kind = {
                "!": "NOT",
                "+": "ADD",
                "-": "SUB",
                "*": "MUL",
                "/": "DIV",
                "%": "MOD",
                "&": "BITAND",
                "|": "BITOR",
                "^": "BITXOR",
                "<": "LT",
                ">": "GT",
                "~": "BITNOT",
            }.get(ch, ch)
            out.append(_Tok(kind, ch))
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                j += 1
            out.append(_Tok("STR", src[i + 1 : j - 1] if j <= n else src[i + 1 :]))
            i = j
            continue
        m = _NUM.match(src, i)
        if m and not (ch.isalpha() or ch == "_"):
            raw = m.group(0)
            if "." in raw or "e" in raw or "E" in raw:
                out.append(_Tok("FLOAT", raw))
            else:
                out.append(_Tok("INT", raw))
            i = m.end()
            continue
        m = _IDENT.match(src, i)
        if m:
            out.append(_Tok("IDENT", m.group(0)))
            i = m.end()
            continue
        i += 1
    return out


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> _Tok | None:
        if self.i >= len(self.toks):
            return None
        return self.toks[self.i]

    def take(self, *kinds: str) -> _Tok | None:
        cur = self.peek()
        if cur is None:
            return None
        if kinds and cur.kind not in kinds:
            return None
        self.i += 1
        return cur

    def parse(self) -> Expr:
        if not self.toks:
            return Expr("RAW", value="")
        expr = self.parse_or()
        return expr

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.peek() and self.peek().kind == "OR":
            self.take("OR")
            left = Expr("OR", (left, self.parse_and()))
        return left

    def parse_and(self) -> Expr:
        left = self.parse_bitor()
        while self.peek() and self.peek().kind == "AND":
            self.take("AND")
            left = Expr("AND", (left, self.parse_bitor()))
        return left

    def parse_bitor(self) -> Expr:
        left = self.parse_bitxor()
        while self.peek() and self.peek().kind == "BITOR":
            self.take("BITOR")
            left = Expr("BITOR", (left, self.parse_bitxor()))
        return left

    def parse_bitxor(self) -> Expr:
        left = self.parse_bitand()
        while self.peek() and self.peek().kind == "BITXOR":
            self.take("BITXOR")
            left = Expr("BITXOR", (left, self.parse_bitand()))
        return left

    def parse_bitand(self) -> Expr:
        left = self.parse_eq()
        while self.peek() and self.peek().kind == "BITAND":
            self.take("BITAND")
            left = Expr("BITAND", (left, self.parse_eq()))
        return left

    def parse_eq(self) -> Expr:
        left = self.parse_rel()
        while self.peek() and self.peek().kind in {"EQ", "NE"}:
            op = self.take().kind
            left = Expr(op, (left, self.parse_rel()))
        return left

    def parse_rel(self) -> Expr:
        left = self.parse_shift()
        while self.peek() and self.peek().kind in {"LT", "LE", "GT", "GE"}:
            op = self.take().kind
            left = Expr(op, (left, self.parse_shift()))
        return left

    def parse_shift(self) -> Expr:
        left = self.parse_add()
        while self.peek() and self.peek().kind in {"SHL", "SHR"}:
            op = self.take().kind
            left = Expr(op, (left, self.parse_add()))
        return left

    def parse_add(self) -> Expr:
        left = self.parse_mul()
        while self.peek() and self.peek().kind in {"ADD", "SUB"}:
            op = self.take().kind
            left = Expr(op, (left, self.parse_mul()))
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_unary()
        while self.peek() and self.peek().kind in {"MUL", "DIV", "MOD"}:
            op = self.take().kind
            left = Expr(op, (left, self.parse_unary()))
        return left

    def parse_unary(self) -> Expr:
        cur = self.peek()
        if cur and cur.kind == "NOT":
            self.take("NOT")
            return Expr("NOT", (self.parse_unary(),))
        if cur and cur.kind == "SUB":
            self.take("SUB")
            inner = self.parse_unary()
            if inner.op == "INT":
                return Expr("INT", value="-" + inner.value if not inner.value.startswith("-") else inner.value[1:])
            return Expr("SUB", (Expr("INT", value="0"), inner))
        return self.parse_postfix()

    def parse_unary_cast(self) -> Expr:
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        expr = self.parse_primary()
        while True:
            cur = self.peek()
            if cur is None:
                return expr
            if cur.kind == "(":
                self.take("(")
                args: list[Expr] = []
                if self.peek() and self.peek().kind != ")":
                    args.append(self.parse_or())
                    while self.take(","):
                        args.append(self.parse_or())
                self.take(")")
                if expr.op == "REF" and expr.value in _CASTS:
                    expr = args[0] if args else expr
                else:
                    expr = Expr("CALL", (expr, *tuple(args)))
                continue
            if cur.kind == "ARROW" or cur.kind == ".":
                self.take()
                ident = self.take("IDENT")
                if ident is None:
                    return expr
                base = expr.value if expr.op == "REF" else ""
                joined = f"{base}.{ident.text}" if base else ident.text
                expr = Expr("REF", value=joined)
                continue
            if cur.kind == "[":
                self.take("[")
                idx = self.parse_or()
                self.take("]")
                expr = Expr("CALL", (Expr("REF", value="[]"), expr, idx))
                continue
            return expr

    def parse_primary(self) -> Expr:
        cur = self.peek()
        if cur is None:
            return Expr("RAW", value="")
        if cur.kind == "(":
            self.take("(")
            expr = self.parse_or()
            self.take(")")
            return expr
        if cur.kind == "INT":
            self.take("INT")
            return Expr("INT", value=_norm_int(cur.text))
        if cur.kind == "FLOAT":
            self.take("FLOAT")
            return Expr("FLOAT", value=cur.text.rstrip("fFlLuU"))
        if cur.kind == "STR":
            self.take("STR")
            return Expr("STR", value=cur.text)
        if cur.kind == "IDENT":
            name = cur.text
            self.take("IDENT")
            while self.peek() and self.peek().kind == "SCOPE":
                self.take("SCOPE")
                nxt = self.take("IDENT")
                if nxt is None:
                    break
                name = f"{name}::{nxt.text}"
            leaf = name.replace("::", ".").rsplit(".", 1)[-1]
            if leaf in _BOOLS:
                return Expr("BOOL", value="true" if _BOOLS[leaf] else "false")
            if leaf in _CASTS:
                if self.peek() and self.peek().kind == "LT":
                    self._skip_angles()
                return Expr("REF", value=leaf)
            if _is_enum_ident(leaf):
                return Expr("ENUM", value=leaf)
            return Expr("REF", value=leaf)
        self.take()
        return Expr("RAW", value=cur.text)

    def _skip_angles(self) -> None:
        if not self.take("LT"):
            return
        depth = 1
        while self.peek() and depth:
            cur = self.take()
            if cur is None:
                return
            if cur.kind == "LT":
                depth += 1
            elif cur.kind == "GT":
                depth -= 1


def _norm_int(raw: str) -> str:
    text = str(raw or "").rstrip("uUlL")
    try:
        if text.lower().startswith("0x"):
            return str(int(text, 16))
        if text.lower().startswith("0b"):
            return str(int(text, 2))
        return str(int(text, 10))
    except ValueError:
        return text


def parse_predicate(text: str) -> Expr:
    """Parse a guard string. Always returns an Expr (RAW on empty / failure)."""
    raw = str(text or "").strip()
    if not raw:
        return Expr("RAW", value="")
    if raw.startswith("for (") or raw.startswith("for("):
        inner = raw[raw.find("(") + 1 : raw.rfind(")")] if ")" in raw else raw
        return parse_predicate(inner)
    try:
        tokens = _tokenize(raw)
        parser = _Parser(tokens)
        expr = parser.parse()
        if expr.op == "RAW" and not expr.value:
            return Expr("RAW", value=raw[:400])
        return expr
    except Exception:  # noqa: BLE001
        return Expr("RAW", value=raw[:400])


def annotate_attrs(text: str) -> dict[str, Any]:
    """Attrs to stamp onto BRANCH / PREDICATE entities."""
    expr = parse_predicate(text)
    payload = {
        "expr_ast": expr.to_dict(),
        "operators": expr.operators(),
        "literals": expr.literals(),
        "references": expr.references(),
        "enum_values": expr.enum_values(),
    }
    return payload


def ast_matches_literal(attrs: dict[str, Any], literal: str) -> bool:
    want = _norm_int(str(literal).strip()) if str(literal).strip().lstrip("-").isdigit() else str(literal).strip()
    got = [str(x) for x in (attrs.get("literals") or [])]
    return want in got or str(literal).strip() in got


def ast_matches_operator(attrs: dict[str, Any], operator: str) -> bool:
    want = str(operator or "").strip().upper()
    return want in {str(x).upper() for x in (attrs.get("operators") or [])}


def ast_matches_value(attrs: dict[str, Any], value: str) -> bool:
    want = str(value or "").strip()
    if not want:
        return False
    pool = [str(x) for x in (attrs.get("enum_values") or [])] + [
        str(x) for x in (attrs.get("literals") or [])
    ]
    if any(e == want or e.endswith("_" + want) or want in e.split("_") for e in pool):
        return True
    low = want.lower()
    return any(e.lower() == low for e in pool)


def ast_matches_symbol(attrs: dict[str, Any], symbol: str) -> bool:
    want = str(symbol or "").strip()
    if not want:
        return False
    leaf = want.replace("::", ".").rsplit(".", 1)[-1]
    refs = [str(x) for x in (attrs.get("references") or [])]
    return leaf in refs or want in refs
