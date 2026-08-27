# -*- coding: utf-8 -*-
"""Restricted C++ expression parser + evaluator over the ExprIR.

Scope is deliberately narrow: the guard expressions that appear in Ascend C
`IsCapable` bodies and tiling branch conditions. Anything outside the grammar
becomes `Unknown(reason)` rather than being silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from ascendc_codemap_mcp.engine.expr_ir import Bin, Call, Const, Expr, Ite, Ref, Select, Un, Unknown, pretty

TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<char>'(?:[^'\\]|\\.)')
  | (?P<num>0[xX][0-9a-fA-F]+|\d+\.\d+|\d+[uUlL]*)
  | (?P<id>[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)
  | (?P<op><<=|>>=|->|\+\+|--|<=|>=|==|!=|&&|\|\||<<|>>|[-+*/%<>!&|^~?:.,()\[\]{}=;])
    """,
    re.VERBOSE,
)

_NAMED_CASTS = ("static_cast", "reinterpret_cast", "const_cast", "dynamic_cast")
_INT_CAST_CALLEES = frozenset(
    {
        "bool",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "unsigned",
        "size_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
    }
)

BIN_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}


@dataclass
class Token:
    kind: str
    text: str


def tokenize(src: str) -> list[Token]:
    out: list[Token] = []
    pos = 0
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        kind = m.lastgroup or "op"
        if kind == "ws":
            continue
        text = m.group()
        if kind == "id":
            text = re.sub(r"\s*::\s*", "::", text)
        out.append(Token(kind, text))
    return _drop_template_arguments(out)


def _drop_template_arguments(toks: list[Token]) -> list[Token]:
    """Erase `<...>` in `name<T>(args)` so it parses as a plain call.

    Without this, `std::get<0>(pair)` is read as two comparisons and the
    callee disappears behind a `<` operator.

    An integer template argument is *not* erased but folded into the callee
    name, because for `std::get<1>` the index is the whole meaning of the call.
    """
    out: list[Token] = []
    i = 0
    while i < len(toks):
        out.append(toks[i])
        if toks[i].kind != "id" or i + 1 >= len(toks) or toks[i + 1].text != "<":
            i += 1
            continue
        depth = 0
        j = i + 1
        while j < len(toks):
            t = toks[j].text
            if t == "<":
                depth += 1
            elif t == ">":
                depth -= 1
                if depth == 0:
                    break
            elif t in (";", "{", "}", "&&", "||"):
                j = len(toks)  # a real comparison, not a template argument list
                break
            j += 1
        if j < len(toks) and j + 1 < len(toks) and toks[j + 1].text == "(":
            inner = toks[i + 2 : j]
            if len(inner) == 1 and inner[0].text.isdigit():
                out[-1] = Token(toks[i].kind, f"{toks[i].text}<{inner[0].text}>")
            i = j + 1
        else:
            i += 1
    return out


class Parser:
    """Recursive-descent / precedence-climbing parser for the guard subset."""

    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Token | None:
        t = self.peek()
        if t is not None:
            self.i += 1
        return t

    def accept(self, text: str) -> bool:
        t = self.peek()
        if t is not None and t.text == text:
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> bool:
        return self.accept(text)

    # -- grammar -----------------------------------------------------------
    def parse(self) -> Expr:
        e = self.parse_ternary()
        return e

    def parse_ternary(self) -> Expr:
        cond = self.parse_binary(0)
        if self.accept("?"):
            then = self.parse_ternary()
            if not self.accept(":"):
                return Unknown("malformed_ternary")
            other = self.parse_ternary()
            return Ite(cond, then, other)
        return cond

    def parse_binary(self, min_prec: int) -> Expr:
        left = self.parse_unary()
        while True:
            t = self.peek()
            if t is None or t.text not in BIN_PRECEDENCE:
                return left
            prec = BIN_PRECEDENCE[t.text]
            if prec < min_prec:
                return left
            self.next()
            right = self.parse_binary(prec + 1)
            left = Bin(t.text, left, right)

    def parse_unary(self) -> Expr:
        t = self.peek()
        if t is None:
            return Unknown("eof")
        if t.text in ("!", "-", "~", "+", "&", "*"):
            self.next()
            return Un(t.text, self.parse_unary())
        if t.kind == "id" and any(t.text.startswith(name) for name in _NAMED_CASTS):
            self.next()
            self._skip_template_args()
            if self.accept("("):
                inner = self.parse_ternary()
                self.accept(")")
                return inner
            return Unknown("malformed_named_cast")
        return self.parse_postfix()

    def _template_args_end(self) -> int | None:
        """Index just past `<...>` when the `<` really opens a template-argument
        list, i.e. it balances and the call parentheses follow.

        `a.b < 1 ? c : d` has a `<` after a member name but is a comparison;
        consuming it as template arguments swallows the rest of the ternary.
        """
        if self.peek() is None or self.peek().text != "<":
            return None
        depth = 0
        j = self.i
        while j < len(self.toks):
            text = self.toks[j].text
            if text == "<":
                depth += 1
            elif text == ">":
                depth -= 1
                if depth == 0:
                    break
            elif text in (";", "{", "}", "&&", "||", "?", ":", ","):
                return None
            j += 1
        if depth != 0 or j + 1 >= len(self.toks):
            return None
        return j + 1 if self.toks[j + 1].text == "(" else None

    def _skip_template_args(self) -> None:
        end = self._template_args_end()
        if end is None:
            return
        self.i = end

    def parse_postfix(self) -> Expr:
        e = self.parse_primary()
        while True:
            t = self.peek()
            if t is None:
                return e
            if t.text in ("->", "."):
                self.next()
                name_tok = self.next()
                if name_tok is None:
                    return Unknown("dangling_member")
                name = name_tok.text
                self._skip_template_args()
                if self.accept("("):
                    args = self.parse_args()
                    e = Call(name, (e,) + tuple(args))
                else:
                    e = Call("field:" + name, (e,))
                continue
            if t.text == "(":
                self.next()
                args = self.parse_args()
                fname = e.symbol if isinstance(e, Ref) else "?"
                if fname in _INT_CAST_CALLEES and len(args) == 1:
                    e = args[0]
                else:
                    e = Call(fname, tuple(args))
                continue
            if t.text == "[":
                self.next()
                idx = self.parse_ternary()
                self.accept("]")
                e = Select(e, idx)
                continue
            return e

    def parse_args(self) -> list[Expr]:
        args: list[Expr] = []
        if self.accept(")"):
            return args
        while True:
            args.append(self.parse_ternary())
            if self.accept(","):
                continue
            self.accept(")")
            return args

    def parse_primary(self) -> Expr:
        t = self.next()
        if t is None:
            return Unknown("eof")
        if t.text == "(":
            e = self.parse_ternary()
            self.accept(")")
            return e
        if t.text == "{":
            # Braced initialiser list: `std::max({a, b, c})`, `{0, 0}`. Without
            # this the `{` derails the rest of the expression.
            items: list[Expr] = []
            if not self.accept("}"):
                while True:
                    items.append(self.parse_ternary())
                    if self.accept(","):
                        continue
                    self.accept("}")
                    break
            return Call("__init_list", tuple(items))
        if t.kind == "str":
            return Const(t.text[1:-1], string_literal=True)
        if t.kind == "char":
            # Deliberately unmarked: a character and a one-character string are
            # written alike here once the quotes are gone, and only the string
            # comparisons are what the distinctness argument is about.
            return Const(t.text[1:-1])
        if t.kind == "num":
            txt = t.text.rstrip("uUlL")
            try:
                return Const(int(txt, 0))
            except ValueError:
                try:
                    return Const(float(txt))
                except ValueError:
                    return Unknown(f"bad_number:{t.text}")
        if t.kind == "id":
            if t.text == "nullptr" or t.text == "NULL":
                return Const(None)
            if t.text == "true":
                return Const(True)
            if t.text == "false":
                return Const(False)
            return Ref(t.text)
        return Unknown(f"unexpected_token:{t.text}")


# Guards longer than this are not tiling predicates; parsing them can hang
# the host closure (mis-extracted nodes / macro soup).
MAX_EXPR_CHARS = 4096


@lru_cache(maxsize=16384)
def parse_expr(src: str) -> Expr:
    """Parse a C++-ish expression. Cached: controllability re-parses the same
    guard / path-condition strings hundreds of times per operator.
    """
    if len(src) > MAX_EXPR_CHARS:
        return Unknown("expr_too_long")
    return Parser(tokenize(src)).parse()


class EvalUnknown(Exception):
    """Raised when evaluation hits a node with no value in the environment."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def evaluate(
    e: Expr,
    env: dict[str, Any],
    *,
    call_hook: Callable[[Call, dict[str, Any]], Any] | None = None,
    symbol_hook: Callable[[str, dict[str, Any]], Any] | None = None,
) -> Any:
    """Evaluate an ExprIR node. Raises EvalUnknown when a symbol is unbound."""
    kw = {"call_hook": call_hook, "symbol_hook": symbol_hook}
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Unknown):
        raise EvalUnknown(e.reason)
    if isinstance(e, Ref):
        if e.symbol in env:
            return env[e.symbol]
        if symbol_hook is not None:
            return symbol_hook(e.symbol, env)
        raise EvalUnknown(f"unbound:{e.symbol}")
    if isinstance(e, Un):
        v = evaluate(e.arg, env, **kw)
        if e.op == "!":
            return not v
        if e.op == "-":
            return -v
        if e.op == "+":
            return +v
        if e.op == "~":
            return ~v
        raise EvalUnknown(f"unary:{e.op}")
    if isinstance(e, Bin):
        if e.op == "&&":
            if not evaluate(e.left, env, **kw):
                return False
            return bool(evaluate(e.right, env, **kw))
        if e.op == "||":
            if evaluate(e.left, env, **kw):
                return True
            return bool(evaluate(e.right, env, **kw))
        left = evaluate(e.left, env, **kw)
        right = evaluate(e.right, env, **kw)
        return _apply_bin(e.op, left, right)
    if isinstance(e, Ite):
        return (
            evaluate(e.then, env, **kw)
            if evaluate(e.cond, env, **kw)
            else evaluate(e.else_, env, **kw)
        )
    if isinstance(e, Select):
        arr = evaluate(e.array, env, **kw)
        idx = evaluate(e.index, env, **kw)
        try:
            return arr[idx]
        except Exception as exc:
            raise EvalUnknown(f"select:{exc}") from exc
    if isinstance(e, Call):
        key = pretty(e)
        if key in env:
            return env[key]
        if call_hook is not None:
            return call_hook(e, env)
        raise EvalUnknown(f"call:{e.func}")
    raise EvalUnknown(f"node:{type(e).__name__}")


def _apply_bin(op: str, a: Any, b: Any) -> Any:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b else 0
    if op == "%":
        return a % b if b else 0
    if op == "&":
        return a & b
    if op == "|":
        return a | b
    if op == "^":
        return a ^ b
    if op == "<<":
        return a << b
    if op == ">>":
        return a >> b
    raise EvalUnknown(f"binop:{op}")


def free_symbols(e: Expr) -> set[str]:
    if isinstance(e, Ref):
        return {e.symbol}
    if isinstance(e, Un):
        return free_symbols(e.arg)
    if isinstance(e, Bin):
        return free_symbols(e.left) | free_symbols(e.right)
    if isinstance(e, Ite):
        return free_symbols(e.cond) | free_symbols(e.then) | free_symbols(e.else_)
    if isinstance(e, Call):
        out: set[str] = set()
        for a in e.args:
            out |= free_symbols(a)
        return out
    if isinstance(e, Select):
        return free_symbols(e.array) | free_symbols(e.index)
    return set()
