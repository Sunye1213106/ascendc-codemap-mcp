# -*- coding: utf-8 -*-
"""Restricted expression IR + Unknown as first-class citizen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Const:
    value: Any
    #: The value was written as a quoted string in the source. Worth keeping
    #: separately from the text, because the quotes are stripped here and
    #: `"TND"` then reads exactly like a reference to something named `TND` --
    #: and in this operator both exist, with unrelated values. Two different
    #: string literals are necessarily different strings, which a pair of names
    #: cannot promise, so only the marked ones may be assumed distinct.
    string_literal: bool = False


@dataclass(frozen=True)
class Ref:
    symbol: str
    version: int = 0
    #: Function the symbol was read in. Expansion inlines definitions across
    #: functions, so a name left unexpanded here may be a local of a function
    #: far from the one being derived; resolving it against the wrong scope
    #: finds no binding and yields UNMAPPED_SYMBOL. Empty means "wherever the
    #: consumer is looking".
    scope: str = ""


@dataclass(frozen=True)
class Un:
    op: str
    arg: "Expr"


@dataclass(frozen=True)
class Bin:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Ite:
    cond: "Expr"
    then: "Expr"
    else_: "Expr"


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple["Expr", ...]


@dataclass(frozen=True)
class Select:
    array: "Expr"
    index: "Expr"


@dataclass(frozen=True)
class Unknown:
    reason: str


Expr = Union[Const, Ref, Un, Bin, Ite, Call, Select, Unknown]


def pretty(e: Expr) -> str:
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, Ref):
        return f"{e.symbol}@{e.version}"
    if isinstance(e, Un):
        return f"({e.op} {pretty(e.arg)})"
    if isinstance(e, Bin):
        return f"({pretty(e.left)} {e.op} {pretty(e.right)})"
    if isinstance(e, Ite):
        return f"(ite {pretty(e.cond)} {pretty(e.then)} {pretty(e.else_)})"
    if isinstance(e, Call):
        return f"{e.func}({', '.join(pretty(a) for a in e.args)})"
    if isinstance(e, Select):
        return f"{pretty(e.array)}[{pretty(e.index)}]"
    if isinstance(e, Unknown):
        return f"Unknown({e.reason})"
    raise TypeError(type(e))
