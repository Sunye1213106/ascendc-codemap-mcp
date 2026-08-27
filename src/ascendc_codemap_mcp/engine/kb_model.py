# -*- coding: utf-8 -*-
"""Extract-time types used by gaps, variable model, and key-field derivation.

This is not a KnowledgeBase graph. Canonical product is the ``.uo`` CodeMap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ids import evidence_id

STATUS_EXTRACTED = "extracted"
STATUS_PARTIAL = "partial"
STATUS_UNRESOLVED = "unresolved"
STATUS_NOT_EXTRACTED = "not_extracted"

VALID_STATUSES = frozenset(
    {STATUS_EXTRACTED, STATUS_PARTIAL, STATUS_UNRESOLVED, STATUS_NOT_EXTRACTED}
)

# Roots a test generator can actually set when constructing a case. Everything
# else may still be "closed" but is not a control knob.
CONTROLLABLE_ROOTS = frozenset(
    {
        "INPUT_SHAPE",
        "INPUT_DTYPE",
        "INPUT_FORMAT",
        "INPUT_VALUE",
        "OPTIONAL_INPUT_PRESENCE",
        "ATTRIBUTE",
        "SESSION_OPTION",
    }
)

# Not knobs, but fixed once the CANN profile and build are chosen, so a case can
# still be constructed against them — they behave as constants at generation
# time rather than as unknowns.
PLATFORM_LOCKED_ROOTS = frozenset(
    {
        "PLATFORM_ARCH",
        "PLATFORM_CORE_COUNT",
        "PLATFORM_MEMORY_SIZE",
        "PLATFORM_L2_SIZE",
        "PLATFORM_AIV_COUNT",
        "COMPILE_INFO",
        "COMPILE_DEFINE",
        "TEMPLATE_LITERAL",
        "CONSTANT",
    }
)

#: How far a derivation got toward something a test case can drive. Orthogonal
#: to `exactness`, which only says whether the *expression* closed: a field can
#: be exact and still be undrivable. `IsTnd` is exactly that — its predicate form is
#: the single comparison `layoutType == 4`, with no free variables at all, but
#: `layoutType` is host state the resolver stopped on instead of the layout
#: attribute behind it, so nothing a generator sets reaches it.
IC_CONTROLLABLE = "controllable"
IC_PLATFORM_LOCKED = "platform_locked"
IC_HOST_STATE = "host_state"
IC_NONE = "none"


def classify_input_closure(roots: Iterable[str]) -> str:
    """Grade a set of input roots by whether a test case can drive them.

    Anything unrecognized counts as host state. Guessing the other way would
    report a dimension as drivable on the strength of a root nobody classified.
    """
    seen = {str(r) for r in roots if str(r)}
    if not seen:
        return IC_NONE
    if seen <= CONTROLLABLE_ROOTS:
        return IC_CONTROLLABLE
    if seen <= (CONTROLLABLE_ROOTS | PLATFORM_LOCKED_ROOTS):
        return IC_PLATFORM_LOCKED
    return IC_HOST_STATE


def input_closure_is_drivable(closure: str) -> bool:
    """A constant field is drivable: nothing needs setting to reach its value."""
    return closure in (IC_CONTROLLABLE, IC_PLATFORM_LOCKED, IC_NONE)


@dataclass
class Evidence:
    id: str
    file: str
    line_start: int
    line_end: int
    snippet: str = ""
    source_hash: str = ""

    @classmethod
    def at(
        cls,
        file: str,
        line: int,
        *,
        snippet: str = "",
        line_end: int | None = None,
        root: str = "",
    ) -> "Evidence":
        end = int(line_end if line_end is not None else line)
        return cls(
            id=evidence_id(file, int(line), end, root),
            file=str(file).replace("\\", "/"),
            line_start=int(line),
            line_end=end,
            snippet=(snippet or "")[:400],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "snippet": self.snippet,
            "source_hash": self.source_hash,
        }


@dataclass
class Domain:
    """Value domain of a Variable.

    `completeness` is the honest part: `open` says the upper bound is a test
    策略 decision, not something the source proves.
    """

    var_id: str
    value_type: str = "int"
    lo: int | None = None
    hi: int | None = None
    values: list[Any] = field(default_factory=list)
    completeness: str = "open"
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.value_type,
            "completeness": self.completeness,
            "source": self.source,
        }
        if self.lo is not None:
            out["lo"] = self.lo
        if self.hi is not None:
            out["hi"] = self.hi
        if self.values:
            out["domain"] = list(self.values)
        return out


@dataclass
class Blocker:
    """One normalization failure, with every node it holds open.

    This is the unit of LLM work. A single unresolved symbol commonly blocks
    dozens of branches, so batching by node would multiply the same question.
    """

    id: str
    text: str
    reason_code: str
    affected_nodes: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hint: str = ""
    #: Variables an answer to this blocker may name. Not a hint: a condition
    #: mentioning anything else is rejected as invented, so a blocker without
    #: this leaves a model guessing at names it cannot see.
    readable_vars: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "text": self.text,
            "reason_code": self.reason_code,
            "affected_node_count": len(self.affected_nodes),
            "affected_nodes": sorted(self.affected_nodes)[:50],
            "evidence_refs": [e.id for e in self.evidence],
            # Inline evidence so investigate / consumers can read a closed pack
            # without a second index lookup.
            "evidence": [e.to_dict() for e in self.evidence[:5]],
            "hint": self.hint,
        }
        if self.readable_vars:
            out["readable_vars"] = list(self.readable_vars)
        return out
