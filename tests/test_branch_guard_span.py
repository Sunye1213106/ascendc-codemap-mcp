# -*- coding: utf-8 -*-
"""A guard has to cover the code it guards, and only that code.

Branches were recorded at their condition line with no body extent, so no site
was ever inside one. Every `when` on a read, a call or a sync edge silently
came back empty, and the cross-layer question those answer -- "the kernel reads
this field, but on which path" -- could not be asked at all.

Taking the condition line as the start of the range is the other half of the
contract: a negated guard is written at the `if` but only holds inside the
`else`, so a span reaching back to the condition would put every site in the
`then` block under the negation of its real guard, which is worse than silence.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)
_REL_UO = Path(".ascendc-codemap/arch35/FlashAttentionScoreGrad.arch35.uo")
KERNEL_DETER = "op_kernel/arch35/flash_attention_score_grad_kernel_deter.h"


def _operator() -> Path | None:
    for root in _CANDIDATES:
        if (root / _REL_UO).is_file():
            return root
    return None


FAG = _operator()

pytestmark = pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")


def _text(payload: dict) -> str:
    return str((payload.get("data") or {}).get("text") or "")


def _branch_rows() -> list[tuple[str, int, int, int]]:
    conn = sqlite3.connect(str(FAG / _REL_UO))
    try:
        rows = conn.execute(
            """
            SELECT name, line_start, line_end, data FROM entity
            WHERE kind = 'BRANCH' AND IFNULL(line_start, 0) > 0
            """
        ).fetchall()
    finally:
        conn.close()
    out: list[tuple[str, int, int, int]] = []
    for name, start, end, raw in rows:
        body = int((json.loads(raw or "{}") or {}).get("guard_body_start") or 0)
        out.append((str(name or ""), int(start or 0), int(end or 0), body))
    return out


def test_guards_carry_the_extent_of_the_branch_they_open() -> None:
    rows = _branch_rows()
    assert rows, "no BRANCH entities"
    with_extent = [r for r in rows if r[3] > 0 and r[2] > r[3]]
    # Clang cannot give a body for the negation implied by an early return, so
    # this is a floor on coverage rather than a demand that every branch has one.
    assert len(with_extent) > len(rows) // 2, (
        f"only {len(with_extent)}/{len(rows)} guards know their own extent"
    )


def test_a_negated_guard_starts_at_its_else_not_at_the_condition() -> None:
    # `if constexpr (BaseClass::IS_TND) { … } else { … }` at 437, else body 453.
    rows = [r for r in _branch_rows() if r[0] == "!(BaseClass::IS_TND)" and r[1] == 437]
    assert rows, "expected the negated IS_TND guard at 437"
    _name, start, end, body = rows[0]
    assert start == 437, "the guard is still cited where it is written"
    assert body > start, "the else body must not start at the condition line"
    assert body <= end


def test_a_kernel_read_names_the_compile_time_path_it_sits_on() -> None:
    status(project=str(FAG), architecture="arch35")
    text = _text(query(project=str(FAG), architecture="arch35", symbol="deterMaxRound"))
    consumers = text.split("Kernel consumers", 1)
    assert len(consumers) == 2, "no kernel consumers section"
    block = consumers[1].split("\n\n", 1)[0]
    assert "CalDeterMaxLoopNum" in block
    assert "DETER_BAND" in block, f"consumer path not named:\n{block}"


def test_a_site_in_the_then_block_is_not_reported_under_the_negation() -> None:
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    engine = UoSqlQuery(str(FAG / _REL_UO))
    # 445 sits inside `if constexpr (BaseClass::IS_TND)`; its else runs 453-467.
    assert "IS_TND" not in engine.site_guard(KERNEL_DETER, 445)
    assert "!BaseClass::IS_TND" in engine.site_guard(KERNEL_DETER, 463)


def test_a_condition_is_not_a_guard_on_itself() -> None:
    """The read inside `if (x > 0)` decides the branch, it does not sit in it.

    629 reads deterMaxRound in its own condition. Naming that condition as the
    guard says the read only happens when it already succeeded, and it buried
    the two compile-time flags that actually select the path.
    """
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    guard = UoSqlQuery(str(FAG / _REL_UO)).site_guard(KERNEL_DETER, 629)
    assert "deterMaxRound" not in guard, f"site guarded by its own condition: {guard}"
    assert "!BaseClass::IS_N_EQUAL" in guard
    assert "DETER_DENSE" in guard


def test_dropping_parens_after_not_never_flips_a_comparison() -> None:
    """`!(a > b)` must not be shortened to `!a > b`, which parses as `(!a) > b`."""
    from ascendc_codemap_mcp.engine.query.sql import _simplify_negation

    for text in ("!(a>0)", "!(a > 0)", "!(a-b)", "!(a==b)", "!(a && b)"):
        assert _simplify_negation(text) == text, text
    assert _simplify_negation("!(!x)") == "x"
    assert _simplify_negation("!(BaseClass::IS_TND)") == "!BaseClass::IS_TND"
    assert _simplify_negation("!(a.b->c)") == "!a.b->c"
