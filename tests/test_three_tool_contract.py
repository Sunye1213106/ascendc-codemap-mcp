# -*- coding: utf-8 -*-
"""One regression per defect the three-tool split was built to remove.

Each of these reproduces a specific thing an OpenCode session got wrong against
a real `.uo`: a card titled after something other than the symbol asked for, a
body printed above the claim that it had no definition, a compiled key space
answered by reading dispatch macros, and a section count that could not be told
apart from a section limit.
"""
from __future__ import annotations

import pytest

from ascendc_codemap_mcp.engine.query.contract import RELATION_FAMILIES, expand_relation
from ascendc_codemap_mcp.engine.query.typed import InvalidQuery, validate_plan
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import fag_operator_root

FAG = fag_operator_root()


def _text(**kwargs) -> str:
    if FAG is None:
        pytest.skip("FAG arch35 .uo missing")
    status(project=str(FAG), architecture="arch35")
    payload = query(project=str(FAG), architecture="arch35", **kwargs)
    return str((payload.get("data") or {}).get("text") or "")


# ---------------------------------------------------------------- contract


def test_a_location_never_outranks_the_symbol_that_was_asked_for() -> None:
    """A symbol plus a location used to be answered about the location.

    Forty calls in one session passed `symbol`, `file`, `line` and `line_end`
    together, and twenty-four came back titled after something else on that
    line — with `line_end` present the symbol was not consulted at all. The
    location is now discarded, named as discarded, and left off the plan.
    """
    plan = validate_plan(
        operation="trace", symbol="SetSplitAxis", file="a.cpp", line=1657, line_end=1719
    )
    assert plan.symbol == "SetSplitAxis"
    assert (plan.file, plan.line, plan.line_end) == ("", 0, 0)
    assert {"file", "line", "line_end"} <= set(plan.dropped), plan.dropped


def test_a_symbol_never_outranks_the_range_that_was_asked_for() -> None:
    plan = validate_plan(operation="source", file="a.cpp", line=10, symbol="DoSparse")
    assert (plan.file, plan.line) == ("a.cpp", 10)
    assert not plan.symbol
    assert "symbol" in plan.dropped


def test_neither_tool_accepts_the_other_ones_seed() -> None:
    """The disjointness has to hold at the schema, not only at runtime."""
    from ascendc_codemap_mcp.engine.query.typed import legal_filters_for

    trace = set(legal_filters_for("trace", ""))
    source = set(legal_filters_for("source", ""))
    assert {"file", "line", "line_end"} & trace == set()
    assert {"symbol", "from_symbol", "to_symbol", "pattern"} & source == set()


def test_trace_needs_only_a_symbol() -> None:
    """The complete call has to also be the shortest one.

    Every optional parameter is one the caller can get wrong, and a narrowing
    parameter got wrong reads as an empty answer rather than as a mistake.
    """
    plan = validate_plan(operation="trace", symbol="isBn2MultiBlk")
    assert plan.symbol == "isBn2MultiBlk"
    assert not plan.relation_families, "no filter means every family"
    assert not plan.file and not plan.line


def test_source_needs_only_a_file_and_line() -> None:
    plan = validate_plan(operation="source", file="op_host/a.cpp", line=1099)
    assert (plan.file, plan.line) == ("op_host/a.cpp", 1099)
    assert not plan.symbol


# ---------------------------------------------------------------- families


def test_relation_families_cover_the_graph_without_naming_it() -> None:
    """Four families, and every one of them resolves to real relation kinds."""
    from ascendc_codemap_mcp.engine.ir.relation import RelationKind

    legal = {k.value for k in RelationKind}
    assert set(RELATION_FAMILIES) == {"call", "data", "control", "compile"}
    for family, kinds in RELATION_FAMILIES.items():
        assert kinds, family
        assert set(kinds) <= legal, (family, sorted(set(kinds) - legal))


def test_a_family_and_a_raw_kind_are_both_accepted() -> None:
    kinds, families = expand_relation("data,control")
    assert families == {"data", "control"}
    assert "WRITES" in kinds and "GUARDED_BY" in kinds
    kinds, families = expand_relation("WRITES")
    assert not families
    assert kinds == {"WRITES"}


def test_an_unknown_relation_fails_by_name_rather_than_walking_nothing() -> None:
    with pytest.raises(InvalidQuery) as caught:
        validate_plan(operation="trace", symbol="X", relation="dataflow")
    assert "DATAFLOW" in str(caught.value)
    assert "data" in caught.value.legal_filters, "the families have to be offered"


# ---------------------------------------------------------------- rendering


def test_a_narrowed_card_says_which_families_it_withheld() -> None:
    """A narrowed answer must not look like a complete one.

    Otherwise the reader cannot tell a symbol with no writes from a question
    that never asked about writes, which is the same collapse that turned
    "not computed" into "none".
    """
    text = _text(operation="trace", symbol="isBn2MultiBlk", relation="data")
    assert "Writes" in text
    assert "Withheld by relation filter" in text
    for family in ("compile", "control"):
        assert family in text.split("Withheld by relation filter")[1]


def test_every_write_section_states_its_own_completeness() -> None:
    text = _text(operation="trace", symbol="isBn2MultiBlk")
    head = next(ln for ln in text.splitlines() if ln.startswith("Writes"))
    assert "of" in head and ("complete" in head or "shown" in head), head


def test_a_write_names_the_call_that_reaches_it() -> None:
    """Three writes to one flag are three candidates until the order is known.

    The order is a property of the calls, not the writes, and recovering it by
    hand cost the session several round trips per flag.
    """
    text = _text(operation="trace", symbol="isBn2MultiBlk")
    assert "reached by" in text
    assert "DoOpTiling" in text


def test_a_printed_body_is_never_called_undefined() -> None:
    """`no resolved caller, and no definition either` above its own source.

    The location card never promised complete callers, but it rendered the
    empty list as a finding anyway, seventeen times in one session.
    """
    text = _text(
        operation="source",
        file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
        line=1099,
    )
    assert "Source" in text
    assert "no definition either" not in text
    if "no resolved caller" in text:
        raise AssertionError("a card that did not compute callers must not report none")


def test_the_card_is_titled_after_the_symbol_that_was_asked_for() -> None:
    text = _text(operation="trace", symbol="DoSparse")
    assert text.splitlines()[0].strip() == "DoSparse"


# ---------------------------------------------------------------- compiled truth


def test_the_compiled_key_space_is_reachable_without_reading_dispatch() -> None:
    """The session answered this from `entry_regbase.h` and got it wrong.

    It claimed FP16/BF16/FP32 x {BN2GS1S2, BN2S2, BN2} for RoPE at D=192. The
    built keys carry neither FP32 nor BN2S2, and the product table knew that.
    """
    text = _text(operation="trace", dim="IsRope", value="1")
    assert "DTemplateNum: {192}" in text
    assert "legal_key_count: 224" in text
    dtype = next(ln for ln in text.splitlines() if "InputDType" in ln)
    assert "1" not in dtype.split("{")[1].split("}")[0].split(", "), dtype


# ---------------------------------------------------------------- source denoise (ses_fa50)


def test_source_span_is_titled_after_the_enclosing_function() -> None:
    """A search hit range used to title the card after BN2_MAX_D on line 1657."""
    text = _text(
        operation="source",
        file="op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp",
        line=1657,
        line_end=1740,
    )
    assert text.splitlines()[0].strip() == "SetSplitAxis"
    assert "BN2_MAX_D" not in text.splitlines()[0]
    assert "not computed for" not in text
    site = text.split("At this site", 1)[-1] if "At this site" in text else ""
    assert "BN2_MAX_D" not in site
    assert "CONTRACT" not in site


def test_source_span_attaches_enclosing_function_callers() -> None:
    text = _text(
        operation="source",
        file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
        line=1077,
        line_end=1149,
    )
    assert text.splitlines()[0].strip() == "DoSparse"
    assert "Called by" in text
    assert "DoOpTiling" in text
    assert "819" in text
    assert "not computed for" not in text
    assert text.splitlines()[0].strip().startswith("!(") is False


def test_unknown_dim_lists_real_compiled_dims() -> None:
    text = _text(operation="trace", dim="DeterBandScheduleMode")
    assert "not a compiled dim" in text.lower()
    assert "IsBn2MultiBlk" in text
    assert "IsRope" in text
    assert "Empty query" not in text
    assert "entity_id" not in text
    assert "from_symbol" not in text


def test_dim_catalog_lists_compiled_dims() -> None:
    text = _text(operation="trace", dim="*")
    assert "IsBn2MultiBlk" in text
    assert "IsRope" in text
    assert "DTemplateNum" in text
