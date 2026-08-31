# -*- coding: utf-8 -*-
"""Kernel resources must answer in their own terms.

Extraction records what a queue, buffer, register or event *is* -- pipe
position, memory space, register class, whether a buffer is the locked half of
a double buffer, how many times an event is set versus waited. The projection
only ever read the edges around those entities, so a card could say where a
queue was backed without saying which pipe position it occupies, and an
unbalanced cross-core event looked like an ordinary field declaration.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)
_REL_UO = Path(".ascendc-codemap/arch35/FlashAttentionScoreGrad.arch35.uo")


def _operator() -> Path | None:
    for root in _CANDIDATES:
        if (root / _REL_UO).is_file():
            return root
    return None


FAG = _operator()
pytestmark = pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")


def _resolve(symbol: str) -> str:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG), architecture="arch35", operation="resolve", symbol=symbol
    )
    return str((payload.get("data") or {}).get("text") or "")


def test_queue_card_states_its_pipe_position_and_space() -> None:
    text = _resolve("inQueuePing")
    assert "Resource" in text
    assert "tposition=VECIN" in text
    assert "memory_space=UB" in text
    # The weaker duplicate of the same backing must not come back.
    assert "UNKNOWN" not in text
    assert text.count("backed by vecInPing") <= 1


def test_register_card_states_its_register_class() -> None:
    text = _resolve("vregReduceSum")
    assert "register_class=VREG" in text
    assert "memory_space=REG" in text
    assert "type_name=RegTensor" in text


def test_buffer_card_shows_the_double_buffer_lock() -> None:
    text = _resolve("ping_")
    assert "wraps_lock=True" in text
    assert "role=storage_wrapper" in text


def test_event_card_reports_its_set_and_wait_sites() -> None:
    text = _resolve("eventIDMte3ToS")
    assert "event_type=MTE3_S" in text
    assert "mechanism=hard_event" in text
    assert "set sites=2  wait sites=2" in text


def test_sites_of_one_event_name_in_two_templates_stay_apart() -> None:
    """Two templates each set once; bare line numbers read as one setting twice."""
    text = _resolve("eventIDMte3ToS")
    block = text.split("Sync pairing", 1)[1]
    assert "flash_attention_score_grad_nz_post.h:273" in block
    assert "flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h:138" in block


def test_guarded_sync_sites_are_not_called_unbalanced() -> None:
    """A cross-core handshake that pairs correctly was reported as a defect.

    Sites are counted per guarded branch, so an `isReuse` pair counts twice
    though only one ever compiles, and a second flag issued at an AIV offset
    does not count at all. Three-versus-four is not a finding.
    """
    text = _resolve("id1_")
    assert "mechanism=cross_core" in text
    assert "set sites=4  wait sites=3" in text
    assert "UNBALANCED" not in text
    assert "sit under a condition" in text


def test_tiling_field_card_names_its_generated_accessors() -> None:
    """The transport hop ends at the struct; the accessor is how it is read back."""
    text = _resolve("deterMaxRound")
    assert "Accessors" in text
    assert "set_deterMaxRound" in text
    assert "SaveToTilingData" in text


def _resolve_site(file: str, line: int) -> str:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        file=file,
        line=line,
    )
    return str((payload.get("data") or {}).get("text") or "")


def test_site_card_profiles_the_pipeline_operations() -> None:
    """What a kernel block does to the pipeline is classified, not just quoted."""
    text = _resolve_site(
        "op_kernel/arch35/flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h", 138
    )
    assert "Pipeline operations" in text
    profile = text.split("Pipeline operations", 1)[1]
    for category in ("sync_signal", "sync_wait", "buffer_acquire", "buffer_release"):
        assert category in profile, f"{category} missing from the pipeline profile"
    # Acquire and release are what a reviewer pairs up, so both carry a count.
    assert "×" in profile


def test_site_card_carries_the_same_resource_facts_as_the_symbol_card() -> None:
    """Asking by file+line must not answer with less than asking by name."""
    text = _resolve_site("op_kernel/arch35/vector_api/vf_post_reduce_sink.h", 37)
    assert "Resources in this unit" in text
    block = text.split("Resources in this unit", 1)[1]
    assert "register_class=VREG" in block
    assert "register_class=UNALIGN_REG" in block
    assert "register_class=MASK_REG" in block


def test_function_symbol_card_carries_the_pipeline_profile() -> None:
    """Both spellings of "read this function" return the same classification."""
    text = _resolve("ProcessVec3")
    assert "Pipeline operations" in text
    assert "buffer_acquire" in text


def test_delegating_function_reports_what_its_callees_do() -> None:
    """A dispatcher's own body is nearly empty; the work is one hop out."""
    text = _resolve("CalBandDeterIndex")
    assert "Pipeline operations via callees (depth 1)" in text


def test_symbol_card_carries_the_same_resource_facts_as_the_site_card() -> None:
    """The converse direction: asking by name must not answer with less.

    Resources were projected only onto the site card, so five review agents
    read a function by name, saw no registers, and spent a second call on the
    same lines by file and line to get them.
    """
    text = _resolve("ReduceSinkVF")
    assert "Resources in this unit" in text
    block = text.split("Resources in this unit", 1)[1]
    assert "register_class=VREG" in block
    assert "register_class=UNALIGN_REG" in block
    assert "register_class=MASK_REG" in block


def test_a_class_read_by_name_declares_its_buffers() -> None:
    """A type is where buffers are declared, so its own card has to say so."""
    text = _resolve("BuffersPolicyDB")
    assert "Resources in this unit" in text
    block = text.split("Resources in this unit", 1)[1]
    assert "ping_" in block
    assert "pong_" in block


def test_a_bare_class_name_is_the_class_not_its_constructor() -> None:
    """`METHOD` sorts before `TYPE`, so `Buffer` used to land on `Buffer::Buffer`."""
    text = _resolve("Buffer")
    head = text.splitlines()
    assert head[0] == "Buffer", head[:2]
    assert head[1] == "TYPE", head[:2]


def test_tiling_data_card_maps_each_field_host_to_kernel() -> None:
    """A TilingData type is the host/kernel ABI, so its card answers as one."""
    text = _resolve("BaseDeterParamRegbase")
    assert "TilingData fields" in text
    block = text.split("TilingData fields", 1)[1]
    row = next(ln for ln in block.splitlines() if "deterMaxRound" in ln)
    assert "host " in row and "kernel " in row
    assert "CalDeterMaxLoopNum" in row


def test_tiling_key_assignments_exclude_operator_inputs() -> None:
    """A key derives from every input transitively; that is not an assignment."""
    text = _resolve("DTemplateNum")
    assigns = text.split("Assignments", 1)[1].split("\n\n", 1)[0]
    for noise in ("head_num", "input_layout", "actual_seq_qlen", "softmax_max"):
        assert noise not in assigns, f"{noise} rendered as an assignment"
    assert "dTemplateType" in assigns


def test_type_card_returns_the_class_body() -> None:
    text = _resolve("BuffersPolicyDB")
    assert "class BuffersPolicyDB" in text
    # Enough of the body to see how the two halves are allocated and paired.
    assert "AllocBuffer" in text
    assert "pong_" in text


def test_a_template_lists_the_events_it_uses_not_the_ones_filed_under_it() -> None:
    """Two templates spelling one event name share one entity.

    It is filed at whichever was indexed first, so listing only what a span
    declares gave the other template its buffers and none of its events, though
    it sets and waits the same four.
    """
    text = _resolve_site("op_kernel/arch35/flash_attention_score_grad_nz_post.h", 273)
    block = text.split("Resources in this unit", 1)[1]
    for event in ("eventIDMte3ToS", "eventIDVToSPing", "eventIDVToSPong", "eventIDSToMte3"):
        assert event in block, f"{event} missing from the card that uses it"


def test_an_operation_category_says_which_calls_it_counted() -> None:
    """A category total sums unlike calls, and the split is the answer wanted.

    `sync_signal x12` covered FetchEventID, GetTPipePtr and SetFlag together, so
    a reader after the flag count had to derive it from the body.
    """
    text = _resolve("ProcessSink")
    row = next(ln for ln in text.splitlines() if "sync_signal" in ln)
    assert "SetFlag ×" in row, row
    assert "FetchEventID ×" in row, row


def test_a_sync_scope_reports_the_flag_calls_it_issues() -> None:
    """Guarded sites are places, not calls, and the call count is the question.

    Told the two site totals were not call counts, a reader read the whole
    function to count the flags -- a number the census over that same function
    already held.
    """
    text = _resolve("id0_")
    assert "Sync pairing" in text
    block = text.split("Sync pairing", 1)[1]
    assert "CrossCoreSetFlag ×" in block, block[:600]
    assert "CrossCoreWaitFlag ×" in block, block[:600]
