# -*- coding: utf-8 -*-
"""Scheme A: path layer + line role, no graph join."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.engine.query.rg import line_role, path_layer, rank_hit
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import write_uo_fixture
from tests.test_query_surface import _add_source_lines


def test_path_layer_own_host_before_common_host() -> None:
    assert path_layer("op_host/arch35/tiling.cpp") < path_layer(
        "../common/op_host/fia_tiling_templates_registry.h"
    )
    assert path_layer("op_host/arch35/tiling.cpp") < path_layer("op_kernel/arch35/k.h")
    assert path_layer("op_kernel/arch35/k.h") < path_layer("../common/op_kernel/attn.h")


def test_line_role_prefers_assign_over_log_and_tpl() -> None:
    assert line_role("    fBaseParams.isBn2MultiBlk =") < line_role(
        '    OP_LOGI(context_, "isBn2MultiBlk[%d]", isBn2MultiBlk);'
    )
    assert line_role("    bool isBn2MultiBlk = false;") < line_role(
        "    ASCENDC_TPL_BOOL_SEL(IsBn2MultiBlk, 0),"
    )
    assert line_role(
        "DeterBandScheduleResult SelectDeterBandSchedule(int64_t k, int64_t m)"
    ) < line_role("    // Keep identical to CalDeterMaxLoopNum on the kernel side.")
    assert line_role("    if (fBaseParams.isBn2MultiBlk) {") < line_role(
        "    ASCENDC_TPL_BOOL_SEL(IsEmptyTensor, 0),"
    )


def test_rank_hit_orders_decision_before_template() -> None:
    hits = [
        (
            "op_kernel/arch35/template_tiling_key.h",
            131,
            "    ASCENDC_TPL_BOOL_SEL(IsEmptyTensor, 0),",
        ),
        (
            "op_host/flash_attention_score_grad_tiling.cpp",
            99,
            "            tilingData->emptyTensorTilingData.set_formerDqNum(aivNum);",
        ),
        (
            "../common/op_host/fia.h",
            10,
            "    bool IsEmptyTensor = false;",
        ),
        (
            "op_host/arch35/common_regbase.cpp",
            200,
            '    OP_LOGI("empty tensor");',
        ),
    ]
    ordered = sorted(hits, key=lambda h: rank_hit(h[0], h[1], h[2], "EmptyTensor"))
    assert "set_formerDqNum" in ordered[0][2]


def test_search_ranks_assignment_ahead_of_tpl_and_common(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    dest = write_uo_fixture(op)
    conn = sqlite3.connect(str(dest))
    try:
        _add_source_lines(
            conn,
            [
                (
                    "../common/op_host/shared.h",
                    10,
                    "    bool isBn2MultiBlk = false;",
                ),
                (
                    "op_kernel/arch35/template_tiling_key.h",
                    40,
                    "    ASCENDC_TPL_BOOL_SEL(IsBn2MultiBlk, 0),",
                ),
                (
                    "op_host/arch35/tiling.cpp",
                    80,
                    '    OP_LOGI(context_, "isBn2MultiBlk[%d]", flag);',
                ),
                (
                    "op_host/arch35/tiling.cpp",
                    50,
                    "    fBaseParams.isBn2MultiBlk = bnSparseLimit && !hasRope;",
                ),
                (
                    "op_host/arch35/tiling.cpp",
                    60,
                    "    if (fBaseParams.isBn2MultiBlk) {",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    status(project=str(op), architecture="arch35")
    text = str(
        (
            query(
                project=str(op),
                architecture="arch35",
                operation="search",
                name="isBn2MultiBlk",
            ).get("data")
            or {}
        ).get("text")
        or ""
    )
    assign = text.find("tiling.cpp:50")
    ctrl = text.find("tiling.cpp:60")
    log = text.find("tiling.cpp:80")
    tpl = text.find("template_tiling_key.h:40")
    common = text.find("shared.h:10")
    assert assign != -1 and ctrl != -1 and log != -1
    assert assign < ctrl < log
    assert tpl == -1 or assign < tpl
    assert common == -1 or assign < common
