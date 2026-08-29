# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from tests.conftest import write_uo_fixture


def test_resolve_codemap_id_from_disk_after_new_registry(tmp_path: Path, monkeypatch) -> None:
    from ascendc_codemap_mcp.service.control import status
    from ascendc_codemap_mcp.service.identity import is_ref, make_id, resolve
    from ascendc_codemap_mcp.service import runtime

    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    cid = make_id("toy_op", "arch35", project=op)
    runtime.registry.clear()
    runtime.registry._loaded = False
    ref = resolve(codemap_id=cid, registry=runtime.registry, require_indexed=True)
    assert is_ref(ref)
    assert ref.op_name == "toy_op"  # type: ignore[union-attr]


def test_paginate_does_not_slice_dim_names() -> None:
    from ascendc_codemap_mcp.service.query import paginate

    names = [f"Dim{i}" for i in range(20)]
    payload = {
        "ok": True,
        "shape": "index",
        "dim_names": names,
        "tiling_data_names": ["Td0"],
        "phases": [],
        "count": 0,
    }
    page, coverage, nxt = paginate(
        payload, limit=8, offset=0, snapshot="cm:x", query="q"
    )
    assert page["dim_names"] == names
    assert page["tiling_data_names"] == ["Td0"]
    assert nxt is None or coverage.get("truncated") is False or page.get("dim_names") == names
