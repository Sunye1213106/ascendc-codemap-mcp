# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ascendc_codemap_mcp.engine.store.reader import close_uo_connections
from ascendc_codemap_mcp.engine.store.schema import SCHEMA_SQL
from ascendc_codemap_mcp.service import runtime

FAG_REL_UO = Path(".ascendc-codemap") / "arch35" / "FlashAttentionScoreGrad.arch35.uo"
FAG_ROOT_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)


def fag_operator_root() -> Path | None:
    """Workspace FAG first; the historical d:\\TEST clone is the fallback."""
    for root in FAG_ROOT_CANDIDATES:
        if (root / FAG_REL_UO).is_file():
            return root
    return None


def write_uo_fixture(
    op: Path,
    *,
    arch: str = "arch35",
    symbol: str = "IsPse",
    revision: str = "abc123",
    entity_id: str = "e1",
    attrs: dict[str, Any] | None = None,
) -> Path:
    dest = op / ".ascendc-codemap" / arch / f"{op.name}.{arch}.uo"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    blob = json.dumps(attrs) if attrs is not None else "{}"
    conn = sqlite3.connect(str(dest))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("architecture", arch))
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("op_name", op.name))
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("schema", "codemap-uo/v3"))
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("source_revision", revision))
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("entity_count", "1"))
        conn.execute(
            "INSERT INTO entity(id, kind, name, status, confidence, file, line_start, line_end, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entity_id,
                "TILING_KEY",
                symbol,
                "verified",
                1.0,
                "op_host/tiling.cpp",
                10,
                12,
                blob,
            ),
        )
        conn.execute(
            "INSERT INTO source_span(id, entity_id, file, line_start, line_end, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"span:{entity_id}", entity_id, "op_host/tiling.cpp", 10, 12, symbol),
        )
        conn.commit()
    finally:
        conn.close()
    return dest


@pytest.fixture(autouse=True)
def _reset_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASCENDC_CODEMAP_CACHE_DIR", str(tmp_path / ".codemap-cache"))
    runtime.registry.clear()
    runtime.registry._loaded = False
    runtime.cache.close_all()
    close_uo_connections()
    yield
    runtime.registry.clear()
    runtime.cache.close_all()
    close_uo_connections()
