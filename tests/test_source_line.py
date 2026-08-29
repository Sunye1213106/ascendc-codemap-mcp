# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

from ascendc_codemap_mcp.engine.store.accel import build_source_line
from ascendc_codemap_mcp.engine.store.writer import operator_root_from_product


def _touch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_operator_root_from_product_is_operator_dir_not_domain(tmp_path: Path) -> None:
    dest = (
        tmp_path
        / "attention"
        / "sparse_flash_attention_grad"
        / ".ascendc-codemap"
        / "arch35"
        / "sparse_flash_attention_grad.arch35.uo"
    )
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    root = operator_root_from_product(dest)
    assert root == (tmp_path / "attention" / "sparse_flash_attention_grad").resolve()
    assert root is not None
    assert root.name == "sparse_flash_attention_grad"


def test_source_line_indexes_sibling_common_not_neighbor_ops(tmp_path: Path) -> None:
    attention = tmp_path / "attention"
    common = attention / "common"
    fag = attention / "flash_attention_score_grad"
    sfag = attention / "sparse_flash_attention_grad"
    _touch(common / "op_kernel" / "arch35" / "pse.h", "int pse;\n")
    _touch(common / "op_kernel" / "arch22" / "old.h", "int old;\n")
    _touch(common / "include" / "util.h", "int util;\n")
    _touch(fag / "op_kernel" / "arch35" / "fag.h", "int fag;\n")
    _touch(sfag / "op_kernel" / "arch35" / "sfag.h", "int sfag;\n")

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE entity(id TEXT, file TEXT)")
    conn.execute("CREATE TABLE source_span(id TEXT, file TEXT)")
    conn.execute(
        "INSERT INTO entity(id, file) VALUES (?, ?)",
        ("e1", "../common/op_kernel/arch35/pse.h"),
    )
    conn.commit()

    files, lines = build_source_line(conn, sfag, architecture="arch35")
    assert files >= 3
    assert lines >= 3
    paths = {row[0] for row in conn.execute("SELECT DISTINCT path FROM source_line")}
    assert "op_kernel/arch35/sfag.h" in paths
    assert "../common/op_kernel/arch35/pse.h" in paths
    assert "../common/include/util.h" in paths
    assert not any("flash_attention_score_grad" in path for path in paths)
    assert "../common/op_kernel/arch22/old.h" not in paths
    assert "fag.h" not in {Path(p).name for p in paths}
