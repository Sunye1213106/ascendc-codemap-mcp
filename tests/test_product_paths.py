# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.paths import PRODUCT_DIR_NAME, product_dir
from ascendc_codemap_mcp.engine.store.reader import find_uo_product


def test_product_dir_layout(tmp_path: Path) -> None:
    op = tmp_path / "my_op"
    op.mkdir()
    got = product_dir(op, "arch35")
    assert got == (op / PRODUCT_DIR_NAME / "arch35").resolve()
    assert ".ascendc-pilot" not in str(got)


def test_find_uo_product_under_new_layout(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    dest = op / PRODUCT_DIR_NAME / "arch35" / "toy_op.arch35.uo"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    found = find_uo_product(op, architecture="arch35")
    assert found == dest.resolve()


def test_product_dir_requires_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ARCHITECTURE_MISSING"):
        product_dir(tmp_path, "")
