# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "src" / "ascendc_codemap_mcp"
_NEW = (
    "engine/passes/compile_policy.py",
    "engine/passes/workspace_abi.py",
    "engine/passes/consumer_role.py",
    "engine/passes/entry_path.py",
    "engine/passes/contracts.py",
    "engine/passes/guarded_calls.py",
    "engine/passes/constexpr_alias.py",
    "engine/passes/host_predicates.py",
    "engine/query/explore.py",
    "engine/query/closure.py",
    "engine/query/completeness.py",
    "engine/query/typed.py",
    "engine/query/predicate_ast.py",
)
_FORBIDDEN = (
    "s1Inner",
    "hasDrop",
    "VECTOR_BASEM",
    "sValueZeroUnderTND",
    "FlashAttentionScoreGrad",
    "PreSfmg",
    "keepProb",
    "dropMask",
    "CUBE_BASEM",
)


def test_new_contract_modules_have_no_fixture_identifiers() -> None:
    for rel in _NEW:
        text = (_ROOT / rel).read_text(encoding="utf-8")
        for needle in _FORBIDDEN:
            assert needle not in text, f"{rel} contains fixture identifier {needle}"
