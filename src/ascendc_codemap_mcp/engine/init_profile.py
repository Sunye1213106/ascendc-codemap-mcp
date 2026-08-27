# -*- coding: utf-8 -*-
"""uo-init product defaults (single path: host IR ∥ one-dtype kernel IR)."""
from __future__ import annotations

import os
from typing import Any


def cold_budget_s() -> float:
    try:
        return float(os.environ.get("UO_COLD_BUDGET_S", "180"))
    except ValueError:
        return 180.0


def default_with_kernel(ctx: dict[str, Any] | None = None) -> bool:
    if isinstance(ctx, dict) and "with_kernel" in ctx:
        return bool(ctx.get("with_kernel"))
    env = os.environ.get("UO_WITH_KERNEL")
    if env is not None and str(env).strip() != "":
        return str(env).strip().lower() not in {"0", "false", "off", "no"}
    return True


def default_kernel_max_variants(ctx: dict[str, Any] | None = None) -> int:
    """Cap kernel dtype walks. ``0`` means all declared variants; product default is 1."""
    if isinstance(ctx, dict) and ctx.get("kernel_max_variants") not in (None, ""):
        try:
            return max(0, int(ctx.get("kernel_max_variants")))
        except (TypeError, ValueError):
            pass
    env = os.environ.get("UO_KERNEL_MAX_VARIANTS")
    if env is not None and str(env).strip() != "":
        try:
            return max(0, int(str(env).strip()))
        except ValueError:
            pass
    return 1
