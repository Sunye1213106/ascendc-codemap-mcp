# -*- coding: utf-8 -*-
"""Process-local uo-init session teardown.

Caches are still module globals for in-run speed. Workflow end (and extract
failure) must drop them so a long-lived Python runner cannot leak TUs or
reuse the wrong operator bundle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def live_ast_count() -> int:
    from ascendc_codemap_mcp.engine import tu_cache

    return tu_cache.live_ast_count()


def end_session(
    op_root: str | Path | None = None,
    architecture: str | None = None,
    *,
    drop_compile_mem: bool = True,
) -> None:
    """Release process-global extract/analyze caches for this workflow."""
    from ascendc_codemap_mcp.engine import include_heal, tu_cache
    from ascendc_codemap_mcp.engine.build import drop_compile_mem as _drop_compile_mem
    from ascendc_codemap_mcp.engine.passes.source_text_cache import clear as clear_source_text
    from ascendc_codemap_mcp.engine.source_index import reset_index_cache

    tu_cache.clear_live_ast()
    try:
        from ascendc_codemap_mcp.engine import pilot_engines as pe

        pe._STORE.clear()
    except Exception:  # noqa: BLE001
        pass
    if drop_compile_mem:
        _drop_compile_mem(Path(op_root) if op_root else None, architecture=architecture)
    clear_source_text()
    reset_index_cache()
    include_heal.reset_index_cache()


def bundle_identity(
    project_root: str | Path,
    ctx: dict[str, Any] | None = None,
    *,
    op_name: str = "",
    architecture: str = "",
    extract_fingerprint: str = "",
) -> tuple[str, str, str]:
    ctx = ctx or {}
    root = str(Path(project_root).expanduser().resolve())
    name = str(op_name or ctx.get("op_name") or "")
    arch = str(
        architecture
        or ctx.get("arch_dir")
        or ctx.get("architecture")
        or ctx.get("arch")
        or ""
    )
    del extract_fingerprint
    return (root, name, arch)
