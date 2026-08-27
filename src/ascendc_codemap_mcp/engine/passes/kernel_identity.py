# -*- coding: utf-8 -*-
"""Keep source-verified Kernel identity stable across lexical scope passes."""

from __future__ import annotations

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind


def preserve_verified_kernel_identity(codemap: CodeMap) -> CodeMap:
    """Restore verified Kernel provenance after the v1 body-scope scan.

    ``kernel_tiling_closure`` first creates a KERNEL from the verified
    ``__global__`` signature. Its generic function scanner may then reuse that
    entity as the body scope. Reuse is correct, but the generic scope attrs must
    not demote the authoritative Kernel provenance: the following v2 refinement
    deliberately purges old generic scope entities by that provenance.

    Preserve the scope origin separately while keeping the entity classified as
    a verified source Kernel. This is operator-agnostic and only applies to
    entities that are already KERNEL and carry a verified source signature.
    """
    restored = 0
    for kernel in codemap.by_kind(EntityKind.KERNEL):
        if not kernel.attrs.get("source_signature"):
            continue
        provenance = str(kernel.attrs.get("provenance") or "")
        if provenance not in {"source_kernel_definition", "source_kernel_definition_v2"}:
            continue
        kernel.attrs["scope_provenance"] = provenance
        kernel.attrs["provenance"] = "source_kernel_signature_verified"
        kernel.attrs["source_definition"] = True
        restored += 1
    meta = dict(codemap.meta.get("kernel_identity") or {})
    meta.update({"schema": "uo-kernel-identity/v1", "verified_provenance_restored": restored})
    codemap.meta["kernel_identity"] = meta
    return codemap
