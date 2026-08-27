# -*- coding: utf-8 -*-
"""Serve TG projections from the graph tables instead of a stored blob.

The kernel / host / tilingdata views used to be embedded in every product as
pre-serialized documents. That cost 4.4 MB per operator and made the document a
second source of truth that could disagree with the graph it came from; the
freshness machinery in `projection_provenance` exists only to police that gap.

Projecting on demand closes the gap by construction. It is affordable because a
projection needs a few kinds rather than the whole graph, so each one reads its
own slice: the kernel view touches 6.8k BRANCH rows and no relations, while
loading the full CodeMap to get at them costs ~40x more.

Callers keep using `load_view_blob_checked`; this module is what answers when
the blob is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap

#: Entity and relation kinds each projector reads. `None` means "every kind":
#: `project_tg_host_view` walks the incoming edges of intermediate nodes without
#: filtering on kind, so narrowing relations there would silently change which
#: host symbols it finds.
_REQUIREMENTS: dict[str, tuple[frozenset[str] | None, frozenset[str] | None]] = {
    "views/kernel.yaml": (
        frozenset({"BRANCH", "TILING_KEY", "TILING_FIELD"}),
        frozenset(),
    ),
    "views/tilingdata.yaml": (
        frozenset({"TILING_FIELD", "METHOD", "MACRO", "FUNCTION", "KERNEL"}),
        frozenset({"READS"}),
    ),
    "ir/tg_host_view.yaml": (
        frozenset(
            {
                "TILING_KEY",
                "FIELD",
                "VARIABLE",
                "PREDICATE",
                "INPUT",
                "COMPILE_VAR",
                "MACRO",
            }
        ),
        None,
    ),
    "ir/operator_graph.yaml": (None, None),
}


#: Views the product no longer ships, because projecting them costs less than
#: storing them. Together they were 4.4 MB of every product and the only place a
#: stale copy of the graph could hide.
#:
#: `ir/operator_graph.yaml` stays stored on purpose: it is 2 KB, and it is a
#: whole-graph histogram, so projecting it would have to load every row.
NOT_SHIPPED: tuple[str, ...] = (
    "views/kernel.yaml",
    "views/tilingdata.yaml",
    "ir/tg_host_view.yaml",
)


def is_projectable(name: str) -> bool:
    """Whether `name` can be rebuilt from the graph without a stored blob."""
    return str(name) in _REQUIREMENTS


def projectable_views() -> tuple[str, ...]:
    return tuple(sorted(_REQUIREMENTS))


def _projector(name: str) -> Callable[..., Any]:
    from ascendc_codemap_mcp.engine.tg_views import (
        project_kernel_view,
        project_operator_graph,
        project_tg_host_view,
        project_tilingdata_view,
    )

    return {
        "views/kernel.yaml": project_kernel_view,
        "views/tilingdata.yaml": project_tilingdata_view,
        "ir/tg_host_view.yaml": project_tg_host_view,
        "ir/operator_graph.yaml": project_operator_graph,
    }[str(name)]


def read_slice(path: str | Path, name: str) -> CodeMap:
    """The graph slice `name`'s projector needs, with canonical identity intact."""
    from ascendc_codemap_mcp.engine.store.reader import read_codemap

    ent_kinds, rel_kinds = _REQUIREMENTS[str(name)]
    return read_codemap(
        path,
        entity_kinds=None if ent_kinds is None else ent_kinds,
        relation_kinds=None if rel_kinds is None else rel_kinds,
    )


def project_view(path: str | Path, name: str) -> Any | None:
    """Build `name` from the graph. None when this view has no projector.

    The fingerprint and provenance stamp come from `meta`, not from the slice,
    so the result is indistinguishable from one projected off the whole graph.
    """
    key = str(name)
    if key not in _REQUIREMENTS:
        return None
    from ascendc_codemap_mcp.engine.projection_provenance import stamp_provenance

    cm = read_slice(path, key)
    fingerprint = str(cm.meta.get("graph_fingerprint") or "")
    view = _projector(key)(cm, fingerprint=fingerprint)
    return stamp_provenance(view, cm)
