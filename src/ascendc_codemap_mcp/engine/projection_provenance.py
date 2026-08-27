# -*- coding: utf-8 -*-
"""Projection provenance: semantic digest + counts + builder.

A projection is only fresh when it is tied to the exact canonical semantic
content. Histogram fingerprints and row counts are useful diagnostics, but are
not semantic identities: two graphs can have identical kinds/counts while
connecting different endpoints or carrying different projection-driving meta.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.tg_views import graph_fingerprint

PROJECTION_SCHEMA = "uo-projection-provenance/v1"
PROJECTION_BUILDER = "uo_init.tg_views"
PROJECTION_BUILDER_VERSION = "2"
CANONICAL_DIGEST_SCHEMA = "uo-canonical-graph-digest/v1"

VIEW_STALE = "VIEW_STALE"
LEGACY_VIEW_UNVERIFIED = "LEGACY_VIEW_UNVERIFIED"

_IDENTITY_META_KEYS = frozenset(
    {
        "graph_fingerprint",
        "canonical_graph_digest",
        "canonical_revision",
    }
)


def canonical_counts(codemap: CodeMap) -> dict[str, int]:
    """Row counts of the canonical graph.

    A CodeMap read with a kind filter carries the counts of the graph it was
    cut from in `meta`, the same way `canonical_graph_digest` reuses a cached
    digest. Falling back to `len()` there would stamp a projection with the
    size of its own slice and make every freshness check fail.
    """
    meta_ec = codemap.meta.get("entity_count")
    meta_rc = codemap.meta.get("relation_count")
    if isinstance(meta_ec, int) and isinstance(meta_rc, int):
        return {"entity_count": meta_ec, "relation_count": meta_rc}
    return {
        "entity_count": len(codemap.entities),
        "relation_count": len(codemap.relations),
    }


def _semantic_meta(codemap: CodeMap) -> dict[str, Any]:
    """Canonical metadata excluding self-referential projection identities."""
    return {
        str(k): v
        for k, v in sorted(codemap.meta.items(), key=lambda item: str(item[0]))
        if str(k) not in _IDENTITY_META_KEYS
    }


def canonical_graph_digest(codemap: CodeMap) -> str:
    """Stable digest over canonical entities, relations and semantic meta.

    Relation endpoints and semantic attributes are included so a rewired graph
    cannot preserve identity merely by keeping the same counts/kind histogram.
    Projection-driving canonical meta is included as well; only identity fields
    derived from this digest are excluded to avoid recursion.

    When ``codemap.meta['canonical_graph_digest']`` is already set (finalize /
    commit after a mutation pop), reuse it. Callers that mutate the graph must
    pop identity keys first — this function never writes the cache itself, so
    a stamp-then-rewire-then-validate sequence still sees a fresh digest.
    """
    cached = codemap.meta.get("canonical_graph_digest")
    if isinstance(cached, str) and cached:
        return cached
    payload = {
        "schema": CANONICAL_DIGEST_SCHEMA,
        "op": codemap.op_name,
        "arch": codemap.architecture,
        "meta": _semantic_meta(codemap),
        "entities": [
            codemap.entities[eid].to_dict()
            for eid in sorted(codemap.entities)
        ],
        "relations": [
            codemap.relations[rid].to_dict()
            for rid in sorted(codemap.relations)
        ],
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stamp_provenance(
    view: Any,
    codemap: CodeMap,
    *,
    builder: str = PROJECTION_BUILDER,
    builder_version: str = PROJECTION_BUILDER_VERSION,
    schema_version: str = PROJECTION_SCHEMA,
) -> Any:
    """Attach provenance block to a dict view; non-dicts returned unchanged."""
    if not isinstance(view, dict):
        return view
    counts = canonical_counts(codemap)
    fp = str(codemap.meta.get("graph_fingerprint") or graph_fingerprint(codemap))
    digest = canonical_graph_digest(codemap)
    revision = str(codemap.meta.get("canonical_revision") or digest[:16])
    out = dict(view)
    if "fingerprint" in out:
        out["fingerprint"] = fp
    source = out.get("source")
    if isinstance(source, dict):
        src = dict(source)
        src["graph_fingerprint"] = fp
        out["source"] = src
    out["provenance"] = {
        "schema": schema_version,
        "canonical_revision": revision,
        "canonical_graph_digest": digest,
        "graph_fingerprint": fp,
        "entity_count": counts["entity_count"],
        "relation_count": counts["relation_count"],
        "schema_version": schema_version,
        "projection_builder": builder,
        "projection_builder_version": builder_version,
    }
    if out.get("schema") == "uo-operator-graph/v1":
        out["node_count"] = counts["entity_count"]
        out["edge_count"] = counts["relation_count"]
        out["fingerprint"] = fp
    return out


def stamp_all_views(views: dict[str, Any], codemap: CodeMap) -> dict[str, Any]:
    return {name: stamp_provenance(payload, codemap) for name, payload in views.items()}


def extract_provenance(view: Any) -> dict[str, Any] | None:
    if not isinstance(view, dict):
        return None
    prov = view.get("provenance")
    if isinstance(prov, dict) and prov.get("canonical_graph_digest"):
        return prov
    fp = view.get("fingerprint")
    if not fp and isinstance(view.get("source"), dict):
        fp = view["source"].get("graph_fingerprint")
    if not fp:
        return None
    return {
        "graph_fingerprint": fp,
        "entity_count": view.get("node_count") or view.get("entity_count"),
        "relation_count": view.get("edge_count") or view.get("relation_count"),
        "canonical_graph_digest": None,
    }


def validate_view_against_codemap(view: Any, codemap: CodeMap) -> dict[str, Any]:
    """Return a fail-closed freshness verdict for one materialized projection."""
    expected_fp = str(codemap.meta.get("graph_fingerprint") or graph_fingerprint(codemap))
    expected_digest = canonical_graph_digest(codemap)
    expected_counts = canonical_counts(codemap)

    if not isinstance(view, dict):
        return {
            "ok": False,
            "reason_code": LEGACY_VIEW_UNVERIFIED,
            "message": "non-dict projection has no verifiable provenance",
            "expected": {
                "canonical_graph_digest": expected_digest,
                **expected_counts,
            },
        }

    prov = extract_provenance(view)
    if prov is None or not prov.get("canonical_graph_digest"):
        return {
            "ok": False,
            "reason_code": LEGACY_VIEW_UNVERIFIED,
            "message": "projection lacks canonical semantic digest; rebuild/update the .uo or use canonical fallback",
            "expected": {
                "canonical_graph_digest": expected_digest,
                "graph_fingerprint": expected_fp,
                **expected_counts,
            },
            "actual": {
                "graph_fingerprint": (prov or {}).get("graph_fingerprint"),
                "entity_count": (prov or {}).get("entity_count"),
                "relation_count": (prov or {}).get("relation_count"),
            },
        }

    actual_digest = str(prov.get("canonical_graph_digest") or "")
    actual_fp = str(prov.get("graph_fingerprint") or "")
    actual_ec = prov.get("entity_count")
    actual_rc = prov.get("relation_count")
    mismatches: list[str] = []
    if actual_digest != expected_digest:
        mismatches.append("canonical_graph_digest")
    if actual_fp and actual_fp != expected_fp:
        mismatches.append("graph_fingerprint")
    if actual_ec is not None and int(actual_ec) != expected_counts["entity_count"]:
        mismatches.append("entity_count")
    if actual_rc is not None and int(actual_rc) != expected_counts["relation_count"]:
        mismatches.append("relation_count")
    if mismatches:
        return {
            "ok": False,
            "reason_code": VIEW_STALE,
            "mismatches": mismatches,
            "expected": {
                "canonical_graph_digest": expected_digest,
                "graph_fingerprint": expected_fp,
                **expected_counts,
            },
            "actual": {
                "canonical_graph_digest": actual_digest,
                "graph_fingerprint": actual_fp,
                "entity_count": actual_ec,
                "relation_count": actual_rc,
            },
        }
    return {"ok": True, "reason_code": ""}
