# -*- coding: utf-8 -*-
"""TG Host View: a disposable projection of HostIR facts for TG/CE search.

Authority lives in the arch-scoped ``.uo`` CodeMap product
(``.ascendc-codemap/<arch>/<op>.<arch>.uo``). This module may write a
working-tree YAML projection during extract; production loaders read the
``.uo`` view_blob only. sqlite / YAML are migrate/test helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

# Durable projection (preferred name).
TG_HOST_VIEW_YAML = "ir/tg_host_view.yaml"
# Legacy read-only fallbacks for old on-disk caches. New writers must not
# create these; load_tg_host_view still accepts them when the preferred file
# is missing.
CODEMAP_YAML = "ir/host_codemap.yaml"  # legacy alias (read-only)
CODEMAP_SQLITE = "indexes/host_codemap.sqlite"  # leftover; unlinked when present
SCHEMA = "tg-host-view/v1"
COMPAT_SCHEMA = "codemap/v2"

#: RHS length for writer rows. v1 capped at 200 and already overflowed.
RHS_LIMIT = 800
#: Guard strings kept per writer.
GUARD_LIMIT = 16

_PLATFORM_RE = re.compile(
    r"\b(npuArch|socVersion|NpuArch|SocVersion|DAV_\w+|Ascend\d+)\b"
)
_CMP_HINTS = (
    ("bn1s1s2", re.compile(r"\bb\b.*\bn1\b.*\bs1\b.*\bs2\b|\bbn1s1s2\b", re.I)),
    ("qkv_bytes", re.compile(r"qkv|dtypeBytes|GetSize|l2Size|L2", re.I)),
    ("s1_mod128", re.compile(r"%\s*128|s1\s*%", re.I)),
    ("band", re.compile(r"preTokens|nextTokens|s1Token|s2Token|pre_tokens", re.I)),
    ("dtype_is_fp32", re.compile(r"DT_FLOAT\b", re.I)),
)


def export_host_codemap(
    host_ir: Any,
    uo_root: str | Path,
    *,
    derive_fields: list[dict[str, Any]] | None = None,
    declared: dict[str, Any] | None = None,
    graph_fingerprint: str = "",
    source_revision: str = "",
    manifest_hash: str = "",
) -> dict[str, Any]:
    """Write the TG host view under ``uo_root`` and rebuild the query cache.

    Prefer :func:`export_tg_host_view` at call sites; this name is kept for
    older imports.
    """
    return export_tg_host_view(
        host_ir,
        uo_root,
        derive_fields=derive_fields,
        declared=declared,
        graph_fingerprint=graph_fingerprint,
        source_revision=source_revision,
        manifest_hash=manifest_hash,
    )


def export_tg_host_view(
    host_ir: Any,
    uo_root: str | Path,
    *,
    derive_fields: list[dict[str, Any]] | None = None,
    declared: dict[str, Any] | None = None,
    graph_fingerprint: str = "",
    source_revision: str = "",
    manifest_hash: str = "",
) -> dict[str, Any]:
    """Project HostIR into ``tg_host_view.yaml`` stamped with the KB fingerprint.

    Does not read ``.probe_cache/*.pkl``. Callers must supply a live HostIR
    (typically from the same in-process extract that later commits ``.uo``).
    """
    root = Path(uo_root)
    payload = host_ir_payload(
        host_ir,
        derive_fields=derive_fields,
        declared=declared,
        graph_fingerprint=graph_fingerprint,
        source_revision=source_revision,
        manifest_hash=manifest_hash,
    )
    view_path = root / TG_HOST_VIEW_YAML
    view_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    view_path.write_text(text, encoding="utf-8")
    # Working-tree YAML is extract scratch. Production loaders read the .uo blob.
    summary = rebuild_codemap_index(root)
    return {
        "ok": True,
        "schema": SCHEMA,
        "yaml": str(view_path),
        "alias_yaml": "",
        "fields": len(payload.get("fields") or []),
        "writers": sum(
            len(f.get("writers") or []) for f in payload.get("fields") or []
        ),
        "predicates": len(payload.get("predicates") or []),
        "graph_fingerprint": str(
            (payload.get("source") or {}).get("graph_fingerprint") or ""
        ),
        **summary,
    }


def host_ir_payload(
    host_ir: Any,
    *,
    derive_fields: list[dict[str, Any]] | None = None,
    declared: dict[str, Any] | None = None,
    graph_fingerprint: str = "",
    source_revision: str = "",
    manifest_hash: str = "",
) -> dict[str, Any]:
    """Serialise the query surfaces the coverage / CE agents need."""
    writers = _writer_rows(host_ir)
    fields = _fields_from_writers(writers, derive_fields or [])
    predicates = _predicates_from_writers(writers)
    platform_gates = [
        p for p in predicates
        if _PLATFORM_RE.search(str(p.get("condition") or p.get("lhs") or ""))
    ]
    return {
        "schema": SCHEMA,
        "compat_schema": COMPAT_SCHEMA,
        "source": {
            "graph_fingerprint": graph_fingerprint or "",
            "manifest_hash": manifest_hash or "",
            "source_revision": source_revision or "",
            "generated_by": "export_tg_host_view",
            "authority": "uo/ir/operator_graph.yaml",
            "role": "tg_host_projection",
        },
        "fields": fields,
        "predicates": predicates,
        "declared_keys": declared or {},
        "platform_gates": platform_gates,
    }


def _writer_rows(host_ir: Any) -> list[dict[str, Any]]:
    expand = getattr(host_ir, "expand_callee_writers", None)
    events = expand() if callable(expand) else list(getattr(host_ir, "writes", ()) or ())
    rows = []
    for w in events:
        guards = list(getattr(w, "guards", lambda: [])() or [])[:GUARD_LIMIT]
        rows.append({
            "path": getattr(w, "path", ""),
            "function": getattr(w, "function", ""),
            "file": getattr(w, "file", ""),
            "line": int(getattr(w, "line", 0) or 0),
            "rhs": str(getattr(w, "rhs", "") or "")[:RHS_LIMIT],
            "guards": guards,
            "via": str(getattr(w, "via", "") or ""),
        })
    return rows


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1] if path else ""


def _fields_from_writers(
    writers: list[dict[str, Any]],
    derive_fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_leaf: dict[str, dict[str, Any]] = {}
    for w in writers:
        leaf = _leaf(w["path"])
        if not leaf:
            continue
        slot = by_leaf.setdefault(leaf, {
            "name": leaf,
            "kind": "host_state",
            "writers": [],
            "reads": [],
            "state_deps": [],
            "exactness": "",
            "note": "",
            "grade": "",
            "domain": [],
        })
        slot["writers"].append({
            "file": w["file"],
            "line": w["line"],
            "function": w["function"],
            "rhs": w["rhs"],
            "guards": w["guards"],
            "via": w["via"] or "direct",
            "path": w["path"],
        })

    # Overlay lightweight metadata from derive_key_fields when provided.
    for f in derive_fields:
        name = str(f.get("name") or "")
        if not name:
            continue
        # Key dims are often PascalCase; writers use camelCase field names.
        leaf = name[0].lower() + name[1:] if name[:2].isupper() is False else name
        # Prefer exact name match against writer leaves, else the dim name itself.
        slot = by_leaf.get(name) or by_leaf.get(leaf) or by_leaf.setdefault(name, {
            "name": name,
            "kind": "key_dim",
            "writers": [],
            "reads": [],
            "state_deps": [],
            "exactness": "",
            "note": "",
            "grade": "",
            "domain": [],
        })
        slot["kind"] = "key_dim"
        slot["exactness"] = str(f.get("exactness") or "")
        slot["note"] = str(f.get("note") or "")
        slot["domain"] = list(f.get("domain") or [])
        roots = f.get("var_roots") or {}
        if isinstance(roots, dict):
            slot["reads"] = [
                {"var": str(v), "root": str(r)} for v, r in roots.items()
            ]
        elif isinstance(roots, list):
            slot["reads"] = [{"var": str(v), "root": ""} for v in roots]
        state = f.get("state_targets") or {}
        if isinstance(state, dict):
            deps = []
            for vals in state.values():
                deps.extend(str(x) for x in (vals or []))
            slot["state_deps"] = sorted(set(deps))
        exact = slot["exactness"]
        if exact in ("exact", "constant"):
            slot["grade"] = "exact_static"
        elif exact == "overapproximated" or str(f.get("status")) == "partial":
            slot["grade"] = "empirical"
        elif slot["state_deps"]:
            slot["grade"] = "observed_exact"

    return [by_leaf[k] for k in sorted(by_leaf)]


def _feature_hint(text: str) -> str:
    for name, pat in _CMP_HINTS:
        if pat.search(text):
            return name
    return ""


def _predicates_from_writers(writers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift guard strings into predicate rows with a feature_hint when possible.

    This is a durable, queryable projection for TG feature engineering,
    not a full predicate normalizer.
    """
    seen: set[tuple] = set()
    out = []
    for w in writers:
        for g in w.get("guards") or []:
            text = str(g).strip()
            if not text:
                continue
            key = (w.get("file"), w.get("line"), text)
            if key in seen:
                continue
            seen.add(key)
            hint = _feature_hint(text)
            out.append({
                "id": f"P{len(out):04d}",
                "file": w.get("file"),
                "line": w.get("line"),
                "function": w.get("function"),
                "condition": text,
                "fields": [_leaf(w.get("path") or "")],
                "feature_hint": hint,
            })
    return out


def load_host_codemap(
    uo_root: str | Path,
    *,
    architecture: str = "",
    op_name: str = "",
) -> dict[str, Any]:
    """Load the TG host view from the ``.uo`` product blob only."""
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    root = Path(uo_root).expanduser().resolve()
    product: Path | None
    if root.is_file() and root.suffix == ".uo":
        product = root
    else:
        product = find_uo_product(root, op_name=op_name, architecture=architecture)
        if product is None and is_product_architecture(root.name) and root.parent.name == ".ascendc-codemap":
            product = find_uo_product(
                root.parent.parent,
                op_name=op_name,
                architecture=architecture or root.name,
            )
    if product is None or product.suffix != ".uo":
        return {}
    from ascendc_codemap_mcp.engine.store.reader import load_production_view

    for key in ("ir/tg_host_view.yaml", "tg_host_view"):
        blob = load_production_view(product, key)
        if isinstance(blob, dict) and blob:
            return blob
    return {}


def migrate_load_host_view_from_yaml(uo_root: str | Path) -> dict[str, Any]:
    """Test/migrate helper: read working-tree YAML (not production authority)."""
    root = Path(uo_root)
    for rel in (TG_HOST_VIEW_YAML, CODEMAP_YAML):
        path = root / rel
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def load_tg_host_view(uo_root: str | Path) -> dict[str, Any]:
    """TG shim: materialize host view from ``.uo`` (no YAML/sqlite fallback)."""
    return load_host_codemap(uo_root)


def rebuild_codemap_index(uo_root: str | Path) -> dict[str, Any]:
    """Unlink leftover host_codemap.sqlite. Production authority is ``.uo``."""
    root = Path(uo_root)
    doc = load_host_codemap(root)
    if not doc:
        doc = migrate_load_host_view_from_yaml(root)
    fp = str((doc.get("source") or {}).get("graph_fingerprint") or "")
    legacy = root / CODEMAP_SQLITE
    if legacy.is_file():
        try:
            legacy.unlink()
        except OSError:
            pass
    return {
        "ok": True,
        "mode": "uo",
        "field_rows": len(doc.get("fields") or []),
        "predicate_rows": len(doc.get("predicates") or []),
        "graph_fingerprint": fp,
    }


@dataclass
class QueryResult:
    """Uniform Codemap query envelope: facts + completeness + evidence."""

    facts: list[Any] = field(default_factory=list)
    completeness: str = "unknown"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": list(self.facts),
            "completeness": self.completeness,
            "evidence": list(self.evidence),
            "fingerprint": self.fingerprint,
            "scope": self.scope,
        }


def default_codemap_completeness(
    *,
    init_profile: str = "",
    closure_mode: str = "",
) -> dict[str, Any]:
    """Product completeness contract stored in KB meta / view blob."""
    del init_profile, closure_mode
    return {
        "schema": "codemap-completeness/v1",
        "host": {
            "functions": {
                "mode": "host_ir",
                "entry_roots_complete": True,
                "call_closure": "partial",
            },
            "writes": "complete",
            "reads": "complete",
        },
        "kernel": {
            "completeness": "partial",
            "dtype_variants": "one",
        },
        "macros": {"completeness": "partial"},
        "lemma_certificate": {
            "assignment_sites_complete": False,
            "call_closure_complete": False,
        },
    }


class CodemapQuery:
    """Unified read API over the ``.uo`` CodeMap product (host view + graph)."""

    def __init__(
        self,
        uo_root: str | Path,
        *,
        architecture: str = "",
        op_name: str = "",
    ):
        from ascendc_codemap_mcp.engine.store.reader import (
            find_uo_product,
            load_production_view,
            read_codemap,
            read_meta,
        )

        root = Path(uo_root).expanduser().resolve()
        if root.is_file() and root.suffix == ".uo":
            product = root
        else:
            product = find_uo_product(root, op_name=op_name, architecture=architecture)
        if product is None or product.suffix != ".uo":
            raise FileNotFoundError(
                f"no .uo product under {root}; expected "
                ".ascendc-codemap/<arch>/<op>.<arch>.uo"
            )
        self.product = product
        self.root = product.parent
        self.db = product
        self._cm = read_codemap(product)
        view = load_production_view(product, "ir/tg_host_view.yaml") or load_production_view(
            product, "tg_host_view"
        )
        self._view: dict[str, Any] = view if isinstance(view, dict) else {}
        self._mode = "uo"
        meta = read_meta(product)
        self._fingerprint = str(
            (self._view.get("source") or {}).get("graph_fingerprint")
            or meta.get("graph_fingerprint")
            or self._cm.meta.get("graph_fingerprint")
            or ""
        )
        completeness = load_production_view(product, "codemap/completeness.yaml")
        self._completeness = (
            completeness
            if isinstance(completeness, dict)
            else default_codemap_completeness()
        )

    def _fields_payload(self) -> list[dict[str, Any]]:
        from ascendc_codemap_mcp.engine.ir.entity import EntityKind

        view_fields = [f for f in (self._view.get("fields") or []) if isinstance(f, dict)]
        by_name: dict[str, dict[str, Any]] = {}
        for field in view_fields:
            name = str(field.get("name") or "")
            if name:
                by_name[name] = dict(field)
        for kind in (EntityKind.TILING_FIELD, EntityKind.FIELD, EntityKind.TILING_KEY):
            for ent in self._cm.by_kind(kind):
                writers = list(
                    ent.attrs.get("host_writer_sites")
                    or ent.attrs.get("producer_sites")
                    or ent.attrs.get("writers")
                    or []
                )
                reads = list(ent.attrs.get("reads") or [])
                row = by_name.setdefault(
                    ent.name,
                    {
                        "name": ent.name,
                        "kind": ent.kind_name(),
                        "exactness": ent.attrs.get("exactness") or "",
                        "grade": ent.attrs.get("grade") or "",
                        "writers": [],
                        "reads": [],
                        "entity_id": ent.id,
                    },
                )
                if writers and not row.get("writers"):
                    row["writers"] = writers
                if reads and not row.get("reads"):
                    row["reads"] = reads
        return list(by_name.values())

    def _result(
        self,
        facts: Iterable[Any],
        *,
        completeness: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        scope: str = "",
    ) -> QueryResult:
        return QueryResult(
            facts=list(facts),
            completeness=completeness or "unknown",
            evidence=list(evidence or []),
            fingerprint=self._fingerprint,
            scope=scope,
        )

    def completeness(self, scope: str = "") -> QueryResult:
        """Return the stored completeness contract (optionally scoped)."""
        payload: Any = self._completeness
        level = "partial"
        if scope:
            parts = scope.split(".")
            cur: Any = self._completeness
            for part in parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            payload = cur
            if isinstance(cur, str):
                level = cur
            elif isinstance(cur, dict):
                level = str(cur.get("completeness") or cur.get("call_closure") or "partial")
        else:
            lemma = self._completeness.get("lemma_certificate") or {}
            if all(
                bool(lemma.get(k))
                for k in (
                    "assignment_sites_complete",
                    "call_closure_complete",
                    "alias_state_exact",
                    "macro_context_complete",
                )
            ):
                level = "complete"
            else:
                level = "partial"
        return self._result(
            [payload] if payload is not None else [],
            completeness=level,
            scope=scope or "codemap",
        )

    def fields(self) -> list[dict[str, Any]]:
        """All host-view fields with writers (and attached guards)."""
        return list(self._fields_payload())

    def writers_of(self, symbol: str) -> list[dict[str, Any]]:
        needle = str(symbol or "").strip()
        out: list[dict[str, Any]] = []
        for field in self._fields_payload():
            name = str(field.get("name") or "")
            if needle and needle not in name and needle not in str(field.get("path") or ""):
                continue
            for writer in field.get("writers") or []:
                if isinstance(writer, dict):
                    out.append(dict(writer))
        return out

    def writers(self, symbol: str) -> QueryResult:
        host = (self._completeness.get("host") or {}).get("writes") or "partial"
        facts = self.writers_of(symbol)
        return self._result(facts, completeness=str(host), scope=f"writers:{symbol}")

    def guards_at(self, file: str, line: int) -> list[str]:
        target = str(file or "").replace("\\", "/")
        out: list[str] = []
        for field in self._fields_payload():
            for writer in field.get("writers") or []:
                if not isinstance(writer, dict):
                    continue
                wfile = str(writer.get("file") or "").replace("\\", "/")
                if target and target not in wfile:
                    continue
                if int(writer.get("line") or 0) != int(line):
                    continue
                for g in writer.get("guards") or []:
                    if g and str(g) not in out:
                        out.append(str(g))
        return out

    def guards(self, file: str, line: int) -> QueryResult:
        facts = [{"guard": g} for g in self.guards_at(file, line)]
        return self._result(facts, completeness="partial", scope=f"guards:{file}:{line}")

    def reads_of(self, field: str) -> list[dict[str, str]]:
        needle = str(field or "").strip()
        out: list[dict[str, str]] = []
        for item in self._fields_payload():
            if needle and str(item.get("name") or "") != needle:
                continue
            for row in item.get("reads") or []:
                if isinstance(row, dict):
                    out.append({"var": str(row.get("var") or ""), "root": str(row.get("root") or "")})
                elif isinstance(row, str):
                    out.append({"var": row, "root": ""})
        return out

    def readers(self, field: str) -> QueryResult:
        host = (self._completeness.get("host") or {}).get("reads") or "partial"
        return self._result(
            self.reads_of(field), completeness=str(host), scope=f"reads:{field}"
        )

    def roots(self, field: str) -> QueryResult:
        roots = sorted({r.get("root") for r in self.reads_of(field) if r.get("root")})
        host = (self._completeness.get("host") or {}).get("reads") or "partial"
        return self._result(
            [{"root": r} for r in roots],
            completeness=str(host),
            scope=f"roots:{field}",
        )

    def predicates(self, *, feature_hint: str | None = None) -> list[dict[str, Any]]:
        rows = [p for p in (self._view.get("predicates") or []) if isinstance(p, dict)]
        if not rows:
            for field in self._fields_payload():
                for pred in field.get("predicates") or []:
                    if isinstance(pred, dict):
                        rows.append(pred)
        if feature_hint:
            rows = [r for r in rows if str(r.get("feature_hint") or "") == feature_hint]
        return rows

    def _function_ents(self, function: str):
        name = str(function or "").strip()
        if not name:
            return []
        hits = self._cm.by_name(name)
        short = name.rsplit("::", 1)[-1]
        extra = [
            e
            for e in self._cm.entities.values()
            if e.name == short or e.name.endswith(f"::{short}")
        ]
        seen: set[str] = set()
        out = []
        for ent in hits + extra:
            if ent.id not in seen:
                seen.add(ent.id)
                out.append(ent)
        return out

    def callers_of(self, function: str) -> list[dict[str, Any]]:
        """Callers of ``function`` from first-class ``CALLS`` relations."""
        from ascendc_codemap_mcp.engine.ir.relation import RelationKind

        targets = {e.id for e in self._function_ents(function)}
        if not targets:
            return []
        rows: list[tuple] = []
        for rel in self._cm.relations.values():
            if rel.kind_name() != RelationKind.CALLS.value:
                continue
            if rel.dst not in targets:
                continue
            src = self._cm.entities.get(rel.src)
            dst = self._cm.entities.get(rel.dst)
            if not src or not dst:
                continue
            data = json.dumps(rel.attrs or {}, ensure_ascii=False)
            rows.append((rel.src, rel.dst, data, src.name, dst.name))
        return _expand_call_rows(rows, want="caller")

    def callees_of(self, function: str) -> list[dict[str, Any]]:
        from ascendc_codemap_mcp.engine.ir.relation import RelationKind

        sources = {e.id for e in self._function_ents(function)}
        if not sources:
            return []
        rows: list[tuple] = []
        for rel in self._cm.relations.values():
            if rel.kind_name() != RelationKind.CALLS.value:
                continue
            if rel.src not in sources:
                continue
            src = self._cm.entities.get(rel.src)
            dst = self._cm.entities.get(rel.dst)
            if not src or not dst:
                continue
            data = json.dumps(rel.attrs or {}, ensure_ascii=False)
            rows.append((rel.src, rel.dst, data, src.name, dst.name))
        return _expand_call_rows(rows, want="callee")

    def callers(self, function: str) -> QueryResult:
        host = (
            ((self._completeness.get("host") or {}).get("functions") or {}).get(
                "call_closure"
            )
            or "partial"
        )
        return self._result(
            self.callers_of(function), completeness=str(host), scope=f"callers:{function}"
        )

    def callees(self, function: str) -> QueryResult:
        host = (
            ((self._completeness.get("host") or {}).get("functions") or {}).get(
                "call_closure"
            )
            or "partial"
        )
        return self._result(
            self.callees_of(function), completeness=str(host), scope=f"callees:{function}"
        )

    def influence(self, symbol: str, *, limit: int = 64) -> QueryResult:
        """Bounded BFS over outbound relations from nodes matching ``symbol``."""
        needle = str(symbol or "")
        seeds = [
            e.id
            for e in self._cm.entities.values()
            if needle in e.name or needle in e.id
        ][:16]
        seen = set(seeds)
        frontier = list(seeds)
        facts: list[dict[str, Any]] = []
        while frontier and len(facts) < limit:
            cur = frontier.pop(0)
            for rel in self._cm.relations.values():
                if rel.src != cur:
                    continue
                facts.append(
                    {
                        "edge_id": rel.id,
                        "kind": rel.kind_name(),
                        "src": rel.src,
                        "dst": rel.dst,
                    }
                )
                if rel.dst not in seen and len(seen) < limit:
                    seen.add(rel.dst)
                    frontier.append(rel.dst)
                if len(facts) >= limit:
                    break
        return self._result(facts, completeness="partial", scope=f"influence:{symbol}")

    def path(self, src: str, dst: str, *, limit: int = 32) -> QueryResult:
        """Shortest node path via relations (ids or names)."""

        def resolve(token: str) -> str | None:
            if token in self._cm.entities:
                return token
            hits = self._cm.by_name(token)
            if hits:
                return hits[0].id
            for ent in self._cm.entities.values():
                if token in ent.name or token in ent.id:
                    return ent.id
            return None

        start = resolve(src)
        goal = resolve(dst)
        if not start or not goal:
            return self._result([], completeness="unknown", scope=f"path:{src}->{dst}")
        prev: dict[str, str | None] = {start: None}
        queue = [start]
        while queue and len(prev) < limit * 4:
            cur = queue.pop(0)
            if cur == goal:
                break
            for rel in self._cm.relations.values():
                if rel.src != cur:
                    continue
                nxt = rel.dst
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        if goal not in prev:
            return self._result([], completeness="partial", scope=f"path:{src}->{dst}")
        chain: list[str] = []
        cur: str | None = goal
        while cur is not None:
            chain.append(cur)
            cur = prev.get(cur)
        chain.reverse()
        return self._result(
            [{"nodes": chain}],
            completeness="partial",
            scope=f"path:{src}->{dst}",
        )

    def search(self, text: str, *, limit: int = 32) -> QueryResult:
        q = str(text or "")
        facts = [
            {"id": e.id, "kind": e.kind_name(), "name": e.name, "match": "node"}
            for e in self._cm.entities.values()
            if q in e.name or q in e.id
        ]
        return self._result(facts[:limit], completeness="partial", scope=f"search:{text}")

    def source(self, node_id: str) -> QueryResult:
        ent = self._cm.entities.get(node_id)
        if ent is None:
            hits = self._cm.by_name(node_id)
            ent = hits[0] if hits else None
        if ent is None:
            return self._result([], completeness="partial", scope=f"source:{node_id}")
        facts = [
            {
                "id": ent.id,
                "file": ent.file,
                "line_start": ent.line_start,
                "line_end": ent.line_end,
                "snippet": str(ent.attrs.get("snippet") or ""),
            }
        ]
        return self._result(facts, completeness="partial", scope=f"source:{node_id}")


def _expand_call_rows(
    rows: list[tuple], *, want: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src, dst, data_json, src_name, dst_name in rows:
        try:
            data = json.loads(data_json) if data_json else {}
        except json.JSONDecodeError:
            data = {}
        sites = data.get("sites") if isinstance(data.get("sites"), list) else None
        if not sites:
            sites = [data]
        for site in sites:
            if not isinstance(site, dict):
                continue
            out.append(
                {
                    "caller": src_name,
                    "callee": dst_name,
                    "file": site.get("file") or data.get("file") or "",
                    "line": int(site.get("line") or data.get("line") or 0),
                    "guards": list(site.get("guards") or data.get("guards") or []),
                    "args": list(site.get("args") or data.get("args") or []),
                    "receiver": site.get("receiver") or data.get("receiver") or "",
                    "peer": dst_name if want == "callee" else src_name,
                }
            )
    return out


def export_codemap_from_bundle(
    bundle_path: str | Path, uo_root: str | Path
) -> dict[str, Any]:
    """Legacy helper: load a pickled host bundle and export its HostIR.

    Production paths must not call this. Prefer live HostIR from
    ``extract_host_bundle`` / in-process ``_STORE``. Kept for migration
    scripts only.
    """
    import pickle

    path = Path(bundle_path)
    raw = pickle.loads(path.read_bytes())
    host_ir = raw.get("host_ir") if isinstance(raw, dict) else raw
    if host_ir is None:
        return {"ok": False, "error": "bundle has no host_ir"}
    derive = None
    if isinstance(raw, dict):
        hd = raw.get("host_derivation") or {}
        derive = hd.get("fields") if isinstance(hd, dict) else None
    declared = None
    try:
        from testcase_agent.closure import workspace as WS
        sch = WS.schema()
        declared = {
            "count": len(WS.declared()),
            "dims": [
                {"name": d.name, "bw": getattr(d, "bw", 0),
                 "domain": list(getattr(d, "value_domain", []) or [])}
                for d in sch.dims
            ],
        }
    except Exception:
        declared = None
    return export_tg_host_view(
        host_ir, uo_root, derive_fields=derive, declared=declared)
