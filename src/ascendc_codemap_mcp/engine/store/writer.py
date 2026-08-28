# -*- coding: utf-8 -*-
"""Write a CodeMap into ``<op>.<arch>.uo`` (SQLite)."""

from __future__ import annotations

from ascendc_codemap_mcp.engine.paths import CANN_MARKER, cann_root, require_architecture
import json
import os
import posixpath
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import (
    shrink_evidence_attrs,
    summarize_trust,
    validate_trust_records,
)
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.store.schema import SCHEMA_SQL, SCHEMA_VERSION


LEGAL_KEY_FLUSH = 3000


def vacuum_uo_enabled(dest: Path) -> bool:
    """Cold init VACUUMs; interactive update does not unless ``UO_VACUUM_UO=1``."""
    env_raw = str(os.environ.get("UO_VACUUM_UO") or "").strip().lower()
    if env_raw:
        return env_raw in {"1", "true", "yes"}
    return not Path(dest).exists()


class _PathBase:
    """Rewrites every spelling of a location into one operator-relative form.

    Extraction produced four bases at once for the same tree: operator-relative
    (`op_kernel/x.h`), ops-root-relative (`<op>/op_kernel/x.h`), the shared
    sibling directory (`common/x.h`), and absolute. Recall joins a span to
    `source_line` by path, so two spellings of one file are two files and only
    one of them can ever be cited; the basename fallback in
    `clamp_spans_to_file_length` was hiding that.

    The operator directory is the origin because the product describes one
    operator. Shared code lives beside it, so it keeps a leading `../` rather
    than being pushed under a base it is not under.
    """

    #: How far above the operator directory a location can still be named
    #: relatively. Shared code sits one or two levels up (`../common`,
    #: `../../common/include`); past that a path is a different checkout.
    ANCESTOR_LEVELS = 3

    __slots__ = ("rules", "op_seg", "active")

    def __init__(self, dest: Path) -> None:
        self.rules: list[tuple[str, str]] = []
        self.op_seg = ""
        self.active = False
        try:
            if dest.parents[1].name != ".ascendc-codemap":
                return
            op_dir = dest.parents[2]
        except IndexError:
            return

        def key(path: Path) -> str:
            return str(path).replace("\\", "/").rstrip("/").lower() + "/"

        self.op_seg = op_dir.name.lower() + "/"
        self.rules.append((key(op_dir), ""))
        for level in range(1, self.ANCESTOR_LEVELS + 1):
            try:
                self.rules.append((key(dest.parents[2 + level]), "../" * level))
            except IndexError:
                break
        cann = _cann_prefix()
        if cann:
            # Toolkit headers are outside every checkout, so they cannot be made
            # relative. Naming the tree instead of the build machine's drive is
            # what lets the product be read somewhere else.
            self.rules.append((cann, CANN_MARKER))
        # Most specific prefix wins, whatever kind it is. A vendored toolkit can
        # sit under the checkout, and `<cann>/` says more about such a header
        # than a count of `../` does.
        self.rules.sort(key=lambda rule: len(rule[0]), reverse=True)
        self.active = True

    def file(self, path: str) -> str:
        """Canonical form of a whole-string path."""
        text = str(path or "").replace("\\", "/")
        if not text:
            return ""
        if self.active:
            low = text.lower()
            for prefix, repl in self.rules:
                if low.startswith(prefix):
                    text = repl + text[len(prefix) :]
                    break
            else:
                if (
                    not _is_absolute(text)
                    and not text.startswith("../")
                    and not _under_operator(text)
                ):
                    # A bare relative path that names no operator subdirectory
                    # is spelled from an ancestor; the shared tree arrives so.
                    if low.startswith(self.op_seg):
                        text = text[len(self.op_seg) :]
                    else:
                        text = "../" + text
        if ".." in text:
            collapsed = posixpath.normpath(text)
            # normpath turns "" into "." and drops a trailing slash.
            text = "" if collapsed == "." else collapsed
        return text

    def inside(self, text: str) -> str:
        """Canonical form of paths embedded in an arbitrary attr string.

        Attr values are not all pure paths -- some are expressions quoting one --
        so this only rewrites the roots it recognizes and never runs `normpath`
        over the whole string.
        """
        if not self.active or not text:
            return text
        if "/" not in text and "\\" not in text:
            return text
        for prefix, repl in self.rules:
            text = _replace_ci(text, prefix, repl)
        return text


#: Top-level directories that belong to an operator rather than to the tree
#: around it. A relative path starting with one of these is already
#: operator-relative.
_OPERATOR_DIRS = frozenset(
    {
        "op_host",
        "op_kernel",
        "op_api",
        "op_graph",
        "examples",
        "tests",
        "docs",
        "op_def",
    }
)


def _cann_prefix() -> str:
    """`<cann_root>/` lowercased, or ``''`` when the tree cannot be located."""
    try:
        root = cann_root()
    except Exception:
        return ""
    if root is None:
        return ""
    return str(root).replace("\\", "/").rstrip("/").lower() + "/"


def _under_operator(text: str) -> bool:
    return text.split("/", 1)[0].lower() in _OPERATOR_DIRS


def _is_absolute(text: str) -> bool:
    return text.startswith("/") or (len(text) > 1 and text[1] == ":")


def _replace_ci(text: str, needle: str, repl: str) -> str:
    """Replace every case-insensitive occurrence of `needle`, slashes unified."""
    lowered = text.replace("\\", "/").lower()
    if needle not in lowered:
        return text
    out: list[str] = []
    index = 0
    width = len(needle)
    while True:
        at = lowered.find(needle, index)
        if at < 0:
            out.append(text[index:])
            break
        out.append(text[index:at])
        out.append(repl)
        index = at + width
    return "".join(out)


def _write_legal_key_tables(conn: sqlite3.Connection, blob: dict[str, Any]) -> None:
    """Materialize compact legal-key rows as relational postings."""
    rows = blob.get("rows") if isinstance(blob, dict) else None
    dim_order = [str(n) for n in ((blob or {}).get("dim_order") or [])]
    if not isinstance(rows, list) or not rows:
        return
    key_rows: list[tuple[Any, ...]] = []
    dim_rows: list[tuple[Any, ...]] = []
    next_kid = 0

    def _flush() -> None:
        if key_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO legal_key(id, packed, hex, sel_group, status) VALUES (?,?,?,?,?)",
                key_rows,
            )
            key_rows.clear()
        if dim_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO legal_key_dim(key_id, dim, value) VALUES (?,?,?)",
                dim_rows,
            )
            dim_rows.clear()

    for row in rows:
        if isinstance(row, dict):
            kid = int(row.get("index") if row.get("index") is not None else next_kid)
            key_rows.append(
                (
                    kid,
                    str(row.get("tiling_key") or ""),
                    str(row.get("tiling_key_hex") or ""),
                    str(row.get("sel_group_id") or ""),
                    str(row.get("status") or "template_admissible"),
                )
            )
            dims = row.get("dims") if isinstance(row.get("dims"), dict) else {}
            for dim, value in dims.items():
                dim_rows.append((kid, str(dim), "" if value is None else str(value)))
        elif isinstance(row, (list, tuple)) and len(row) >= 4:
            kid = int(row[0] if row[0] is not None else next_kid)
            key_rows.append(
                (
                    kid,
                    str(row[1] or ""),
                    str(row[2] or "") if len(row) > 2 else "",
                    str(row[4] or "") if len(row) > 4 else "",
                    str(row[5] or "template_admissible") if len(row) > 5 else "template_admissible",
                )
            )
            values = row[3] if isinstance(row[3], list) else []
            for i, dim in enumerate(dim_order):
                value = values[i] if i < len(values) else ""
                dim_rows.append((kid, dim, "" if value is None else str(value)))
        else:
            continue
        next_kid = max(next_kid, kid + 1)
        if len(key_rows) >= LEGAL_KEY_FLUSH:
            _flush()
    _flush()


def uo_product_dir(op_root: str | Path, *, architecture: str = "") -> Path:
    """Arch-scoped UO tree that holds the ``*.uo`` product and work files.

    ``architecture`` is required in production; when omitted, fall back to
    pilot path discovery (env / active_run / sole arch).
    """
    from ascendc_codemap_mcp.engine.paths import product_dir

    root = Path(op_root).expanduser().resolve()
    arch = (architecture or "").strip()
    if not arch:
        raise ValueError("ARCHITECTURE_MISSING_IN_RUN_STATE: architecture required")
    return product_dir(root, arch)


def uo_product_path(op_root: str | Path, op_name: str, architecture: str) -> Path:
    safe_op = (op_name or "operator").replace("/", "_").replace("\\", "_")
    arch = require_architecture(architecture)
    return uo_product_dir(op_root, architecture=arch) / f"{safe_op}.{arch}.uo"


def _trust_rows(codemap: CodeMap) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ent in codemap.entities.values():
        row = dict(ent.attrs)
        row["id"] = ent.id
        row["status"] = ent.status
        rows.append(row)
    for rel in codemap.relations.values():
        row = dict(rel.attrs)
        row["id"] = rel.id
        row["status"] = rel.status
        rows.append(row)
    return rows


def _drop_unproven_direct_selection_edges(codemap: CodeMap) -> int:
    """Prevent legacy Cartesian TilingKey→Kernel edges entering the product."""
    removed: list[str] = []
    for rid, rel in list(codemap.relations.items()):
        if rel.kind_name() not in {RelationKind.SELECTS.value, RelationKind.LAUNCHES.value}:
            continue
        src = codemap.entities.get(rel.src)
        dst = codemap.entities.get(rel.dst)
        if not src or not dst:
            continue
        if src.kind_name() != EntityKind.TILING_KEY.value or dst.kind_name() != EntityKind.KERNEL.value:
            continue
        if rel.attrs.get("provenance") or rel.attrs.get("legacy_kind") or rel.attrs.get("evidence"):
            continue
        removed.append(rid)
    for rid in removed:
        codemap.relations.pop(rid, None)
    if removed:
        codemap.meta["dropped_unproven_direct_key_kernel_edges"] = len(removed)
    return len(removed)


def _canonicalize_views(codemap: CodeMap, views: dict[str, Any] | None) -> dict[str, Any]:
    """Return only projections that are safe to stamp with current authority."""
    from ascendc_codemap_mcp.engine.canonical_tpl_projection import TPL_VIEW_NAMES, project_tpl_views_from_codemap
    from ascendc_codemap_mcp.engine.projection_provenance import validate_view_against_codemap
    from ascendc_codemap_mcp.engine.tg_views import finalize_tg_views

    incoming = dict(views or {})
    rebuilt_names = {
        "ir/operator_graph.yaml",
        "ir/tg_host_view.yaml",
        "views/kernel.yaml",
        "views/tilingdata.yaml",
        *TPL_VIEW_NAMES,
    }
    seed: dict[str, Any] = {}

    tpl_views = project_tpl_views_from_codemap(codemap)
    if tpl_views:
        seed.update(tpl_views)
    elif any(name in incoming for name in TPL_VIEW_NAMES):
        # A caller supplied a materialized TPL domain but the canonical graph
        # cannot reproduce it.  Never drop/re-stamp that D silently: require a
        # TPL re-extract/backfill so the authority becomes self-contained.
        raise ValueError(
            "TPL_CANONICAL_FACTS_INCOMPLETE: caller supplied TPL views but canonical "
            "TILING_KEY/ARGS_SEL TEMPLATE facts cannot rebuild them"
        )

    for name, payload in incoming.items():
        if name in rebuilt_names or name == "summary":
            continue
        check = validate_view_against_codemap(payload, codemap)
        if not check.get("ok"):
            raise ValueError(
                "VIEW_STALE_ON_COMMIT: extension projection cannot be proven fresh: "
                + json.dumps({"name": name, **check}, ensure_ascii=False)[:800]
            )
        seed[name] = payload

    return finalize_tg_views(codemap, existing=seed)


_CORE_VIEW_NAMES = (
    "ir/operator_graph.yaml",
    "ir/tg_host_view.yaml",
    "views/kernel.yaml",
    "views/tilingdata.yaml",
)


def _ensure_graph_identities(codemap: CodeMap) -> None:
    """Compute fingerprint/digest once after a canonical mutation (or pop)."""
    from ascendc_codemap_mcp.engine.projection_provenance import canonical_graph_digest
    from ascendc_codemap_mcp.engine.tg_views import graph_fingerprint

    if not str(codemap.meta.get("graph_fingerprint") or ""):
        codemap.meta["graph_fingerprint"] = graph_fingerprint(codemap)
    if not str(codemap.meta.get("canonical_graph_digest") or ""):
        codemap.meta["canonical_graph_digest"] = canonical_graph_digest(codemap)
    if not str(codemap.meta.get("canonical_revision") or ""):
        codemap.meta["canonical_revision"] = str(codemap.meta["canonical_graph_digest"])[:16]


def _views_match_current_identity(views: dict[str, Any] | None, codemap: CodeMap) -> bool:
    if not isinstance(views, dict):
        return False
    digest = str(codemap.meta.get("canonical_graph_digest") or "")
    if not digest:
        return False
    for name in _CORE_VIEW_NAMES:
        payload = views.get(name)
        if not isinstance(payload, dict):
            return False
        prov = payload.get("provenance")
        if not isinstance(prov, dict) or str(prov.get("canonical_graph_digest") or "") != digest:
            return False
    return True


_JSON_DUMP = {"ensure_ascii": False, "separators": (",", ":")}


def _attrs_json(
    attrs: dict[str, Any],
    base: "_PathBase | None" = None,
    build_context_id: str = "",
) -> str:
    cleaned: dict[str, Any] = {}
    source = shrink_evidence_attrs(attrs or {}, build_context_id=build_context_id)
    for key, value in source.items():
        if key == "type_text":
            continue
        cleaned[key] = _trim_attr(value, key=key, base=base)
    return json.dumps(cleaned, default=str, **_JSON_DUMP)


#: Attrs whose value is a predicate someone downstream parses. Clipping these
#: does not shorten a label, it produces an expression that is still
#: syntactically inviting but no longer means what the source said --
#: `OP_CHECK_IF(..., return ge::GRAPH` reads as a condition and is not one.
_KEEP_ATTR_KEYS = frozenset(
    {
        "rhs",
        "condition",
        "expression",
        "predicate",
        "finite_predicate",
        "guards",
    }
)


#: Attr keys whose value is a whole path, so it gets the same canonical form as
#: the `file` column rather than only having its root rewritten.
_PATH_ATTR_KEYS = frozenset(
    {
        "file",
        "path",
        "decl_file",
        "callee_decl_file",
        "header",
        "source_file",
        "defined_in",
        "include",
    }
)


def _trim_attr(
    value: Any, *, depth: int = 0, key: str = "", base: "_PathBase | None" = None
) -> Any:
    if depth > 4:
        return value
    if isinstance(value, str):
        # Attrs carry as many locations as the columns do (`callee_decl_file`,
        # `write_sites[].file`, `sites[].file`), and a reader cannot join two
        # spellings of one file. Keys known to be whole paths get the full
        # canonical form; anything else only has recognized roots rewritten, so
        # an expression that quotes a path is not mangled.
        if base is not None and base.active:
            value = base.file(value) if key in _PATH_ATTR_KEYS else base.inside(value)
        if key in _KEEP_ATTR_KEYS:
            return value
        if len(value) > 400:
            return value[:400]
        return value
    if isinstance(value, list):
        child_key = "rhs" if key == "packing_value_sites" else key
        return [
            _trim_attr(item, depth=depth + 1, key=child_key, base=base)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(k): _trim_attr(v, depth=depth + 1, key=str(k), base=base)
            for k, v in value.items()
        }
    return value


def _persistable_cm_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    drop = {"walk_cache_stats", "gaps", "quality"}
    for key, value in dict(meta or {}).items():
        if key == "kernel_root_trace" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in drop}
        out[key] = value
    return out


def _entity_snippet(ent: Any) -> str:
    kind_name = ""
    if hasattr(ent, "kind_name"):
        kind_name = str(ent.kind_name() or "")
    if kind_name == "BRANCH":
        return ""
    existing = str(getattr(ent, "attrs", {}).get("snippet") or "")[:400]
    if existing.strip():
        return existing.strip()[:400]
    file = str(getattr(ent, "file", "") or "")
    line = int(getattr(ent, "line_start", 0) or 0)
    if not file or line <= 0:
        return ""
    try:
        from ascendc_codemap_mcp.engine.passes.source_text_cache import cached_snippet

        return cached_snippet(file, line)
    except Exception:
        return ""


def detect_source_revision(root: str | Path) -> str:
    """Return ``git rev-parse HEAD`` for ``root``, or empty when git is unavailable."""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        return ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return ""
    if proc.returncode:
        return ""
    return str(proc.stdout or "").strip()


def write_codemap(
    codemap: CodeMap,
    path: str | Path,
    *,
    views: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist CodeMap to ``path`` (``.uo`` SQLite). Overwrites atomically.

    Commit order: canonical mutation → clear stale graph identity → rebuild all
    canonical projections → semantic digest/provenance validation → atomic
    replace. Caller-provided materialized views never acquire a new digest
    unless rebuilt or already proven against the current canonical graph.
    """
    dest = Path(path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    removed = _drop_unproven_direct_selection_edges(codemap)
    trust_rows = _trust_rows(codemap)
    trust_errors = validate_trust_records(trust_rows)
    if trust_errors:
        raise ValueError(
            "TRUST_INVARIANT: lexical/heuristic facts cannot be derived/authoritative: "
            + "; ".join(trust_errors[:5])
        )
    trust_summary = summarize_trust(trust_rows)
    codemap.meta["trust_summary"] = trust_summary
    # A CodeMap read from a previous product may carry its former identities.
    # Any canonical mutation invalidates those values; projection finalization
    # below recomputes all three from the post-mutation graph.
    if removed:
        for identity_key in ("graph_fingerprint", "canonical_graph_digest", "canonical_revision"):
            codemap.meta.pop(identity_key, None)

    from ascendc_codemap_mcp.engine.projection_provenance import (
        stamp_provenance,
        validate_view_against_codemap,
    )

    _ensure_graph_identities(codemap)
    if removed == 0 and _views_match_current_identity(views, codemap):
        finalized = dict(views or {})
    else:
        finalized = _canonicalize_views(codemap, views)

    from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap

    if removed == 0 and summary:
        strict_summary = dict(summary)
    else:
        strict_summary = dict(summary or {})
        strict_summary.update(dict(audit_codemap(codemap)["summary"]))
    strict_summary = stamp_provenance(strict_summary, codemap)

    stale: list[dict[str, Any]] = []
    for name, payload in finalized.items():
        check = validate_view_against_codemap(payload, codemap)
        if not check.get("ok"):
            stale.append({"name": name, **check})
    summary_check = validate_view_against_codemap(strict_summary, codemap)
    if not summary_check.get("ok"):
        stale.append({"name": "summary", **summary_check})
    if stale:
        raise ValueError(
            "VIEW_STALE_ON_COMMIT: projections drifted from canonical before write: "
            + json.dumps(stale[:5], ensure_ascii=False)[:800]
        )

    conn = sqlite3.connect(str(tmp))
    try:
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA journal_mode=OFF")
        conn.executescript(SCHEMA_SQL)
        product_meta: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "trust_model": "v2",
            "authority": "uo",
            "op_name": codemap.op_name,
            "architecture": codemap.architecture,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entity_count": str(len(codemap.entities)),
            "relation_count": str(len(codemap.relations)),
            **{k: _jsonable(v) for k, v in (meta or {}).items()},
            **{f"cm_{k}": _jsonable(v) for k, v in _persistable_cm_meta(codemap.meta).items()},
        }
        if not str(product_meta.get("source_revision") or "").strip():
            try:
                if dest.parents[1].name == ".ascendc-codemap":
                    revision = detect_source_revision(dest.parents[2])
                    if revision:
                        product_meta["source_revision"] = revision
            except IndexError:
                pass
        _write_meta(
            conn,
            product_meta,
        )
        path_base = _PathBase(dest)
        # One value for the whole product, and `meta` already carries it. Rows
        # store it only when they disagree with the product they are in.
        context_id = str(product_meta.get("cm_build_context_id") or "")
        variants = [
            (
                ent.id,
                ent.name,
                codemap.architecture,
                _attrs_json(ent.attrs, path_base, context_id),
            )
            for ent in codemap.by_kind("BUILD_VARIANT")
        ]
        if variants:
            conn.executemany(
                "INSERT OR REPLACE INTO build_variant(id, name, architecture, data) VALUES (?,?,?,?)",
                variants,
            )
        entity_rows = []
        file_rows = []
        span_rows = []
        for ent in codemap.entities.values():
            file_path = path_base.file(ent.file)
            entity_rows.append(
                (
                    ent.id,
                    ent.kind_name(),
                    ent.name,
                    ent.status,
                    float(ent.confidence),
                    file_path,
                    int(ent.line_start),
                    int(ent.line_end),
                    _attrs_json(ent.attrs, path_base, context_id),
                )
            )
            if file_path:
                file_rows.append(
                    (file_path, file_path, "", ent.attrs.get("layer") or "")
                )
                snippet = _entity_snippet(ent)
                if snippet and int(ent.line_start or 0) > 0:
                    span_rows.append(
                        (
                            f"span:{ent.id}",
                            ent.id,
                            file_path,
                            int(ent.line_start),
                            int(ent.line_end or ent.line_start),
                            snippet,
                        )
                    )
        if entity_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO entity("
                "id, kind, name, status, confidence, file, line_start, line_end, data"
                ") VALUES (?,?,?,?,?,?,?,?,?)",
                entity_rows,
            )
        if file_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO file(id, path, sha256, role) VALUES (?,?,?,?)",
                file_rows,
            )
        if span_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO source_span("
                "id, entity_id, file, line_start, line_end, snippet"
                ") VALUES (?,?,?,?,?,?)",
                span_rows,
            )
        rel_rows = [
            (
                rel.id,
                rel.kind_name(),
                rel.src,
                rel.dst,
                rel.status,
                float(rel.confidence),
                _attrs_json(rel.attrs, path_base, context_id),
            )
            for rel in codemap.relations.values()
            if rel.src in codemap.entities and rel.dst in codemap.entities
        ]
        if rel_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO relation("
                "id, kind, src, dst, status, confidence, data"
                ") VALUES (?,?,?,?,?,?,?)",
                rel_rows,
            )
        from ascendc_codemap_mcp.engine.query.legal_key_cache import compact_legal_key_blob

        from ascendc_codemap_mcp.engine.store.view_projection import NOT_SHIPPED

        view_rows = []
        for name, payload in finalized.items():
            # Validated above with the rest, then dropped rather than embedded:
            # the reader projects these from the graph on demand. Skipping the
            # insert instead of deleting it afterwards keeps 4.4 MB of pages
            # from being written and then freed.
            if name in NOT_SHIPPED:
                continue
            stored = payload
            if name == "tiling/legal_key_index.jsonl" and isinstance(payload, dict):
                stored = compact_legal_key_blob(payload)
            view_rows.append(
                (
                    str(name),
                    str((stored or {}).get("schema") or "") if isinstance(stored, dict) else "",
                    json.dumps(stored, default=str, **_JSON_DUMP),
                )
            )
        view_rows.append(
            (
                "summary",
                "codemap-summary/v1",
                json.dumps(strict_summary, default=str, **_JSON_DUMP),
            )
        )
        conn.executemany(
            "INSERT OR REPLACE INTO view_blob(name, schema_id, data) VALUES (?,?,?)",
            view_rows,
        )
        legal_blob = finalized.get("tiling/legal_key_index.jsonl")
        if isinstance(legal_blob, dict):
            _write_legal_key_tables(conn, compact_legal_key_blob(legal_blob))
        # Acceleration tables are part of the product contract, not a bonus: the
        # query path branches on their presence, so a silent build failure
        # degrades every later answer instead of failing here where it is cheap
        # to see. Errors propagate on purpose.
        from ascendc_codemap_mcp.engine.store.accel import (
            ACCEL_VERSION,
            build_name_leaf,
            build_source_fts,
            build_source_line,
            build_template_blocks,
            clamp_spans_to_file_length,
            patch_sel_lines,
        )

        # Same derivation `_PathBase` used above; if there is no operator tree
        # to be relative to there is none to index either.
        op_root = dest.parents[3] if path_base.active else None

        accel_stats: dict[str, Any] = {}
        # source_line first: it is the only record of how long each file is, and
        # the span clamp has to run before anything copies a span out of
        # `entity` (patch_sel_lines writes spans, build_template_blocks reads
        # them).
        if op_root is None:
            # Without the operator root there is no tree to index. Record it so
            # the gap is visible instead of looking like an empty operator.
            accel_stats["source_line_skipped"] = "op_root_unresolved"
        else:
            files, lines = build_source_line(
                conn, op_root, architecture=codemap.architecture
            )
            accel_stats["source_files"] = files
            accel_stats["source_lines"] = lines
            for key, value in clamp_spans_to_file_length(conn).items():
                accel_stats[f"span_{key}"] = value
            # After source_line is final: the index is keyed on its rowids.
            accel_stats["source_fts"] = build_source_fts(conn)
        accel_stats["name_leaf_rows"] = build_name_leaf(conn)
        accel_stats["sel_lines_patched"] = patch_sel_lines(conn, op_root)
        accel_stats["template_blocks"] = build_template_blocks(conn)
        accel_stats["accel_version"] = ACCEL_VERSION
        conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            [(f"accel_{k}", str(v)) for k, v in accel_stats.items()],
        )
        conn.commit()
        vacuum = vacuum_uo_enabled(dest)
        if vacuum:
            old_isolation = conn.isolation_level
            conn.isolation_level = None
            try:
                conn.execute("VACUUM")
            finally:
                conn.isolation_level = old_isolation
    finally:
        conn.close()

    # Query connections are pooled per (path, thread). Those handles block
    # unlink on Windows, so whoever replaces the product must release them
    # rather than relying on callers to remember.
    from ascendc_codemap_mcp.engine.store.reader import close_uo_connections

    close_uo_connections(dest)
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)
    return {
        "ok": True,
        "path": str(dest),
        "schema": SCHEMA_VERSION,
        "entities": len(codemap.entities),
        "relations": len(codemap.relations),
    }


def _write_meta(conn: sqlite3.Connection, items: dict[str, Any]) -> None:
    for key, value in items.items():
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            (str(key), _jsonable(value)),
        )


def _jsonable(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
