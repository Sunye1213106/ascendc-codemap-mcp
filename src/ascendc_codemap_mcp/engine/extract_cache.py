# -*- coding: utf-8 -*-
"""Scope/content fingerprints for deterministic incremental extraction.

The cache contract is semantic: unchanged source/build identity may reuse a
content-addressed TU result, while changed input must invalidate it. Wall-clock
budgets are deliberately not part of UO correctness.

Fingerprints use a two-layer stamp: ``mtime_ns`` + ``size`` first, then sha256
only for files whose stamp drifted. ``/uo-update`` consumes the same per-file
delta so detect/plan/extract do not disagree on which confirmed sources moved.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.tu_cache import sha256_file, tu_cache_dir, uo_cache_root

_META_NAME = "extract_fingerprint.yaml"
_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
_EXTRACTOR_VERSION = "dep-include-v1"


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    return mtime_ns, int(st.st_size)


def _index_stamps(rows: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, dict):
        items = rows.items()
        for key, value in items:
            if isinstance(value, dict):
                path = str(value.get("path") or key).replace("\\", "/")
                out[path] = value
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "").replace("\\", "/")
        if path:
            out[path] = row
    return out


def collect_source_stamps(
    project_root: Path,
    rel_paths: list[str],
    *,
    previous: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-file mtime/size/sha rows. Matching stamps reuse the stored sha."""
    root = Path(project_root).expanduser().resolve()
    prev_rows = (previous or {}).get("source_stamps") if isinstance(previous, dict) else None
    prev = _index_stamps(prev_rows if prev_rows is not None else previous)
    rows: list[dict[str, Any]] = []
    for rel in sorted({path.replace("\\", "/") for path in rel_paths}):
        path = root / rel
        stamp = _file_stamp(path) if path.is_file() else None
        if stamp is None:
            rows.append(
                {
                    "path": rel,
                    "mtime_ns": 0,
                    "size": 0,
                    "sha": "missing",
                    "hashed": False,
                }
            )
            continue
        mtime_ns, size = stamp
        prior = prev.get(rel) or {}
        prior_sha = str(prior.get("sha") or "")
        try:
            prior_mtime = int(prior.get("mtime_ns") or -1)
            prior_size = int(prior.get("size") or -1)
        except (TypeError, ValueError):
            prior_mtime, prior_size = -1, -1
        if (
            prior_sha
            and prior_sha != "missing"
            and prior_mtime == mtime_ns
            and prior_size == size
        ):
            rows.append(
                {
                    "path": rel,
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "sha": prior_sha,
                    "hashed": False,
                }
            )
        else:
            rows.append(
                {
                    "path": rel,
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "sha": sha256_file(path),
                    "hashed": True,
                }
            )
    return rows


def persist_source_stamps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "mtime_ns": int(row.get("mtime_ns") or 0),
            "size": int(row.get("size") or 0),
            "sha": str(row.get("sha") or ""),
        }
        for row in rows
    ]


def content_fingerprint(
    project_root: Path,
    rel_paths: list[str],
    *,
    previous: dict[str, Any] | None = None,
) -> str:
    """Return a stable hash over sorted ``(relative_path, file_sha)`` pairs."""
    rows = collect_source_stamps(project_root, rel_paths, previous=previous)
    return _stable_hash([[row["path"], row["sha"]] for row in rows])[:32]


def stamp_changed_paths(
    now_rows: list[dict[str, Any]] | None,
    prev_rows: list[dict[str, Any]] | None,
) -> list[str]:
    """Confirmed-source paths whose sha (or presence) drifted."""
    prev_sha = {
        path: str(row.get("sha") or "")
        for path, row in _index_stamps(prev_rows or []).items()
    }
    changed: list[str] = []
    now_paths: set[str] = set()
    for path, row in _index_stamps(now_rows or []).items():
        now_paths.add(path)
        if prev_sha.get(path) != str(row.get("sha") or ""):
            changed.append(path)
    for path in prev_sha:
        if path not in now_paths:
            changed.append(path)
    return sorted(changed)


def _quoted_includes(path: Path) -> list[str]:
    out: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        for inc in _QUOTED_INCLUDE_RE.findall(line):
            out.append(inc.replace("\\", "/"))
    return out


def resolve_quoted_include(inc: str, from_rel: str, all_rels: set[str]) -> str | None:
    needle = str(inc or "").replace("\\", "/")
    if not needle:
        return None
    parent = str(Path(from_rel).parent / needle).replace("\\", "/")
    if parent in all_rels:
        return parent
    if needle in all_rels:
        return needle
    name = Path(needle).name.lower()
    matches = [
        rel
        for rel in all_rels
        if rel.replace("\\", "/").endswith("/" + needle) or Path(rel).name.lower() == name
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches)[0]
    return None


def collect_include_graph(project_root: Path, rel_paths: list[str]) -> dict[str, list[str]]:
    root = Path(project_root).expanduser().resolve()
    rels = sorted({str(p).replace("\\", "/") for p in rel_paths if p})
    all_rels = set(rels)
    graph: dict[str, list[str]] = {}
    for rel in rels:
        found: list[str] = []
        seen: set[str] = set()
        for inc in _quoted_includes(root / rel):
            resolved = resolve_quoted_include(inc, rel, all_rels)
            if resolved and resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
        graph[rel] = found
    return graph


def include_reverse_closure(
    changed: list[str], graph: dict[str, list[str]] | None
) -> list[str]:
    """Expand changed files to TUs that include them (reverse include closure)."""
    seeds = {str(p).replace("\\", "/") for p in (changed or []) if p}
    if not seeds or not graph:
        return sorted(seeds)
    reverse: dict[str, set[str]] = {}
    for src, includes in graph.items():
        src_n = str(src).replace("\\", "/")
        for dst in includes or []:
            reverse.setdefault(str(dst).replace("\\", "/"), set()).add(src_n)
    out = set(seeds)
    queue = list(seeds)
    while queue:
        cur = queue.pop()
        for parent in reverse.get(cur) or ():
            if parent not in out:
                out.add(parent)
                queue.append(parent)
    return sorted(out)


def compute_extract_fingerprint(
    project_root: Path,
    *,
    uo_root: Path | None = None,
    arch: str | None = None,
    build_fingerprint: str = "",
) -> dict[str, Any]:
    """Combine scope identity, source bytes and build identity."""
    from ascendc_codemap_mcp.engine.update.artifacts import current_scope_identity, resolve_uo_root

    root = Path(project_root).expanduser().resolve()
    uo = Path(uo_root) if uo_root is not None else resolve_uo_root(root)
    if arch and not uo_root:
        candidate = root / ".ascendc-codemap" / arch
        if candidate.is_dir():
            uo = candidate
    scope = current_scope_identity(uo)
    rels = list(scope.get("confirmed_sources") or [])
    if not rels:
        # Never fall back to an arch-blind glob — that is how foreign-arch
        # sources leak into confirmed_sources. Callers must finish prepare
        # (Clang-complete scope_set.yaml) before extract fingerprinting.
        raise RuntimeError(
            "SCOPE_CONFIRMED_SOURCES_MISSING: no Clang-confirmed file list under "
            f"{uo}; run prepare until clang_scope_status=complete writes "
            "summary/scope_set.yaml confirmed_source_files"
        )
    from ascendc_codemap_mcp.engine.tu_cache import CACHE_VERSION

    previous = load_extract_fingerprint(uo)
    stamps = collect_source_stamps(root, rels, previous=previous)
    content_fp = _stable_hash([[row["path"], row["sha"]] for row in stamps])[:32]
    extract_fp = _stable_hash(
        {
            "scope_fingerprint": scope.get("scope_fingerprint"),
            "content_fingerprint": content_fp,
            "build_fingerprint": build_fingerprint or "",
            "confirmed_sources": rels,
            "walk_cache_version": CACHE_VERSION,
        }
    )[:32]
    include_graph = collect_include_graph(root, rels)
    persisted = persist_source_stamps(stamps)
    return {
        "scope_fingerprint": scope.get("scope_fingerprint") or "",
        "scope_revision": scope.get("scope_revision") or 0,
        "content_fingerprint": content_fp,
        "build_fingerprint": build_fingerprint or "",
        "extract_fingerprint": extract_fp,
        "confirmed_sources": rels,
        "uo_root": str(uo),
        "source_stamps": persisted,
        "include_graph": include_graph,
        "extractor_version": _EXTRACTOR_VERSION,
        "hashed_files": [row["path"] for row in stamps if row.get("hashed")],
        "reused_stamp_files": [
            row["path"]
            for row in stamps
            if not row.get("hashed") and row.get("sha") != "missing"
        ],
    }


def fingerprint_meta_path(uo_root: Path) -> Path:
    return Path(uo_root) / "cache" / _META_NAME


def load_extract_fingerprint(uo_root: Path) -> dict[str, Any]:
    path = fingerprint_meta_path(uo_root)
    if not path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def store_extract_fingerprint(uo_root: Path, meta: dict[str, Any]) -> Path:
    import yaml

    path = fingerprint_meta_path(uo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["stored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")
    manifest = path.with_name("dependency_manifest.yaml")
    slim = {
        "extractor_version": payload.get("extractor_version") or _EXTRACTOR_VERSION,
        "include_graph": payload.get("include_graph") or {},
        "source_stamps": [
            {"path": row.get("path"), "sha": row.get("sha")}
            for row in (payload.get("source_stamps") or [])
            if isinstance(row, dict)
        ],
        "arch": payload.get("arch") or "",
        "walk_cache_version": payload.get("walk_cache_version") or "",
    }
    try:
        manifest.write_text(
            yaml.safe_dump(slim, allow_unicode=True, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        pass
    return path


def sources_unchanged(
    project_root: Path,
    *,
    uo_root: Path | None = None,
    arch: str | None = None,
    build_fingerprint: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Return whether the stored extraction identity still matches source/build input."""
    now = compute_extract_fingerprint(
        project_root,
        uo_root=uo_root,
        arch=arch,
        build_fingerprint=build_fingerprint,
    )
    uo = Path(now["uo_root"])
    previous = load_extract_fingerprint(uo)
    if not previous:
        return False, now
    unchanged = str(previous.get("extract_fingerprint") or "") == str(now.get("extract_fingerprint") or "")
    now["previous_extract_fingerprint"] = previous.get("extract_fingerprint") or ""
    now["unchanged"] = unchanged
    return unchanged, now


def skip_reextract_for_unchanged_tus(
    project_root: Path,
    *,
    uo_root: Path | None = None,
    arch: str | None = None,
    build_fingerprint: str = "",
) -> dict[str, Any]:
    """Describe the deterministic reuse decision for confirmed translation units."""
    unchanged, meta = sources_unchanged(
        project_root,
        uo_root=uo_root,
        arch=arch,
        build_fingerprint=build_fingerprint,
    )
    rels = list(meta.get("confirmed_sources") or [])
    previous_fp = str(meta.get("previous_extract_fingerprint") or "")
    if unchanged:
        changed: list[str] = []
        kept = list(rels)
    elif not previous_fp:
        changed = list(rels)
        kept = []
    else:
        previous = load_extract_fingerprint(Path(meta["uo_root"]))
        changed = stamp_changed_paths(
            meta.get("source_stamps") or [],
            previous.get("source_stamps") or [],
        )
        graph = meta.get("include_graph") or previous.get("include_graph") or {}
        if not graph:
            graph = collect_include_graph(project_root, rels)
        changed = include_reverse_closure(changed, graph)
        changed_set = set(changed)
        kept = [path for path in rels if path not in changed_set]
    return {
        "skip_reextract": unchanged,
        "unchanged_tus": kept,
        "changed_or_cold": changed,
        "fingerprint": meta,
        "previous_extract_fingerprint": previous_fp,
        "tu_cache_dir": str(tu_cache_dir(project_root, arch)),
        "cache_root": str(uo_cache_root(project_root, arch)),
    }


def align_scoped_changes(
    files: list[dict[str, Any]],
    skip_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep uo-update's change_set on the same stamp delta extract uses.

    Out-of-scope and non-confirmed rows stay for review. Confirmed sources
    follow ``changed_or_cold`` once a previous extract fingerprint exists.
    """
    previous_fp = str(skip_plan.get("previous_extract_fingerprint") or "")
    if not previous_fp:
        return list(files)
    changed = {
        str(path).replace("\\", "/")
        for path in (skip_plan.get("changed_or_cold") or [])
    }
    confirmed = {
        str(path).replace("\\", "/")
        for path in ((skip_plan.get("fingerprint") or {}).get("confirmed_sources") or [])
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").replace("\\", "/")
        if not path:
            continue
        if not item.get("in_scope"):
            out.append(item)
            seen.add(path)
            continue
        if path not in confirmed:
            out.append(item)
            seen.add(path)
            continue
        if path in changed:
            out.append(item)
            seen.add(path)
    for path in sorted(changed):
        if path in seen:
            continue
        out.append(
            {
                "path": path,
                "status": "M",
                "in_scope": True,
                "role": _infer_role(path),
                "suspicious_out_of_scope": False,
            }
        )
        seen.add(path)
    return out


def _infer_role(path: str) -> str:
    try:
        from ascendc_codemap_mcp.engine.update.artifacts import infer_role

        return str(infer_role(path) or "")
    except Exception:  # noqa: BLE001
        return ""
