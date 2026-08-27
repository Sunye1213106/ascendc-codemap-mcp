# -*- coding: utf-8 -*-
"""uo-update helpers: freshness, fingerprints, scope identity (new KB contract)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.yaml_io import read_yaml

SOURCE_SUFFIXES = {".cpp", ".cc", ".c", ".h", ".hpp", ".py", ".cuh", ".cu"}
OPERATOR_PATH_MARKERS = ("op_host", "op_kernel", "op_api", "common", "tiling")

_WIN_GIT_CANDIDATES = (
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)


def change_set_path(uo_root: Path) -> Path:
    return uo_root / "diff" / "change_set.yaml"


def update_plan_path(uo_root: Path) -> Path:
    return uo_root / "summary" / "update_plan.yaml"


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_executable() -> str:
    """Resolve git even when OpenCode/acp inherit a thin PATH (Windows)."""
    explicit = (os.environ.get("GIT_EXECUTABLE") or os.environ.get("GIT") or "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        found_explicit = shutil.which(explicit)
        if found_explicit:
            return found_explicit
    found = shutil.which("git")
    if found:
        return found
    if os.name == "nt":
        for candidate in _WIN_GIT_CANDIDATES:
            if Path(candidate).is_file():
                return candidate
    return "git"


def run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [git_executable(), *args],
            cwd=Path(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None


def git_toplevel(repo_root: Path) -> Path | None:
    proc = run_git(repo_root, ["rev-parse", "--show-toplevel"])
    if proc is None or proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return Path(text) if text else None


def git_operator_scope(op_root: Path) -> tuple[Path, list[str], str]:
    """Run git at the repo root, pathspec-limited to the operator tree.

    Returns ``(git_cwd, pathspec_args, prefix_slash)``. Git name-status from a
    nested operator directory is repo-root relative; CodeMap scope is
    operator-relative, so callers must strip ``prefix_slash``.
    """
    op = Path(op_root).expanduser().resolve()
    top = git_toplevel(op)
    if top is None:
        return op, [], ""
    top = top.resolve()
    try:
        prefix = op.relative_to(top).as_posix().replace("\\", "/").strip("/")
    except ValueError:
        return op, [], ""
    if prefix in {"", "."}:
        return top, [], ""
    return top, ["--", prefix], prefix + "/"


def _strip_git_prefix(path: str, prefix: str) -> str | None:
    posix = path.replace("\\", "/")
    while posix.startswith("./"):
        posix = posix[2:]
    posix = posix.lstrip("/")
    if not prefix:
        return posix
    if posix.startswith(prefix):
        return posix[len(prefix):]
    return None


def git_head(repo_root: Path) -> str:
    proc = run_git(repo_root, ["rev-parse", "HEAD"])
    if proc is None or proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def revision_sha(rev: Any) -> str:
    """Strip a ``<sha>+dirty:<fp>`` head label down to the commit SHA."""
    text = str(rev or "").strip()
    if "+dirty:" in text:
        return text.split("+dirty:", 1)[0].strip()
    return text


def parse_name_status(stdout: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][:1].upper()
        path = parts[-1].replace("\\", "/").strip()
        if status and path:
            rows.append((status, path))
    return rows


def _git_failed_hard(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    err = f"{proc.stderr or ''} {proc.stdout or ''}".lower()
    return proc.returncode == 128 or "not a git repository" in err


def git_diff_name_status(repo_root: Path, *diff_args: str) -> list[tuple[str, str]] | None:
    """Return name-status rows, ``None`` if git is missing / not a repo."""
    proc = run_git(repo_root, ["diff", "--name-status", *diff_args])
    if proc is None or _git_failed_hard(proc):
        return None
    if proc.returncode != 0:
        return []
    return parse_name_status(proc.stdout)


def git_untracked_files(repo_root: Path, *extra: str) -> list[str] | None:
    proc = run_git(repo_root, ["ls-files", "--others", "--exclude-standard", *extra])
    if proc is None or _git_failed_hard(proc):
        return None
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def is_kb_artifact_path(path: str) -> bool:
    norm = path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if norm.startswith(".git/") or "/.git/" in f"/{norm}" or norm == ".git":
        return True
    if (
        norm == ".ascendc-codemap"
        or norm.startswith(".ascendc-codemap/")
        or "/.ascendc-codemap/" in f"/{norm}"
    ):
        return True
    if (
        norm == ".ascendc-pilot"
        or norm.startswith(".ascendc-pilot/")
        or "/.ascendc-pilot/" in f"/{norm}"
    ):
        return True
    return False


def _merge_name_status(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    by_path: dict[str, str] = {}
    for status, path in rows:
        norm = path.replace("\\", "/").strip()
        if not status or not norm or is_kb_artifact_path(norm):
            continue
        by_path[norm] = status
    return [(status, path) for path, status in sorted(by_path.items())]


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_git_changes(
    repo_root: Path,
    *,
    base: str = "",
    head: str = "",
) -> dict[str, Any]:
    """Committed range plus unstaged / staged / untracked (working tree).

    ``git_ok`` is False when git cannot be executed or ``repo_root`` is not a
    repository. An empty working tree on a real repo is still ``git_ok``.
    """
    git_cwd, pathspec, prefix = git_operator_scope(repo_root)
    git_ok = True
    committed: list[tuple[str, str]] = []
    worktree: list[tuple[str, str]] = []
    base_sha = revision_sha(base)
    head_sha = revision_sha(head) or git_head(repo_root)

    def _scoped(rows: list[tuple[str, str]] | None) -> list[tuple[str, str]] | None:
        if rows is None:
            return None
        out: list[tuple[str, str]] = []
        for status, path in rows:
            rel = _strip_git_prefix(path, prefix)
            if rel:
                out.append((status, rel))
        return out

    if base_sha and head_sha and base_sha != head_sha:
        rows = _scoped(git_diff_name_status(git_cwd, f"{base_sha}..{head_sha}", *pathspec))
        if rows is None:
            git_ok = False
        else:
            committed = rows

    for spec in (("HEAD",), ("--cached",)):
        rows = _scoped(git_diff_name_status(git_cwd, *spec, *pathspec))
        if rows is None:
            git_ok = False
        else:
            worktree.extend(rows)

    untracked = git_untracked_files(git_cwd, *pathspec)
    if untracked is None:
        git_ok = False
    else:
        for path in untracked:
            rel = _strip_git_prefix(path, prefix)
            if rel:
                worktree.append(("A", rel))

    worktree_rows = _merge_name_status(worktree)
    merged = _merge_name_status(committed + worktree)
    worktree_fingerprint = _stable_hash(
        [
            [
                status,
                path,
                "deleted" if status == "D" else _file_digest(Path(repo_root) / path),
            ]
            for status, path in worktree_rows
        ]
    )[:32]
    return {
        "git_ok": git_ok,
        "rows": merged,
        "worktree_rows": worktree_rows,
        "worktree_dirty": bool(worktree_rows),
        "worktree_fingerprint": worktree_fingerprint,
        "head_sha": git_head(repo_root) or head_sha,
        "base_sha": base_sha,
    }


def infer_role(path: str) -> str:
    lower = path.replace("\\", "/").lower()
    if "template_tiling_key" in lower or lower.endswith("tiling_key.h"):
        return "tilingkey"
    if "/op_kernel/" in f"/{lower}" or lower.startswith("op_kernel/"):
        return "kernel"
    if "/op_host/" in f"/{lower}" or lower.startswith("op_host/"):
        return "host"
    if "tiling" in lower:
        return "tiling"
    if "/op_api/" in f"/{lower}" or lower.startswith("op_api/"):
        return "api"
    if "/common/" in f"/{lower}" or lower.startswith("common/"):
        return "common"
    if lower.endswith(".py") and ("cpu_impl" in lower or "golden" in lower):
        return "golden"
    if lower.endswith((".h", ".hpp")):
        return "headers"
    return "other"


def _extract_file_list(doc: dict[str, Any]) -> dict[str, str]:
    raw: list[Any] = []
    frozen = doc.get("frozen_scope")
    if isinstance(frozen, dict):
        raw = (
            frozen.get("confirmed_source_files")
            or frozen.get("confirmed_file_list")
            or frozen.get("files")
            or []
        )
    if not raw:
        raw = (
            doc.get("confirmed_source_files")
            or doc.get("confirmed_file_list")
            or doc.get("files")
            or []
        )
    out: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            out[item.replace("\\", "/")] = infer_role(item)
            continue
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("file") or "").replace("\\", "/")
        if not path:
            continue
        role = str(item.get("role") or "").strip() or infer_role(path)
        out[path] = role
    return out


def _scope_files_from_scope_set(doc: dict[str, Any]) -> dict[str, str]:
    """Prefer explicit confirmed_source_files (op-relative); fall back to files[]."""
    confirmed = doc.get("confirmed_source_files")
    if isinstance(confirmed, list) and confirmed:
        out: dict[str, str] = {}
        for item in confirmed:
            path = str(item or "").replace("\\", "/").strip()
            if path:
                out[path] = infer_role(path)
        if out:
            return out
    return _extract_file_list(doc)


def load_scope_index(uo_root: Path) -> dict[str, str]:
    """Load Clang-confirmed source list. Prefer prepare's summary/scope_set.yaml."""
    man = read_yaml(uo_root / "manifest.yaml")
    run_id = str(man.get("current_run_id") or "")
    candidates: list[Path] = [
        uo_root / "summary" / "scope_set.yaml",
    ]
    if run_id:
        candidates.append(uo_root / "runs" / run_id / "scope" / "scope_set.yaml")
        candidates.append(uo_root / "runs" / run_id / "scope" / "receipt.yaml")
        candidates.append(uo_root / "runs" / run_id / "scope" / "scope_validated.yaml")
    # Pilot run tree mirrors under .ascendc-codemap/<arch>/runs (sibling of uo/).
    pilot_runs = uo_root.parent / "runs"
    if run_id:
        candidates.append(pilot_runs / run_id / "scope" / "scope_set.yaml")
    if pilot_runs.is_dir():
        candidates.extend(sorted(pilot_runs.glob("*/scope/scope_set.yaml"), reverse=True))
    runs = uo_root / "runs"
    if runs.is_dir():
        candidates.extend(sorted(runs.glob("*/scope/scope_set.yaml"), reverse=True))
        candidates.extend(sorted(runs.glob("*/scope/receipt.yaml"), reverse=True))
        candidates.extend(sorted(runs.glob("*/scope/scope_validated.yaml"), reverse=True))
    for path in candidates:
        doc = read_yaml(path)
        if not doc:
            continue
        files = _scope_files_from_scope_set(doc)
        if files:
            return files
    return {}


def current_scope_identity(uo_root: Path) -> dict[str, Any]:
    scope_index = load_scope_index(uo_root)
    rels = sorted(scope_index)
    man = read_yaml(uo_root / "manifest.yaml")
    revision = man.get("scope_revision")
    if revision is None:
        run_id = str(man.get("current_run_id") or "")
        snap = read_yaml(uo_root / "runs" / run_id / "scope" / "scope_snapshot.yaml") if run_id else {}
        revision = snap.get("scope_revision", 0)
    confirmed_sources_hash = _stable_hash(rels)[:32]
    computed_fp = _stable_hash({"confirmed_sources": rels, "scope_revision": revision})[:32]
    return {
        "scope_revision": revision or 0,
        "scope_fingerprint": computed_fp,
        "confirmed_sources_hash": confirmed_sources_hash,
        "confirmed_sources": rels,
    }


def compute_change_set_fingerprint(
    *,
    head_revision: str,
    base_revision: str,
    scope_fingerprint: str,
    changed_files: list[Any],
) -> str:
    paths = sorted(
        {
            str(item.get("path") or "")
            for item in changed_files
            if isinstance(item, dict)
        }
    )
    return _stable_hash(
        {
            "head": head_revision,
            "base": base_revision,
            "scope": scope_fingerprint,
            "files": paths,
        }
    )[:32]


def compute_plan_fingerprint(
    *,
    head_revision: str,
    base_revision: str,
    scope_fingerprint: str,
    change_set_fingerprint: str,
    mode: str,
    affected_layers: list[Any],
) -> str:
    return _stable_hash(
        {
            "head_revision": head_revision,
            "base_revision": base_revision,
            "scope_fingerprint": scope_fingerprint,
            "change_set_fingerprint": change_set_fingerprint,
            "mode": mode,
            "affected_layers": sorted(str(x) for x in (affected_layers or [])),
        }
    )[:32]


def load_change_set_if_fresh(
    uo_root: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    path = change_set_path(uo_root)
    if not path.is_file():
        return None
    doc = read_yaml(path)
    if not doc:
        return None
    manifest = read_yaml(uo_root / "manifest.yaml") or {}
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    base_expected = revision_sha(source.get("revision") or "")
    cs_base = revision_sha(doc.get("base_revision") or "")
    cs_head = revision_sha(doc.get("head_sha") or doc.get("head_revision") or "")
    cs_scope_fp = str(doc.get("scope_fingerprint") or "").strip()
    cs_fp = str(doc.get("change_set_fingerprint") or doc.get("fingerprint") or "").strip()
    if "worktree_fingerprint" not in doc:
        return None
    if not cs_base or not cs_head or not cs_scope_fp or not cs_fp:
        return None
    if base_expected and cs_base != base_expected:
        return None
    root = Path(repo_root) if repo_root is not None else Path(str(source.get("root") or "")).resolve()
    if not str(root) or not root.is_dir():
        return None
    head_now = git_head(root)
    if head_now and cs_head != head_now:
        return None
    inspected = inspect_git_changes(root, base=cs_base, head=cs_head)
    if not inspected.get("git_ok"):
        return None
    if str(doc.get("worktree_fingerprint") or "") != str(inspected.get("worktree_fingerprint") or ""):
        return None
    scope_now = current_scope_identity(uo_root)
    if cs_scope_fp != str(scope_now.get("scope_fingerprint") or ""):
        return None
    expected = compute_change_set_fingerprint(
        head_revision=str(doc.get("head_revision") or cs_head),
        base_revision=str(doc.get("base_revision") or cs_base),
        scope_fingerprint=cs_scope_fp,
        changed_files=list(doc.get("files") or []),
    )
    if cs_fp != expected:
        return None
    return doc


def load_update_plan_if_fresh(
    uo_root: Path,
    *,
    change_set: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = update_plan_path(uo_root)
    if not path.is_file() or change_set is None:
        return None
    doc = read_yaml(path)
    if not doc:
        return None
    for key in (
        "head_revision",
        "base_revision",
        "scope_fingerprint",
        "change_set_fingerprint",
        "plan_fingerprint",
    ):
        if not str(doc.get(key) or "").strip():
            return None
    for key in ("head_revision", "base_revision", "scope_fingerprint", "change_set_fingerprint"):
        if str(doc.get(key) or "").strip() != str(change_set.get(key) or "").strip():
            return None
    scope_now = current_scope_identity(uo_root)
    if str(doc.get("scope_fingerprint") or "") != str(scope_now.get("scope_fingerprint") or ""):
        return None
    expected = compute_plan_fingerprint(
        head_revision=str(doc.get("head_revision") or ""),
        base_revision=str(doc.get("base_revision") or ""),
        scope_fingerprint=str(doc.get("scope_fingerprint") or ""),
        change_set_fingerprint=str(doc.get("change_set_fingerprint") or ""),
        mode=str(doc.get("mode") or ""),
        affected_layers=list(doc.get("affected_layers") or []),
    )
    if str(doc.get("plan_fingerprint") or "") != expected:
        return None
    return doc


def resolve_uo_root(project_root: Path, *, architecture: str = "") -> Path:
    """Arch-scoped working tree: ``.ascendc-codemap/<arch>/``.

    The durable product is ``<op>.<arch>.uo`` inside this directory. Top-level
    ``.ascendc-codemap/uo/`` is not a production path.
    """
    root = Path(project_root).expanduser().resolve()
    arch = (architecture or "").strip()
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    if arch:
        from ascendc_codemap_mcp.engine.paths import product_dir

        return product_dir(root, arch)
    product_root = root / ".ascendc-codemap"
    if product_root.is_dir():
        arch_dirs = sorted(
            p
            for p in product_root.iterdir()
            if p.is_dir() and is_product_architecture(p.name)
        )
        with_product = [p for p in arch_dirs if any(p.glob("*.uo"))]
        chosen = (
            with_product[0]
            if len(with_product) == 1
            else (arch_dirs[0] if len(arch_dirs) == 1 else None)
        )
        if chosen is not None:
            return chosen
    import os

    env_arch = (
        os.environ.get("ASCENDC_CODEMAP_ARCH")
        or os.environ.get("UO_ARCH")
        or os.environ.get("ASCENDC_ARCH")
        or ""
    ).strip()
    return root / ".ascendc-codemap" / (env_arch or "_missing_arch")


def source_content_fingerprint(
    project_root: Path,
    *,
    uo_root: Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Scope identity + confirmed-source content hash (incremental extract).

    Thin wrapper around :func:`uo_init.extract_cache.compute_extract_fingerprint`
    so update/plan callers can share the same fingerprint without importing the
    cache package by name.
    """
    from ascendc_codemap_mcp.engine.extract_cache import compute_extract_fingerprint

    return compute_extract_fingerprint(
        project_root, uo_root=uo_root, arch=arch
    )
