# -*- coding: utf-8 -*-
"""Decide which files an operator's analysis is allowed to look at.

Principle: the user chooses the analysis target (operator + arch); Clang
decides the authoritative source closure. Layout scan bootstraps owned files
and entry TUs; regex include walking is diagnostic only. After
``enrich_with_clang`` succeeds, SHARED files come solely from
``tu.get_includes()`` — never a regex∪clang union.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.source_layout import (
    ARCH_DIR_RE as ARCH_SEGMENT_RE,
    architecture_in_scope,
    architectures_match,
    is_other_arch_path,
    is_variant_architecture,
)

SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp", ".hxx"})
SCANNED_SUFFIXES = SOURCE_SUFFIXES | HEADER_SUFFIXES

# Path segments that never carry production behaviour. Matched per segment, so
# a directory named `st` is dropped while `fast_path.cpp` is not.
EXCLUDED_SEGMENTS = frozenset(
    {"test", "tests", "ut", "st", "example", "examples", "third_party", "build", "dist"}
)

# The four directories the Ascend C layout gives an operator. Their presence is
# what makes a file operator-owned; the file name plays no part.
OP_SEGMENTS = frozenset({"op_api", "op_graph", "op_host", "op_kernel"})

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)

# Only the head of a file is scanned for includes; no translation unit puts
# them past this point, and reading whole kernel headers would dominate.
INCLUDE_SCAN_BYTES = 200_000

ROLE_API = "api"
ROLE_GRAPH = "graph"
ROLE_HOST_DEF = "host_def"
ROLE_HOST_INFERSHAPE = "host_infershape"
ROLE_HOST_TILING = "host_tiling"
ROLE_HOST_OTHER = "host_other"
ROLE_KERNEL_ENTRY = "kernel_entry"
ROLE_KERNEL_OTHER = "kernel_other"
ROLE_HEADER = "header"

# Glob/stem/directory may recall a file. These are not semantic identity.
HINT_TILING = "maybe_tiling"
HINT_DEF = "maybe_def"
HINT_INFERSHAPE = "maybe_infershape"
HINT_APT = "maybe_apt"
HINT_KERNEL = "maybe_kernel_tu"

_HINT_FROM_ROLE = {
    ROLE_HOST_TILING: HINT_TILING,
    ROLE_HOST_DEF: HINT_DEF,
    ROLE_HOST_INFERSHAPE: HINT_INFERSHAPE,
}

SIDE_HOST = "host"
SIDE_KERNEL = "kernel"

# Classification of a path relative to the analysis universe.
KIND_OWNED = "OWNED"
KIND_SHARED = "SHARED"
KIND_EXTERNAL = "EXTERNAL_LIBRARY"
KIND_SYSTEM = "SYSTEM"


@dataclass(frozen=True)
class ScopeFile:
    """One file the analysis may read.

    ``role`` is layout bootstrap (which compiler SIDE / which directory).
    ``role_hints`` are glob/stem/directory candidates, never semantic identity.
    """

    path: Path
    role: str
    side: str
    is_tu: bool
    shared: bool = False
    kind: str = KIND_OWNED
    provenance: str = "layout"
    role_hints: tuple[str, ...] = ()

    @property
    def is_header(self) -> bool:
        return not self.is_tu


@dataclass
class ScopeSet:
    """Every file in scope for one operator on one architecture."""

    op_dir: Path
    workspace_root: Path
    arch_dir: str
    files: list[ScopeFile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = {_key(f.path) for f in self.files}

    def contains(self, path: str | Path | None) -> bool:
        """Membership for a path clang reports, whose spelling we do not control."""
        if not path:
            return False
        return _key(path) in self._index

    def select(
        self,
        *,
        role: str | Iterable[str] | None = None,
        hint: str | Iterable[str] | None = None,
        side: str | None = None,
        tu_only: bool = False,
    ) -> list[ScopeFile]:
        requested = {role} if isinstance(role, str) else (set(role) if role else None)
        hints = {hint} if isinstance(hint, str) else (set(hint) if hint else None)
        layout_roles: set[str] | None = None
        if requested:
            layout_roles = set()
            hint_from_role: set[str] = set()
            for item in requested:
                mapped = _HINT_FROM_ROLE.get(item)
                if mapped:
                    hint_from_role.add(mapped)
                else:
                    layout_roles.add(item)
            if hint_from_role:
                hints = (hints or set()) | hint_from_role
            if not layout_roles:
                layout_roles = None
        out = []
        for f in self.files:
            if layout_roles is not None and hints is not None:
                if f.role not in layout_roles and not hints.intersection(f.role_hints):
                    continue
            elif layout_roles is not None:
                if f.role not in layout_roles:
                    continue
            elif hints is not None:
                if not hints.intersection(f.role_hints):
                    continue
            if side is not None and f.side != side:
                continue
            if tu_only and not f.is_tu:
                continue
            out.append(f)
        return out

    def paths(self, **kw) -> list[Path]:
        return [f.path for f in self.select(**kw)]

    def confirmed_source_files(self) -> list[str]:
        """Project-relative paths of the Clang-confirmed file set (op_dir root)."""
        out: list[str] = []
        for f in self.files:
            try:
                out.append(f.path.relative_to(self.op_dir).as_posix())
            except ValueError:
                try:
                    out.append(f.path.relative_to(self.workspace_root).as_posix())
                except ValueError:
                    out.append(f.path.as_posix())
        return sorted(set(out))

    def to_dict(self) -> dict:
        def rel(p: Path) -> str:
            try:
                return p.relative_to(self.workspace_root).as_posix()
            except ValueError:
                return p.as_posix()

        files = [
            {
                "path": rel(f.path),
                "role": f.role,
                "side": f.side,
                "is_tu": f.is_tu,
                "shared": f.shared,
                "kind": f.kind,
                "provenance": f.provenance,
                "role_hints": list(f.role_hints),
            }
            for f in self.files
        ]
        return {
            "op_dir": self.op_dir.as_posix(),
            "workspace_root": self.workspace_root.as_posix(),
            "arch_dir": self.arch_dir,
            "files": files,
            "confirmed_source_files": self.confirmed_source_files(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScopeSet":
        workspace_root = Path(data["workspace_root"])
        op_dir = Path(data["op_dir"])
        files: list[ScopeFile] = []
        for row in data.get("files") or []:
            rel = row.get("path") or ""
            path = Path(rel)
            if not path.is_absolute():
                path = (workspace_root / rel).resolve()
            shared = bool(row.get("shared"))
            kind = str(row.get("kind") or (KIND_SHARED if shared else KIND_OWNED))
            hints = tuple(str(h) for h in (row.get("role_hints") or ()) if str(h))
            files.append(
                ScopeFile(
                    path=path,
                    role=str(row.get("role") or ""),
                    side=str(row.get("side") or ""),
                    is_tu=bool(row.get("is_tu")),
                    shared=shared,
                    kind=kind,
                    provenance=str(row.get("provenance") or "layout"),
                    role_hints=hints or _role_hints_of(path),
                )
            )
        return cls(
            op_dir=op_dir,
            workspace_root=workspace_root,
            arch_dir=str(data.get("arch_dir") or ""),
            files=files,
            notes=list(data.get("notes") or []),
        )


def _key(path: str | Path) -> str:
    """Comparable form of a path clang reports.

    Clang keeps the include spelling, so ``op_kernel/./foo.h`` and
    ``op_kernel/foo.h`` must be the same file. Case and separators also differ
    on Windows.
    """
    text = str(path or "").replace("\\", "/")
    if not text:
        return ""
    drive, rest = "", text
    if len(text) >= 2 and text[1] == ":":
        drive, rest = text[:2], text[2:]
        if rest.startswith("/"):
            rest = rest[1:]
    parts: list[str] = []
    for part in rest.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    body = "/".join(parts)
    if drive:
        return f"{drive}/{body}".lower()
    if text.startswith("/"):
        return ("/" + body).lower()
    return body.lower()


def _excluded(rel_parts: Iterable[str]) -> bool:
    return bool({p.lower() for p in rel_parts} & EXCLUDED_SEGMENTS)


def _walk_sources(root: Path) -> list[Path]:
    """Source and header files under one tree, skipping test-like folders."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    stack = [root]
    while stack:
        here = stack.pop()
        try:
            entries = list(here.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            if entry.is_dir():
                if name.lower() in EXCLUDED_SEGMENTS:
                    continue
                stack.append(entry)
            elif entry.suffix.lower() in SCANNED_SUFFIXES:
                out.append(entry)
    return sorted(out)


def resolve_workspace(op_dir: Path) -> tuple[Path, str, list[str]]:
    """Where the operator sits and where its domain keeps shared code.

    Returns `(workspace_root, common_rel, notes)`; `common_rel` is empty when
    the domain has no shared tree. Three layouts are tried in the order they
    occur in Ascend C repositories: a `common` beside the operator package, one
    inside it, then one further up.
    """
    notes: list[str] = []
    op_like = (op_dir / "op_host").is_dir() or (op_dir / "op_kernel").is_dir()

    sibling = op_dir.parent / "common"
    if op_like and sibling.is_dir():
        notes.append(f"common_beside_operator: {sibling.as_posix()}")
        return op_dir.parent, "common", notes

    inner = op_dir / "common"
    if inner.is_dir():
        notes.append(f"common_inside_operator: {inner.as_posix()}")
        return op_dir, "common", notes

    cur = op_dir.parent
    for _ in range(3):
        cand = cur.parent / "common"
        if cand.is_dir() and op_like:
            notes.append(f"common_above_operator: {cand.as_posix()}")
            return cur.parent, "common", notes
        if cur.parent == cur:
            break
        cur = cur.parent

    notes.append("no_common_tree")
    return op_dir.parent if op_like else op_dir, "", notes


def _operator_files(op_dir: Path) -> list[Path]:
    """Everything under the operator's four layout directories."""
    out: list[Path] = []
    for segment in sorted(OP_SEGMENTS):
        out.extend(_walk_sources(op_dir / segment))
    return out


def filter_architecture(paths: Iterable[Path], arch_dir: str) -> list[Path]:
    """Drop `archNN` folders other than the requested one.

    A path with no `archNN` segment is architecture-neutral and stays.
    """
    arch = (arch_dir or "").strip()
    if not is_variant_architecture(arch):
        return list(paths)
    out: list[Path] = []
    for path in paths:
        segments = [p.lower() for p in path.parts]
        arch_segments = [p for p in segments if ARCH_SEGMENT_RE.match(p)]
        if not arch_segments or any(architecture_in_scope(p, arch) for p in arch_segments):
            out.append(path)
    return out


def _include_targets(text: str) -> list[str]:
    return [m.group(1).replace("\\", "/").strip() for m in INCLUDE_RE.finditer(text)]


def _read_head(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:INCLUDE_SCAN_BYTES]
    except OSError:
        return ""


def _shared_index(shared: Iterable[Path], workspace_root: Path) -> tuple[dict, dict]:
    by_rel: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in shared:
        try:
            rel = path.relative_to(workspace_root).as_posix()
        except ValueError:
            rel = path.as_posix()
        by_rel[_key(rel)] = path
        by_name.setdefault(path.name.lower(), []).append(path)
    return by_rel, by_name


def _resolve_include(
    include: str, *, source: Path, workspace_root: Path,
    by_rel: dict[str, Path], by_name: dict[str, list[Path]],
) -> Path | None:
    """Which shared file an include line names, if any.

    Three ways, narrowing: resolved against the including file, matched as a
    workspace-relative path, then matched on a trailing path fragment. Never on
    the bare file name -- `matmul.h` exists in three trees here, and picking one
    by name would attach a file the operator does not compile.
    """
    candidate = (source.parent / include).resolve()
    for known in by_name.get(candidate.name.lower(), ()):
        if _key(known) == _key(candidate):
            return known

    try:
        rel = candidate.relative_to(workspace_root).as_posix()
    except (ValueError, OSError):
        rel = ""
    if rel and _key(rel) in by_rel:
        return by_rel[_key(rel)]

    if "/" in include:
        tail = _key(include)
        for known in by_name.get(Path(include).name.lower(), ()):
            if _key(known).endswith("/" + tail):
                return known
    return None


def prune_shared_by_includes(
    operator_files: Iterable[Path], shared: Iterable[Path], workspace_root: Path
) -> list[Path]:
    """Shared files reachable from the operator through `#include`.

    Transitive: a common header that pulls another common header brings it in.
    Bounded by the shared set itself, so the walk cannot leave the domain.
    """
    shared = list(shared)
    if not shared:
        return []
    by_rel, by_name = _shared_index(shared, workspace_root)

    selected: dict[str, Path] = {}
    frontier = list(operator_files)
    seen: set[str] = set()
    while frontier:
        source = frontier.pop()
        marker = _key(source)
        if marker in seen:
            continue
        seen.add(marker)
        for include in _include_targets(_read_head(source)):
            hit = _resolve_include(
                include,
                source=source,
                workspace_root=workspace_root,
                by_rel=by_rel,
                by_name=by_name,
            )
            if hit is not None and _key(hit) not in selected:
                selected[_key(hit)] = hit
                frontier.append(hit)
    return sorted(selected.values())


def _role_hints_of(path: Path) -> tuple[str, ...]:
    """Candidate labels from directory/stem. Never a semantic conclusion."""
    stem = path.stem.lower()
    name = path.name.lower()
    segments = {p.lower() for p in path.parts}
    hints: list[str] = []
    if "op_tiling" in segments or "_tiling" in stem:
        hints.append(HINT_TILING)
    if stem.endswith("_def"):
        hints.append(HINT_DEF)
    if "infershape" in stem or stem.endswith("_proto"):
        hints.append(HINT_INFERSHAPE)
    if name.endswith("_apt.cpp"):
        hints.append(HINT_APT)
    if "op_kernel" in segments and path.suffix.lower() in SOURCE_SUFFIXES:
        hints.append(HINT_KERNEL)
    return tuple(hints)


def _make_scope_file(
    path: Path,
    *,
    role: str | None = None,
    side: str | None = None,
    is_tu: bool | None = None,
    shared: bool = False,
    kind: str = KIND_OWNED,
    provenance: str = "layout",
    role_hints: tuple[str, ...] | None = None,
) -> ScopeFile:
    return ScopeFile(
        path=path,
        role=_role_of(path) if role is None else role,
        side=_side_of(path) if side is None else side,
        is_tu=(path.suffix.lower() in SOURCE_SUFFIXES) if is_tu is None else is_tu,
        shared=shared,
        kind=kind,
        provenance=provenance,
        role_hints=_role_hints_of(path) if role_hints is None else role_hints,
    )


def _copy_scope_file(prev: ScopeFile, **overrides: Any) -> ScopeFile:
    data: dict[str, Any] = {
        "path": prev.path,
        "role": prev.role,
        "side": prev.side,
        "is_tu": prev.is_tu,
        "shared": prev.shared,
        "kind": prev.kind,
        "provenance": prev.provenance,
        "role_hints": prev.role_hints,
    }
    data.update(overrides)
    return ScopeFile(**data)


def _role_of(path: Path) -> str:
    """Layout bootstrap: directory and suffix, not the file's stem identity."""
    suffix = path.suffix.lower()
    segments = {p.lower() for p in path.parts}

    if "op_api" in segments:
        return ROLE_API
    if "op_graph" in segments:
        return ROLE_GRAPH
    if suffix in HEADER_SUFFIXES:
        return ROLE_HEADER
    if "op_kernel" in segments:
        return ROLE_KERNEL_ENTRY
    if "op_host" in segments:
        return ROLE_HOST_OTHER
    return ROLE_HOST_OTHER


def _side_of(path: Path) -> str:
    """Which compiler configuration this file needs.

    Only `op_kernel` is built by the device compiler; the API layer, the
    prototype and tiling are all host code.
    """
    return SIDE_KERNEL if "op_kernel" in {p.lower() for p in path.parts} else SIDE_HOST


def entry_architecture(path: Path) -> str:
    """Which `archNN` a kernel entry compiles, read from what it includes.

    A repository can keep one entry per architecture beside each other. Their
    names carry no reliable marker -- one may end in `_apt`, the other not --
    but each includes only its own architecture's headers. Mixed markers
    (preprocessor-gated ``arch38/`` plus ``*_arch22.h``) yield empty so the
    entry is kept for the requested arch.
    """
    from ascendc_codemap_mcp.engine.source_layout import entry_include_architecture

    return entry_include_architecture(_read_head(path))


def scan(op_dir: str | Path, *, arch_dir: str = "") -> ScopeSet:
    """Everything the analysis may read for one operator on one architecture."""
    op_dir = Path(op_dir).expanduser().resolve()
    workspace_root, common_rel, notes = resolve_workspace(op_dir)

    owned = _operator_files(op_dir)
    owned = filter_architecture(owned, arch_dir)

    shared: list[Path] = []
    if common_rel:
        pool = _walk_sources(workspace_root / common_rel)
        pool = filter_architecture(pool, arch_dir)
        shared = prune_shared_by_includes(owned, pool, workspace_root)
        notes.append(f"shared_available={len(pool)} shared_included={len(shared)}")

    from_common = {_key(p) for p in shared}
    files: list[ScopeFile] = []
    for path in owned + shared:
        is_shared = _key(path) in from_common
        files.append(
            _make_scope_file(
                path,
                shared=is_shared,
                kind=KIND_SHARED if is_shared else KIND_OWNED,
                provenance="include_regex" if is_shared else "layout",
            )
        )

    files = _drop_foreign_arch_entries(files, arch_dir, notes)
    files.sort(key=lambda f: f.path.as_posix())
    return ScopeSet(
        op_dir=op_dir,
        workspace_root=workspace_root,
        arch_dir=arch_dir,
        files=files,
        notes=notes,
    )


def _drop_foreign_arch_entries(
    files: list[ScopeFile], arch_dir: str, notes: list[str]
) -> list[ScopeFile]:
    """Drop kernel TUs that belong to another architecture.

    Files under ``archNN/`` are owned by that folder. Root-level entries
    (``op_kernel/foo.cpp``) are classified from includes. Never drop the last
    remaining kernel TU: some trees keep one ``.cpp`` and put the other arch
    in headers.
    """
    arch = (arch_dir or "").strip()
    if not is_variant_architecture(arch):
        return files
    from ascendc_codemap_mcp.engine.source_layout import path_owned_architecture

    def _owns(path: Path) -> str:
        return path_owned_architecture(path) or entry_architecture(path)

    def _keep_tu(path: Path) -> bool:
        owned = path_owned_architecture(path)
        if owned:
            return architectures_match(owned, arch)
        includes = entry_architecture(path)
        if includes and not architecture_in_scope(includes, arch):
            return False
        return True

    kernel_tus = [f for f in files if f.role == ROLE_KERNEL_ENTRY and f.is_tu]
    kept_tus = [f for f in kernel_tus if _keep_tu(f.path)]
    if kernel_tus and not kept_tus:
        kept_tus = [
            f
            for f in kernel_tus
            if not path_owned_architecture(f.path)
            or architectures_match(path_owned_architecture(f.path), arch)
        ]
        for f in kept_tus:
            notes.append(
                f"kernel_entry_kept_last_tu: {f.path.name} includes another arch "
                "but is the only kernel TU"
            )
    keep_ids = {id(f) for f in kept_tus}
    out: list[ScopeFile] = []
    for f in files:
        if f.role != ROLE_KERNEL_ENTRY or not f.is_tu:
            out.append(f)
            continue
        if id(f) in keep_ids:
            out.append(f)
            continue
        notes.append(f"kernel_entry_other_arch: {f.path.name} builds {_owns(f.path)}")
    return out


def classify_path(
    path: str | Path,
    *,
    op_dir: Path,
    workspace_root: Path,
    foreign_markers: tuple[str, ...] = (
        "cann-asc-devkit",
        "/_cann/",
        "cann-metadef",
        "bisheng",
        "/usr/include",
        "/usr/lib",
    ),
) -> str:
    """Classify a Clang-reported path into OWNED / SHARED / EXTERNAL / SYSTEM.

    SHARED is any project source under ``workspace_root`` that is not under the
    operator directory (``common/``, sibling shared trees, etc.). Directory
    name alone is not required — the compile graph is.
    """
    text = str(path).replace("\\", "/")
    low = text.lower()
    if any(m in low for m in foreign_markers):
        return KIND_EXTERNAL if "cann" in low or "bisheng" in low else KIND_SYSTEM
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    try:
        resolved.relative_to(op_dir.resolve())
        return KIND_OWNED
    except ValueError:
        pass
    try:
        resolved.relative_to(workspace_root.resolve())
        return KIND_SHARED
    except ValueError:
        return KIND_SYSTEM


@dataclass
class ClangIncludeResult:
    """Outcome of asking Clang for one TU's include graph."""

    ok: bool
    paths: list[Path] = field(default_factory=list)
    error: str = ""
    probe: dict[str, Any] | None = None


@dataclass
class ClangEnrichment:
    """Authoritative (or incomplete) Source Scope after Clang include closure."""

    scope: ScopeSet
    status: str  # complete | incomplete | skipped
    tus_expected: int = 0
    tus_parsed: int = 0
    errors: list[str] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.status == "complete"


def load_prepared_scope(op_dir: str | Path, arch_dir: str | None) -> ScopeSet | None:
    """Reuse prepare's clang-complete ``scope_set.yaml`` so extract does not re-parse.

    ``UO_FORCE_SCOPE_ENRICH=1`` ignores the receipt and re-walks includes.
    """
    import os

    import yaml

    from ascendc_codemap_mcp.engine.paths import require_architecture

    if str(os.environ.get("UO_FORCE_SCOPE_ENRICH") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    try:
        arch = require_architecture(arch_dir)
    except Exception:
        return None
    root = Path(op_dir).expanduser().resolve()
    uo = root / ".ascendc-codemap" / arch
    cand_path = uo / "summary" / "scope_candidates.yaml"
    scope_path = uo / "summary" / "scope_set.yaml"
    if not scope_path.is_file():
        return None
    status = ""
    if cand_path.is_file():
        try:
            cand = yaml.safe_load(cand_path.read_text(encoding="utf-8")) or {}
            status = str((cand or {}).get("clang_scope_status") or "")
        except Exception:  # noqa: BLE001
            return None
    if status != "complete":
        return None
    try:
        data = yaml.safe_load(scope_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict) or not data.get("files"):
            return None
        return ScopeSet.from_dict(data)
    except Exception:  # noqa: BLE001
        return None


def _probe_from_parsed_tu(tu: Any, path: str, op_dir: str) -> dict[str, Any]:
    """Same cleanliness score as ``probe_diagnostics``, from an already-parsed TU."""
    from ascendc_codemap_mcp.engine.diag_scope import score_tu_diagnostics

    try:
        diags = list(tu.diagnostics)
    except Exception:  # noqa: BLE001
        diags = []
    scored = score_tu_diagnostics(diags, path, op_dir)
    return {**scored, "skipped_bodies": False}


def clang_include_paths(
    tu_path: str | Path,
    args: list[str],
    *,
    op_dir: str | Path = "",
    side: str = "kernel",
    walk_ctx: Any = None,
) -> ClangIncludeResult:
    """Files Clang actually included while parsing ``tu_path``.

    Requires ``PARSE_DETAILED_PROCESSING_RECORD``. Distinguishes parse failure
    from a successful parse that simply pulled no project headers. Also scores
    diagnostics so prepare does not re-parse the same TU for ``probe``.
    """
    try:
        from clang import cindex
    except ImportError:
        return ClangIncludeResult(ok=False, error="libclang_not_installed")
    path_s = str(tu_path)
    try:
        from ascendc_codemap_mcp.engine.bisheng_attrs import parse_unsaved_kwargs

        idx = cindex.Index.create()
        tu = idx.parse(
            path_s,
            args=list(args),
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            **parse_unsaved_kwargs(op_dir, side=side),
        )
        try:
            from ascendc_codemap_mcp.engine.perf import bump

            bump("clang_tu_parse")
        except Exception:  # noqa: BLE001
            pass
        if walk_ctx is not None:
            try:
                from ascendc_codemap_mcp.engine import tu_cache as _tu_cache

                ast_key = _tu_cache.parse_cache_key(
                    path_s,
                    walk_ctx,
                    side=side,
                    dtype_variant="DT_FLOAT16",
                    parse_flags=list(args),
                )
                _tu_cache.store_live_ast(ast_key, idx, tu, side=side)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        return ClangIncludeResult(
            ok=False, error=f"clang_parse_failed:{Path(tu_path).name}:{str(exc)[:160]}"
        )
    out: list[Path] = []
    seen: set[str] = set()
    try:
        inclusions = tu.get_includes()
    except Exception as exc:  # noqa: BLE001
        return ClangIncludeResult(
            ok=False,
            error=f"clang_get_includes_failed:{Path(tu_path).name}:{str(exc)[:160]}",
        )
    for inc in inclusions:
        try:
            included = inc.include
            if included is None:
                continue
            name = getattr(included, "name", None) or str(included)
        except Exception:  # noqa: BLE001
            continue
        if not name:
            continue
        key = _key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(name))
    probe = _probe_from_parsed_tu(tu, path_s, str(op_dir or ""))
    # Walking stays in extract: prepare-side host+kernel walks share the GIL
    # and inflate kernel ast_walk from ~16s (isolated) to ~70s.
    return ClangIncludeResult(ok=True, paths=out, probe=probe)


def enrich_with_clang(
    scope: ScopeSet,
    *,
    host_args: list[str] | None = None,
    kernel_args: list[str] | None = None,
    host_tus: Iterable[Path] | None = None,
    kernel_tu: Path | None = None,
    walk_ctx: Any = None,
) -> ClangEnrichment:
    """Build Source Scope from Clang's authoritative include closure only.

    Final ``scope.files`` = entry TUs that were parsed ∪ OWNED/SHARED paths
    reported by ``tu.get_includes()``. Layout-owned files that this parse never
    referenced are dropped. Architecture is used only to choose which entry TUs
    to parse (foreign kernel entries are rejected here); common headers that
    Clang actually includes stay in scope.
    """
    notes = list(scope.notes)
    arch = (scope.arch_dir or "").strip().lower()
    jobs: list[tuple[Path, list[str], str]] = []
    for path in host_tus or ():
        if path is not None and Path(path).is_file():
            jobs.append((Path(path), list(host_args or []), SIDE_HOST))
    if kernel_tu is not None and Path(kernel_tu).is_file():
        from ascendc_codemap_mcp.engine.source_layout import path_owned_architecture

        kpath = Path(kernel_tu)
        owns_path = path_owned_architecture(kpath)
        owns_inc = entry_architecture(kpath)
        skip = False
        if owns_path:
            skip = bool(arch and owns_path != arch)
        elif owns_inc and arch and owns_inc != arch:
            notes.append(
                f"kernel_entry_kept_last_tu: {kpath.name} includes {owns_inc}"
            )
        if skip:
            notes.append(
                f"kernel_entry_other_arch: {kpath.name} builds "
                f"{owns_path or owns_inc}; not parsed for clang scope"
            )
        else:
            jobs.append((kpath, list(kernel_args or []), SIDE_KERNEL))

    if not jobs:
        notes.append("clang_enrichment_skipped: no_entry_tus")
        empty = ScopeSet(
            op_dir=scope.op_dir,
            workspace_root=scope.workspace_root,
            arch_dir=scope.arch_dir,
            files=[],
            notes=notes,
        )
        return ClangEnrichment(
            scope=empty,
            status="skipped",
            tus_expected=0,
            tus_parsed=0,
            errors=["no_entry_tus"],
        )

    # Seed only with entry TUs that will be (or were) parsed — not full layout.
    index: dict[str, ScopeFile] = {}
    layout_by_key = {_key(f.path): f for f in scope.files}
    for tu_path, _args, side in jobs:
        key = _key(tu_path)
        prev = layout_by_key.get(key)
        try:
            resolved = tu_path.resolve()
        except OSError:
            resolved = Path(tu_path)
        index[key] = _make_scope_file(
            resolved,
            role=prev.role if prev is not None else None,
            side=prev.side if prev is not None else side,
            is_tu=True,
            shared=False,
            kind=KIND_OWNED,
            provenance="clang_tu",
            role_hints=prev.role_hints if prev is not None else None,
        )

    added = 0
    external_hits = 0
    parsed = 0
    errors: list[str] = []
    probes: list[dict[str, Any]] = []
    op_dir_s = str(scope.op_dir)

    def _parse_one(job: tuple[Path, list[str], str]) -> tuple[Path, str, ClangIncludeResult]:
        tu_path, args, side = job
        return tu_path, side, clang_include_paths(
            tu_path, args, op_dir=op_dir_s, side=side, walk_ctx=walk_ctx
        )

    parsed_jobs: list[tuple[Path, str, ClangIncludeResult]]
    if len(jobs) <= 1:
        parsed_jobs = [_parse_one(j) for j in jobs]
    else:
        from concurrent.futures import ThreadPoolExecutor

        workers = min(len(jobs), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parsed_jobs = list(pool.map(_parse_one, jobs))

    for tu_path, side, result in parsed_jobs:
        if result.probe:
            probes.append(
                {
                    "file": tu_path.as_posix(),
                    "side": side,
                    **result.probe,
                }
            )
        if not result.ok:
            err = result.error or f"clang_includes_failed:{tu_path.name}"
            errors.append(err)
            notes.append(err)
            continue
        parsed += 1
        notes.append(f"clang_includes={len(result.paths)} from {tu_path.name}")
        for inc in result.paths:
            kind = classify_path(
                inc, op_dir=scope.op_dir, workspace_root=scope.workspace_root
            )
            if kind in {KIND_EXTERNAL, KIND_SYSTEM}:
                external_hits += 1
                continue
            if arch and is_other_arch_path(inc, arch):
                # Clang will include a cousin architecture's tiling header
                # for types. That path is not this product's source.
                continue
            key = _key(inc)
            if key in index:
                prev = index[key]
                if prev.provenance == "clang_tu":
                    continue
                if prev.shared or kind == KIND_SHARED:
                    index[key] = _copy_scope_file(
                        prev,
                        shared=True,
                        kind=KIND_SHARED,
                        provenance="clang_include",
                    )
                continue
            try:
                resolved = Path(inc).resolve()
            except OSError:
                resolved = Path(inc)
            is_shared = kind == KIND_SHARED
            layout_prev = layout_by_key.get(key)
            index[key] = _make_scope_file(
                resolved,
                role=(
                    layout_prev.role
                    if layout_prev is not None
                    else (_role_of(resolved) if not is_shared else ROLE_HEADER)
                ),
                side=(
                    layout_prev.side
                    if layout_prev is not None
                    else (side if is_shared else _side_of(resolved))
                ),
                is_tu=(
                    layout_prev.is_tu
                    if layout_prev is not None
                    else resolved.suffix.lower() in SOURCE_SUFFIXES
                ),
                shared=is_shared,
                kind=kind,
                provenance="clang_include",
                role_hints=(
                    layout_prev.role_hints
                    if layout_prev is not None
                    else (_role_hints_of(resolved) if not is_shared else ())
                ),
            )
            added += 1

    status = "complete" if parsed == len(jobs) and not errors else "incomplete"
    files = sorted(index.values(), key=lambda f: f.path.as_posix())
    notes.append(
        f"clang_scope_status={status} tus_parsed={parsed}/{len(jobs)} "
        f"clang_shared_added={added} clang_external_seen={external_hits} "
        f"confirmed_count={len(files)} regex_shared_replaced=1"
    )
    out_scope = ScopeSet(
        op_dir=scope.op_dir,
        workspace_root=scope.workspace_root,
        arch_dir=scope.arch_dir,
        files=files,
        notes=notes,
    )
    return ClangEnrichment(
        scope=out_scope,
        status=status,
        tus_expected=len(jobs),
        tus_parsed=parsed,
        errors=errors,
        probes=probes,
    )
