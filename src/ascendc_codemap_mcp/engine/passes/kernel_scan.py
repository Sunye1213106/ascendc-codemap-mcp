# -*- coding: utf-8 -*-
"""Shared Kernel source scan helpers for Root Trace (Clang walk + lexical)."""

from __future__ import annotations

from ascendc_codemap_mcp.engine.paths import require_architecture
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ids import rel_posix
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

from ascendc_codemap_mcp.engine.source_index.builder import (  # noqa: F401  re-export for quality / identity
    _CALL_RE,
    _CXX_CALL_SKIP,
    _is_false_lexical_callee,
    _is_tpl_dsl_file,
    _strip_line_noise,
)

_WALK_CACHE_LIMIT = 48


def caller_allowed(caller: str, reachable: set[str], *, filter_strict: bool) -> bool:
    if not filter_strict or not reachable or not caller:
        return True
    short = caller.split("::")[-1]
    return caller in reachable or short in reachable


def norm_file(path: str, root: str = "") -> str:
    text = str(path or "").replace("\\", "/")
    if text.startswith("/mnt/") and len(text) >= 7 and text[5].isalpha() and text[6] == "/":
        text = f"{text[5].upper()}:{text[6:]}"
    return rel_posix(text, root)


def reachable_function_names(codemap: CodeMap) -> tuple[set[str], bool]:
    from collections import deque

    starts = {
        e.id
        for e in codemap.by_kind(EntityKind.KERNEL)
        if e.attrs.get("source_definition") or e.attrs.get("source_signature")
    }
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if rel.kind_name() != RelationKind.CALLS.value:
            continue
        prov = str(rel.attrs.get("provenance") or "")
        if prov not in {
            "source_kernel_call_bound",
            "source_kernel_macro_call_bound",
            "source_kernel_call_refined",
        } and not prov.startswith("source_kernel"):
            continue
        adj[rel.src].add(rel.dst)
    seen = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    names: set[str] = set()
    non_kernel = 0
    for eid in seen:
        ent = codemap.entities.get(eid)
        if ent is None:
            continue
        names.add(ent.name)
        short = str(ent.attrs.get("short_name") or "").strip()
        if short:
            names.add(short)
        if "::" in ent.name:
            names.add(ent.name.rsplit("::", 1)[-1])
        if ent.kind_name() != EntityKind.KERNEL.value:
            non_kernel += 1
    if not names:
        for e in codemap.by_kind(EntityKind.KERNEL):
            names.add(e.name)
    return names, non_kernel >= 3


_KERNEL_SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}


def architecture_kernel_files(source_root: Path, architecture: str) -> list[Path]:
    """Kernel sources for this architecture, including unused headers.

    Covers ``op_kernel/<arch>/`` and arch-neutral files under ``op_kernel/``
    (helpers, 220-gated headers beside an ``_apt.cpp`` entry, infra). A
    different ``arch*`` directory is never mixed in. Prepare's confirmed TU
    is one ORIG_DTYPE walk; TQue / DataCopy / Cast in the rest of this
    architecture's kernel tree still belong in the CodeMap.
    """
    from ascendc_codemap_mcp.engine.source_layout import (
        architecture_in_scope,
        entry_include_architecture,
        is_foreign_arch_entry_tu,
        is_other_arch_path,
    )

    arch = require_architecture(architecture)
    kernel_root = Path(source_root) / "op_kernel"
    if not kernel_root.is_dir():
        return []
    try:
        kernel_root_key = kernel_root.resolve()
    except OSError:
        kernel_root_key = kernel_root
    out: list[Path] = []
    for path in sorted(kernel_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _KERNEL_SOURCE_SUFFIXES:
            continue
        if is_foreign_arch_entry_tu(path, arch) or is_other_arch_path(path, arch):
            continue
        try:
            parent_key = path.parent.resolve()
        except OSError:
            parent_key = path.parent
        if (
            parent_key == kernel_root_key
            and path.suffix.lower() in {".cpp", ".cc", ".cxx"}
        ):
            try:
                owns = entry_include_architecture(read_text(path))
            except OSError:
                owns = ""
            if owns and not architecture_in_scope(owns, arch):
                continue
        out.append(path)
    return out


# Cube glue headers (family ``common/cgmct``, CANN ``lib/matmul``). Not every
# ascendc/ header — that would inflate lexical source n far past the graph.
_KERNEL_API_INCLUDE_HINTS = (
    "cgmct/",
    "/cgmct/",
    "lib/matmul",
    "/lib/matmul/",
    "/matmul/",
)
_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_CORPUS_FILE_CAP = 8192
# Family ``common/`` / cube-template includes stay one hop. Recursing that
# forest (cgmct / matmul) inflates lexical Cast/DataCopy far past Clang sites.
_CORPUS_ONE_HOP_OWNERS = frozenset({"sibling_op", "family_common"})


def _kernel_api_include_roots(source_root: Path) -> list[Path]:
    """Quoted-include search roots: file parent is tried first, then these.

    Cube templates live under the family ``common/`` tree (``*/common/cgmct``)
    as well as CANN package includes. No operator-name branches.
    """
    from ascendc_codemap_mcp.engine.tpl_dsl import cann_include_search_roots

    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            return
        if not key.is_dir() or key in seen:
            return
        seen.add(key)
        roots.append(key)

    for path in cann_include_search_roots():
        add(path)
    root = Path(source_root)
    add(root)
    add(root / "common")
    add(root / "op_kernel")
    add(root.parent)
    add(root.parent / "common")
    add(root.parent / "common" / "utils")
    add(root.parent / "common" / "inc")
    add(root.parent.parent / "common")
    return roots


def _resolved(path: Path) -> Path:
    from ascendc_codemap_mcp.engine.paths import resolved

    return resolved(path)


def _under(path: Path, root: Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(root))
        return True
    except ValueError:
        return False


def kernel_file_owner(path: Path | str, source_root: Path) -> str:
    """Classify a kernel file: this_op | family_common | sibling_op | cann.

    Fusion wrappers ``#include`` a sibling operator's ``.cpp``; those files stay
    in the corpus (locate must still find EnQue) but they are not this op's tree.
    No operator-name branches.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(source_root) / resolved
    resolved = _resolved(resolved)
    root = _resolved(Path(source_root))
    if _under(resolved, root):
        return "this_op"
    parent = root.parent
    if _under(resolved, parent / "common"):
        return "family_common"
    if _under(resolved, parent):
        posix = f"/{str(resolved).replace(chr(92), '/').lower()}/"
        if "/common/" in posix:
            return "family_common"
        return "sibling_op"
    posix = str(resolved).replace("\\", "/").lower()
    if any(h in posix for h in _KERNEL_API_INCLUDE_HINTS):
        return "family_common"
    return "cann"


def _corpus_should_follow(resolved: Path, source_root: Path) -> str:
    """Return owner if this include should enter the corpus, else empty.

    Sibling-operator files are included one hop (the ``#include "*.cpp"`` itself)
    so fusion TUs stay locate-able. Recursing the sibling tree inflates lexical
    source n far past Clang call sites; walk-cited files fill the rest.
    """
    owner = kernel_file_owner(resolved, source_root)
    if owner in {"this_op", "family_common"}:
        return owner
    if owner == "sibling_op":
        return owner
    rel = str(resolved).replace("\\", "/").lower()
    if any(h in rel for h in _KERNEL_API_INCLUDE_HINTS):
        return "family_common"
    return ""


def _add_corpus_file(
    path: Path,
    *,
    seen: set[Path],
    out: list[Path],
    pending: list[Path],
) -> bool:
    try:
        key = path.resolve()
    except OSError:
        key = path
    if key in seen or not path.is_file():
        return False
    seen.add(key)
    out.append(path)
    pending.append(path)
    return True


def _quoted_include_targets(path: Path, search_roots: list[Path]) -> list[Path]:
    try:
        text = read_text(path)
    except OSError:
        return []
    parent = path.parent
    found: list[Path] = []
    seen: set[Path] = set()
    for inc in _QUOTED_INCLUDE_RE.findall(text):
        posix = inc.replace("\\", "/")
        for cand in [parent / posix, *(root / posix for root in search_roots)]:
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            if not resolved.is_file() or resolved in seen:
                continue
            if resolved.suffix.lower() not in _KERNEL_SOURCE_SUFFIXES:
                continue
            seen.add(resolved)
            found.append(resolved)
            break
    return found


def walk_cited_kernel_files(source_root: Path, architecture: str) -> list[Path]:
    """Files named by cached Clang walks (the TU Clang actually saw)."""
    from ascendc_codemap_mcp.engine.source_layout import is_foreign_arch_entry_tu, is_other_arch_path

    try:
        from ascendc_codemap_mcp.engine import tu_cache
    except Exception:  # noqa: BLE001
        return []
    arch = require_architecture(architecture)
    try:
        walks = tu_cache.iter_cached_walks(
            source_root, arch, path_substr="op_kernel", limit=_WALK_CACHE_LIMIT
        )
    except Exception:  # noqa: BLE001
        return []
    raw: list[str] = []
    for wr in walks or []:
        raw.extend(_walk_cited_raw_paths(wr))
    out: list[Path] = []
    seen: set[Path] = set()
    root = Path(source_root)
    # A file is cited once per call site, decl, control and field that mentions
    # it, so this arrives as tens of thousands of entries naming a few dozen
    # files. Deduplicating the spelling first matters because everything below
    # is per entry: a filesystem probe and two architecture classifications.
    # The existing `seen` set only dedups *after* all of that has been paid.
    for text in dict.fromkeys(raw):
        path = _resolve_cited_file(text, root)
        if path is None:
            continue
        if is_foreign_arch_entry_tu(path, arch) or is_other_arch_path(path, arch):
            continue
        key = _resolved(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _walk_cited_raw_paths(wr: Any) -> list[str]:
    rows: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            rows.append(text)

    add(getattr(wr, "path", "") or "")
    for site in getattr(wr, "call_sites", None) or []:
        add(getattr(site, "file", "") or "")
        add(getattr(site, "callee_decl_file", "") or "")
        if isinstance(site, dict):
            add(site.get("file") or "")
            add(site.get("callee_decl_file") or "")
    for decl in getattr(wr, "local_decls", None) or []:
        add(getattr(decl, "file", "") or (decl.get("file") if isinstance(decl, dict) else ""))
    for decl in getattr(wr, "type_decls", None) or []:
        add(getattr(decl, "file", "") or (decl.get("file") if isinstance(decl, dict) else ""))
    for decl in getattr(wr, "alias_decls", None) or []:
        add(getattr(decl, "file", "") or (decl.get("file") if isinstance(decl, dict) else ""))
    fds = getattr(wr, "field_decls", None) or {}
    if isinstance(fds, dict):
        fds = fds.values()
    for fd in fds:
        add(getattr(fd, "file", "") or (fd.get("file") if isinstance(fd, dict) else ""))
    for ctrl in getattr(wr, "controls", None) or []:
        add(getattr(ctrl, "file", "") or (ctrl.get("file") if isinstance(ctrl, dict) else ""))
    fns = getattr(wr, "functions", None) or {}
    if isinstance(fns, dict):
        fns = fns.values()
    for rec in fns:
        add(getattr(rec, "file", "") or (rec.get("file") if isinstance(rec, dict) else ""))
    return rows


def _resolve_cited_file(raw: str, source_root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_file():
        return path
    cand = source_root / text
    if cand.is_file():
        return cand
    return None


def kernel_corpus(
    source_root: Path,
    architecture: str,
    extra_files: Iterable[Path] | None = None,
    *,
    include_walks: bool = True,
    deadline: float | None = None,
) -> list[Path]:
    """One file set for lexical ``source_api`` and OPERATION minting.

    Starts at this-arch kernel files and follows quoted includes: this operator
    is a full BFS; sibling ``.cpp`` and family ``common/`` / cube templates are
    one hop. Walk-cited this-op / sibling files are unioned so fusion wrappers
    still see IFA/PFA call sites. CANN headers stay out of the lexical corpus.
    """
    from ascendc_codemap_mcp.engine.source_layout import is_foreign_arch_entry_tu, is_other_arch_path

    arch = require_architecture(architecture)
    root = Path(source_root)
    seen: set[Path] = set()
    out: list[Path] = []
    pending: list[Path] = []
    for path in architecture_kernel_files(root, arch):
        _add_corpus_file(path, seen=seen, out=out, pending=pending)
    extras: list[Path] = [Path(p) for p in (extra_files or [])]
    if include_walks:
        extras.extend(walk_cited_kernel_files(root, arch))
    for path in extras:
        if not path.is_file():
            continue
        if is_foreign_arch_entry_tu(path, arch) or is_other_arch_path(path, arch):
            continue
        # Walk-cited CANN / cube-template headers are already Clang CallExprs.
        # Union only this-op and sibling files so FIA locate still sees IFA.
        owner = kernel_file_owner(path, root)
        if owner not in {"this_op", "sibling_op"}:
            continue
        # Walk-cited / extra files enter the scan set without re-expanding
        # their include trees (Clang already named what it used).
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    search_roots = _kernel_api_include_roots(root)
    while pending and len(out) < _CORPUS_FILE_CAP:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        path = pending.pop(0)
        for resolved in _quoted_include_targets(path, search_roots):
            if is_foreign_arch_entry_tu(resolved, arch) or is_other_arch_path(resolved, arch):
                continue
            owner = _corpus_should_follow(resolved, root)
            if not owner:
                continue
            if owner in _CORPUS_ONE_HOP_OWNERS:
                try:
                    key = resolved.resolve()
                except OSError:
                    key = resolved
                if key in seen:
                    continue
                seen.add(key)
                out.append(resolved)
                continue
            _add_corpus_file(resolved, seen=seen, out=out, pending=pending)
    return out


def kernel_api_scan_files(source_root: Path, architecture: str) -> list[Path]:
    """Architecture kernel tree plus quoted-include closure (cgmct / matmul / sibling .cpp)."""
    return kernel_corpus(source_root, architecture)


def selected_kernel_files(codemap: CodeMap, source_root: Path) -> list[Path]:
    meta = codemap.meta.get("kernel_tiling_closure") or {}
    listed = meta.get("selected_kernel_files") or []
    out: list[Path] = []
    seen: set[str] = set()
    for item in listed:
        p = Path(str(item))
        if not p.is_file():
            cand = source_root / item
            if cand.is_file():
                p = cand
            else:
                continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    arch = require_architecture(codemap.architecture)
    if out:
        from ascendc_codemap_mcp.engine.source_layout import keep_lexical_kernel_path

        return [p for p in out if keep_lexical_kernel_path(p, arch)]
    return architecture_kernel_files(source_root, arch)


def site_dedupe_key(site: Any, *, root: str = "") -> tuple[str, int, str]:
    d = site if isinstance(site, dict) else None
    if d is None:
        file = str(getattr(site, "file", "") or "")
        line = int(getattr(site, "line", 0) or 0)
        callee = str(getattr(site, "callee", "") or "").split("::")[-1]
    else:
        file = str(d.get("file") or "")
        line = int(d.get("line") or 0)
        callee = str(d.get("callee") or "").split("::")[-1]
    return (norm_file(file, root), line, callee)


def site_as_dict(site: Any) -> dict[str, Any]:
    if isinstance(site, dict):
        return site
    return {
        "caller": getattr(site, "caller", "") or "",
        "callee": getattr(site, "callee", "") or "",
        "file": getattr(site, "file", "") or "",
        "line": int(getattr(site, "line", 0) or 0),
        "column": int(getattr(site, "column", 0) or 0),
        "args": list(getattr(site, "args", None) or []),
        "template_args": list(getattr(site, "template_args", None) or []),
        "receiver": getattr(site, "receiver", "") or "",
        "path_conditions": getattr(site, "path_conditions", None) or (),
        "entry_reachable": bool(getattr(site, "entry_reachable", True)),
        "caller_usr": getattr(site, "caller_usr", "") or "",
        "caller_qualified": getattr(site, "caller_qualified", "") or "",
        "callee_usr": getattr(site, "callee_usr", "") or "",
        "callee_qualified": getattr(site, "callee_qualified", "") or "",
        "callee_decl_file": getattr(site, "callee_decl_file", "") or "",
        "receiver_type": getattr(site, "receiver_type", "") or "",
        "receiver_canonical_type": getattr(site, "receiver_canonical_type", "") or "",
        "provenance": getattr(site, "provenance", "") or "clang_walk_cache",
    }


def _targs_quality(targs: list[str] | None) -> int:
    joined = " ".join(str(t) for t in (targs or []))
    score = 0
    if "HardEvent" in joined or "PIPE_" in joined:
        score += 2
    if joined.strip():
        score += 1
    return score


def _enrich_site_templates(dst: Any, src: dict[str, Any]) -> Any:
    src_t = list(src.get("template_args") or [])
    src_a = list(src.get("args") or [])
    if isinstance(dst, dict):
        if _targs_quality(src_t) > _targs_quality(list(dst.get("template_args") or [])):
            dst["template_args"] = src_t
        if (not dst.get("args")) and src_a:
            dst["args"] = src_a
        if not dst.get("receiver") and src.get("receiver"):
            dst["receiver"] = src["receiver"]
        for key in (
            "receiver_type",
            "receiver_canonical_type",
            "callee_qualified",
            "callee_decl_file",
            "callee_decl_line",
            "identity_kind",
            "callee_return_type",
        ):
            if not dst.get(key) and src.get(key):
                dst[key] = src[key]
        return dst
    d = site_as_dict(dst)
    if _targs_quality(src_t) > _targs_quality(list(d.get("template_args") or [])):
        d["template_args"] = src_t
    if (not d.get("args")) and src_a:
        d["args"] = src_a
    if not d.get("receiver") and src.get("receiver"):
        d["receiver"] = src["receiver"]
    for key in (
        "receiver_type",
        "receiver_canonical_type",
        "callee_qualified",
        "callee_decl_file",
        "callee_decl_line",
        "identity_kind",
        "callee_return_type",
    ):
        if not d.get(key) and src.get(key):
            d[key] = src[key]
    return d if (d.get("template_args") or d.get("args") or d.get("callee_qualified")) else dst


def merge_lexical_sites(
    walk_calls: list[Any],
    lexical: list[dict[str, Any]],
    *,
    root: str,
) -> tuple[list[Any], int]:
    index: dict[tuple[str, int, str], int] = {}
    out: list[Any] = list(walk_calls)
    for i, s in enumerate(out):
        index[site_dedupe_key(s, root=root)] = i
    added = 0
    for site in lexical:
        key = site_dedupe_key(site, root=root)
        if key in index:
            i = index[key]
            out[i] = _enrich_site_templates(out[i], site)
            continue
        index[key] = len(out)
        row = dict(site) if isinstance(site, dict) else site
        if isinstance(row, dict):
            row.setdefault("call_kind", "unresolved_call")
            row.setdefault("semantic_state", "candidate")
        out.append(row)
        added += 1
    return out, added


_MAX_WALK_CONFIRMS = 256


def confirm_lexical_from_walks(
    sites: list[Any],
    walks: list[Any],
    *,
    max_attempts: int = _MAX_WALK_CONFIRMS,
) -> int:
    """Stamp USR onto lexical sites from tu_cache. No extra parse.

    Same callee spelling × same cached functions: at most one confirm. Ambiguous
    names stay unresolved.
    """
    by_name: dict[str, tuple[str, str, str]] = {}
    ambiguous: set[str] = set()
    for wr in walks or []:
        fns = getattr(wr, "functions", None) or {}
        items = fns.items() if isinstance(fns, dict) else []
        for name, fr in items:
            usr = str(getattr(fr, "usr", "") or "")
            if not usr:
                continue
            qn = str(getattr(fr, "qualified_name", "") or name)
            file_s = str(getattr(fr, "file", "") or "")
            for key in {str(name), str(name).split("::")[-1]}:
                if not key or key in ambiguous:
                    continue
                prev = by_name.get(key)
                if prev is not None and prev[0] != usr:
                    ambiguous.add(key)
                    by_name.pop(key, None)
                    continue
                by_name[key] = (usr, qn, file_s)
    confirmed = 0
    for site in sites or []:
        if confirmed >= max_attempts:
            break
        if not isinstance(site, dict):
            continue
        if site.get("callee_usr"):
            continue
        callee = str(site.get("callee") or "").split("::")[-1]
        hit = by_name.get(callee)
        if hit is None:
            continue
        site["callee_usr"], site["callee_qualified"], site["callee_decl_file"] = hit
        site["identity_kind"] = "walk_cache_confirm"
        site["call_kind"] = "resolved_call"
        confirmed += 1
    return confirmed


def _function_names_from_walk(wr: Any) -> set[str]:
    """Callers the kernel walk already closed over (methods, not just KERNEL entities)."""
    names: set[str] = set()
    fns = getattr(wr, "functions", None) or {}
    keys = fns.keys() if isinstance(fns, dict) else fns
    for raw in keys:
        text = str(raw or "")
        if not text:
            continue
        names.add(text)
        names.add(text.split("::")[-1])
        if "<" in text:
            names.add(text.split("<", 1)[0])
    return names


def collect_call_sites_from_walks(
    source_root: Path,
    *,
    architecture: str,
    reachable: set[str],
    filter_strict: bool,
    deadline: float,
) -> tuple[list[Any], list[Any], list[Any], str]:
    from ascendc_codemap_mcp.engine import tu_cache

    walks = tu_cache.iter_cached_walks(
        source_root, architecture, path_substr="op_kernel", limit=_WALK_CACHE_LIMIT
    )
    if time.perf_counter() > deadline:
        return [], [], [], "budget_exhausted_before_walk_cache"
    if not walks:
        return [], [], [], "no_walk_cache"

    allowed = set(reachable)
    for wr in walks:
        allowed |= _function_names_from_walk(wr)

    calls: list[Any] = []
    index: dict[tuple[str, int, str], int] = {}
    decls: list[Any] = []
    controls: list[Any] = []
    root = str(source_root)
    for wr in walks:
        if time.perf_counter() > deadline:
            break
        for n, site in enumerate(getattr(wr, "call_sites", None) or []):
            if n % 400 == 0 and time.perf_counter() > deadline:
                break
            caller = str(getattr(site, "caller", "") or "")
            callee = str(getattr(site, "callee", "") or "").split("::")[-1]
            if not caller_allowed(caller, allowed, filter_strict=filter_strict):
                continue
            # Root Trace needs the full source call graph. Terminal AscendC/CANN
            # classification happens later; do not filter by registry primitives.
            if not callee or not callee.isidentifier():
                continue
            key = site_dedupe_key(site, root=root)
            if key in index:
                i = index[key]
                merged = _enrich_site_templates(calls[i], site_as_dict(site))
                if isinstance(merged, dict):
                    merged["instantiation_n"] = int(merged.get("instantiation_n") or 1) + 1
                    calls[i] = merged
                else:
                    calls[i] = merged
                continue
            row = site_as_dict(site)
            row["instantiation_n"] = int(row.get("instantiation_n") or 1)
            index[key] = len(calls)
            calls.append(row)
        decls.extend(getattr(wr, "local_decls", None) or [])
        controls.extend(getattr(wr, "controls", None) or [])
    provenance = "clang_walk_cache"
    if len(walks) >= _WALK_CACHE_LIMIT:
        provenance = "clang_walk_cache_partial"
    return calls, decls, controls, provenance


def collect_type_graph_from_walks(
    source_root: Path,
    *,
    architecture: str,
    deadline: float,
) -> dict[str, list[dict[str, Any]]]:
    """Clang-first type / member / alias / base facts from walk cache."""
    from ascendc_codemap_mcp.engine import tu_cache

    walks = tu_cache.iter_cached_walks(
        source_root, architecture, path_substr="op_kernel", limit=_WALK_CACHE_LIMIT
    )
    members: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    types: list[dict[str, Any]] = []
    bases: list[dict[str, Any]] = []
    if time.perf_counter() > deadline:
        return {"members": members, "aliases": aliases, "types": types, "bases": bases}
    for wr in walks:
        if time.perf_counter() > deadline:
            break
        for fd in (getattr(wr, "field_decls", None) or {}).values():
            host = str(getattr(fd, "host", "") or "")
            name = str(getattr(fd, "name", "") or "")
            if not host or not name:
                continue
            type_text = str(getattr(fd, "type_text", "") or "")
            members.append(
                {
                    "owner": host,
                    "owner_qualified": str(getattr(fd, "owner_qualified", "") or host),
                    "member": name,
                    "type_text": type_text,
                    "canonical_type": str(getattr(fd, "canonical_type", "") or ""),
                    "referenced_type_usr": str(getattr(fd, "referenced_type_usr", "") or ""),
                    "base_type": type_text.split("<", 1)[0].split("::")[-1].strip()
                    if type_text
                    else "",
                    "file": str(getattr(fd, "file", "") or ""),
                    "line": int(getattr(fd, "line", 0) or 0),
                    "column": int(getattr(fd, "column", 0) or 0),
                    "provenance": "clang_field_decl",
                }
            )
        for ad in getattr(wr, "alias_decls", None) or []:
            aliases.append(
                {
                    "alias": str(getattr(ad, "name", "") or ""),
                    "qualified_name": str(getattr(ad, "qualified_name", "") or ""),
                    "target": str(getattr(ad, "target_type", "") or ""),
                    "canonical_type": str(getattr(ad, "canonical_type", "") or ""),
                    "target_usr": str(getattr(ad, "target_usr", "") or ""),
                    "file": str(getattr(ad, "file", "") or ""),
                    "line": int(getattr(ad, "line", 0) or 0),
                    "column": int(getattr(ad, "column", 0) or 0),
                    "provenance": "clang_alias_decl",
                }
            )
        for td in getattr(wr, "type_decls", None) or []:
            types.append(
                {
                    "name": str(getattr(td, "name", "") or ""),
                    "qualified_name": str(getattr(td, "qualified_name", "") or ""),
                    "usr": str(getattr(td, "usr", "") or ""),
                    "kind": str(getattr(td, "kind", "") or "class"),
                    "file": str(getattr(td, "file", "") or ""),
                    "line": int(getattr(td, "line", 0) or 0),
                    "column": int(getattr(td, "column", 0) or 0),
                    "provenance": "clang_type_decl",
                }
            )
        for bd in getattr(wr, "base_decls", None) or []:
            bases.append(
                {
                    "derived": str(getattr(bd, "derived_name", "") or ""),
                    "derived_usr": str(getattr(bd, "derived_usr", "") or ""),
                    "base": str(getattr(bd, "base_name", "") or ""),
                    "base_usr": str(getattr(bd, "base_usr", "") or ""),
                    "canonical_type": str(getattr(bd, "canonical_type", "") or ""),
                    "file": str(getattr(bd, "file", "") or ""),
                    "line": int(getattr(bd, "line", 0) or 0),
                    "column": int(getattr(bd, "column", 0) or 0),
                    "provenance": "clang_base_decl",
                }
            )
    return {"members": members, "aliases": aliases, "types": types, "bases": bases}


def lexical_source_call_sites(
    files: list[Path],
    *,
    reachable: set[str],
    filter_strict: bool,
    root: str,
    deadline: float,
    primitives_only: bool = False,
) -> list[dict[str, Any]]:
    """Collect source-scope identifier call sites from the shared SourceIndex."""
    from ascendc_codemap_mcp.engine.source_index import get_or_build

    if time.perf_counter() > deadline:
        return []
    index = get_or_build(files, root=root, deadline=deadline)
    sites: list[dict[str, Any]] = []
    for site in index.calls_for(files, primitives_only=primitives_only):
        func = str(site.get("caller") or "")
        if not caller_allowed(func, reachable, filter_strict=filter_strict):
            continue
        row = dict(site)
        row["entry_reachable"] = (
            caller_allowed(func, reachable, filter_strict=True) if reachable else True
        )
        sites.append(row)
    return sites


def lexical_buffer_decls(
    files: list[Path],
    *,
    reachable: set[str],
    filter_strict: bool,
    deadline: float,
) -> list[dict[str, Any]]:
    from ascendc_codemap_mcp.engine.source_index import get_or_build

    if time.perf_counter() > deadline:
        return []
    index = get_or_build(files, root="", deadline=deadline)
    decls: list[dict[str, Any]] = []
    for decl in index.buffers_for(files):
        func = str(decl.get("function") or "")
        if not caller_allowed(func, reachable, filter_strict=filter_strict):
            continue
        decls.append(decl)
    return decls
