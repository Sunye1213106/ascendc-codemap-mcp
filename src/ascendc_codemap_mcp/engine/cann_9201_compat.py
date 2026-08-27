# -*- coding: utf-8 -*-
"""Compile arch-920r1 as __NPU_ARCH__=9201; overlay CANN headers that still gate 3510.

CANN 9.1/9.2 ship DAV_9201 in some compiler headers but not in
``kernel_tpipe.h`` / ``kernel_tpipe_base.h`` / ``platform_config``.
Do not remap the compile macro to 3510.

When a header still tests ``__NPU_ARCH__ == 3510`` or ``!= 3510`` and does
not treat 9201 as a sibling of 3510, copy it under
``.ascendc-codemap/<arch>/cache/cann_9201_overlay/<tree>/...`` preserving
the path relative to ``asc/`` or ``tikcpp/tikcfw/``. Prepend the mirrored
``-I`` dirs ahead of the CANN include roots so quoted ``../../impl/...``
includes from an overlay copy still hit the overlay impl, not the unpatched
original.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import re
import yaml

from ascendc_codemap_mcp.engine.source_layout import canonicalize_architecture

_OVERLAY_LOGIC = 4
_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_HAS_9201 = re.compile(r"__NPU_ARCH__\s*==\s*9201")
_HAS_3510_EQ = re.compile(r"__NPU_ARCH__\s*==\s*3510")
_HAS_3510_NE = re.compile(r"__NPU_ARCH__\s*!=\s*3510")
_ALREADY_EQ = re.compile(
    r"\(__NPU_ARCH__\s*==\s*3510\)\s*\|\|\s*\(__NPU_ARCH__\s*==\s*9201\)"
)
_ALREADY_NE = re.compile(
    r"\(__NPU_ARCH__\s*!=\s*3510\)\s*&&\s*\(__NPU_ARCH__\s*!=\s*9201\)"
)
_HEADER_SUFFIXES = {".h", ".hpp", ".hh"}
_KEEP_DAV = frozenset({"dav_3510", "dav_c310"})
_TREE_RELS = (
    ("asc", "cann-asc-devkit/x86_64-linux/asc"),
    ("tikcfw", "cann-asc-devkit/x86_64-linux/tikcpp/tikcfw"),
)
_PROBE_BASENAMES = (
    "kernel_tpipe.h",
    "kernel_reg_compute_intf.h",
    "kernel_reg_compute_utils.h",
    "kernel_tensor.h",
    "kernel_operator.h",
    "sys_macros.h",
)


def overlay_dir(op_dir: str | Path, arch_dir: str) -> Path:
    arch = str(arch_dir or "").strip()
    return (
        Path(op_dir).expanduser().resolve()
        / ".ascendc-codemap" / arch
        / "cache"
        / "cann_9201_overlay"
    )


def expand_3510_gates(text: str) -> str:
    """Make 9201 take the 3510 branch. Also keep 9201 out of ``!= 3510``. Idempotent."""
    placeholders: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"\x00PILOT{len(placeholders) - 1}\x00"

    held = _ALREADY_EQ.sub(_hold, text)
    held = _ALREADY_NE.sub(_hold, held)
    held = _HAS_3510_EQ.sub("((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201))", held)
    held = _HAS_3510_NE.sub("((__NPU_ARCH__ != 3510) && (__NPU_ARCH__ != 9201))", held)
    for i, orig in enumerate(placeholders):
        held = held.replace(f"\x00PILOT{i}\x00", orig)
    return held


def _iter_header_trees(ctx: Any) -> list[tuple[str, Path]]:
    from ascendc_codemap_mcp.engine.paths import resolve_cann_relative

    cann = Path(getattr(ctx, "cann_root", "") or "")
    if not cann.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for key, rel in _TREE_RELS:
        root = resolve_cann_relative(cann, rel)
        try:
            if not root.is_dir():
                continue
            stamp = str(root.resolve())
        except OSError:
            continue
        if stamp in seen:
            continue
        seen.add(stamp)
        out.append((key, root))
    return out


def _skip_header(path: Path) -> bool:
    for part in path.parts:
        low = part.lower()
        if low.startswith("dav_") and low not in _KEEP_DAV:
            return True
    return False


def _logical_rel(path: Path, root: Path) -> str | None:
    """Path relative to the tree as walked, not as resolved through junctions.

    ``Path.resolve()`` on a Windows reparse point can escape the tree; falling
    back to ``path.name`` then drops ``impl/utils/common_types.h`` onto
    ``tikcfw/common_types.h`` and shadows the kernel ``TPosition`` header.
    """
    try:
        return Path(os.path.normpath(str(path))).relative_to(
            Path(os.path.normpath(str(root)))
        ).as_posix()
    except ValueError:
        return None


def _quoted_target(src: Path, inc: str) -> Path | None:
    rel = str(inc or "").replace("\\", "/").split("#", 1)[0].strip()
    if not rel or rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        return None
    cand = Path(os.path.normpath(str(src.parent / rel)))
    try:
        if cand.is_file():
            return cand
    except OSError:
        return None
    return None


def _tree_dest(path: Path, trees: list[tuple[str, Path]]) -> str | None:
    for key, root in trees:
        rel = _logical_rel(path, root)
        if rel:
            return f"{key}/{rel}"
    return None


def _expand_quoted_closure(
    sources: list[tuple[str, Path]], trees: list[tuple[str, Path]]
) -> list[tuple[str, Path]]:
    """Copy quoted includes that resolve inside an overlay tree (incl. dav_3510)."""
    seen = {rel for rel, _ in sources}
    queue = list(sources)
    while queue:
        _rel, src = queue.pop()
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for inc in _QUOTED_INCLUDE_RE.findall(text):
            target = _quoted_target(src, inc)
            if target is None or _skip_header(target):
                continue
            dest = _tree_dest(target, trees)
            if not dest or dest in seen:
                continue
            seen.add(dest)
            item = (dest, target)
            sources.append(item)
            queue.append(item)
    return sources


def _collect_overlay_sources(ctx: Any) -> list[tuple[str, Path]]:
    """``(tree_key/rel_from_tree, src)`` for gated headers plus same-dir siblings."""
    sources: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for key, root in _iter_header_trees(ctx):
        try:
            files = list(root.rglob("*"))
        except OSError:
            continue
        gated_here: list[tuple[str, Path]] = []
        for path in files:
            try:
                if not path.is_file() or path.suffix.lower() not in _HEADER_SUFFIXES:
                    continue
            except OSError:
                continue
            if _skip_header(path):
                continue
            rel = _logical_rel(path, root)
            if not rel:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if expand_3510_gates(text) == text:
                continue
            dest = f"{key}/{rel}"
            if dest in seen:
                continue
            seen.add(dest)
            gated_here.append((dest, path))
        sources.extend(gated_here)
        for _rel, src in gated_here:
            try:
                siblings = list(src.parent.iterdir())
            except OSError:
                continue
            for sib in siblings:
                try:
                    if not sib.is_file() or sib.suffix.lower() not in _HEADER_SUFFIXES:
                        continue
                except OSError:
                    continue
                if _skip_header(sib):
                    continue
                sib_rel = _logical_rel(sib, root)
                if not sib_rel:
                    continue
                dest = f"{key}/{sib_rel}"
                if dest in seen:
                    continue
                seen.add(dest)
                sources.append((dest, sib))
    return _expand_quoted_closure(sources, _iter_header_trees(ctx))


def _mapped_overlay_includes(
    overlay_root: Path, ctx: Any, trees: list[tuple[str, Path]]
) -> list[str]:
    """Mirror kernel ``-I`` entries that sit under an overlaid CANN tree."""
    resolve = getattr(ctx, "resolve_path", None)
    raw = ((getattr(ctx, "raw", None) or {}).get("kernel") or {}).get("includes") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        try:
            inc = Path(resolve(str(item)) if callable(resolve) else str(item))
            resolved = inc.resolve()
        except OSError:
            continue
        for key, root in trees:
            try:
                rel = resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            mapped = overlay_root / key / rel
            try:
                if not mapped.is_dir():
                    break
                posix = str(mapped.resolve()).replace("\\", "/").rstrip("/")
            except OSError:
                break
            low = posix.lower()
            if low not in seen:
                seen.add(low)
                out.append(posix)
            break
    return out


def _ini_report(ctx: Any) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.platform_ini import list_profiles

    report: dict[str, Any] = {
        "npu_arch": 9201,
        "headers": "missing",
        "ini": "missing",
        "overlay_files": [],
        "native_files": [],
        "sku_fallback": "",
        "overlay_file_count": 0,
        "logic": _OVERLAY_LOGIC,
    }
    cann = Path(getattr(ctx, "cann_root", "") or "")
    if not cann.is_dir():
        report["ini"] = "no_cann"
        report["headers"] = "no_cann"
        return report
    try:
        native_ini = list_profiles(cann, npu_arch=9201)
    except OSError:
        native_ini = []
    if native_ini:
        report["ini"] = "native"
    else:
        report["ini"] = "sku_fallback"
        report["sku_fallback"] = "Ascend950PR_9589"
    return report


def _sample_probe_files(ctx: Any) -> tuple[list[str], list[str]]:
    overlay_names: list[str] = []
    native_names: list[str] = []
    for key, root in _iter_header_trees(ctx):
        for name in _PROBE_BASENAMES:
            found = None
            for sub in ("", "include/basic_api", "interface", "impl"):
                cand = root / sub / name if sub else root / name
                try:
                    if cand.is_file():
                        found = cand
                        break
                except OSError:
                    continue
            if found is None:
                continue
            try:
                text = found.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel_body = _logical_rel(found, root)
            if not rel_body:
                continue
            rel = f"{key}/{rel_body}"
            if _HAS_9201.search(text) and not _HAS_3510_EQ.search(text):
                native_names.append(rel)
            elif _HAS_3510_EQ.search(text) or _HAS_3510_NE.search(text):
                overlay_names.append(rel)
    return overlay_names, native_names


def _cache_key(ctx: Any) -> dict[str, str]:
    cann = Path(getattr(ctx, "cann_root", "") or "")
    try:
        cann_s = str(cann.resolve()).replace("\\", "/") if cann.is_dir() else str(cann)
    except OSError:
        cann_s = str(cann)
    return {"logic": str(_OVERLAY_LOGIC), "cann_root": cann_s}


def _load_cached_overlay(dest_root: Path, ctx: Any) -> dict[str, Any] | None:
    probe_path = dest_root / "probe.yaml"
    if not probe_path.is_file():
        return None
    try:
        loaded = yaml.safe_load(probe_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(loaded, dict):
        return None
    want = _cache_key(ctx)
    if str(loaded.get("logic") or "") != want["logic"]:
        return None
    if str(loaded.get("cann_root") or "").replace("\\", "/") != want["cann_root"]:
        return None
    dirs = [str(p).replace("\\", "/").rstrip("/") for p in (loaded.get("overlay_includes") or [])]
    if not dirs:
        return None
    if not all(Path(p).is_dir() for p in dirs):
        return None
    loaded["overlay_dir"] = str(dest_root).replace("\\", "/")
    loaded["probe_path"] = str(probe_path).replace("\\", "/")
    loaded["_cached"] = True
    return loaded


def _clear_overlay_payload(dest_root: Path) -> None:
    if not dest_root.is_dir():
        return
    for child in list(dest_root.iterdir()):
        try:
            if child.name == "probe.yaml":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            continue


def _write_probe(dest_root: Path, info: dict[str, Any]) -> None:
    public = {k: v for k, v in info.items() if not str(k).startswith("_")}
    (dest_root / "probe.yaml").write_text(
        yaml.safe_dump(public, allow_unicode=True, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def probe_cann_9201(ctx: Any) -> dict[str, Any]:
    """Inspect CANN headers and platform_config for native 9201 support."""
    report = _ini_report(ctx)
    overlay_names, native_names = _sample_probe_files(ctx)
    report["overlay_files"] = overlay_names
    report["native_files"] = native_names
    if overlay_names:
        report["headers"] = "overlay"
    elif native_names:
        report["headers"] = "native"
    return report


def materialize_9201_overlay(ctx: Any, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write overlay headers (tree-preserving) and return the public probe record."""
    op_dir = str(getattr(ctx, "op_dir", "") or "")
    arch = str(getattr(ctx, "arch_dir", "") or "")
    if not op_dir or not arch:
        return dict(report or _ini_report(ctx))
    dest_root = overlay_dir(op_dir, arch)
    dest_root.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_overlay(dest_root, ctx)
    if cached is not None:
        return cached

    info = dict(report or _ini_report(ctx))
    trees = _iter_header_trees(ctx)
    sources = _collect_overlay_sources(ctx)
    _clear_overlay_payload(dest_root)

    written: list[str] = []
    for rel, src in sources:
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
            dest.write_text(expand_3510_gates(text), encoding="utf-8")
            written.append(rel.replace("\\", "/"))
        except OSError:
            continue

    sample_overlay, sample_native = _sample_probe_files(ctx)
    includes = _mapped_overlay_includes(dest_root, ctx, trees)
    info["overlay_files"] = written[:24] if written else sample_overlay
    info["native_files"] = sample_native
    info["overlay_file_count"] = len(written)
    info["overlay_includes"] = includes
    info["overlay_dir"] = str(dest_root).replace("\\", "/") if written else ""
    info["logic"] = _OVERLAY_LOGIC
    info.update(_cache_key(ctx))
    if written:
        info["headers"] = "overlay"
    elif sample_native:
        info["headers"] = "native"
    info["probe_path"] = str(dest_root / "probe.yaml").replace("\\", "/")
    _write_probe(dest_root, info)
    return info


def _prepend_overlay_includes(ctx: Any, dirs: list[str]) -> None:
    current = [str(p).replace("\\", "/").rstrip("/") for p in (ctx.overlay_includes or [])]
    seen = {p.lower() for p in current}
    new: list[str] = []
    for item in dirs:
        posix = str(item).replace("\\", "/").rstrip("/")
        if not posix or posix.lower() in seen:
            continue
        seen.add(posix.lower())
        new.append(posix)
    if new:
        ctx.overlay_includes = new + current


def attach_9201_overlay(ctx: Any) -> dict[str, Any]:
    """If this BuildContext is 920r1, probe CANN and prepend overlay ``-I``."""
    arch = str(getattr(ctx, "arch_dir", "") or "")
    if canonicalize_architecture(arch) != "arch-920r1":
        return {}
    cann = Path(getattr(ctx, "cann_root", "") or "")
    if not cann.is_dir():
        report = {"npu_arch": 9201, "headers": "no_cann", "ini": "no_cann"}
        ctx.cann_9201 = report
        return report
    try:
        report = materialize_9201_overlay(ctx)
    except OSError as exc:
        report = {"npu_arch": 9201, "headers": "probe_failed", "error": str(exc)[:200]}
        ctx.cann_9201 = report
        return report
    ctx.cann_9201 = report
    _prepend_overlay_includes(ctx, list(report.get("overlay_includes") or []))
    return report
