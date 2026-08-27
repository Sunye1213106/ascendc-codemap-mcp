# -*- coding: utf-8 -*-
"""Operator-source discovery that is not FAG-directory-shaped.

FAG keeps Host under ``op_host/archXX/`` and kernel entries that
``#include "archXX/..."``. Other ops use ``op_host/op_tiling/``,
``./archXX/`` includes, and ``extern "C" __global__``. Walks that only
accept the FAG spelling drop KERNEL / packing / TilingData.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

# Module scope on purpose. These were imported inside the functions below, and
# the functions run tens of thousands of times per analyze, so every call paid a
# `sys.modules` lookup for a module that is always already loaded. `paths` pulls
# in nothing from this package, so there is no cycle to avoid here.
from ascendc_codemap_mcp.engine.paths import ops_root, resolved


def _text(path: Path | str) -> str:
    from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

    return read_text(path)


# Hardware-generation directory / path token:
#   arch35, arch22     — published DAV_NNNN → first two digits
#   arch-920r1         — unpublished DAV_9201 (hyphenated product name)
#   arch920r1          — same identity, unhyphenated spelling on disk / in intent
# More-specific ``rN`` spellings must precede bare ``archNN`` so ``arch920r1``
# is not consumed as ``arch920``.
ARCH_NAME = r"arch(?:-\d+r\d+|\d+r\d+|\d+)"
ARCH_DIR_RE = re.compile(rf"^{ARCH_NAME}$")
ARCH_IN_PATH_RE = re.compile(rf"(?:^|/)({ARCH_NAME})(?:/|$)")
# Path segment `/arch22/` or filename token `_arch22.h` / `foo_arch35_bar.h`.
_ARCH_TOKEN_RE = re.compile(rf"(?:^|[/_.-])({ARCH_NAME})(?:[/_.-]|$)")
# Product slot when the tree has no ``arch*`` folders. Not a hardware generation:
# official repos split implementations under arch22/arch35/…; a third-party tree
# without those folders is one implementation and is built together.
UNIFIED_ARCH_DIR = "default"
_ARCH_ALIAS = {
    "arch-920r1": "arch-920r1",
    "arch920r1": "arch-920r1",
    "dav_9201": "arch-920r1",
    "dav-9201": "arch-920r1",
    "9201": "arch-920r1",
}
# One-way: unpublished 920r1 may read arch35 sources. arch35 never reads 920r1.
_SOURCE_COUSINS: dict[str, frozenset[str]] = {
    "arch-920r1": frozenset({"arch35"}),
}
_CPP_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_ANY_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)

# template <...> __global__, extern "C" __global__, or a plain __global__.
# Qualifier order is not operator-specific: both `__global__ __aicore__` and
# `__aicore__ __global__` (and `__global__` alone) appear in ops-transformer.
# Do not use DOTALL ``.*?`` here: IFA kernel TUs are multi-MB and that form
# spends tens of seconds backtracking.
_KERNEL_QUALS = r"(?:__global__\s+(?:__aicore__\s+)?|__aicore__\s+__global__\s+)"
GLOBAL_KERNEL_RE = re.compile(
    r"(?:template\s*<(?P<tpl>[^>]{0,800})>\s*)?"
    r"(?:extern\s+\"C\"\s+)?"
    rf"{_KERNEL_QUALS}void\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]{0,16000})\)(?:\s|//[^\n]*)*\{",
)
KERNEL_ENTRY_NAME_RE = re.compile(rf"{_KERNEL_QUALS}void\s+([A-Za-z_]\w*)")

_TILING_HEADER_GLOBS = (
    "*tiling_data*.h",
    "*tiling_data*.hpp",
    "*_tiling.h",
    "*_tiling.hpp",
    "*tiling*.h",
)


def canonicalize_architecture(value: str | None) -> str:
    """Map aliases onto the product arch name. Unknown input is returned as-is.

    ``arch920r1`` / ``DAV_9201`` / ``9201`` → ``arch-920r1``. Published
    ``arch35`` stays ``arch35``. Empty stays empty.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    low = re.sub(r"\s+", "", raw).lower()
    if low in _ARCH_ALIAS:
        return _ARCH_ALIAS[low]
    compact = low.replace("-", "").replace("_", "")
    if compact in {"dav9201"}:
        return "arch-920r1"
    m = re.fullmatch(r"arch-?(\d+)r(\d+)", low)
    if m:
        return f"arch-{m.group(1)}r{m.group(2)}"
    m = re.fullmatch(r"arch(\d+)", low)
    if m:
        return f"arch{m.group(1)}"
    return raw


def is_variant_architecture(value: str | None) -> bool:
    """True when ``value`` names an ``arch*`` implementation folder.

    ``arch35`` / ``arch-920r1`` split official ops-transformer trees.
    ``default`` and empty are the single-implementation product slot.
    """
    canon = canonicalize_architecture(value)
    return bool(canon) and ARCH_DIR_RE.match(canon) is not None


def is_product_architecture(value: str | None) -> bool:
    """True for an ``arch*`` variant folder or the unified ``default`` product slot.

    Product paths live under ``.ascendc-codemap/<slot>/``. ``default`` is not an
    on-disk source folder; it names the single-implementation product tree.
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    if raw == UNIFIED_ARCH_DIR:
        return True
    return is_variant_architecture(raw)


def architectures_match(left: str | None, right: str | None) -> bool:
    """True when both names are the same compile identity (hyphen optional)."""
    a = canonicalize_architecture(left)
    b = canonicalize_architecture(right)
    return bool(a) and a == b


def identity_arch_names(architecture: str | None) -> frozenset[str]:
    """On-disk spellings of this architecture, not ISA cousins."""
    raw = str(architecture or "").strip()
    if not raw:
        return frozenset()
    canon = canonicalize_architecture(raw)
    names = {raw, raw.lower(), canon}
    if canon == "arch-920r1":
        names.update({"arch-920r1", "arch920r1"})
    return frozenset(n for n in names if n)


def arch_scope_names(architecture: str | None) -> frozenset[str]:
    """Identity folders plus one-way source cousins (920r1 may read arch35)."""
    ident = identity_arch_names(architecture)
    canon = canonicalize_architecture(architecture)
    extra = _SOURCE_COUSINS.get(canon, frozenset())
    return ident | extra


def architecture_in_scope(name: str | None, architecture: str | None) -> bool:
    """True when ``name`` is this arch or a permitted cousin folder."""
    token = str(name or "").strip()
    if not token:
        return False
    scope = arch_scope_names(architecture)
    low = {s.lower() for s in scope}
    if token in scope or token.lower() in low:
        return True
    return canonicalize_architecture(token) in {canonicalize_architecture(s) for s in scope}


def match_on_disk_architecture(pin: str | None, known: Iterable[str]) -> str:
    """Resolve a user pin onto an existing ``arch*`` folder name.

    Exact disk spelling wins; ``arch920r1`` matches ``arch-920r1``.
    """
    raw = str(pin or "").strip()
    names = [str(n).strip() for n in known if str(n).strip()]
    if not raw or not names:
        return raw
    if raw in names:
        return raw
    by_l = {n.lower(): n for n in names}
    hit = by_l.get(raw.lower())
    if hit:
        return hit
    canon = canonicalize_architecture(raw)
    hits = [n for n in names if canonicalize_architecture(n) == canon]
    if len(hits) == 1:
        return hits[0]
    if "arch-920r1" in hits:
        return "arch-920r1"
    return hits[0] if hits else raw


def iter_identity_arch_dirs(parent: Path, architecture: str) -> list[Path]:
    """Existing identity ``arch*`` folders under ``parent`` (not cousins)."""
    out: list[Path] = []
    seen: set[str] = set()
    for name in sorted(identity_arch_names(architecture)):
        d = Path(parent) / name
        try:
            if not d.is_dir():
                continue
            key = str(d.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def iter_cousin_arch_dirs(parent: Path, architecture: str) -> list[Path]:
    """Existing one-way cousin folders (``arch35`` when analysing 920r1)."""
    canon = canonicalize_architecture(architecture)
    extra = _SOURCE_COUSINS.get(canon, frozenset())
    out: list[Path] = []
    seen: set[str] = set()
    for name in sorted(extra):
        d = Path(parent) / name
        try:
            if not d.is_dir():
                continue
            key = str(d.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def iter_arch_source_dirs(parent: Path, architecture: str) -> list[Path]:
    """Identity folders first, then cousins. Used for host tiling / headers."""
    return iter_identity_arch_dirs(parent, architecture) + iter_cousin_arch_dirs(
        parent, architecture
    )


def path_owned_architecture(path: Path) -> str:
    """``archNN`` folder the file sits in. Empty when the path is arch-neutral.

    A file under ``op_kernel/arch22/`` belongs to arch22 even if it includes a
    shared ``*_arch35.h`` (A2/A3 pipeline reuse). Include-derived architecture
    is only for entries that live *above* the arch folders.
    """
    found = [part.lower() for part in Path(path).parts if ARCH_DIR_RE.match(part)]
    if len(found) == 1:
        return found[0]
    return ""


@lru_cache(maxsize=1 << 15)
def _is_other_arch_path(path_text: str, arch: str) -> bool:
    if not is_variant_architecture(arch):
        return False
    for part in Path(path_text).parts:
        if ARCH_DIR_RE.match(part) and not architecture_in_scope(part, arch):
            return True
    return False


def is_other_arch_path(path: Path | str, architecture: str) -> bool:
    """True when a path segment is an ``arch*`` folder outside this scope.

    Cousin folders (``arch35`` while analysing ``arch-920r1``) are in scope.

    Memoized on the spelling: this decides a question about a path's *text*,
    not about anything on disk, and callers ask it about the same few dozen
    files tens of thousands of times while classifying walk-cited references.
    """
    return _is_other_arch_path(str(path), str(architecture or "").strip())


def keep_lexical_kernel_path(path: Path | str, architecture: str) -> bool:
    """True when METHOD/CALLS / SourceIndex may scan this kernel file.

    Clang may confirm a foreign-arch tiling header for types. That path stays
    in ``selected_kernel_files``. Lexical body scans must not mint a second
    architecture's kernel graph from ``op_kernel/archNN/**``.
    """
    return not is_other_arch_path(path, architecture)


def include_root_owned_architecture(path: Path | str) -> str:
    """Arch folder a ``-I`` root sits in. Empty when the directory is arch-neutral.

    ``op_kernel/arch35`` → ``arch35``; ``op_kernel`` → ``""``.
    """
    return path_owned_architecture(Path(path))


_ENTRY_TU_SUFFIXES = {".cpp", ".cc", ".cxx"}


@lru_cache(maxsize=1 << 15)
def _is_foreign_arch_entry_tu(path_text: str, architecture: str) -> bool:
    p = Path(path_text)
    if p.suffix.lower() not in _ENTRY_TU_SUFFIXES:
        return False
    if not is_variant_architecture(architecture):
        return False
    owned = path_owned_architecture(p)
    if not owned:
        return False
    return not architectures_match(owned, architecture)


def is_foreign_arch_entry_tu(path: Path | str, architecture: str) -> bool:
    """True for another architecture's compile unit, not an included header.

    Cousin ``arch35/*.cpp`` stays a foreign entry when analysing 920r1: the
    sources may be included, but they are not this arch's kernel TU.

    Memoized for the same reason as `is_other_arch_path`: purely a question
    about the spelling, asked repeatedly about a small set of files.
    """
    return _is_foreign_arch_entry_tu(str(path), str(architecture or "").strip())


def includes_architecture(text: str, architecture: str) -> bool:
    """True when the TU pulls the current arch, including ``./arch35/``."""
    names = arch_scope_names(architecture)
    if not names:
        return False
    blob = text.replace("\\", "/")
    return any(f"{name}/" in blob for name in names)


def arch_tokens_in_include(include: str) -> set[str]:
    """``archNN`` markers in an include path or filename (not ``architecture.h``)."""
    text = "/" + (include or "").replace("\\", "/")
    return {m.group(1).lower() for m in _ARCH_TOKEN_RE.finditer(text)}


def arch_number(architecture: str) -> int:
    """Numeric rank for apt-vs-plain entry picking. ``arch-920r1`` → 920."""
    raw = canonicalize_architecture(architecture) or str(architecture or "").strip().lower()
    m = re.fullmatch(r"arch(\d+)", raw)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"arch-(\d+)r\d+", raw)
    if m:
        return int(m.group(1))
    return 0


def pick_kernel_entry(targets: list[Path], architecture: str) -> Path | None:
    """Pick the kernel TU for this architecture.

    A file under ``op_kernel/archNN/`` is owned by that folder. Root-level
    entries (``foo.cpp`` vs ``foo_apt.cpp``) use include-derived architecture
    when it is unique; otherwise apt vs plain is a candidate ranking by
    arch generation, not a semantic identity for the TU.

    ``arch-920r1`` may keep a root ``*_apt.cpp`` that ``#include "arch35/..."``.
    A compile unit sitting in a cousin folder is last-resort only.
    """
    arch = str(architecture or "").strip()
    arch_n = arch_number(arch)
    matching: list[Path] = []
    unscoped: list[Path] = []
    cousin_hits: list[Path] = []
    for raw in targets:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            owned = path_owned_architecture(path)
        except OSError:
            owned = ""
        include_owned = ""
        if not owned:
            try:
                include_owned = entry_include_architecture(_text(path))
            except OSError:
                include_owned = ""
            owned = include_owned
        if owned and arch and not architecture_in_scope(owned, arch):
            continue
        if owned and architectures_match(owned, arch):
            matching.append(path)
        elif include_owned and architecture_in_scope(include_owned, arch):
            unscoped.append(path)
        elif owned:
            cousin_hits.append(path)
        else:
            unscoped.append(path)
    pool = matching or unscoped or cousin_hits
    if not pool:
        return None
    apt = [p for p in pool if p.name.endswith("_apt.cpp")]
    plain = [p for p in pool if not p.name.endswith("_apt.cpp")]
    chosen = (apt or plain) if arch_n >= 35 else (plain or apt)
    return sorted(chosen, key=lambda p: p.as_posix())[0]


def follow_repo_includes(
    seeds: Iterable[Path],
    *,
    repo_root: Path,
    architecture: str = "",
) -> list[Path]:
    """Quoted includes under the ops repo (sibling operators), not CANN."""
    root = Path(repo_root).resolve()
    out: list[Path] = []
    seen: set[Path] = set()
    pending = [Path(p) for p in seeds if Path(p).is_file()]
    while pending:
        path = pending.pop()
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        for included in resolve_quoted_includes(path):
            try:
                rel = included.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if "/tests/" in f"/{rel}/" or rel.startswith("tests/"):
                continue
            if is_other_arch_path(included, architecture):
                continue
            pending.append(included)
            if included.suffix.lower() in {".h", ".hpp", ".hh"}:
                out.append(included)
    return out


def entry_include_architecture(text: str) -> str:
    """Which ``archNN`` a root-level kernel entry builds, from its includes.

    Entries sit above ``archNN/`` folders, so the path alone cannot tell.
    Matching ``scope_scan.entry_architecture``: one concrete arch wins; mixed
    or absent markers yield empty so a preprocessor-gated entry (arch22 header
    plus an ``arch38/`` include behind ``#if``) is not rejected.
    """
    found: set[str] = set()
    for inc in _ANY_INCLUDE_RE.findall(text or ""):
        found |= arch_tokens_in_include(inc)
    if len(found) == 1:
        return next(iter(found))
    return ""


def quoted_include_basenames(path: Path) -> set[str]:
    """Basenames from ``#include "..."`` in ``path`` (not angle includes)."""
    try:
        text = _text(path)
    except OSError:
        return set()
    return {Path(inc.replace("\\", "/")).name.lower() for inc in _QUOTED_INCLUDE_RE.findall(text)}


def resolve_quoted_includes(path: Path) -> list[Path]:
    """Quoted includes resolved relative to the including file."""
    parent = Path(path).parent
    out: list[Path] = []
    try:
        text = _text(path)
    except OSError:
        return out
    for inc in _QUOTED_INCLUDE_RE.findall(text):
        cand = resolved(parent / inc.replace("\\", "/"))
        if cand.is_file():
            out.append(cand)
    return out


def iter_cpp(root: Path, *, recursive: bool = True) -> Iterator[Path]:
    if not root.is_dir():
        return
    it = root.rglob("*") if recursive else root.glob("*")
    for path in it:
        if path.is_file() and path.suffix.lower() in _CPP_SUFFIXES:
            yield path


def _resolve_confirmed_path(op: Path, rel: str) -> Path | None:
    """Resolve a prepare-confirmed path against the operator, family, or ops root.

    Clang records sibling-operator includes (``moe_distribute_dispatch_v2/...``)
    relative to the family folder, not ``op_dir``. ``op / rel`` then misses the
    file and TilingData structs living next door drop out of analyze.
    """
    rel_path = Path(str(rel or "").replace("\\", "/"))
    if not str(rel_path):
        return None
    candidates = [op / rel_path, op.parent / rel_path]
    try:
        repo = ops_root()
        if repo is not None:
            candidates.append(Path(repo) / rel_path)
    except Exception:  # noqa: BLE001
        pass
    seen: set[Path] = set()
    for cand in candidates:
        hit = resolved(cand)
        if hit in seen:
            continue
        seen.add(hit)
        if hit.is_file():
            return hit
    return None


#: Keyed by operator, architecture and the scope set's mtime/size. Prepare
#: writes that file and analyze reads it, and `run_full_init` runs both in one
#: interpreter, so keying on the operator alone would serve analyze a list built
#: before prepare rewrote it. Stat is the cheap part; parsing the YAML and
#: resolving sixty paths is what this avoids.
_CONFIRMED_MEMO: dict[tuple[str, str, int, int], tuple[tuple[Path, str], ...]] = {}
_CONFIRMED_LOCK = threading.Lock()


def _confirmed_with_rel(root: Path, architecture: str) -> tuple[tuple[Path, str], ...]:
    """Confirmed files paired with their operator-relative spelling.

    Empty means "prepare has not produced a usable list", which callers read as
    permission to fall back to layout heuristics.

    One analyze asked for this set 53 times. Each ask re-read the same 16 KB
    scope set, re-resolved the same sixty paths against three candidate bases,
    and recomputed the same sixty relative spellings -- none of which can change
    while a stage runs. The relative spelling is cached alongside the path
    because every consumer classifies on it, and recomputing it was the rest of
    the cost once the parse was gone.
    """
    arch = str(architecture or "").strip()
    if not arch:
        return ()
    op = resolved(root)
    scope_path = op / ".ascendc-codemap" / arch / "summary" / "scope_set.yaml"
    try:
        stamp = scope_path.stat()
    except OSError:
        return ()
    key = (str(op), arch, stamp.st_mtime_ns, stamp.st_size)
    with _CONFIRMED_LOCK:
        hit = _CONFIRMED_MEMO.get(key)
    if hit is not None:
        return hit

    try:
        from ascendc_codemap_mcp.engine.yaml_io import read_yaml

        doc = read_yaml(scope_path)
    except Exception:  # noqa: BLE001
        return ()
    raw = doc.get("confirmed_source_files") if isinstance(doc, dict) else None
    if not isinstance(raw, list) or not raw:
        return ()
    rows: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for item in raw:
        rel = str(item or "").replace("\\", "/").strip()
        if not rel:
            continue
        cand = _resolve_confirmed_path(op, rel)
        if cand is None or cand in seen:
            continue
        seen.add(cand)
        rows.append((cand, _posix_rel(op, cand)))
    built = tuple(rows)
    with _CONFIRMED_LOCK:
        _CONFIRMED_MEMO[key] = built
    return built


def load_confirmed_source_files(root: Path, architecture: str) -> list[Path] | None:
    """Clang-confirmed files from prepare, or None when that list is not ready.

    Analyze/stub scans must not invent a second file universe. Layout heuristics
    remain only as bootstrap before ``summary/scope_set.yaml`` exists.
    """
    rows = _confirmed_with_rel(root, architecture)
    return [path for path, _rel in rows] or None


def _posix_rel(root: Path, path: Path) -> str:
    try:
        return resolved(path).relative_to(resolved(root)).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def _is_kernel_scope_rel(rel: str) -> bool:
    posix = rel.replace("\\", "/")
    return (
        posix.startswith("op_kernel/")
        or "/op_kernel/" in posix
        or posix.startswith("common/op_kernel/")
        or "/common/op_kernel/" in posix
    )


def _is_host_scope_rel(rel: str) -> bool:
    posix = rel.replace("\\", "/")
    return posix.startswith("op_host/") or "/op_host/" in posix


def host_ir_keeps_file(path: str | Path, architecture: str) -> bool:
    """Whether a Host IR definition site belongs in this product.

    Clang indexes every function in the include closure. TilingData headers
    under ``op_kernel/`` (including a foreign ``arch22/`` tree while this
    product is arch35) then show up as host FUNCTION entities: named, located,
    and reachable from nothing, because they were never called from Host code.
    The definition's path is the compiler's own record of which translation
    the symbol belongs to -- kernel-side and other-arch files are not Host.
    """
    text = str(path or "").replace("\\", "/")
    if not text:
        return True
    if is_other_arch_path(text, architecture):
        return False
    return not _is_kernel_scope_rel(text)


def _is_generated_rel(rel: str) -> bool:
    posix = rel.replace("\\", "/")
    return posix.startswith(".ascendc-codemap/") or "/.ascendc-codemap/" in posix


def _confirmed_subset(
    root: Path,
    architecture: str,
    *,
    predicate,
) -> list[Path] | None:
    rows = _confirmed_with_rel(root, architecture)
    if not rows:
        return None
    out: list[Path] = []
    seen: set[Path] = set()
    for path, rel in rows:
        if _is_generated_rel(rel):
            continue
        if not predicate(rel, path):
            continue
        key = resolved(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def selected_kernel_files(
    root: Path,
    architecture: str,
    *,
    kernel_entry: Path | None = None,
) -> list[Path]:
    """Kernel files for this architecture.

    After prepare, this is the kernel-side Clang set (included headers from
    another ``arch*`` folder stay when Clang confirmed them). Before that,
    layout heuristics bootstrap entry TUs and the current arch folder.
    """
    confirmed = _confirmed_subset(
        root,
        architecture,
        predicate=lambda rel, _path: _is_kernel_scope_rel(rel),
    )
    if confirmed is not None:
        out = list(confirmed)
        if kernel_entry is not None and kernel_entry.is_file():
            if not is_foreign_arch_entry_tu(kernel_entry, architecture):
                key = kernel_entry.resolve()
                if key not in {p.resolve() for p in out}:
                    owns = entry_include_architecture(_text(kernel_entry))
                    if not owns or architecture_in_scope(owns, architecture):
                        out.append(kernel_entry)
        return out

    out: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    add(kernel_entry)
    kernel_root = Path(root) / "op_kernel"
    for folder in iter_identity_arch_dirs(kernel_root, architecture):
        for path in sorted(iter_cpp(folder)):
            add(path)
    arch = str(architecture or "").strip().lower()
    arch_n = arch_number(arch)
    if kernel_root.is_dir():
        root_tus: list[tuple[Path, str, str]] = []
        for path in sorted(iter_cpp(kernel_root, recursive=False)):
            try:
                text = _text(path)
            except OSError:
                continue
            owns = entry_include_architecture(text)
            if owns and arch and not architecture_in_scope(owns, architecture):
                continue
            root_tus.append((path, text, owns))
        apt_here = any(p.name.endswith("_apt.cpp") for p, _t, _o in root_tus)
        for path, text, owns in root_tus:
            if includes_architecture(text, architecture):
                add(path)
                continue
            if path.name.endswith("_apt.cpp") and arch_n >= 35:
                add(path)
                continue
            if arch_n and arch_n < 35 and not path.name.endswith("_apt.cpp"):
                if "__aicore__" in text or "GET_TILING_DATA" in text:
                    add(path)
                continue
            if not apt_here and not owns and ("__aicore__" in text or "GET_TILING_DATA" in text):
                add(path)
    op_root = Path(root).resolve()
    pending = list(out)
    while pending:
        path = pending.pop()
        for included in resolve_quoted_includes(path):
            if is_other_arch_path(included, architecture):
                continue
            try:
                included.resolve().relative_to(op_root)
            except ValueError:
                continue
            before = len(seen)
            add(included)
            if len(seen) > before:
                pending.append(included)
    return out


def selected_host_files(root: Path, architecture: str) -> list[Path]:
    """Host sources for this arch, including ``op_host/op_tiling/``."""
    confirmed = _confirmed_subset(
        root,
        architecture,
        predicate=lambda rel, _path: _is_host_scope_rel(rel),
    )
    if confirmed is not None:
        return confirmed
    host_root = Path(root) / "op_host"
    out: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(iter_cpp(host_root)):
        if is_other_arch_path(path, architecture):
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def selected_tiling_headers(root: Path, architecture: str) -> list[Path]:
    """TilingData headers under op_host / op_kernel, current arch only."""
    def _tiling_header(rel: str, path: Path) -> bool:
        if path.suffix.lower() not in {".h", ".hpp", ".hh"}:
            return False
        return "tiling" in path.name.lower() or "tiling" in rel.lower()

    confirmed = _confirmed_subset(root, architecture, predicate=_tiling_header)
    if confirmed is not None:
        return confirmed
    hits: list[Path] = []
    seen: set[Path] = set()
    for base in (Path(root) / "op_host", Path(root) / "op_kernel"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".h", ".hpp", ".hh"}:
                continue
            if is_other_arch_path(path, architecture):
                continue
            name = path.name.lower()
            if "tiling" not in name:
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            hits.append(path)
    return hits


def _kernel_include_closure(root: Path, architecture: str) -> list[Path]:
    """Quoted-include walk from current-arch kernel entries (no other-arch)."""
    kernel_files = list(selected_kernel_files(root, architecture))
    by_key = {resolved(p): p for p in kernel_files}
    entries: list[Path] = []
    for path in kernel_files:
        if path.suffix.lower() not in {".cpp", ".cc", ".cxx"}:
            continue
        if is_foreign_arch_entry_tu(path, architecture):
            continue
        try:
            text = _text(path)
        except OSError:
            continue
        if GLOBAL_KERNEL_RE.search(text):
            entries.append(path)
    order: list[Path] = []
    seen: set[Path] = set()
    pending = list(entries)
    while pending:
        path = pending.pop(0)
        key = resolved(path)
        if key in seen:
            continue
        seen.add(key)
        order.append(path)
        for inc in resolve_quoted_includes(path):
            if is_other_arch_path(inc, architecture):
                continue
            hit = resolved(inc)
            if hit in seen:
                continue
            pending.append(by_key.get(hit, inc))
    return order


def _path_is_under(path: Path, root: Path) -> bool:
    """True when ``path`` lives in this operator tree, not a sibling op include."""
    try:
        resolved(path).relative_to(resolved(root))
        return True
    except ValueError:
        return False


def _first_tpl_marker_file(
    root: Path, architecture: str, marker: str
) -> list[Path]:
    kernel_files = list(selected_kernel_files(root, architecture))
    for path in kernel_files:
        if path.suffix.lower() not in {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}:
            continue
        try:
            text = _text(path)
        except OSError:
            continue
        if marker not in text:
            continue
        for inc in resolve_quoted_includes(path):
            if is_other_arch_path(inc, architecture):
                continue
            if not _path_is_under(inc, root):
                continue
            try:
                inc_text = _text(inc)
            except OSError:
                continue
            if marker in inc_text:
                return [inc]
        if _path_is_under(path, root):
            return [path]
    return []


_PACKING_CAST_WORDS = frozenset(
    {
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "int",
        "bool",
        "true",
        "false",
        "sizeof",
    }
)
_LINE_MARKER_RE = re.compile(r"^#\s+(?:\d+\s+)?\"([^\"]+)\"", re.MULTILINE)


def _host_get_tpl_packing_arities(root: Path, architecture: str) -> list[int]:
    """Argument counts of Host ``GET_TPL_TILING_KEY(...)`` calls (longest first)."""
    from ascendc_codemap_mcp.engine.tpl_dsl import _balanced_paren_body, _split_args

    arities: list[int] = []
    for path in selected_host_files(root, architecture):
        try:
            text = _text(path)
        except OSError:
            continue
        for match in re.finditer(r"\bGET_TPL_TILING_KEY\s*\(", text):
            try:
                body = _balanced_paren_body(text, match.end() - 1)
            except ValueError:
                continue
            n = len(_split_args(body))
            if n:
                arities.append(n)
    arities.sort(reverse=True)
    return arities


def _host_get_tpl_dim_names(root: Path, architecture: str) -> list[str]:
    """Identifier order from the first Host ``GET_TPL_TILING_KEY(...)`` call."""
    from ascendc_codemap_mcp.engine.tpl_dsl import _balanced_paren_body, _split_args

    for path in selected_host_files(root, architecture):
        try:
            text = _text(path)
        except OSError:
            continue
        match = re.search(r"\bGET_TPL_TILING_KEY\s*\(", text)
        if not match:
            continue
        try:
            body = _balanced_paren_body(text, match.end() - 1)
        except ValueError:
            continue
        names: list[str] = []
        for arg in _split_args(body):
            idents = [
                tok
                for tok in re.findall(r"[A-Za-z_]\w*", arg)
                if tok not in _PACKING_CAST_WORDS
            ]
            if not idents:
                continue
            names.append(idents[-1] if len(idents) > 1 else idents[0])
        if names:
            return names
    return []


def _decl_dim_names(text: str) -> list[str]:
    from ascendc_codemap_mcp.engine.tpl_dsl import parse_args_decl

    return [dim.name for dim in parse_args_decl(text).dims]


def _tpl_decl_rank(
    path: Path, text: str, packing: list[str], arities: list[int]
) -> tuple[int, int, int, int, int]:
    """Lower is better: GET_TPL arity, not a nested variant, packing names, header, more dims."""
    dims = _decl_dim_names(text)
    n = len(dims)
    best_arity = arities[0] if arities else 0
    arity_gap = abs(n - best_arity) if best_arity else 0
    nested = 0 if path.parent.name == "op_kernel" else 1
    overlap = sum(1 for name in packing if name in dims) if packing else 0
    name = path.name.lower()
    return (
        arity_gap,
        nested,
        -overlap,
        0 if "tiling_key" in name else 1,
        0 if path.suffix.lower() in {".h", ".hpp", ".hh"} else 1,
        -n,
    )


def tpl_decl_candidates_from_preprocess(stdout: str, root: Path) -> list[Path]:
    """Files clang.exe ``-E`` actually included that still contain ARGS_DECL."""
    root_r = root.expanduser().resolve()
    hits: list[Path] = []
    seen: set[Path] = set()
    for match in _LINE_MARKER_RE.finditer(stdout or ""):
        raw = match.group(1).replace("\\\\", "/").replace("\\", "/")
        cand = Path(raw)
        if not cand.is_file():
            continue
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        try:
            resolved.relative_to(root_r)
        except ValueError:
            continue
        try:
            text = _text(resolved)
        except OSError:
            continue
        if "ASCENDC_TPL_ARGS_DECL" not in text:
            continue
        seen.add(resolved)
        hits.append(resolved)
    return hits


def list_tpl_decl_candidates(root: Path, architecture: str) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in _kernel_include_closure(root, architecture):
        if not _path_is_under(path, root):
            continue
        try:
            key = path.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        try:
            text = _text(path)
        except OSError:
            continue
        if "ASCENDC_TPL_ARGS_DECL" not in text:
            continue
        seen.add(key)
        hits.append((path, text))
    return hits


def _scan_op_kernel_tpl_decls(root: Path, architecture: str) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    kernel_root = root / "op_kernel"
    if not kernel_root.is_dir():
        return hits
    paths = list(kernel_root.rglob("*tiling_key*.h")) + list(kernel_root.glob("*.h"))
    for path in paths:
        if not path.is_file() or is_other_arch_path(path, architecture):
            continue
        try:
            key = path.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        try:
            text = _text(path)
        except OSError:
            continue
        if "ASCENDC_TPL_ARGS_DECL" not in text:
            continue
        seen.add(key)
        hits.append((path, text))
    return hits


def tpl_decl_files(root: Path, architecture: str) -> list[Path]:
    """One TPL ARGS_DECL schema: packing-aligned header, not the first literal hit.

    Layout globs and Clang scope often also list sibling ``*_tiling_key.h``
    files (apt vs non-apt, ifdef-gated variants). Merging those schemas
    inflates TILING_KEY counts so GET_TPL_TILING_KEY packing never matches.
    Fusion wrappers that ``#include "../../../other_op/...tiling_key.h"`` must
    not inherit that sibling's ARGS_DECL as this operator's source-declared keys.
    An entry ``.cpp`` may contain a small C++-template ARGS_DECL that must not
    crowd out the real ``*_tiling_key.h`` schema.
    """
    packing = _host_get_tpl_dim_names(root, architecture)
    arities = _host_get_tpl_packing_arities(root, architecture)
    hits = list_tpl_decl_candidates(root, architecture)
    if arities:
        target = arities[0]

        def _gap(item: tuple[Path, str]) -> int:
            return abs(len(_decl_dim_names(item[1])) - target)

        best = min((_gap(item) for item in hits), default=99)
        if best > 0:
            seen = {p.resolve() for p, _ in hits}
            for extra in _scan_op_kernel_tpl_decls(root, architecture):
                try:
                    key = extra[0].resolve()
                except OSError:
                    continue
                if key in seen:
                    continue
                if _gap(extra) < best:
                    hits.append(extra)
                    seen.add(key)
    if not hits:
        fallback = _first_tpl_marker_file(root, architecture, "ASCENDC_TPL_ARGS_DECL")
        return fallback
    hits.sort(key=lambda item: _tpl_decl_rank(item[0], item[1], packing, arities))
    return [hits[0][0]]


def tpl_sel_files(root: Path, architecture: str) -> list[Path]:
    """ARGS_SEL headers reachable from the current-arch kernel entry.

    DECL and SEL are often split: the entry includes ``archNN/*_tiling_key.h``
    (SEL) which includes ``*_tiling_key_decl.h`` (DECL). Stopping at the first
    ARGS_DECL file drops the selections, so commit cannot rebuild TPL views.
    """
    hits: list[Path] = []
    seen: set[Path] = set()
    for path in _kernel_include_closure(root, architecture):
        if not _path_is_under(path, root):
            continue
        try:
            text = _text(path)
        except OSError:
            continue
        if "ASCENDC_TPL_ARGS_SEL" not in text:
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        hits.append(path)
    if hits:
        return hits
    return _first_tpl_marker_file(root, architecture, "ASCENDC_TPL_ARGS_SEL")


def select_tpl_decl_header(root: Path, architecture: str) -> Path | None:
    hits = tpl_decl_files(root, architecture)
    return hits[0] if hits else None
