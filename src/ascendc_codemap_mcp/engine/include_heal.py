# -*- coding: utf-8 -*-
"""Heal missing-header clang probes by adding -I roots to BuildContext.

``build_context.yaml`` is a generic ``-I/-D`` baseline. Operators still include
CANN / family headers from directories that yaml never listed. Prepare used to
fail as ``clang_probe_unclean`` / ``SCOPE_VALIDATE_BLOCKED``; this module finds
the header in the CANN tree (extracted ``cann-*`` or official install) or ops
repo, adds the matching include directory, and retries. Per-operator extras are
persisted so extract uses the same flags. The shared yaml is never rewritten.
When the script still cannot resolve a header, a staged LLM Action writes
``staging.yaml``; deterministic ``heal_promote`` appends validated ``-I`` dirs
to extras (``source: heal_promote``). Official CANN packages are complete; do
not treat a hardcoded relative path as a missing vendor file.

Disable with ``UO_INCLUDE_HEAL=0``. Round cap: ``UO_INCLUDE_HEAL_ROUNDS`` (8).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from ascendc_codemap_mcp.engine.paths import require_architecture
from ascendc_codemap_mcp.engine.source_layout import (
    arch_tokens_in_include,
    architectures_match,
    include_root_owned_architecture,
    is_other_arch_path,
    path_owned_architecture,
)

MISSING_RE = re.compile(
    r"""['"<]([^'"><\s]+?\.(?:h|hpp|hh|inc|cuh))['">]\s+file not found""",
    re.IGNORECASE,
)
UNKNOWN_TYPE_RE = re.compile(r"unknown type name '([A-Za-z_]\w*)'")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
_TYPE_DECL_RE = re.compile(
    r"\b(?:struct|class|enum(?:\s+class)?|union)\s+([A-Za-z_]\w*)\b"
    r"|\busing\s+([A-Za-z_]\w*)\s*="
)
# Headers that declare Cube/SoftMax tiling PODs and MatmulConfig. Not stubs.
TYPE_HEADER_HINTS = (
    "kernel_tiling/kernel_tiling.h",
    "lib/matmul/matmul_config.h",
    "tiling/matmul/matmul_config.h",
    "adv_api/matmul/matmul_config.h",
    "lib/matrix/matmul/matmul_config.h",
)
# Prelude / language names. Do not hunt CANN headers or stub them here.
SKIP_UNKNOWN_TYPES = frozenset(
    {
        "int",
        "char",
        "void",
        "bool",
        "float",
        "double",
        "long",
        "short",
        "unsigned",
        "signed",
        "auto",
        "size_t",
        "ptrdiff_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "Dim3",
        "cce",
        "half",
        "half2",
        "float2",
        "__callee__",
        "__simt_callee__",
        "__simd_callee__",
        "__simt_vf__",
        "__simd_vf__",
    }
)
HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".inc", ".cuh")
# CANN layout moved highlevel_api matmul headers. Operators still include the
# old path; forward to the file that exists in the current unpack.
INCLUDE_PREFIX_ALIASES = (
    ("lib/matrix/matmul/", "lib/matmul/"),
    # Current CANN unpack ships hccl/hccl.h; operators still include the
    # older hccl/hcom.h spelling (file is not in the tree).
    ("hccl/hcom.h", "hccl/hccl.h"),
)
SKIP_DIR_NAMES = {
    ".git",
    ".svn",
    ".ascendc-codemap",
    "__pycache__",
    "bin",
    "lib",
    "lib64",
    "python",
    "python3",
    "share",
    "tools",
    "tests",
    "test",
    "build",
    "output",
    "cmake-build-debug",
    "cmake-build-release",
    "node_modules",
}
# Kernel trap: this -I makes ../../../../include/... resolve under impl/include.
FORBIDDEN_INCLUDE_SUBSTR = (
    "ascendc/include/basic_api",
)
# Relative to each cann-* package (host tuple substituted at search time).
CANN_PACKAGE_RELS = (
    "x86_64-linux/include",
    "x86_64-linux/include/base",
    "x86_64-linux/pkg_inc",
    "x86_64-linux/pkg_inc/base",
    "x86_64-linux/include/op_common",
    "x86_64-linux/include/op_common/op_host",
    "x86_64-linux/include/aclnn",
    "x86_64-linux/asc/include",
    "x86_64-linux/asc/include/adv_api",
    "x86_64-linux/asc/include/adv_api/hccl/internal/hcomm",
    "x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc",
    "x86_64-linux/asc/impl",
    "x86_64-linux/ascendc/include",
    "x86_64-linux/ascendc/include/highlevel_api",
    "x86_64-linux/tikcpp/tikcfw",
    "x86_64-linux/third_party/include",
    "x86_64-linux/include/nlohmann",
)


def _cann_package_rels(root: Path) -> tuple[str, ...]:
    from ascendc_codemap_mcp.engine.paths import cann_host_dir

    host = cann_host_dir(root) or "x86_64-linux"
    if host == "x86_64-linux":
        return CANN_PACKAGE_RELS
    return tuple(rel.replace("x86_64-linux", host) for rel in CANN_PACKAGE_RELS)
STD_HEADERS = frozenset(
    {
        "algorithm",
        "array",
        "atomic",
        "cassert",
        "cctype",
        "cerrno",
        "chrono",
        "cmath",
        "csignal",
        "cstdarg",
        "cstddef",
        "cstdint",
        "cstdio",
        "cstdlib",
        "cstring",
        "ctime",
        "cwchar",
        "deque",
        "exception",
        "filesystem",
        "functional",
        "initializer_list",
        "iomanip",
        "ios",
        "iosfwd",
        "iostream",
        "istream",
        "iterator",
        "limits",
        "list",
        "map",
        "memory",
        "mutex",
        "new",
        "numeric",
        "optional",
        "ostream",
        "queue",
        "set",
        "sstream",
        "stack",
        "stdexcept",
        "string",
        "string_view",
        "system_error",
        "thread",
        "tuple",
        "type_traits",
        "typeinfo",
        "unordered_map",
        "unordered_set",
        "utility",
        "variant",
        "vector",
        "climits",
        "cfloat",
        "complex",
        "condition_variable",
        "future",
        "random",
        "regex",
        "shared_mutex",
        "span",
        "stdalign.h",
        "stdbool.h",
        "stddef.h",
        "stdint.h",
        "stdio.h",
        "stdlib.h",
        "string.h",
        "math.h",
        "assert.h",
        "errno.h",
        "limits.h",
        "float.h",
        "time.h",
        "ctype.h",
        "wchar.h",
    }
)
_ENV_ENABLE = "UO_INCLUDE_HEAL"
_ENV_ROUNDS = "UO_INCLUDE_HEAL_ROUNDS"
_INDEX_CACHE: dict[tuple[str, ...], dict[str, list[str]]] = {}
_TYPE_INDEX_CACHE: dict[tuple[str, ...], dict[str, list[str]]] = {}


def reset_index_cache() -> None:
    _INDEX_CACHE.clear()
    _TYPE_INDEX_CACHE.clear()


@dataclass
class MissingInclude:
    name: str
    side: str  # host | kernel
    reason: str = ""
    candidates: list[str] = field(default_factory=list)


INCLUDE_UNIQUE = "unique"
INCLUDE_AMBIGUOUS = "INCLUDE_AMBIGUOUS"
INCLUDE_UNRESOLVED = "unresolved"

_LAST_INCLUDE_STATUS = INCLUDE_UNRESOLVED
_LAST_INCLUDE_CANDIDATES: list[str] = []


def last_include_resolution() -> tuple[str, list[str]]:
    return _LAST_INCLUDE_STATUS, list(_LAST_INCLUDE_CANDIDATES)


def _set_include_resolution(status: str, candidates: Iterable[Path | str] | None = None) -> None:
    global _LAST_INCLUDE_STATUS, _LAST_INCLUDE_CANDIDATES
    _LAST_INCLUDE_STATUS = str(status or INCLUDE_UNRESOLVED)
    _LAST_INCLUDE_CANDIDATES = [_posix(Path(p)) for p in (candidates or [])]


@dataclass
class HealHit:
    include: str
    include_dir: str
    found: str
    side: str
    round: int = 0
    source: str = "probe"  # probe | bootstrap


@dataclass
class HealReport:
    rounds: int = 0
    healed: list[HealHit] = field(default_factory=list)
    unresolved: list[MissingInclude] = field(default_factory=list)
    added_host: list[str] = field(default_factory=list)
    added_kernel: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rounds": self.rounds,
            "added_host": list(self.added_host),
            "added_kernel": list(self.added_kernel),
            "healed": [
                {
                    "include": h.include,
                    "include_dir": h.include_dir,
                    "found": h.found,
                    "side": h.side,
                    "round": h.round,
                    "source": h.source,
                }
                for h in self.healed
            ],
            "unresolved": [
                {
                    "include": u.name,
                    "side": u.side,
                    "reason": u.reason,
                    "candidates": list(u.candidates),
                }
                for u in self.unresolved
            ],
        }


def heal_enabled() -> bool:
    raw = os.environ.get(_ENV_ENABLE, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def max_rounds() -> int:
    raw = os.environ.get(_ENV_ROUNDS, "8").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, min(n, 16))


def extras_summary_path(op_dir: str | Path, arch_dir: str | None) -> Path:
    arch = require_architecture(arch_dir)
    return Path(op_dir) / ".ascendc-codemap" / arch / "summary" / "build_context_extras.yaml"


def extras_run_path(op_dir: str | Path, arch_dir: str | None, run_id: str) -> Path:
    arch = require_architecture(arch_dir)
    rid = str(run_id or "default").strip() or "default"
    return (
        Path(op_dir)
        / ".ascendc-codemap" / arch
        / "runs"
        / rid
        / "scope"
        / "build_context_extras.yaml"
    )


SOURCE_SCRIPT = "uo_init.include_heal"
SOURCE_HEAL_PROMOTE = "heal_promote"
SOURCE_MIXED = "mixed"


def _posix(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _norm_key(path: str | Path) -> str:
    return _posix(path).lower()


def _unique_dirs(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items or []:
        p = _posix(raw)
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _identity_include_dirs(items: Iterable[Any], arch_dir: str | None) -> list[str]:
    """Drop ``-I`` roots that belong to another ``arch*`` folder."""
    arch = str(arch_dir or "").strip()
    out: list[str] = []
    for raw in _unique_dirs(items):
        if arch and is_other_arch_path(raw, arch):
            continue
        out.append(raw)
    return out


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def load_extras_payload(op_dir: str | Path, arch_dir: str | None) -> dict[str, Any]:
    path = extras_summary_path(op_dir, arch_dir)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _str_list(data: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        raw = data.get(key)
        if isinstance(raw, list) and raw:
            return _unique_dirs(raw)
    return []


def promoted_include_dirs(data: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    payload = data if isinstance(data, dict) else {}
    host = _str_list(payload, "promoted_host", "promoted_host_include_dirs")
    kernel = _str_list(payload, "promoted_kernel", "promoted_kernel_include_dirs")
    if not host and not kernel and str(payload.get("source") or "") == SOURCE_HEAL_PROMOTE:
        host = _str_list(payload, "host")
        kernel = _str_list(payload, "kernel")
    return host, kernel


def _write_extras_payload(
    op_dir: str | Path,
    arch_dir: str | None,
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
) -> Path:
    summary = extras_summary_path(op_dir, arch_dir)
    _dump_yaml(summary, payload)
    if run_id:
        _dump_yaml(extras_run_path(op_dir, arch_dir, run_id), payload)
    return summary


def _unlink_if_file(path: Path) -> None:
    if path.is_file():
        path.unlink()


def clear_saved_extras(op_dir: str | Path, arch_dir: str | None, *, run_id: str | None = None) -> None:
    """Drop script-healed extras. Keep ``heal_promote`` ``-I`` dirs across prepare reruns."""
    payload = load_extras_payload(op_dir, arch_dir)
    host, kernel = promoted_include_dirs(payload)
    fi_host = _str_list(payload, "promoted_host_force_include")
    fi_kernel = _str_list(payload, "promoted_kernel_force_include")
    if not host and not kernel and not fi_host and not fi_kernel:
        _unlink_if_file(extras_summary_path(op_dir, arch_dir))
        if run_id:
            _unlink_if_file(extras_run_path(op_dir, arch_dir, run_id))
        return
    kept = {
        "version": 2,
        "source": SOURCE_HEAL_PROMOTE,
        "host": list(host),
        "kernel": list(kernel),
        "promoted_host": list(host),
        "promoted_kernel": list(kernel),
        "host_force_include": list(fi_host),
        "kernel_force_include": list(fi_kernel),
        "promoted_host_force_include": list(fi_host),
        "promoted_kernel_force_include": list(fi_kernel),
    }
    _write_extras_payload(op_dir, arch_dir, kept, run_id=run_id)


def save_extras(
    ctx: Any,
    report: HealReport,
    *,
    run_id: str | None = None,
    source: str = SOURCE_SCRIPT,
) -> Path | None:
    """Persist extra -I so extract_host reloads the same BuildContext."""
    op_dir = getattr(ctx, "op_dir", "") or ""
    arch_dir = getattr(ctx, "arch_dir", "") or ""
    if not op_dir or not arch_dir:
        return None
    existing = load_extras_payload(op_dir, arch_dir)
    p_host, p_kernel = promoted_include_dirs(existing)
    p_host = _identity_include_dirs(p_host, arch_dir)
    p_kernel = _identity_include_dirs(p_kernel, arch_dir)
    p_fi_host = _str_list(existing, "promoted_host_force_include")
    p_fi_kernel = _str_list(existing, "promoted_kernel_force_include")
    extra_h = _identity_include_dirs(getattr(ctx, "extra_host_includes", None) or [], arch_dir)
    extra_k = _identity_include_dirs(getattr(ctx, "extra_kernel_includes", None) or [], arch_dir)
    extra_fi_h = _unique_dirs(getattr(ctx, "extra_host_force_includes", None) or [])
    extra_fi_k = _unique_dirs(getattr(ctx, "extra_kernel_force_includes", None) or [])
    src = str(source or SOURCE_SCRIPT).strip() or SOURCE_SCRIPT
    if src == SOURCE_HEAL_PROMOTE:
        p_host = _unique_dirs([*p_host, *extra_h])
        p_kernel = _unique_dirs([*p_kernel, *extra_k])
        p_fi_host = _unique_dirs([*p_fi_host, *extra_fi_h])
        p_fi_kernel = _unique_dirs([*p_fi_kernel, *extra_fi_k])
        extra_h = _unique_dirs([*p_host, *extra_h])
        extra_k = _unique_dirs([*p_kernel, *extra_k])
        extra_fi_h = _unique_dirs([*p_fi_host, *extra_fi_h])
        extra_fi_k = _unique_dirs([*p_fi_kernel, *extra_fi_k])
    else:
        extra_h = _unique_dirs([*p_host, *extra_h])
        extra_k = _unique_dirs([*p_kernel, *extra_k])
        extra_fi_h = _unique_dirs([*p_fi_host, *extra_fi_h])
        extra_fi_k = _unique_dirs([*p_fi_kernel, *extra_fi_k])
        for item in p_host:
            ctx.add_include(item, side="host")
        for item in p_kernel:
            ctx.add_include(item, side="kernel")
        add_fi = getattr(ctx, "add_force_include", None)
        if callable(add_fi):
            for item in p_fi_host:
                add_fi(item, side="host")
            for item in p_fi_kernel:
                add_fi(item, side="kernel")
    out_source = src
    if p_host or p_kernel:
        if src == SOURCE_HEAL_PROMOTE:
            out_source = SOURCE_HEAL_PROMOTE
        elif src == SOURCE_SCRIPT:
            out_source = SOURCE_MIXED if (p_host or p_kernel) else SOURCE_SCRIPT
        else:
            out_source = SOURCE_MIXED
    payload = {
        "version": 2,
        "source": out_source,
        "host": extra_h,
        "kernel": extra_k,
        "promoted_host": p_host,
        "promoted_kernel": p_kernel,
        "host_force_include": extra_fi_h,
        "kernel_force_include": extra_fi_k,
        "promoted_host_force_include": p_fi_host,
        "promoted_kernel_force_include": p_fi_kernel,
        **report.to_dict(),
    }
    return _write_extras_payload(op_dir, arch_dir, payload, run_id=run_id)


def apply_saved_extras(ctx: Any) -> list[str]:
    """Merge persisted extras into ``ctx``. Returns newly applied dirs."""
    op_dir = getattr(ctx, "op_dir", "") or ""
    arch_dir = getattr(ctx, "arch_dir", "") or ""
    if not op_dir or not arch_dir:
        return []
    data = load_extras_payload(op_dir, arch_dir)
    if not data:
        return []
    applied: list[str] = []
    host_dirs = _unique_dirs(
        [*_str_list(data, "host"), *_str_list(data, "promoted_host", "promoted_host_include_dirs")]
    )
    kernel_dirs = _unique_dirs(
        [
            *_str_list(data, "kernel"),
            *_str_list(data, "promoted_kernel", "promoted_kernel_include_dirs"),
        ]
    )
    for item in host_dirs:
        if ctx.add_include(str(item), side="host"):
            applied.append(str(item))
    for item in kernel_dirs:
        if ctx.add_include(str(item), side="kernel"):
            applied.append(str(item))
    for side, key in (("host", "host_force_include"), ("kernel", "kernel_force_include")):
        extra_keys = (
            ("promoted_host_force_include",)
            if side == "host"
            else ("promoted_kernel_force_include",)
        )
        for item in _unique_dirs([*(data.get(key) or []), *_str_list(data, *extra_keys)]):
            add_fi = getattr(ctx, "add_force_include", None)
            if callable(add_fi) and add_fi(str(item), side=side):
                applied.append(str(item))
    return applied


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def allowed_include_dir(
    path: str | Path,
    *,
    cann_root: str = "",
    ops_root: str | None = None,
    op_dir: str | None = None,
) -> bool:
    """True when ``path`` exists and sits under cann_root / ops_root / op_dir."""
    raw = Path(str(path or "").strip())
    if not str(raw):
        return False
    try:
        resolved = raw.resolve()
    except OSError:
        return False
    if not resolved.is_dir():
        return False
    if is_forbidden_include_dir(str(resolved)):
        return False
    roots: list[Path] = []
    for item in (cann_root, ops_root, op_dir):
        text = str(item or "").strip()
        if text:
            roots.append(Path(text))
    return any(_path_under(resolved, root) for root in roots)


def _staging_dirs(staging: dict[str, Any], *keys: str) -> list[str]:
    out: list[str] = []
    for key in keys:
        raw = staging.get(key)
        if isinstance(raw, list):
            out.extend(str(x) for x in raw)
        elif isinstance(raw, str) and raw.strip():
            out.append(raw)
    for row in staging.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        d = str(row.get("dir") or row.get("include_dir") or "").strip()
        if d:
            out.append(d)
    return _unique_dirs(out)


def promote_include_dirs(
    ctx: Any,
    staging: dict[str, Any] | None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Validate staged ``-I`` dirs and append them as ``heal_promote`` extras.

    Does not rewrite shared ``spec/build_context.yaml``.
    """
    data = staging if isinstance(staging, dict) else {}
    host_dirs = _staging_dirs(data, "host", "host_include_dirs")
    kernel_dirs = _staging_dirs(data, "kernel", "kernel_include_dirs")
    # evidence rows may name a side
    evidence_host: list[str] = []
    evidence_kernel: list[str] = []
    for row in data.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        d = str(row.get("dir") or row.get("include_dir") or "").strip()
        if not d:
            continue
        side = str(row.get("side") or "").strip().lower()
        if side == "kernel":
            evidence_kernel.append(d)
        elif side == "host":
            evidence_host.append(d)
    host_dirs = _unique_dirs([*host_dirs, *evidence_host])
    kernel_dirs = _unique_dirs([*kernel_dirs, *evidence_kernel])
    if not host_dirs and not kernel_dirs:
        return {
            "ok": False,
            "error": "INCLUDE_HEAL_STAGING_EMPTY",
            "reason_code": "INCLUDE_HEAL_STAGING_EMPTY",
            "message_zh": "staging.yaml 没有可 promote 的 host/kernel -I 目录",
        }
    cann = str(getattr(ctx, "cann_root", "") or "")
    ops = str(getattr(ctx, "ops_root", "") or "") or None
    op_dir = str(getattr(ctx, "op_dir", "") or "") or None
    rejected: list[dict[str, str]] = []
    accepted_host: list[str] = []
    accepted_kernel: list[str] = []

    def _accept(raw: str, side: str) -> None:
        p = Path(str(raw).strip())
        if not allowed_include_dir(p, cann_root=cann, ops_root=ops, op_dir=op_dir):
            rejected.append({"dir": _posix(raw), "side": side, "reason": "outside_cann_or_ops_or_missing"})
            return
        posix = _posix(p.resolve())
        if ctx.add_include(posix, side=side):
            bucket = accepted_kernel if side == "kernel" else accepted_host
            if posix not in bucket:
                bucket.append(posix)
        else:
            bucket = accepted_kernel if side == "kernel" else accepted_host
            if posix not in bucket:
                bucket.append(posix)

    for item in host_dirs:
        _accept(item, "host")
    for item in kernel_dirs:
        _accept(item, "kernel")
    if rejected and not accepted_host and not accepted_kernel:
        return {
            "ok": False,
            "error": "INCLUDE_HEAL_PROMOTE_REJECTED",
            "reason_code": "INCLUDE_HEAL_PROMOTE_REJECTED",
            "rejected": rejected,
            "message_zh": "staging 里的 -I 不在 cann_root / ops 树内，或目录不存在；未写入 extras",
        }
    extras = save_extras(
        ctx,
        HealReport(enabled=True, rounds=1),
        run_id=run_id,
        source=SOURCE_HEAL_PROMOTE,
    )
    return {
        "ok": True,
        "engine": "heal_promote",
        "reason_code": "INCLUDE_HEAL_PROMOTED",
        "accepted_host": accepted_host,
        "accepted_kernel": accepted_kernel,
        "rejected": rejected,
        "extras_path": extras.as_posix() if extras else None,
        "message_zh": "已把校验通过的 -I 写入 build_context_extras.yaml；下一轮 prepare/extract 会带上",
    }


def parse_missing_includes(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in MISSING_RE.finditer(str(text or "")):
        name = match.group(1).replace("\\", "/").strip().lstrip("./")
        if not name or ".." in name.split("/"):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def missing_includes_from_probes(
    probes: Iterable[dict[str, Any]] | None,
    errors: Iterable[str] | None = None,
) -> list[MissingInclude]:
    out: list[MissingInclude] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, side: str) -> None:
        side_n = "kernel" if str(side).lower() == "kernel" else "host"
        key = (name.lower(), side_n)
        if key in seen:
            return
        seen.add(key)
        out.append(MissingInclude(name=name, side=side_n))

    for row in probes or []:
        if not isinstance(row, dict):
            continue
        side = str(row.get("side") or "host")
        chunks = [str(s) for s in (row.get("samples") or [])]
        chunks.extend(str(s) for s in (row.get("heal_hints") or []))
        if row.get("error"):
            chunks.append(str(row.get("error")))
        for chunk in chunks:
            for name in parse_missing_includes(chunk):
                add(name, side)
    for err in errors or []:
        for name in parse_missing_includes(str(err)):
            add(name, "host")
    return out


def parse_unknown_types(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in UNKNOWN_TYPE_RE.finditer(str(text or "")):
        name = match.group(1).strip()
        if not name or name in SKIP_UNKNOWN_TYPES or len(name) < 3:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def unknown_types_from_probes(
    probes: Iterable[dict[str, Any]] | None,
    errors: Iterable[str] | None = None,
) -> list[MissingInclude]:
    out: list[MissingInclude] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, side: str) -> None:
        side_n = "kernel" if str(side).lower() == "kernel" else "host"
        key = (name.lower(), side_n)
        if key in seen:
            return
        seen.add(key)
        out.append(MissingInclude(name=name, side=side_n))

    for row in probes or []:
        if not isinstance(row, dict):
            continue
        side = str(row.get("side") or "host")
        chunks = [str(s) for s in (row.get("samples") or [])]
        chunks.extend(str(s) for s in (row.get("heal_hints") or []))
        if row.get("error"):
            chunks.append(str(row.get("error")))
        for chunk in chunks:
            for name in parse_unknown_types(chunk):
                add(name, side)
    for err in errors or []:
        for name in parse_unknown_types(str(err)):
            add(name, "kernel")
    return out


def _header_declares_type(path: Path, type_name: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    pat = re.compile(
        rf"\b(?:struct|class|enum(?:\s+class)?|union)\s+{re.escape(type_name)}\b"
        rf"|\busing\s+(?:[A-Za-z_]\w*::)*{re.escape(type_name)}\s*;"
        rf"|\busing\s+{re.escape(type_name)}\s*="
    )
    return bool(pat.search(text))


def _type_index(roots: list[Path]) -> dict[str, list[str]]:
    key = tuple(_norm_key(r) for r in roots)
    cached = _TYPE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx: dict[str, list[str]] = {}
    for root in roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not _skip_walk_dir(d)]
                for fn in filenames:
                    low = fn.lower()
                    if not low.endswith(HEADER_SUFFIXES):
                        continue
                    path = Path(dirpath) / fn
                    if is_forbidden_include_dir(path):
                        continue
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    found_path = str(path)
                    for match in _TYPE_DECL_RE.finditer(text):
                        name = match.group(1) or match.group(2)
                        if not name or name in SKIP_UNKNOWN_TYPES or len(name) < 3:
                            continue
                        bucket = idx.setdefault(name, [])
                        if found_path not in bucket:
                            bucket.append(found_path)
        except OSError:
            continue
    _TYPE_INDEX_CACHE[key] = idx
    return idx


def find_type_header(ctx: Any, type_name: str, *, side: str) -> HealHit | None:
    """Locate the CANN (or kernel) header that declares ``type_name``."""
    name = str(type_name or "").strip()
    if not name or name in SKIP_UNKNOWN_TYPES:
        _set_include_resolution(INCLUDE_UNRESOLVED, [])
        return None
    # Hinted CANN headers first — no full-tree walk for SoftMaxTiling / TCubeTiling.
    for rel in TYPE_HEADER_HINTS:
        hit = find_include_dir(ctx, rel, side=side)
        if hit is None:
            continue
        found = Path(hit.found)
        if not _header_declares_type(found, name):
            continue
        if is_forbidden_include_dir(found):
            continue
        return HealHit(
            include=name,
            include_dir=hit.include_dir,
            found=hit.found,
            side=side,
        )
    roots = search_roots(ctx)
    ranked: list[Path] = []
    for raw in _type_index(roots).get(name, []):
        path = Path(raw)
        posix = _posix(path).lower()
        if side == "kernel" and "/op_host/" in posix:
            continue
        if is_forbidden_include_dir(path):
            continue
        ranked.append(path)
    status, viable = _pick_unique_header(
        ranked,
        roots=roots,
        rel=name,
        arch_dir=str(getattr(ctx, "arch_dir", "") or ""),
    )
    _set_include_resolution(status, viable)
    if status != INCLUDE_UNIQUE or not viable:
        return None
    found = viable[0]
    if not _header_declares_type(found, name):
        _set_include_resolution(INCLUDE_UNRESOLVED, viable)
        return None
    return HealHit(
        include=name,
        include_dir=_posix(found.parent),
        found=_posix(found),
        side=side,
    )


def heal_unknown_types(
    ctx: Any,
    missing: Iterable[MissingInclude],
    *,
    round_no: int = 0,
    source: str = "probe",
) -> list[HealHit]:
    """Force-include the CANN header that declares an unknown type (not prelude stubs)."""
    hits: list[HealHit] = []
    add_fi = getattr(ctx, "add_force_include", None)
    for item in missing:
        hit = find_type_header(ctx, item.name, side=item.side)
        if hit is None:
            status, cands = last_include_resolution()
            item.reason = status if status == INCLUDE_AMBIGUOUS else (item.reason or INCLUDE_UNRESOLVED)
            item.candidates = cands
            continue
        hit.side = item.side
        hit.round = round_no
        hit.source = source
        added = False
        if callable(add_fi) and add_fi(hit.found, side=item.side):
            added = True
        if added:
            hits.append(hit)
    return hits


def scan_source_includes(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for match in INCLUDE_RE.finditer(text):
        name = match.group(1).replace("\\", "/").strip()
        if not name or name in STD_HEADERS:
            continue
        if ".." in name.split("/"):
            continue
        base = name.rsplit("/", 1)[-1].lower()
        if base in STD_HEADERS:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def aliased_include_name(include_name: str) -> str | None:
    rel = include_name.replace("\\", "/").strip().lstrip("./")
    for old, new in INCLUDE_PREFIX_ALIASES:
        if rel.startswith(old):
            return new + rel[len(old):]
    return None


def alias_cache_root(ctx: Any) -> Path | None:
    op_dir = Path(getattr(ctx, "op_dir", "") or "")
    arch = str(getattr(ctx, "arch_dir", "") or "")
    if not op_dir or not arch:
        return None
    return op_dir / ".ascendc-codemap" / arch / "cache" / "include_alias"


def materialize_include_alias(
    ctx: Any, include_name: str, aliased: str, *, side: str
) -> HealHit | None:
    """Forward ``lib/matrix/matmul/X`` to the file that exists as ``lib/matmul/X``."""
    root = alias_cache_root(ctx)
    if root is None:
        return None
    rel = include_name.replace("\\", "/").strip().lstrip("./")
    dest = root.joinpath(*rel.split("/"))
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f'#pragma once\n#include "{aliased}"\n', encoding="utf-8")
    except OSError:
        return None
    include_dir = _posix(root)
    return HealHit(
        include=rel,
        include_dir=include_dir,
        found=_posix(dest),
        side=side,
    )


def is_forbidden_include_dir(path: str | Path) -> bool:
    key = _norm_key(path)
    return any(tok in key for tok in FORBIDDEN_INCLUDE_SUBSTR)


def include_dir_for(found: Path, include_name: str) -> Path:
    """Directory to put on -I so ``#include "include_name"`` opens ``found``."""
    resolved = found.resolve() if found.exists() else found
    rel = include_name.replace("\\", "/").strip().strip("/")
    blob = _posix(resolved)
    suffix = "/" + rel
    if blob.lower().endswith(suffix.lower()):
        parent = blob[: -len(rel)].rstrip("/")
        return Path(parent)
    return resolved.parent


def header_resolves(ctx: Any, include_name: str, *, side: str, tu_dir: Path | None = None) -> bool:
    rel = include_name.replace("\\", "/").strip()
    if tu_dir is not None:
        local = tu_dir / rel
        if local.is_file():
            return True
    includes = ctx.kernel_includes() if side == "kernel" else ctx.host_includes()
    for root in includes:
        cand = Path(root) / rel
        if cand.is_file():
            return True
    return False


def search_roots(ctx: Any) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: str | Path | None) -> None:
        if not path:
            return
        p = Path(str(path))
        key = _norm_key(p)
        if key in seen:
            return
        try:
            if not p.is_dir():
                return
        except OSError:
            return
        seen.add(key)
        roots.append(p)

    for p in list(ctx.host_includes() or []) + list(ctx.kernel_includes() or []):
        add(p)
    cann = Path(getattr(ctx, "cann_root", "") or "")
    if cann.is_dir():
        from ascendc_codemap_mcp.engine.paths import cann_host_dir

        rels = _cann_package_rels(cann)
        try:
            packages = [p for p in cann.iterdir() if p.is_dir()]
        except OSError:
            packages = []
        for pkg in packages:
            for rel in rels:
                add(pkg / rel)
        for rel in rels:
            add(cann / rel)
        add(cann / "asc" / "include")
        add(cann / "asc" / "impl")
        add(cann / "include")
        host = cann_host_dir(cann)
        if host:
            add(cann / host / "asc" / "include")
            add(cann / host / "asc" / "impl")
            add(cann / host / "include")
    ops = Path(getattr(ctx, "ops_root", "") or "")
    if ops.is_dir():
        add(ops)
        add(ops / "common")
        add(ops / "common" / "include")
        add(ops / "common" / "include" / "op_kernel")
        add(ops / "3rd")
        add(ops / "3rdparty")
        add(ops / "3rdparty" / "include")
        # Sibling family commons (ffn includes headers that live under mc2/common).
        try:
            for fam in ops.iterdir():
                if not fam.is_dir() or _skip_walk_dir(fam.name):
                    continue
                add(fam / "common")
                add(fam / "common" / "utils")
                add(fam / "common" / "inc")
                add(fam / "3rd")
                add(fam / "3rdparty")
        except OSError:
            pass
    op_dir = Path(getattr(ctx, "op_dir", "") or "")
    if op_dir.is_dir():
        add(op_dir)
        add(op_dir / "op_host")
        add(op_dir / "op_kernel")
        add(op_dir.parent)
        add(op_dir.parent / "common")
        add(op_dir.parent / "common" / "utils")
        add(op_dir.parent / "common" / "inc")
        add(op_dir.parent / "3rd")
        add(op_dir.parent / "3rdparty")
    return roots


def _skip_walk_dir(name: str) -> bool:
    return name.lower() in SKIP_DIR_NAMES or name.startswith(".")


def _basename_index(roots: list[Path]) -> dict[str, list[str]]:
    key = tuple(_norm_key(r) for r in roots)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx: dict[str, list[str]] = {}
    for root in roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not _skip_walk_dir(d)]
                for fn in filenames:
                    low = fn.lower()
                    if not low.endswith(HEADER_SUFFIXES):
                        continue
                    idx.setdefault(low, []).append(str(Path(dirpath) / fn))
        except OSError:
            continue
    _INDEX_CACHE[key] = idx
    return idx


def _posix_under_roots(found: Path, roots: list[Path] | None) -> str:
    """Path relative to the first matching search root.

    Scoring must not look at the absolute path: a checkout folder named
    ``TEST`` (``D:/PR-review/TEST/ops-transformer/...``) would otherwise
    match the ``/test/`` penalty and drop real family headers.
    """
    if not roots:
        return _posix(found).lower()
    try:
        resolved = found.resolve()
    except OSError:
        resolved = found
    for root in roots:
        try:
            return _posix(resolved.relative_to(Path(root).resolve())).lower()
        except (ValueError, OSError):
            continue
    return _posix(found).lower()


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = _norm_key(path)
        except OSError:
            key = _posix(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _is_test_layout_header(found: Path, roots: list[Path]) -> bool:
    scoped = "/" + _posix_under_roots(found, roots).replace("\\", "/").strip("/") + "/"
    return "/tests/" in scoped or "/test/" in scoped


def _viable_header(found: Path, *, roots: list[Path]) -> bool:
    try:
        if not found.is_file():
            return False
    except OSError:
        return False
    if is_forbidden_include_dir(found) or is_forbidden_include_dir(found.parent):
        return False
    if _is_test_layout_header(found, roots):
        return False
    return True


def _clang_join_hits(roots: list[Path], rel: str) -> list[Path]:
    """O(search paths): ``root / include_spelling``, the way Clang uses -I."""
    hits: list[Path] = []
    rel_n = rel.replace("\\", "/").strip().lstrip("./")
    if not rel_n or ".." in rel_n.split("/"):
        return []
    for root in roots:
        cand = root / rel_n
        try:
            if cand.is_file():
                hits.append(cand)
        except OSError:
            continue
    return _dedupe_paths(hits)


def _basename_hits(roots: list[Path], rel: str) -> list[Path]:
    """Cold-start basename index. Never a semantic pick among many."""
    rel_n = rel.replace("\\", "/").strip().lstrip("./")
    base = rel_n.rsplit("/", 1)[-1].lower()
    if not base:
        return []
    hits: list[Path] = []
    for raw in _basename_index(roots).get(base, []):
        path = Path(raw)
        posix = _posix(path).replace("\\", "/").lower()
        if "/" in rel_n and not posix.endswith("/" + rel_n.lower()):
            continue
        hits.append(path)
    return _dedupe_paths(hits)


def _pick_unique_header(
    hits: list[Path], *, roots: list[Path], rel: str, arch_dir: str = ""
) -> tuple[str, list[Path]]:
    viable = [p for p in _dedupe_paths(hits) if _viable_header(p, roots=roots)]
    if not viable:
        return INCLUDE_UNRESOLVED, []
    arch = str(arch_dir or "").strip()
    tokens = {t.lower() for t in arch_tokens_in_include(rel)}
    if arch and len(viable) > 1:
        identity = [
            p
            for p in viable
            if architectures_match(path_owned_architecture(p), arch)
        ]
        if len(identity) == 1:
            return INCLUDE_UNIQUE, identity
        if identity:
            return INCLUDE_AMBIGUOUS, identity
        scoped = [p for p in viable if not is_other_arch_path(p, arch)]
        if len(scoped) == 1:
            return INCLUDE_UNIQUE, scoped
        if scoped:
            return INCLUDE_AMBIGUOUS, scoped
        other = [p for p in viable if is_other_arch_path(p, arch)]
        if other and len(other) == len(viable):
            # Every hit is another arch* folder. Bare ``foo.h`` must not put
            # that folder on -I; an explicit ``arch22/foo.h`` spelling is ok.
            if not tokens:
                return INCLUDE_UNRESOLVED, viable
            named = [p for p in other if path_owned_architecture(p) in tokens]
            if len(named) == 1:
                return INCLUDE_UNIQUE, named
            if named:
                return INCLUDE_AMBIGUOUS, named
            return INCLUDE_UNRESOLVED, viable
        return INCLUDE_AMBIGUOUS, viable
    if len(viable) > 1:
        return INCLUDE_AMBIGUOUS, viable
    owned = path_owned_architecture(viable[0])
    if (
        arch
        and owned
        and is_other_arch_path(viable[0], arch)
        and arch not in tokens
        and owned not in tokens
    ):
        return INCLUDE_UNRESOLVED, viable
    return INCLUDE_UNIQUE, viable


def find_include_dir(ctx: Any, include_name: str, *, side: str) -> HealHit | None:
    rel = include_name.replace("\\", "/").strip().lstrip("./")
    if not rel or ".." in rel.split("/"):
        _set_include_resolution(INCLUDE_UNRESOLVED, [])
        return None
    roots = search_roots(ctx)
    aliased = aliased_include_name(rel)
    hits = _clang_join_hits(roots, rel)
    source = "clang_join"
    if aliased:
        alias_hits = _clang_join_hits(roots, aliased)
        if alias_hits:
            hits = _dedupe_paths(hits + alias_hits)
            source = "alias_join"
    if not hits:
        hits = _basename_hits(roots, rel)
        source = "unique_basename"
        if aliased:
            hits = _dedupe_paths(hits + _basename_hits(roots, aliased))
    status, viable = _pick_unique_header(
        hits,
        roots=roots,
        rel=aliased or rel,
        arch_dir=str(getattr(ctx, "arch_dir", "") or ""),
    )
    _set_include_resolution(status, viable)
    if status != INCLUDE_UNIQUE or not viable:
        return None
    found = viable[0]
    posix = _posix(found).replace("\\", "/").lower()
    if aliased and posix.endswith("/" + aliased.lower()):
        hit = materialize_include_alias(ctx, rel, aliased, side=side)
        if hit is not None:
            return hit
    include_dir = include_dir_for(found, rel)
    if is_forbidden_include_dir(include_dir) or not include_dir.is_dir():
        _set_include_resolution(INCLUDE_UNRESOLVED, viable)
        return None
    owned_root = include_root_owned_architecture(include_dir)
    arch = str(getattr(ctx, "arch_dir", "") or "").strip()
    if owned_root and arch and is_other_arch_path(include_dir, arch):
        _set_include_resolution(INCLUDE_UNRESOLVED, viable)
        return None
    return HealHit(
        include=rel,
        include_dir=_posix(include_dir),
        found=_posix(found),
        side=side,
        source=source,
    )


def heal_missing_includes(
    ctx: Any,
    missing: Iterable[MissingInclude],
    *,
    round_no: int = 0,
    source: str = "probe",
) -> list[HealHit]:
    hits: list[HealHit] = []
    for item in missing:
        hit = find_include_dir(ctx, item.name, side=item.side)
        if hit is None:
            status, cands = last_include_resolution()
            item.reason = status if status == INCLUDE_AMBIGUOUS else (item.reason or INCLUDE_UNRESOLVED)
            item.candidates = cands
            continue
        hit.side = item.side
        hit.round = round_no
        hit.source = source
        if ctx.add_include(hit.include_dir, side=item.side):
            hits.append(hit)
        elif not any(h.include.lower() == hit.include.lower() and h.side == hit.side for h in hits):
            # Already on -I (maybe from yaml this round); still record if it was the miss.
            continue
    return hits


def bootstrap_operator_includes(ctx: Any, tus: Iterable[Path]) -> list[HealHit]:
    missing: list[MissingInclude] = []
    seen: set[tuple[str, str]] = set()
    for tu in tus:
        path = Path(tu)
        if not path.is_file():
            continue
        side = "kernel" if "op_kernel" in _posix(path).lower() else "host"
        tu_dir = path.parent
        for name in scan_source_includes(path):
            if header_resolves(ctx, name, side=side, tu_dir=tu_dir):
                continue
            key = (name.lower(), side)
            if key in seen:
                continue
            seen.add(key)
            missing.append(MissingInclude(name=name, side=side))
    return heal_missing_includes(ctx, missing, round_no=0, source="bootstrap")


def enrich_scope_with_heal(
    *,
    ctx: Any,
    host_tus: Iterable[Path],
    kernel_tu: Path | None,
    enrich_fn: Callable[[], Any],
    run_id: str | None = None,
) -> tuple[Any, HealReport]:
    """Run Clang include enrichment, healing ``file not found`` between rounds.

    ``enrich_fn`` must read ``ctx.host_args()`` / ``kernel_args()`` live so each
    retry sees newly added -I. Returns the last enrichment and a persistable
    report. Caller still owns probe fallback / candidates.yaml.
    """
    from ascendc_codemap_mcp.engine.progress import emit

    report = HealReport(enabled=heal_enabled())
    tus = [Path(p) for p in host_tus if p is not None]
    if kernel_tu is not None:
        tus.append(Path(kernel_tu))

    if not report.enabled:
        enrichment = enrich_fn()
        save_extras(ctx, report, run_id=run_id)
        return enrichment, report

    boot = bootstrap_operator_includes(ctx, tus)
    if boot:
        report.healed.extend(boot)
        report.rounds = 1
        for hit in boot:
            bucket = report.added_kernel if hit.side == "kernel" else report.added_host
            if hit.include_dir not in bucket:
                bucket.append(hit.include_dir)
        emit(
            "prepare include-heal bootstrap "
            + ", ".join(f"{h.include} -> {h.include_dir}" for h in boot[:4])
        )

    enrichment = None
    last_missing: list[MissingInclude] = []
    last_types: list[MissingInclude] = []
    rounds = max_rounds()
    for rnd in range(1, rounds + 1):
        enrichment = enrich_fn()
        probes = list(getattr(enrichment, "probes", None) or [])
        errors = list(getattr(enrichment, "errors", None) or [])
        last_missing = missing_includes_from_probes(probes, errors)
        last_types = unknown_types_from_probes(probes, errors)
        if not last_missing and not last_types:
            report.unresolved = []
            break
        added = heal_missing_includes(ctx, last_missing, round_no=rnd, source="probe")
        added.extend(heal_unknown_types(ctx, last_types, round_no=rnd, source="unknown_type"))
        if not added:
            report.unresolved = last_missing + last_types
            break
        report.healed.extend(added)
        report.rounds = max(report.rounds, rnd)
        for hit in added:
            bucket = report.added_kernel if hit.side == "kernel" else report.added_host
            label = hit.found if hit.source == "unknown_type" else hit.include_dir
            if label not in bucket:
                bucket.append(label)
        emit(
            f"prepare include-heal round {rnd}: "
            + ", ".join(f"{h.include} -> {h.found or h.include_dir}" for h in added[:4])
        )
    else:
        report.unresolved = last_missing + last_types

    if enrichment is None:
        enrichment = enrich_fn()
    save_extras(ctx, report, run_id=run_id)
    return enrichment, report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from ascendc_codemap_mcp.engine.build_context import BuildContext
    from ascendc_codemap_mcp.engine.op_spec import discover

    ap = argparse.ArgumentParser(
        prog="uo-heal-includes",
        description="Discover missing-header -I dirs and write build_context extras.",
    )
    ap.add_argument("--op-dir", required=True)
    ap.add_argument("--arch-dir", required=True)
    ap.add_argument("--cann-root", default=None)
    ap.add_argument("--ops-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Also run libclang include enrichment (same as prepare's heal loop).",
    )
    args = ap.parse_args(argv)
    spec = discover(args.op_dir, arch_dir=args.arch_dir)
    ctx = BuildContext.load(
        cann_root=args.cann_root,
        ops_root=args.ops_root,
        op_dir=str(spec.op_dir),
        arch_dir=spec.arch_dir,
        apply_saved_extras=False,
    )
    hosts = [p for p in spec.host_targets if p.exists()]
    kernel = spec.kernel_entry if spec.kernel_entry and spec.kernel_entry.exists() else None
    clear_saved_extras(spec.op_dir, spec.arch_dir, run_id=args.run_id)
    from ascendc_codemap_mcp.engine.include_heal import apply_saved_extras as _apply_extras

    _apply_extras(ctx)
    if args.probe:
        from ascendc_codemap_mcp.engine import scope_scan as sscan

        base_scope = spec.scope
        if base_scope is None:
            base_scope = sscan.scan(spec.op_dir, arch_dir=spec.arch_dir)

        def _enrich():
            return sscan.enrich_with_clang(
                base_scope,
                host_args=ctx.host_args(),
                kernel_args=ctx.kernel_args(
                    dtype_variant="DT_FLOAT16", source_path=kernel
                ),
                host_tus=hosts,
                kernel_tu=kernel,
            )

        _enr, report = enrich_scope_with_heal(
            ctx=ctx,
            host_tus=hosts,
            kernel_tu=kernel,
            enrich_fn=_enrich,
            run_id=args.run_id,
        )
    else:
        tus = list(hosts) + ([kernel] if kernel is not None else [])
        report = HealReport(enabled=heal_enabled())
        boot = bootstrap_operator_includes(ctx, tus)
        report.healed.extend(boot)
        report.rounds = 1 if boot else 0
        for hit in boot:
            bucket = report.added_kernel if hit.side == "kernel" else report.added_host
            if hit.include_dir not in bucket:
                bucket.append(hit.include_dir)
        save_extras(ctx, report, run_id=args.run_id)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    extras = extras_summary_path(spec.op_dir, spec.arch_dir)
    if extras.is_file():
        print(f"wrote {extras.as_posix()}", file=__import__("sys").stderr)
    return 0 if not report.unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
