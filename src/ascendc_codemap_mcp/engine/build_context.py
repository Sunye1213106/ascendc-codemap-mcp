# -*- coding: utf-8 -*-
"""Resolve build_context.yaml placeholders into concrete clang/libclang args."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.paths import require_architecture
import yaml

from ascendc_codemap_mcp.engine import paths

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec"
DEFAULT_CONTEXT = SPEC_DIR / "build_context.yaml"
FUNCTION_LIKE_QUALIFIERS = {
    "__in_pipe__",
    "__out_pipe__",
    "__inout_pipe__",
    "LAUNCH_BOUND",
    "__launch_bounds__",
}


def dtype_macro_for_source(
    source_path: str | Path | None,
    *,
    op_dir: str | Path | None = None,
    ops_root: str | Path | None = None,
    macros: dict[str, str] | None = None,
) -> str | None:
    """First ``ORIG_DTYPE_*`` in the kernel include closure, if any."""
    from ascendc_codemap_mcp.engine.kernel_gates import discover_kernel_gates

    gates = discover_kernel_gates(
        source_path, op_dir=op_dir, ops_root=ops_root, macros=macros
    )
    return gates.orig_dtypes[0] if gates.orig_dtypes else None


def source_uses_dtype_variants(
    source_path: str | Path | None,
    *,
    op_dir: str | Path | None = None,
    ops_root: str | Path | None = None,
    macros: dict[str, str] | None = None,
) -> bool:
    from ascendc_codemap_mcp.engine.kernel_gates import source_uses_kernel_gates

    return source_uses_kernel_gates(
        source_path, op_dir=op_dir, ops_root=ops_root, macros=macros
    )



def _sub(s: str, mapping: dict[str, str]) -> str:
    out = s
    # multi-pass so nested placeholders resolve
    for _ in range(4):
        prev = out
        for k, v in mapping.items():
            out = out.replace("{" + k + "}", v)
        if out == prev:
            break
    return out


def _dedupe_includes(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        p = str(raw).replace("\\", "/").rstrip("/")
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


@dataclass
class BuildContext:
    raw: dict[str, Any]
    cann_root: str
    ops_root: str
    compat_root: str
    op_dir: str = ""
    arch_dir: str = ""
    repo_root: str = ""
    extra_host_includes: list[str] = field(default_factory=list)
    extra_kernel_includes: list[str] = field(default_factory=list)
    extra_host_force_includes: list[str] = field(default_factory=list)
    extra_kernel_force_includes: list[str] = field(default_factory=list)
    overlay_includes: list[str] = field(default_factory=list)
    cann_9201: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        *,
        cann_root: str | None = None,
        ops_root: str | None = None,
        op_dir: str | None = None,
        arch_dir: str = "",
        repo_root: str | None = None,
        apply_saved_extras: bool = True,
    ) -> "BuildContext":
        p = Path(path) if path else DEFAULT_CONTEXT
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        defaults = dict(raw.get("defaults") or {})
        rr = repo_root or str(paths.repo_root())
        # The spec file carries no machine-specific defaults; where the external
        # trees live is a property of the checkout, resolved by uo_init.paths.
        cann_fallback = cann_root or defaults.get("cann_root") or paths.cann_root() or ""
        ops_fallback = ops_root or defaults.get("ops_root") or paths.ops_root() or ""
        mapping = {
            "repo_root": rr.replace("\\", "/"),
            "cann_root": str(cann_fallback).replace("\\", "/"),
            "ops_root": str(ops_fallback).replace("\\", "/"),
            "compat_root": "",
            "op_dir": (op_dir or "").replace("\\", "/"),
            "arch_dir": arch_dir,
        }
        # Always prefer the in-package compat/ (shim + prelude) unless overridden
        cr = str(SPEC_DIR / "compat").replace("\\", "/")
        if defaults.get("compat_root"):
            cr = _sub(defaults["compat_root"], mapping).replace("\\", "/")
        mapping["compat_root"] = cr
        if cann_root:
            mapping["cann_root"] = cann_root.replace("\\", "/")
        if ops_root:
            mapping["ops_root"] = ops_root.replace("\\", "/")
        obj = cls(
            raw=raw,
            cann_root=mapping["cann_root"],
            ops_root=mapping["ops_root"],
            compat_root=mapping["compat_root"],
            op_dir=mapping["op_dir"],
            arch_dir=arch_dir,
            repo_root=rr.replace("\\", "/"),
        )
        if apply_saved_extras:
            from ascendc_codemap_mcp.engine.include_heal import apply_saved_extras as _apply_extras

            _apply_extras(obj)
        try:
            from ascendc_codemap_mcp.engine.cann_9201_compat import attach_9201_overlay

            attach_9201_overlay(obj)
        except OSError:
            pass
        return obj

    def mapping(self) -> dict[str, str]:
        return {
            "cann_root": self.cann_root,
            "ops_root": self.ops_root,
            "compat_root": self.compat_root,
            "op_dir": self.op_dir,
            "arch_dir": self.arch_dir,
            "repo_root": self.repo_root,
        }

    def resolve_path(self, template: str) -> str:
        out = _sub(template, self.mapping()).replace("\\", "/")
        root = Path(self.cann_root) if self.cann_root else None
        host = paths.cann_host_dir(root) if root else None
        if host and host != "x86_64-linux":
            out = out.replace("/x86_64-linux/", f"/{host}/")
        return paths.adapt_cann_fs_path(out, root)

    def sysroot_includes(self) -> list[str]:
        return [self.resolve_path(p) for p in self.raw.get("sysroot_includes") or []]

    def _ops_family_includes(self) -> list[str]:
        """Family ``common/`` and ``3rd/`` next to the operator (mc2/common, gmm/common, …)."""
        ops = (self.ops_root or "").replace("\\", "/").rstrip("/")
        op_dir = (self.op_dir or "").replace("\\", "/").rstrip("/")
        if not ops or not op_dir:
            return []
        ops_l = ops.lower()
        op_l = op_dir.lower()
        if not (op_l == ops_l or op_l.startswith(ops_l + "/")):
            return []
        rest = op_dir[len(ops):].lstrip("/")
        family = rest.split("/")[0] if rest else ""
        if not family or family in {"common", "tests", "scripts", "examples"}:
            return []
        base = f"{ops}/{family}"
        return [
            f"{base}/common",
            f"{base}/common/utils",
            f"{base}/common/inc",
            f"{base}/3rd",
        ]

    def _arch_cousin_includes(self) -> list[str]:
        """On-disk cousin arch folders (920r1 may ``-I`` ``op_*/arch35``)."""
        op = (self.op_dir or "").replace("\\", "/").rstrip("/")
        if not op:
            return []
        from ascendc_codemap_mcp.engine.source_layout import iter_cousin_arch_dirs

        out: list[str] = []
        for side in ("op_kernel", "op_host"):
            for folder in iter_cousin_arch_dirs(Path(op) / side, self.arch_dir):
                out.append(str(folder).replace("\\", "/"))
        return out

    def add_include(self, path: str, *, side: str) -> bool:
        """Append a runtime extra -I. Returns False when already present or empty.

        Refuse another architecture's folder (``op_kernel/arch22`` while
        ``arch_dir`` is ``arch-920r1``). Cousin ``arch35`` is allowed for 920r1.
        Neutral roots such as ``op_kernel`` stay.
        """
        p = str(path or "").replace("\\", "/").rstrip("/")
        if not p:
            return False
        from ascendc_codemap_mcp.engine.source_layout import is_other_arch_path

        arch = str(self.arch_dir or "").strip()
        if arch and is_other_arch_path(p, arch):
            return False
        current = self.kernel_includes() if side == "kernel" else self.host_includes()
        if p.lower() in {x.replace("\\", "/").rstrip("/").lower() for x in current}:
            return False
        target = self.extra_kernel_includes if side == "kernel" else self.extra_host_includes
        target.append(p)
        return True

    def add_force_include(self, path: str, *, side: str) -> bool:
        p = str(path or "").replace("\\", "/")
        if not p:
            return False
        current = self.kernel_force_includes() if side == "kernel" else self.host_force_includes()
        if p.lower() in {x.replace("\\", "/").lower() for x in current}:
            return False
        target = (
            self.extra_kernel_force_includes if side == "kernel" else self.extra_host_force_includes
        )
        target.append(p)
        return True

    def host_includes(self) -> list[str]:
        out = [self.resolve_path(p) for p in (self.raw.get("host") or {}).get("includes") or []]
        out.extend(self._ops_family_includes())
        out.extend(self._arch_cousin_includes())
        out.extend(self.extra_host_includes)
        return _dedupe_includes(out)

    def kernel_includes(self) -> list[str]:
        out = list(self.overlay_includes)
        out.extend(self.resolve_path(p) for p in (self.raw.get("kernel") or {}).get("includes") or [])
        out.extend(self._ops_family_includes())
        out.extend(self._arch_cousin_includes())
        out.extend(self.extra_kernel_includes)
        return _dedupe_includes(out)

    def host_defines(self) -> dict[str, str]:
        return dict((self.raw.get("host") or {}).get("defines") or {})

    def kernel_defines(self) -> dict[str, str]:
        out = dict((self.raw.get("kernel") or {}).get("defines") or {})
        from ascendc_codemap_mcp.engine.platform_ini import kernel_macros_for_arch

        out.update(kernel_macros_for_arch(self.arch_dir))
        return out

    def erase_qualifiers(self) -> list[str]:
        return list((self.raw.get("kernel") or {}).get("erase_qualifiers") or [])

    def dtype_variants(self) -> dict[str, Any]:
        return dict((self.raw.get("kernel") or {}).get("dtype_variants") or {})

    def force_includes(self) -> list[str]:
        return self.kernel_force_includes()

    def kernel_force_includes(self) -> list[str]:
        out: list[str] = []
        for p in (self.raw.get("kernel") or {}).get("force_include") or []:
            resolved = self.resolve_path(p)
            try:
                if not Path(resolved).is_file():
                    continue
            except OSError:
                continue
            out.append(resolved)
        out.extend(self.extra_kernel_force_includes)
        return _dedupe_includes(out)

    def host_force_includes(self) -> list[str]:
        out = [self.resolve_path(p) for p in (self.raw.get("host") or {}).get("force_include") or []]
        out.extend(self.extra_host_force_includes)
        return _dedupe_includes(out)

    def base_flags(self) -> list[str]:
        flags = list(self.raw.get("base_flags") or [])
        std = self.raw.get("std") or "c++17"
        target = self.raw.get("target") or "aarch64-linux-gnu"
        out = list(flags)
        if "-std=c++17" not in " ".join(out):
            out += [f"-std={std}"]
        if "--target" not in " ".join(out):
            out += [f"--target={target}"]
        return out

    def host_args(self) -> list[str]:
        args = list(self.base_flags())
        for d, v in self.host_defines().items():
            args.append(f"-D{d}" if v == "" else f"-D{d}={v}")
        for fi in self.host_force_includes():
            args += ["-include", fi]
        for p in self.sysroot_includes():
            args += ["-isystem", p]
        for p in self.host_includes():
            args += ["-I", p]
        return args

    def to_dict(self) -> dict[str, Any]:
        """Pickle-safe snapshot for ProcessPool workers."""
        return {
            "raw": self.raw,
            "cann_root": self.cann_root,
            "ops_root": self.ops_root,
            "compat_root": self.compat_root,
            "op_dir": self.op_dir,
            "arch_dir": self.arch_dir,
            "repo_root": self.repo_root,
            "extra_host_includes": list(self.extra_host_includes),
            "extra_kernel_includes": list(self.extra_kernel_includes),
            "extra_host_force_includes": list(self.extra_host_force_includes),
            "extra_kernel_force_includes": list(self.extra_kernel_force_includes),
            "overlay_includes": list(self.overlay_includes),
            "cann_9201": dict(self.cann_9201),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildContext":
        return cls(
            raw=dict(data.get("raw") or {}),
            cann_root=str(data.get("cann_root") or ""),
            ops_root=str(data.get("ops_root") or ""),
            compat_root=str(data.get("compat_root") or ""),
            op_dir=str(data.get("op_dir") or ""),
            arch_dir=require_architecture(data.get("arch_dir")),
            repo_root=str(data.get("repo_root") or ""),
            extra_host_includes=list(data.get("extra_host_includes") or []),
            extra_kernel_includes=list(data.get("extra_kernel_includes") or []),
            extra_host_force_includes=list(data.get("extra_host_force_includes") or []),
            extra_kernel_force_includes=list(data.get("extra_kernel_force_includes") or []),
            overlay_includes=list(data.get("overlay_includes") or []),
            cann_9201=dict(data.get("cann_9201") or {}),
        )

    def kernel_args(
        self,
        dtype_variant: str | None = None,
        *,
        source_path: str | Path | None = None,
        orig_assignment: dict[str, str] | None = None,
    ) -> list[str]:
        args = list(self.base_flags())
        for q in self.erase_qualifiers():
            if q in FUNCTION_LIKE_QUALIFIERS:
                args.append(f"-D{q}(...)=")
            else:
                args.append(f"-D{q}=")
        for d, v in self.kernel_defines().items():
            args.append(f"-D{d}" if v == "" else f"-D{d}={v}")
        dv = self.dtype_variants()
        if dtype_variant:
            from ascendc_codemap_mcp.engine.kernel_gates import discover_kernel_gates

            gates = discover_kernel_gates(
                source_path,
                op_dir=self.op_dir,
                ops_root=self.ops_root,
                macros=self.kernel_defines(),
            )
            args.extend(
                gates.clang_defines(
                    dtype_variant,
                    dv.get("dt_enum_defines") or {},
                    orig_assignment=orig_assignment,
                )
            )
        for fi in self.kernel_force_includes():
            args += ["-include", fi]
        for p in self.sysroot_includes():
            args += ["-isystem", p]
        for p in self.kernel_includes():
            args += ["-I", p]
        return args
