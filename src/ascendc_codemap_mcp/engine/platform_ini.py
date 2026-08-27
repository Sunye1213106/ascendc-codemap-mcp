# -*- coding: utf-8 -*-
"""Load locked platform constants from CANN ``platform_config/*.ini``.

Compilation target (arch / SKU) is fixed for a uo-init run, so ``aicNum`` /
``l2_size`` are not free variables — they are closed constants from the INI.
"""
from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from ascendc_codemap_mcp.engine.source_layout import canonicalize_architecture

# arch_dir short name → NpuArch number (`NpuArch=3510` in INI / DAV_NNNN).
ARCH_TO_NPU_ARCH = {
    "arch35": 3510,
    "arch22": 2201,
    "arch32": 3202,
    "arch50": 5001,
    "arch-920r1": 9201,  # DAV_9201 unpublished; first-class identity
}

# Default SKU when the run only names an arch.
# arch-920r1 has no published NpuArch=9201 INI yet; lock the same SKU name
# as arch35 so load_platform_profile can resolve a concrete profile.
DEFAULT_SKU_BY_ARCH = {
    "arch35": "Ascend950PR_9589",  # Server, cube_core_cnt=32
    "arch-920r1": "Ascend950PR_9589",
    "arch22": "Ascend910B2",
}

# Kernel -D set for a BuildVariant. Never silently fall back to 3510.
# arch-920r1 compiles as DAV_9201 (__NPU_ARCH__=9201). CANN headers that still
# gate on 3510 are widened by cann_9201_compat overlay, not by this table.
ARCH_KERNEL_MACROS: dict[str, dict[str, str]] = {
    "arch35": {"__NPU_ARCH__": "3510", "__DAV_C310__": "", "__CCE_AICORE__": "310"},
    "arch-920r1": {"__NPU_ARCH__": "9201", "__DAV_C310__": "", "__CCE_AICORE__": "310"},
    "arch22": {"__NPU_ARCH__": "2201", "__DAV_C220__": "", "__CCE_AICORE__": "220"},
    "arch32": {"__NPU_ARCH__": "3202"},
    "arch50": {"__NPU_ARCH__": "5001"},
}


def dav_name_for_arch(arch_dir: str | None) -> str | None:
    """``arch-920r1`` → ``DAV_9201``. None when the arch_dir is unknown."""
    arch = canonicalize_architecture(arch_dir)
    npu = ARCH_TO_NPU_ARCH.get(arch)
    return f"DAV_{npu}" if npu is not None else None


def kernel_macros_for_arch(arch_dir: str | None) -> dict[str, str]:
    """Clang -D map for this architecture. Unknown arch: NPU number only."""
    arch = canonicalize_architecture(arch_dir) or str(arch_dir or "").strip()
    if not arch:
        return {}
    known = ARCH_KERNEL_MACROS.get(arch)
    if known is not None:
        return dict(known)
    npu = ARCH_TO_NPU_ARCH.get(arch)
    if npu is not None:
        return {"__NPU_ARCH__": str(npu)}
    return {}

_PLATFORM_DIR_RE = re.compile(r"platform_config$", re.I)


@dataclass(frozen=True)
class PlatformProfile:
    soc_version: str
    npu_arch: int
    cube_core_cnt: int
    vector_core_cnt: int
    ai_core_cnt: int
    l2_size: int
    memory_size: int
    ini_path: str
    sku_fallback: str = ""
    npu_arch_source: str = "ini"

    @property
    def aic_num(self) -> int:
        return self.cube_core_cnt or self.ai_core_cnt


def find_platform_config_dirs(cann_root: str | Path) -> list[Path]:
    root = Path(cann_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for p in root.rglob("platform_config"):
        if p.is_dir() and _PLATFORM_DIR_RE.search(p.name):
            found.append(p)
    # Prefer runtime package paths when several trees exist.
    found.sort(key=lambda p: (0 if "npu-runtime" in str(p).lower() else 1, str(p)))
    return found


def _parse_ini(path: Path) -> PlatformProfile | None:
    cp = configparser.ConfigParser()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # INI files use `#` comments; ConfigParser needs them stripped or allow.
    cleaned = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    try:
        cp.read_string(cleaned)
    except configparser.Error:
        return None
    if "SoCInfo" not in cp and "version" not in cp:
        return None

    def _get(section: str, key: str, default: str = "0") -> str:
        if section in cp and key in cp[section]:
            return cp[section][key].strip()
        return default

    soc = _get("version", "SoC_version", path.stem)
    try:
        npu = int(_get("version", "NpuArch", "0"))
    except ValueError:
        npu = 0
    try:
        cube = int(_get("SoCInfo", "cube_core_cnt", "0"))
        vec = int(_get("SoCInfo", "vector_core_cnt", "0"))
        ai = int(_get("SoCInfo", "ai_core_cnt", "0"))
        l2 = int(_get("SoCInfo", "l2_size", "0"))
        mem = int(_get("SoCInfo", "memory_size", "0"))
    except ValueError:
        return None
    return PlatformProfile(
        soc_version=soc,
        npu_arch=npu,
        cube_core_cnt=cube,
        vector_core_cnt=vec,
        ai_core_cnt=ai,
        l2_size=l2,
        memory_size=mem,
        ini_path=str(path).replace("\\", "/"),
    )


def list_profiles(
    cann_root: str | Path,
    *,
    npu_arch: int | None = None,
) -> list[PlatformProfile]:
    out: list[PlatformProfile] = []
    seen: set[str] = set()
    for d in find_platform_config_dirs(cann_root):
        for ini in sorted(d.glob("*.ini")):
            prof = _parse_ini(ini)
            if prof is None:
                continue
            if npu_arch is not None and prof.npu_arch != npu_arch:
                continue
            if prof.soc_version in seen:
                continue
            seen.add(prof.soc_version)
            out.append(prof)
    return out


def load_platform_profile(
    cann_root: str | Path,
    *,
    arch_dir: str = "",
    platform_sku: str | None = None,
) -> PlatformProfile:
    """Resolve a locked SKU profile or raise if the INI cannot be found."""
    arch = canonicalize_architecture(arch_dir) or str(arch_dir or "").strip()
    npu = ARCH_TO_NPU_ARCH.get(arch)
    sku = platform_sku or DEFAULT_SKU_BY_ARCH.get(arch)
    profiles = list_profiles(cann_root, npu_arch=npu)
    if not profiles and sku:
        # Unpublished NpuArch (e.g. 9201) has no INI; resolve the named SKU.
        profiles = [
            p
            for p in list_profiles(cann_root)
            if p.soc_version == sku
            or p.soc_version.startswith(sku)
            or Path(p.ini_path).stem == sku
        ]
    found: PlatformProfile | None = None
    if sku:
        for p in profiles:
            if p.soc_version == sku or p.soc_version.startswith(sku):
                found = p
                break
        if found is None:
            # Allow bare stem match against filename when SoC_version differs.
            for p in profiles:
                if Path(p.ini_path).stem == sku:
                    found = p
                    break
    if found is None and profiles:
        # Prefer the arch default if listed among NpuArch matches, else first.
        pref = DEFAULT_SKU_BY_ARCH.get(arch)
        for p in profiles:
            if pref and p.soc_version == pref:
                found = p
                break
        if found is None:
            found = profiles[0]
    if found is None:
        raise FileNotFoundError(
            f"no platform_config INI under {cann_root} for arch={arch_dir} sku={sku!r}"
        )
    if npu is not None and found.npu_arch != npu:
        return replace(
            found,
            npu_arch=npu,
            sku_fallback=found.soc_version,
            npu_arch_source="sku_fallback",
        )
    return found


def cube_core_domain(profiles: Iterable[PlatformProfile]) -> list[int]:
    vals = sorted({p.aic_num for p in profiles if p.aic_num > 0})
    return vals
