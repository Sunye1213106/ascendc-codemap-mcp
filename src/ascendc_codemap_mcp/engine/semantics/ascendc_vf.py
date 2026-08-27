# -*- coding: utf-8 -*-
"""CANN Reg / VF compute API spellings loaded from installed headers.

VF / MicroAPI / SIMD-Reg (``namespace MicroAPI = Reg``) is **not** on arch22.
CANN ``kernel_reg_compute_intf.h`` only includes the family when
``__NPU_ARCH__`` is 3510 / 5102 / 3003 / 3113 / 9201 (arch35 and later
cousins). Unpublished DAV_9201 (``arch-920r1``) shares the arch35 VF ISA.
arch22 (DAV_2201) keeps Level-2 LocalTensor vector APIs
(``kernel_operator_vec_*_intf.h``) only.

``CreateMask`` / ``CreateAddrReg`` return MaskReg/AddrReg, not void — the
scanner must not require ``inline void``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Matches ``__simd_callee__ inline void LoadAlign(`` and
# ``__simd_callee__ inline MaskReg CreateMask(``.
_SIMD_FN_RE = re.compile(
    r"__simd_callee__\s+(?:inline\s+)?(?:[\w:<>,\s*&]+?)\s+([A-Za-z_]\w*)\s*\("
)
_LEVEL2_RE = re.compile(
    r"__aicore__\s+inline\s+void\s+([A-Za-z_]\w*)\s*\("
)

# Older Level-2 names still used in operators; map onto the current Reg spelling.
VF_ALIASES: dict[str, str] = {
    "FusedExpSub": "ExpSub",
    "FusedMulDstAdd": "MulDstAdd",
}

# Spellings that also exist as project scalar/logic helpers or cube/generic
# Load/Store/Move. Never prove from the name alone.
AMBIGUOUS_VF_ROOTS: frozenset[str] = frozenset(
    {"Min", "Max", "Or", "And", "Xor", "Not", "Load", "Store", "Move", "Gather", "Scatter"}
)

_SKIP_SIMD_NAMES = frozenset(
    {
        "Print",
        "RegTensor",
        "MaskReg",
        "AddrReg",
        "UnalignReg",
        "UnalignRegForLoad",
        "UnalignRegForStore",
    }
)

# kernel_reg_compute_intf.h: VF/Reg headers are compiled in only for these.
# 9201 is unpublished; same VF ISA as 3510, distinct DAV identity.
VF_NPU_ARCHS: frozenset[int] = frozenset({3510, 5102, 3003, 3113, 9201})

# Always keep fused / mask / load-store spellings even if a unpack omits them.
# DataCopyScatter is the impl alias of VF Scatter (not a Level-2 DataCopy).
_ALWAYS_VF_REG: frozenset[str] = frozenset(
    {
        "ExpSub",
        "MulDstAdd",
        "AbsSub",
        "MulsCast",
        "CreateMask",
        "UpdateMask",
        "CreateAddrReg",
        "MoveMask",
        "LoadAlign",
        "StoreAlign",
        "LoadUnAlign",
        "StoreUnAlign",
        "LoadUnAlignPre",
        "StoreUnAlignPost",
        "DataCopyScatter",
        "GatherB",
        "LocalMemBar",
    }
)


def architecture_npu_arch(architecture: str) -> int | None:
    """Map ``arch35`` / ``3510`` / ``arch-920r1`` onto the CANN ``__NPU_ARCH__`` number."""
    from ascendc_codemap_mcp.engine.platform_ini import ARCH_TO_NPU_ARCH
    from ascendc_codemap_mcp.engine.source_layout import canonicalize_architecture

    raw = str(architecture or "").strip()
    if not raw:
        return None
    canon = canonicalize_architecture(raw)
    if canon in ARCH_TO_NPU_ARCH:
        return int(ARCH_TO_NPU_ARCH[canon])
    low = raw.lower()
    if low in ARCH_TO_NPU_ARCH:
        return int(ARCH_TO_NPU_ARCH[low])
    if raw.isdigit():
        return int(raw)
    m = re.fullmatch(r"arch(\d{2})", canon or low)
    if not m:
        return None
    prefix = m.group(1)
    known = set(ARCH_TO_NPU_ARCH.values()) | set(VF_NPU_ARCHS)
    for npu in sorted(known):
        if str(npu).startswith(prefix):
            return int(npu)
    return None


def architecture_has_vf(architecture: str) -> bool:
    """False on arch22 (DAV_2201). Unknown arch fails open (do not hide APIs).

    ``arch-920r1`` is first-class: ``architecture_npu_arch`` maps it to 9201,
    which is listed in ``VF_NPU_ARCHS``.
    """
    raw = str(architecture or "").strip()
    npu = architecture_npu_arch(raw)
    if npu is not None and npu in VF_NPU_ARCHS:
        return True
    if npu is None:
        return True
    return False


def _header_roots(cann: Path) -> list[Path]:
    from ascendc_codemap_mcp.engine.paths import resolve_cann_relative

    rels = (
        "cann-asc-devkit/x86_64-linux/tikcpp/tikcfw/interface/reg_compute",
        "cann-asc-devkit/x86_64-linux/tikcpp/tikcfw/interface",
        "cann-asc-devkit/x86_64-linux/ascendc/include/basic_api/interface/reg_compute",
        "cann-asc-devkit/x86_64-linux/asc/include/basic_api/reg_compute",
        "cann-asc-devkit/x86_64-linux/asc/include/basic_api",
        "cann-asc-devkit/x86_64-linux/include/ascendc/basic_api/interface/reg_compute",
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for rel in rels:
        d = resolve_cann_relative(cann, rel)
        if d.is_dir() and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _keep_name(name: str) -> bool:
    return (
        bool(name)
        and name[0].isupper()
        and name not in _SKIP_SIMD_NAMES
        and not name.endswith("Impl")
    )


def _scan_simd_file(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return {n for n in _SIMD_FN_RE.findall(text) if _keep_name(n)}


def _scan_l2_file(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return {n for n in _LEVEL2_RE.findall(text) if _keep_name(n)}


def _scan_glob(folders: list[Path], pattern: str, scanner) -> set[str]:
    names: set[str] = set()
    for folder in folders:
        for path in folder.glob(pattern):
            if path.is_file():
                names.update(scanner(path))
    return names


@lru_cache(maxsize=1)
def cann_vf_reg_api_names() -> frozenset[str]:
    """Reg / MicroAPI spellings from ``kernel_reg_compute_*.h`` (arch35+)."""
    from ascendc_codemap_mcp.engine.paths import cann_root

    names: set[str] = set(_ALWAYS_VF_REG)
    names.update(VF_ALIASES)
    names.update(VF_ALIASES.values())
    root = cann_root()
    if root is not None:
        folders = _header_roots(root)
        names.update(_scan_glob(folders, "kernel_reg_compute_*.h", _scan_simd_file))
    return frozenset(n for n in names if _keep_name(n))


@lru_cache(maxsize=1)
def cann_l2_vector_api_names() -> frozenset[str]:
    """Level-2 LocalTensor vector APIs — present on arch22 and arch35."""
    from ascendc_codemap_mcp.engine.paths import cann_root

    names: set[str] = set()
    root = cann_root()
    if root is not None:
        folders = _header_roots(root)
        names.update(_scan_glob(folders, "kernel_operator_vec_*_intf.h", _scan_l2_file))
    return frozenset(n for n in names if _keep_name(n))


@lru_cache(maxsize=1)
def cann_vf_api_names() -> frozenset[str]:
    """Reg/VF plus Level-2 vector names and Fused* aliases (architecture-neutral set)."""
    names: set[str] = set(cann_vf_reg_api_names())
    names.update(cann_l2_vector_api_names())
    names.update({"Or", "And", "Xor"})
    return frozenset(names)


def vf_only_api_names() -> frozenset[str]:
    """VF/Reg spellings that do **not** exist as Level-2 LocalTensor APIs."""
    return frozenset(cann_vf_reg_api_names() - cann_l2_vector_api_names())


def vf_root_spelling(callee: str) -> str:
    short = str(callee or "").split("::")[-1]
    if "<" in short:
        short = short.split("<", 1)[0]
    return VF_ALIASES.get(short, short)


def is_vf_only_api(callee: str) -> bool:
    return vf_root_spelling(callee) in vf_only_api_names()


def is_cann_vf_api(callee: str, architecture: str = "") -> bool:
    """True when *callee* is a CANN vector/Reg API **legal on this architecture**.

    arch22: Level-2 names only. arch35+: Level-2 and VF/Reg.
    Empty architecture fails open (used by lexical unit tests).
    """
    short = vf_root_spelling(callee)
    if short not in cann_vf_api_names():
        return False
    if architecture and not architecture_has_vf(architecture):
        return short in cann_l2_vector_api_names()
    return True


def is_ambiguous_vf_name(callee: str) -> bool:
    return vf_root_spelling(callee) in AMBIGUOUS_VF_ROOTS
