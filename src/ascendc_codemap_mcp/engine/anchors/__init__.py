# -*- coding: utf-8 -*-
"""L1 deterministic anchors: opdef / registry / kernel entry."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Operator kernels dispatch through an `INVOKE_<OP>_<VARIANT>` macro. clang
# expands these away, so they are recovered textually; the operator tag varies
# per operator, hence no fixed prefix.
INVOKE_MACRO_RE = re.compile(r"\bINVOKE_[A-Z][A-Z0-9_]{2,}\b")


class ValidationError(ValueError):
    pass


@dataclass
class Evidence:
    file: str
    line: int
    snippet: str

    def validate(self) -> None:
        if not self.file or not self.line or not self.snippet:
            raise ValidationError("anchor requires file/line/snippet")


@dataclass
class Anchor:
    role: str
    symbol: str
    evidence: Evidence
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.evidence.validate()


REG_INPUT = re.compile(r"""(?:this->|\w+\.)Input\(\s*"([^"]+)"\s*\)""")
REG_OUTPUT = re.compile(r"""(?:this->|\w+\.)Output\(\s*"([^"]+)"\s*\)""")
REG_ATTR = re.compile(r"""(?:this->|\w+\.)Attr\(\s*"([^"]+)"\s*\)""")

REG_TILING = re.compile(
    r"REGISTER_TILING_TEMPLATE_WITH_ARCH\s*\(\s*"
    r"(\w+)\s*,\s*(\w+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*\)",
    re.MULTILINE,
)


def _line_of(text: str, idx: int) -> int:
    return text[:idx].count("\n") + 1


def extract_opdef(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    inputs = [m.group(1) for m in REG_INPUT.finditer(text)]
    outputs = [m.group(1) for m in REG_OUTPUT.finditer(text)]
    attrs = [m.group(1) for m in REG_ATTR.finditer(text)]

    def uniq(seq: list[str]) -> list[str]:
        seen: list[str] = []
        for x in seq:
            if x not in seen:
                seen.append(x)
        return seen

    anchors = []
    for kind, names, raw in (
        ("input", uniq(inputs), inputs),
        ("output", uniq(outputs), outputs),
        ("attr", uniq(attrs), attrs),
    ):
        for name in names:
            m = re.search(rf'this->{kind.title() if kind != "attr" else "Attr"}\(\s*"{name}"', text)
            # simpler search
            pat = {
                "input": rf'(?:this->|[\w.]+\.)Input\(\s*"{name}"',
                "output": rf'(?:this->|[\w.]+\.)Output\(\s*"{name}"',
                "attr": rf'(?:this->|[\w.]+\.)Attr\(\s*"{name}"',
            }[kind]
            mm = re.search(pat, text)
            line = _line_of(text, mm.start()) if mm else 1
            snip = mm.group(0) if mm else name
            a = Anchor(
                role=f"opdef_{kind}",
                symbol=name,
                evidence=Evidence(file=str(path), line=line, snippet=snip),
            )
            a.validate()
            anchors.append(a)
    return {
        "inputs_unique": uniq(inputs),
        "outputs_unique": uniq(outputs),
        "attrs_unique": uniq(attrs),
        "inputs_raw_count": len(inputs),
        "outputs_raw_count": len(outputs),
        "attrs_raw_count": len(attrs),
        "anchors": anchors,
    }


def extract_registry(root: str | Path, op_name: str) -> list[dict]:
    """Registry sites for one operator. `op_name` is required: a default would
    silently return an empty list for every other operator."""
    root = Path(root)
    hits = []
    skip_dirs = {".ascendc-codemap", ".git", ".svn", "__pycache__", "cache", "build", "output"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for name in filenames:
            if not name.endswith((".cpp", ".h", ".hpp")):
                continue
            path = Path(dirpath) / name
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text_j = re.sub(r"\\\r?\n", " ", text)
            for m in REG_TILING.finditer(text_j):
                op, cls, arch, pri = m.groups()
                if op != op_name:
                    continue
                hits.append(
                    {
                        "op": op,
                        "class": cls,
                        "arch_expr": arch.strip(),
                        "priority": int(pri),
                        "file": str(path).replace("\\", "/"),
                        "line": text_j[: m.start()].count("\n") + 1,
                    }
                )
    hits.sort(key=lambda r: (r["arch_expr"], r["priority"]))
    return hits


def arch_bucket(arch_expr: str) -> str:
    # Distinct DAV identities — do not collapse 9201 into 3510.
    if "9201" in arch_expr:
        return "DAV_9201"
    if "3510" in arch_expr:
        return "DAV_3510"
    if "2201" in arch_expr or "2002" in arch_expr:
        return "DAV_2201_family"
    return arch_expr


def extract_kernel_entry(path: str | Path, entry_name: str | None = None) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.tpl_bind import parse_kernel_nttps

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    nttps = parse_kernel_nttps(text, entry_name)
    invoke = INVOKE_MACRO_RE.findall(text)
    return {
        "file": str(path).replace("\\", "/"),
        "nttps": [{"type": t, "name": n} for t, n in nttps],
        "nttp_arity": len(nttps),
        "invoke_macros": sorted(set(invoke)),
    }


def build_anchors_yaml(
    opdef_path: str | Path,
    host_root: str | Path,
    kernel_entry: str | Path,
    *,
    op_name: str,
    entry_name: str | None = None,
) -> dict[str, Any]:
    opdef = extract_opdef(opdef_path)
    regs = extract_registry(host_root, op_name)
    by_arch: dict[str, list] = {}
    for r in regs:
        by_arch.setdefault(arch_bucket(r["arch_expr"]), []).append(r)
    for k in by_arch:
        by_arch[k] = sorted(by_arch[k], key=lambda x: x["priority"])
    entry = extract_kernel_entry(kernel_entry, entry_name)
    return {
        "opdef": {
            "inputs": opdef["inputs_unique"],
            "outputs": opdef["outputs_unique"],
            "attrs": opdef["attrs_unique"],
            "inputs_raw_count": opdef["inputs_raw_count"],
            "outputs_raw_count": opdef["outputs_raw_count"],
        },
        "registry_by_arch": by_arch,
        "kernel_entry": entry,
    }
