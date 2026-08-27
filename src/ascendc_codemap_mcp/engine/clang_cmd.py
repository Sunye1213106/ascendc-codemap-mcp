# -*- coding: utf-8 -*-
"""Locate a clang driver and run ``clang -E``. Product analyze uses this for TPL headers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

CLANG_SEARCH = [
    "clang",
    "clang++",
    "clang.exe",
    r"C:\ProgramData\miniconda3\envs\uoclang\Library\bin\clang.exe",
    "/usr/bin/clang",
    "/usr/bin/clang++",
]


def find_clang(explicit: str | None = None) -> str | None:
    """Locate a clang driver binary."""
    import os
    import shutil

    env = [
        os.environ.get("CLANG_EXE"),
        os.environ.get("UO_CLANG"),
        (str(Path(os.environ["LLVM_BIN"]) / "clang.exe") if os.environ.get("LLVM_BIN") else None),
        (str(Path(os.environ["LLVM_HOME"]) / "bin" / "clang.exe") if os.environ.get("LLVM_HOME") else None),
    ]
    common = [
        r"C:\Program Files\LLVM\bin\clang.exe",
        r"C:\Program Files (x86)\LLVM\bin\clang.exe",
        r"C:\msys64\clang64\bin\clang.exe",
        r"C:\msys64\ucrt64\bin\clang.exe",
    ]
    for cand in ([explicit] if explicit else []) + env + common + CLANG_SEARCH:
        if not cand:
            continue
        found = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if found:
            return found
    return None


def clang_preprocess(
    path: str | Path,
    args: Iterable[str] | None = None,
    *,
    clang_exe: str | None = None,
    timeout_s: int = 60,
) -> str | None:
    """Run clang ``-E``. None if the driver is missing."""
    import subprocess

    exe = find_clang(clang_exe)
    if exe is None:
        return None
    cmd = [exe, "-E", *[str(a) for a in (args or ())], str(path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max(5, int(timeout_s)),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not proc.stdout:
        return None
    return proc.stdout
