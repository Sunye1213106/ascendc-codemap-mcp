# -*- coding: utf-8 -*-
"""Child-process entry for KernelIR. Must not be ``kernel_ir.py`` itself.

``python -m uo_init.kernel_ir`` would pickle KernelIR as ``__main__.KernelIR``,
which the parent cannot unpickle. This wrapper imports the real class.
"""
from __future__ import annotations

import sys

from ascendc_codemap_mcp.engine.kernel_ir import run_kernel_ir_job


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m uo_init.kernel_ir_job IN.pkl OUT.pkl")
    run_kernel_ir_job(sys.argv[1], sys.argv[2])
