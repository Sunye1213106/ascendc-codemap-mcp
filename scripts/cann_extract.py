#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unpack a CANN toolkit ``.run`` without executing the installer.

Same entry as ``python -m ascendc_codemap_mcp cann-extract``.
Works from a checkout even before ``pip install -e .``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ascendc_codemap_mcp.cann_extract import main

if __name__ == "__main__":
    raise SystemExit(main())
