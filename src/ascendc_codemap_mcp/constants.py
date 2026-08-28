# -*- coding: utf-8 -*-
"""Product identity for AscendC CodeMap MCP."""
from __future__ import annotations

PRODUCT_NAME = "ascendc-codemap-mcp"
SERVER_NAME = "ascendc-codemap-mcp"
SERVER_VERSION = "0.4.0"
PROTOCOL = "2026-07-28"
PRODUCT_DIR_NAME = ".ascendc-codemap"
MCP_MARK_BEGIN = "# >>> ascendc-codemap-mcp >>>"
MCP_MARK_END = "# <<< ascendc-codemap-mcp <<<"
AGENTS_MARK_BEGIN = "<!-- ascendc-codemap-mcp:start -->"
AGENTS_MARK_END = "<!-- ascendc-codemap-mcp:end -->"
CODEX_ENV_VARS = (
    "ASCENDC_CODEMAP_CANN_ROOT",
    "ASCENDC_CODEMAP_CACHE_DIR",
    "ASCEND_HOME_PATH",
    "ASCEND_CANN_PACKAGE_PATH",
)
SKILL_NAMES = ("index-operator", "update-operator", "query-codemap")
