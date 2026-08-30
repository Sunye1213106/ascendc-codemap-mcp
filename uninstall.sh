#!/usr/bin/env bash
# AscendC CodeMap MCP uninstaller
#
# Usage:
#   ./uninstall.sh                 # all hosts (default)
#   ./uninstall.sh opencode|cursor|claude|codex|all
#
# Removes only this product's MCP entry and skills. Does not glob other
# agents/plugins.
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="${1:-all}"
PLATFORM="${PLATFORM#uninstall-}"

case "$PLATFORM" in
  opencode|cursor|claude|codex|all) ;;
  *)
    echo "Usage: $0 opencode|cursor|claude|codex|all" >&2
    exit 2
    ;;
esac

resolve_python() {
  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    command -v "$PYTHON"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "ERROR: python3 or python >= 3.10 required" >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
cd "$BUNDLE_ROOT"

echo "Uninstalling AscendC CodeMap MCP ($PLATFORM)"
"$PYTHON_BIN" -m ascendc_codemap_mcp uninstall --host "$PLATFORM"
echo "Uninstalled $PLATFORM"
