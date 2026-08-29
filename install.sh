#!/usr/bin/env bash
# AscendC CodeMap MCP installer
#
# Usage:
#   ./install.sh                          # OpenCode (default)
#   ./install.sh opencode|cursor|claude|codex|all
#   ./install.sh uninstall-opencode
#   ./uninstall.sh opencode
#   SKIP_PIP=1 ./install.sh opencode
#   PYTHON=python3.12 ./install.sh opencode
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="${1:-opencode}"
SKIP_PIP="${SKIP_PIP:-0}"

if [[ "$PLATFORM" == uninstall-* ]]; then
  exec "$BUNDLE_ROOT/uninstall.sh" "${PLATFORM#uninstall-}"
fi

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

if [[ "$SKIP_PIP" != "1" ]]; then
  "$PYTHON_BIN" -m pip install -e "$BUNDLE_ROOT"
fi

"$PYTHON_BIN" -m ascendc_codemap_mcp install --host "$PLATFORM"

echo "Installed AscendC CodeMap MCP ($PLATFORM)"
echo "Keep this checkout; pip -e installs point at it. Fully quit and reopen the Host."
