#!/usr/bin/env bash
# Refresh AscendC CodeMap MCP for OpenCode.
#
# Always uninstalls the previous OpenCode Host bits first, then reinstalls.
# Default: skip pip. FORCE_PIP=1 to reinstall the editable package.
#
#   1. Fully quit OpenCode
#   2. ./refresh-opencode.sh
#   3. Start OpenCode again
#
#   ./refresh-opencode.sh --what-if
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$BUNDLE_ROOT"

SKIP_PIP="${SKIP_PIP:-1}"
if [[ "${FORCE_PIP:-0}" == "1" ]]; then
  SKIP_PIP=0
fi

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

if [[ "${1:-}" == "--what-if" || "${1:-}" == "--WhatIf" ]]; then
  echo "[WhatIf] Would run:"
  echo "  1. ./uninstall.sh opencode"
  echo "  2. ./install.sh opencode"
  echo "  pip: $([[ "$SKIP_PIP" == "1" ]] && echo skip || echo reinstall)"
  echo "  3. Verify MCP entry + CodeMap skills"
  exit 0
fi

echo "=== AscendC CodeMap OpenCode refresh ==="
echo "Repo: $BUNDLE_ROOT"

echo "[1/3] Uninstall OpenCode CodeMap bits..."
"$BUNDLE_ROOT/uninstall.sh" opencode

echo "[2/3] Reinstall OpenCode CodeMap bits..."
SKIP_PIP="$SKIP_PIP" "$BUNDLE_ROOT/install.sh" opencode

echo "[3/3] Verify install matches current repo..."
PYTHON_BIN="$(resolve_python)"
CODEMAP_BUNDLE_ROOT="$BUNDLE_ROOT" "$PYTHON_BIN" - <<'PY'
import os
import pathlib

from ascendc_codemap_mcp.constants import AGENTS_MARK_BEGIN, PRODUCT_NAME, SKILL_NAMES
from ascendc_codemap_mcp.install import opencode
from ascendc_codemap_mcp.install.jsonutil import read_json

root = pathlib.Path(os.environ["CODEMAP_BUNDLE_ROOT"]).resolve()
mod = pathlib.Path(__import__("ascendc_codemap_mcp").__file__).resolve()
print("MODULE=" + str(mod))
assert str(mod).lower().startswith(str(root).lower()), (mod, root)

path = opencode.config_path()
assert path.is_file(), path
entry = (read_json(path).get("mcp") or {}).get(PRODUCT_NAME)
assert isinstance(entry, dict), entry
blob = str(entry)
assert "ascendc_codemap_mcp" in blob or "ascendc-codemap-mcp" in blob, entry
print("CONFIG=" + str(path))

home = opencode.home()
for name in SKILL_NAMES:
    skill = home / "skills" / f"ascendc-codemap-{name}" / "SKILL.md"
    assert skill.is_file(), skill
agents = home / "AGENTS.md"
text = agents.read_text(encoding="utf-8") if agents.is_file() else ""
assert AGENTS_MARK_BEGIN in text, agents
print("VERIFY_OK")
PY

echo "=== Refresh complete ==="
echo "Fully start OpenCode and confirm MCP server ascendc-codemap-mcp is connected."
