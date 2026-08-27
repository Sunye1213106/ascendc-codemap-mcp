#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 -m pip install -e "$ROOT"
ascendc-codemap-mcp install "$@"
