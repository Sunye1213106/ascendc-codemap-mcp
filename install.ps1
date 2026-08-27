$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python -m pip install -e $Root
ascendc-codemap-mcp install @args
