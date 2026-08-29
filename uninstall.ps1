# AscendC CodeMap MCP uninstaller (Windows)
#
# Usage:
#   .\uninstall.ps1                 # OpenCode (default)
#   .\uninstall.ps1 opencode|cursor|claude|codex|all
#
# Removes only this product's MCP entry and skills. Does not glob other
# agents/plugins (Pilot leftovers, cannbot-auth.js, …).
param(
  [Parameter(Position = 0)]
  [string]$Platform = "opencode"
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BundleRoot

if ($Platform -like "uninstall-*") {
  $Platform = $Platform.Substring("uninstall-".Length)
}
if ($Platform -notin @("opencode", "cursor", "claude", "codex", "all")) {
  throw "Usage: .\uninstall.ps1 opencode|cursor|claude|codex|all"
}

function Get-PythonExe {
  if (-not [string]::IsNullOrWhiteSpace($env:PYTHON)) {
    $named = Get-Command $env:PYTHON -ErrorAction SilentlyContinue
    if ($named -and $named.Source) { return $named.Source }
    if (Test-Path -LiteralPath $env:PYTHON) { return $env:PYTHON }
  }
  foreach ($name in @("python", "python3")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
  }
  throw "python >= 3.10 required (set `$env:PYTHON if needed)"
}

function Stop-LeftoverCodeMapMcp {
  # Kill leftover stdio/HTTP MCP servers only — never this uninstall CLI.
  $exclude = '\b(install|uninstall|doctor|index|update|query|status|discover|cann-extract|extract-cann)\b'
  $stopped = 0
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match '^(python|pythonw|ascendc-codemap-mcp)\.exe$' -and
      ([string]$_.CommandLine) -match 'ascendc_codemap_mcp|ascendc-codemap-mcp' -and
      ([string]$_.CommandLine) -notmatch $exclude
    } |
    ForEach-Object {
      Write-Host ("  stopping leftover MCP PID {0}" -f $_.ProcessId)
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      $stopped++
    }
  if ($stopped -gt 0) { Start-Sleep -Milliseconds 400 }
  return $stopped
}

Write-Host "Uninstalling AscendC CodeMap MCP ($Platform)"
[void](Stop-LeftoverCodeMapMcp)

$Python = Get-PythonExe
& $Python -m ascendc_codemap_mcp uninstall --host $Platform
if ($LASTEXITCODE -ne 0) { throw "ascendc_codemap_mcp uninstall failed" }

Write-Host "Uninstalled $Platform"
exit 0
