# Refresh AscendC CodeMap MCP for OpenCode.
#
# Always uninstalls the previous OpenCode Host bits first, then reinstalls.
# Default (testing): skip pip. Use -ForcePip to reinstall the editable package.
#
# Use after MCP / skill changes, before re-testing in OpenCode:
#   1. Fully quit OpenCode (not just close a chat tab)
#   2. From this repo root:
#        .\refresh-opencode.ps1
#   3. Start OpenCode again
#
# Options:
#   -SkipPip     (default) Skip pip reinstall
#   -ForcePip    Reinstall editable ascendc-codemap-mcp
#   -WhatIf      Show plan only
#
param(
  [switch]$SkipPip,
  [switch]$ForcePip,
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BundleRoot
$InstallPs1 = Join-Path $BundleRoot "install.ps1"
$UninstallPs1 = Join-Path $BundleRoot "uninstall.ps1"

$sw = [Diagnostics.Stopwatch]::StartNew()
# -SkipPip is the default; -ForcePip is the only way to reinstall packages.
$doPip = [bool]$ForcePip -and -not $SkipPip

function Invoke-RepoScript {
  param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$Arg = "",
    [hashtable]$EnvExtra = @{}
  )
  # install.ps1 / uninstall.ps1 use `exit` — must run in a child process.
  $envAssign = ""
  foreach ($k in $EnvExtra.Keys) {
    $val = [string]$EnvExtra[$k]
    $envAssign += "`$env:$k='$val'; "
  }
  if ($envAssign) {
    $cmd = if ($Arg) { "$envAssign & `"$Script`" $Arg; exit `$LASTEXITCODE" } else { "$envAssign & `"$Script`"; exit `$LASTEXITCODE" }
    $p = Start-Process -FilePath "powershell.exe" `
      -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) `
      -WorkingDirectory $BundleRoot `
      -Wait -PassThru -NoNewWindow
  } else {
    $fileArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Script)
    if ($Arg) { $fileArgs += $Arg }
    $p = Start-Process -FilePath "powershell.exe" `
      -ArgumentList $fileArgs `
      -WorkingDirectory $BundleRoot `
      -Wait -PassThru -NoNewWindow
  }
  if ($null -eq $p -or $p.ExitCode -ne 0) {
    $code = if ($null -eq $p) { "null" } else { $p.ExitCode }
    throw "$(Split-Path $Script -Leaf) $Arg failed with exit $code"
  }
}

function Assert-True([bool]$Cond, [string]$Msg) {
  if (-not $Cond) { throw "VERIFY FAIL: $Msg" }
  Write-Host "  OK  $Msg"
}

Write-Host ""
Write-Host "=== AscendC CodeMap OpenCode refresh ==="
Write-Host "Repo: $BundleRoot"
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "Prerequisite: OpenCode must be fully exited (MCP is loaded at process start)."
$oc = @(Get-CimInstance Win32_Process -Filter "Name = 'opencode.exe'" -ErrorAction SilentlyContinue)
if ($oc.Count -gt 0) {
  Write-Host ("NOTE: {0} opencode.exe still running; stopping leftover serve processes so files can be replaced." -f $oc.Count)
  foreach ($p in $oc) {
    $cmd = [string]$p.CommandLine
    if ($cmd -match 'serve') {
      Write-Host ("  stopping PID {0} {1}" -f $p.ProcessId, $cmd.Trim())
      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
  }
  Start-Sleep -Milliseconds 400
}
Write-Host ""

if ($WhatIf) {
  Write-Host "[WhatIf] Would run:"
  Write-Host "  1. uninstall.ps1 opencode"
  Write-Host "  2. install.ps1 opencode"
  Write-Host ("  pip: " + $(if ($doPip) { "reinstall" } else { "skip" }))
  Write-Host "  3. Verify MCP entry + CodeMap skills in OpenCode home"
  exit 0
}

if (-not (Test-Path -LiteralPath $InstallPs1)) {
  throw "Missing install.ps1 at $InstallPs1"
}
if (-not (Test-Path -LiteralPath $UninstallPs1)) {
  throw "Missing uninstall.ps1 at $UninstallPs1"
}

$envExtra = @{}
if (-not $doPip) {
  $envExtra["SKIP_PIP"] = "1"
}

# --- 1) Uninstall (always) ---
Write-Host "[1/3] Uninstall OpenCode CodeMap bits..."
Invoke-RepoScript -Script $UninstallPs1 -Arg "opencode"

# --- 2) Reinstall ---
Write-Host ""
Write-Host "[2/3] Reinstall OpenCode CodeMap bits..."
if ($doPip) {
  Write-Host "  pip: FORCED reinstall"
} else {
  Write-Host "  pip: SKIPPED (editable package assumed current)"
}

Invoke-RepoScript -Script $InstallPs1 -Arg "opencode" -EnvExtra $envExtra

# --- 3) Verify this install matches THIS repo ---
Write-Host ""
Write-Host "[3/3] Verify install matches current repo..."

$pyCheck = @"
import pathlib, sys
from ascendc_codemap_mcp.constants import AGENTS_MARK_BEGIN, PRODUCT_NAME, SKILL_NAMES
from ascendc_codemap_mcp.install import opencode
from ascendc_codemap_mcp.install.jsonutil import read_json

root = pathlib.Path(r'''$BundleRoot''').resolve()
mod = pathlib.Path(__import__('ascendc_codemap_mcp').__file__).resolve()
print('MODULE=' + str(mod))
assert str(mod).lower().startswith(str(root).lower()), (mod, root)

path = opencode.config_path()
assert path.is_file(), path
entry = (read_json(path).get('mcp') or {}).get(PRODUCT_NAME)
assert isinstance(entry, dict), entry
blob = str(entry)
assert 'ascendc_codemap_mcp' in blob or 'ascendc-codemap-mcp' in blob, entry
print('CONFIG=' + str(path))

home = opencode.home()
for name in SKILL_NAMES:
    skill = home / 'skills' / f'ascendc-codemap-{name}' / 'SKILL.md'
    assert skill.is_file(), skill
agents = home / 'AGENTS.md'
text = agents.read_text(encoding='utf-8') if agents.is_file() else ''
assert AGENTS_MARK_BEGIN in text, agents
print('VERIFY_OK')
"@

$pyOut = & python -c $pyCheck 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host $pyOut
  throw "VERIFY FAIL: OpenCode MCP / skills check failed"
}
Assert-True ("$pyOut" -match "VERIFY_OK") "MCP entry + CodeMap skills installed"
Write-Host $pyOut

$sw.Stop()
Write-Host ""
Write-Host "=== Refresh complete ==="
Write-Host ("Elapsed     : {0:N1}s" -f $sw.Elapsed.TotalSeconds)
Write-Host "Next steps  :"
Write-Host "  1. Fully start OpenCode"
Write-Host "  2. Confirm MCP server ascendc-codemap-mcp is connected"
Write-Host "  3. Use skills ascendc-codemap-index-operator / query-codemap / update-operator"
Write-Host ""
