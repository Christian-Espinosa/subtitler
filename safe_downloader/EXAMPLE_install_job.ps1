# Ejemplo de job idempotente para usar con watch_download.ps1
# Sustituye URLs / comandos por los de tu descarga real.
param(
  [string]$WorkDir = ""
)

$ErrorActionPreference = 'Continue'
if (-not $WorkDir) { $WorkDir = Split-Path -Parent $PSScriptRoot }
$dl = Join-Path $WorkDir ".download_cache"
New-Item -ItemType Directory -Force -Path $dl | Out-Null

function Log([string]$m) {
  $log = Join-Path $WorkDir "safe_download.log"
  Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [job] $m" -Encoding utf8
  Write-Host $m
}

# --- EJEMPLO: descargar un archivo grande con resume ---
# $url = "https://example.com/big.whl"
# $out = Join-Path $dl "big.whl"
# Log "curl resume $out"
# & curl.exe -L --retry 50 --retry-all-errors --retry-delay 5 --connect-timeout 60 -C - -o $out $url
# if ($LASTEXITCODE -ne 0) { Log "curl FAILED $LASTEXITCODE"; exit $LASTEXITCODE }

# --- EJEMPLO: pip con timeout alto ---
# $py = "C:\Path\To\Python312\python.exe"
# & $py -m pip install --default-timeout 1000 --retries 15 some-package
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "EXAMPLE job: no-op (edit this file for a real download)"
exit 0
