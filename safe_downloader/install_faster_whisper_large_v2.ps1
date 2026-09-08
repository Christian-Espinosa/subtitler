# Download Systran/faster-whisper-large-v2 via hf-mirror (WhisperX default for --model large-v2)
param(
  [string]$WorkDir = ""
)

$ErrorActionPreference = 'Continue'
$py = "C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"
$root = if ($WorkDir) { $WorkDir } else { Split-Path -Parent $PSScriptRoot }
$log = Join-Path $root "safe_download.log"
$modelId = "Systran/faster-whisper-large-v2"
$hubRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Systran--faster-whisper-large-v2"
$snapRoot = Join-Path $hubRoot "snapshots"

function Log([string]$m) {
  Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [whisper-lv2] $m" -Encoding utf8
  Write-Host $m
}

function Test-Ready {
  if (-not (Test-Path $snapRoot)) { return $false }
  foreach ($s in Get-ChildItem $snapRoot -Directory -EA SilentlyContinue) {
    $bin = Join-Path $s.FullName "model.bin"
    $cfg = Join-Path $s.FullName "config.json"
    if ((Test-Path $cfg) -and (Test-Path $bin) -and ((Get-Item $bin).Length -gt 1GB)) {
      return $true
    }
  }
  return $false
}

if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
Log "HF_ENDPOINT=$($env:HF_ENDPOINT)"

if (Test-Ready) {
  Log "faster-whisper-large-v2 already present"
  Log "=== DONE OK ==="
  exit 0
}

# Prefer hub snapshot_download via mirror
$dl = Join-Path $root "_fw_download.py"
@'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
p = snapshot_download("Systran/faster-whisper-large-v2")
print("cached", p)
'@ | Set-Content $dl -Encoding utf8

Log "snapshot_download $modelId"
& $py $dl
$snapOk = ($LASTEXITCODE -eq 0)

if (-not (Test-Ready)) {
  Log "fallback curl from hf-mirror"
  New-Item -ItemType Directory -Force -Path $snapRoot | Out-Null
  $snap = Get-ChildItem $snapRoot -Directory -EA SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($snap) { $snapPath = $snap.FullName } else {
    $snapPath = Join-Path $snapRoot "main-local"
    New-Item -ItemType Directory -Force -Path $snapPath | Out-Null
  }
  $base = "https://hf-mirror.com/$modelId/resolve/main"
  # Typical CTranslate2 layout
  foreach ($f in @(
    "config.json",
    "tokenizer.json",
    "vocabulary.txt",
    "preprocessor_config.json",
    "model.bin"
  )) {
    $dest = Join-Path $snapPath $f
    Log "curl -C - $f"
    & curl.exe -L --retry 40 --retry-delay 8 -C - --connect-timeout 60 --speed-time 180 --speed-limit 500 -o $dest "$base/$f"
    $sz = 0
    if (Test-Path $dest) { $sz = [math]::Round((Get-Item $dest).Length / 1MB, 1) }
    Log "curl $f exit=$LASTEXITCODE sizeMB=$sz"
  }
}

Remove-Item $dl -Force -EA SilentlyContinue

if (-not (Test-Ready)) {
  Log "model NOT ready"
  exit 1
}
Log "=== DONE OK ==="
exit 0
