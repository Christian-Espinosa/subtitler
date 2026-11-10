# Job: ensure NLLB-200 distilled 600M (pytorch_model.bin) for translate_srt.py
param(
  [string]$WorkDir = ""
)

$ErrorActionPreference = 'Continue'
$py = if ($env:PY312) { $env:PY312 } else { (& py -3.12 -c "import sys;print(sys.executable)") }
$root = if ($WorkDir) { $WorkDir } else { Split-Path -Parent $PSScriptRoot }
$log = Join-Path $root "safe_download.log"
$modelId = "facebook/nllb-200-distilled-600M"
$hubRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--facebook--nllb-200-distilled-600M"
$snapRoot = Join-Path $hubRoot "snapshots"

function Log([string]$m) {
  Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [nllb] $m" -Encoding utf8
  Write-Host $m
}

function Get-SnapPath {
  if (-not (Test-Path $snapRoot)) { return $null }
  $snaps = @(Get-ChildItem $snapRoot -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
  if ($snaps.Count -eq 0) { return $null }
  return $snaps[0].FullName
}

function Test-ModelReady {
  $snap = Get-SnapPath
  if (-not $snap) { return $false }
  $cfg = Join-Path $snap "config.json"
  $bin = Join-Path $snap "pytorch_model.bin"
  $tok = Join-Path $snap "tokenizer.json"
  $spm = Join-Path $snap "sentencepiece.bpe.model"
  if (-not (Test-Path $cfg)) { return $false }
  if (-not ((Test-Path $tok) -or (Test-Path $spm))) { return $false }
  if (-not (Test-Path $bin)) { return $false }
  return ((Get-Item $bin).Length -gt 1GB)
}

if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
Log "HF_ENDPOINT=$($env:HF_ENDPOINT)"

foreach ($pkg in @("sentencepiece", "accelerate", "protobuf")) {
  Log "pip ensure $pkg"
  & $py -m pip install --default-timeout 1000 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple $pkg
  if ($LASTEXITCODE -ne 0) {
    & $py -m pip install --default-timeout 1000 --retries 10 $pkg
  }
  if ($LASTEXITCODE -ne 0) { Log "FAILED $pkg"; exit 1 }
}

if (Test-ModelReady) {
  Log "Model already present"
} else {
  New-Item -ItemType Directory -Force -Path $snapRoot | Out-Null
  $snap = Get-SnapPath
  if (-not $snap) {
    $snap = Join-Path $snapRoot "main-local"
    New-Item -ItemType Directory -Force -Path $snap | Out-Null
  }
  $base = "https://hf-mirror.com/$modelId/resolve/main"
  foreach ($f in @(
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin"
  )) {
    $dest = Join-Path $snap $f
    Log "curl -C - $f"
    & curl.exe -L --retry 40 --retry-delay 8 -C - --connect-timeout 60 --speed-time 180 --speed-limit 500 -o $dest "$base/$f"
    $sz = 0
    if (Test-Path $dest) { $sz = [math]::Round((Get-Item $dest).Length / 1MB, 1) }
    Log "curl $f exit=$LASTEXITCODE sizeMB=$sz"
  }
}

if (-not (Test-ModelReady)) {
  Log "NLLB not ready"
  exit 1
}

$smoke = Join-Path $root "_nllb_smoke.py"
$snap = Get-SnapPath
@'
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
m = sys.argv[1]
tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(m, local_files_only=True)
tok.src_lang = "eng_Latn"
enc = tok(["Hello, how are you?"], return_tensors="pt", padding=True)
bos = tok.convert_tokens_to_ids("spa_Latn")
gen = model.generate(**enc, forced_bos_token_id=bos, max_length=64)
print(tok.batch_decode(gen, skip_special_tokens=True))
print("OK")
'@ | Set-Content -Path $smoke -Encoding utf8

Log "Smoke translate eng->spa"
& $py $smoke $snap
$rc = $LASTEXITCODE
Remove-Item $smoke -Force -ErrorAction SilentlyContinue
if ($rc -ne 0) { Log "smoke FAILED"; exit $rc }
Log "=== DONE OK ==="
exit 0
