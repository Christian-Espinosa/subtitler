# Job: ensure distilgpt2 (+ tokenizer) available for arbitrate_en_srt.py
# Used with safe_downloader/watch_download.ps1 if HF is slow.
param(
  [string]$WorkDir = ""
)

$ErrorActionPreference = 'Continue'
$py = "C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"
$root = if ($WorkDir) { $WorkDir } else { Split-Path -Parent $PSScriptRoot }
$log = Join-Path $root "safe_download.log"

function Log([string]$m) {
  Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [distilgpt2] $m" -Encoding utf8
  Write-Host $m
}

# Prefer China HF mirror (huggingface.co often times out)
if (-not $env:HF_ENDPOINT) {
  $env:HF_ENDPOINT = "https://hf-mirror.com"
}

$indexes = @(
  "https://pypi.tuna.tsinghua.edu.cn/simple",
  "https://mirrors.aliyun.com/pypi/simple",
  "https://pypi.org/simple"
)

$okPip = $false
foreach ($idx in $indexes) {
  Log "pip ensure huggingface_hub via $idx"
  & $py -m pip install --default-timeout 1000 --retries 15 -i $idx --trusted-host ([uri]$idx).Host "huggingface_hub>=0.20"
  if ($LASTEXITCODE -eq 0) { $okPip = $true; break }
  Log "pip failed on $idx exit=$LASTEXITCODE"
}
if (-not $okPip) { exit 1 }

Log "Downloading distilgpt2 snapshot"
& $py -c @"
from huggingface_hub import snapshot_download
p = snapshot_download('distilgpt2')
print('cached', p)
"@
if ($LASTEXITCODE -ne 0) {
  Log "Official HF failed; retry hf-mirror.com"
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  & $py -c @"
from huggingface_hub import snapshot_download
p = snapshot_download('distilgpt2')
print('cached', p)
"@
}
if ($LASTEXITCODE -ne 0) { Log "distilgpt2 download FAILED"; exit $LASTEXITCODE }

Log "Smoke perplexity (local cache only)"
& $py -c @"
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root = Path.home() / '.cache/huggingface/hub/models--distilgpt2/snapshots'
snaps = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
m = str(next(s for s in snaps if (s/'config.json').exists())) if snaps else 'distilgpt2'
kw = {'local_files_only': True} if Path(m).is_dir() else {}
tok = AutoTokenizer.from_pretrained(m, **kw)
model = AutoModelForCausalLM.from_pretrained(m, **kw)
text = 'This is a coherent English sentence.'
enc = tok(text, return_tensors='pt')
out = model(**enc, labels=enc['input_ids'])
print('model', m)
print('ppl', torch.exp(out.loss).item())
print('OK')
"@
exit $LASTEXITCODE