# Plantilla watchdog autónomo para descargas / installs largas.
# Default: revisa cada 5 minutos, sin prompts, con log + lock.
#
# Ejemplo:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\watch_download.ps1 `
#     -IntervalMinutes 5 `
#     -WorkDir "C:\proyecto" `
#     -InstallScript ".\EXAMPLE_install_job.ps1" `
#     -SuccessCommand 'C:\Python312\python.exe -c "import torch; assert torch.cuda.is_available()"'

param(
  [int]$IntervalMinutes = 5,
  [string]$WorkDir = "",
  [string]$LogName = "safe_download.log",
  [string]$LockName = ".safe_download.lock",
  [string]$InstallScript = "",
  [string]$SuccessCommand = "",
  [string]$ProgressGlob = "",  # optional: file glob whose growing size means progress (e.g. ".wheels\*.whl")
  [int]$StallTicksToRestart = 2
)

$ErrorActionPreference = 'Continue'

if (-not $WorkDir) { $WorkDir = (Get-Location).Path }
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$log = Join-Path $WorkDir $LogName
$lock = Join-Path $WorkDir $LockName
$intervalSec = [Math]::Max(60, $IntervalMinutes * 60)

function WLog([string]$msg) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

function Test-Success {
  if (-not $SuccessCommand) { return $false }
  cmd /c $SuccessCommand | Out-Null
  return ($LASTEXITCODE -eq 0)
}

function Get-ProgressBytes {
  if (-not $ProgressGlob) { return 0L }
  $sum = 0L
  Get-ChildItem -Path (Join-Path $WorkDir $ProgressGlob) -ErrorAction SilentlyContinue | ForEach-Object { $sum += $_.Length }
  return $sum
}

function Get-WorkerProcs {
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -match 'curl\.exe' -or
      $_.CommandLine -match 'pip install' -or
      ($InstallScript -and $_.CommandLine -match [regex]::Escape([IO.Path]::GetFileName($InstallScript)))
    ) -and $_.CommandLine -notmatch 'watch_download\.ps1'
  }
}

if (-not (Test-Path $WorkDir)) { New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null }
Add-Content -Path $log -Value "" -Encoding utf8
WLog "Watchdog start PID=$PID interval=${IntervalMinutes}m workdir=$WorkDir"

$script:lastBytes = Get-ProgressBytes
$script:stallTicks = 0

while ($true) {
  if (Test-Success) {
    WLog "Success probe OK"
    WLog "=== DONE OK ==="
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
    exit 0
  }

  $workers = @(Get-WorkerProcs)
  $bytes = Get-ProgressBytes
  if ($bytes -gt $script:lastBytes) { $script:stallTicks = 0 } else { $script:stallTicks++ }
  $script:lastSizeNote = $bytes
  $script:lastBytes = $bytes

  WLog ("tick workers={0} progressMB={1:N1} stallTicks={2}" -f $workers.Count, ($bytes/1MB), $script:stallTicks)

  $needRun = $false
  if ($workers.Count -eq 0) {
    WLog "No worker process"
    $needRun = $true
  } elseif ($ProgressGlob -and $script:stallTicks -ge $StallTicksToRestart) {
    WLog "Progress stalled ~$($script:stallTicks * $IntervalMinutes) min - restart workers"
    foreach ($w in $workers) {
      try { Stop-Process -Id $w.ProcessId -Force -ErrorAction Stop; WLog "Killed $($w.ProcessId)" } catch {}
    }
    Start-Sleep -Seconds 2
    $needRun = $true
  }

  if ($needRun) {
    if (Test-Path $lock) {
      $age = (Get-Date) - (Get-Item $lock).LastWriteTime
      if ($age.TotalMinutes -lt ($IntervalMinutes * 0.8)) {
        WLog "Lock fresh ($([int]$age.TotalSeconds)s) - skip spawn"
      } else {
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        $needRun = $true
      }
    }

    if ($needRun -and $InstallScript) {
      $installPath = if ([IO.Path]::IsPathRooted($InstallScript)) { $InstallScript } else { Join-Path $WorkDir $InstallScript }
      if (-not (Test-Path $installPath)) {
        WLog "ERROR InstallScript missing: $installPath"
      } else {
        Set-Content -Path $lock -Value $PID -Encoding utf8
        WLog "Launch install: $installPath"
        # Blocking run so this watchdog owns the attempt; on failure we sleep and retry
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installPath -WorkDir $WorkDir
        $code = $LASTEXITCODE
        WLog "InstallScript exit=$code"
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        if (Test-Success) {
          WLog "=== DONE OK ==="
          exit 0
        }
      }
    } elseif ($needRun -and -not $InstallScript) {
      WLog "WARNING: -InstallScript not set; watchdog only monitors"
    }
  }

  Start-Sleep -Seconds $intervalSec
}
