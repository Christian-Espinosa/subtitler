@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =============================================================================
rem  CONFIG — easy to change
rem =============================================================================
set "PY312=C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"
set "WHISPER_MODEL=large-v2"
set "BATCH_SIZE=8"
set "COMPUTE_TYPE=float16"
rem Optional default language: es / en / zh  (empty = auto-detect)
set "DEFAULT_LANGUAGE="
rem Hugging Face mirror (needed when huggingface.co is blocked)
set "HF_ENDPOINT=https://hf-mirror.com"
rem =============================================================================

rem  subtitelar.bat — single WhisperX pass → "[name] subs.srt"
rem  Double-click: asks for path (quotes OK).

title Subtitler - modo simple
cd /d "%~dp0"

if not exist "%PY312%" (
  echo [ERROR] No se encontro Python 3.12 en:
  echo   %PY312%
  goto :end_fail
)

set "RAW="
set "REST="

if "%~1"=="" goto :ask_path
set "RAW=%~1"
shift
goto :gather_rest

:ask_path
echo.
echo ========================================
echo  Subtitular ^(1 pasada WhisperX^)
echo ========================================
echo.
echo Pega la ruta del video o audio y pulsa Enter.
echo Puedes incluir comillas: "D:\ruta\video.mp4"
echo.
set /p "RAW=Ruta: "
goto :after_path

:gather_rest
if "%~1"=="" goto :after_path
set "REST=!REST! %1"
shift
goto :gather_rest

:after_path
if defined RAW set "INPUT=!RAW:"=!"
for /f "tokens=* delims= " %%A in ("!INPUT!") do set "INPUT=%%A"

if not defined INPUT (
  echo [ERROR] No se indico ninguna ruta.
  goto :end_fail
)

if not exist "!INPUT!" (
  echo [ERROR] No existe el archivo:
  echo   !INPUT!
  echo.
  echo Comprueba la ruta. Las comillas se eliminan solas.
  goto :end_fail
)

set "SUBTITLER_INPUT=!INPUT!"
set "IN_DIR="
set "IN_NAME="
for /f "delims=" %%I in ('"%PY312%" -c "from pathlib import Path; import os; p=Path(os.environ['SUBTITLER_INPUT']); print(p.parent); print(p.stem)"') do (
  if not defined IN_DIR (
    set "IN_DIR=%%I"
  ) else (
    set "IN_NAME=%%I"
  )
)

set "TMP_SRT=!IN_DIR!\!IN_NAME!.srt"
set "OUT_SRT=!IN_DIR!\!IN_NAME! subs.srt"

set "LANG_ARGS="
if defined DEFAULT_LANGUAGE set "LANG_ARGS=--language %DEFAULT_LANGUAGE%"

echo ========================================
echo  Subtitulando con WhisperX
echo  Entrada : !INPUT!
echo  Salida  : !OUT_SRT!
echo ========================================
echo.

"%PY312%" -m whisperx "!INPUT!" ^
  --model %WHISPER_MODEL% ^
  --output_dir "!IN_DIR!" ^
  --output_format srt ^
  --compute_type %COMPUTE_TYPE% ^
  --batch_size %BATCH_SIZE% ^
  --print_progress True ^
  %LANG_ARGS% ^
  !REST!

if errorlevel 1 (
  echo.
  echo [ERROR] WhisperX fallo. Revisa el mensaje anterior.
  goto :end_fail
)

if not exist "!TMP_SRT!" (
  echo [ERROR] No se genero el SRT esperado:
  echo   !TMP_SRT!
  goto :end_fail
)

if exist "!OUT_SRT!" del /f /q "!OUT_SRT!"
move /y "!TMP_SRT!" "!OUT_SRT!" >nul
if errorlevel 1 (
  echo [ERROR] No se pudo renombrar a:
  echo   !OUT_SRT!
  goto :end_fail
)

echo.
echo Listo:
echo   !OUT_SRT!
goto :end_ok

:end_fail
echo.
pause
exit /b 1

:end_ok
echo.
pause
exit /b 0
