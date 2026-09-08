@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =============================================================================
rem  CONFIG — easy to change
rem =============================================================================
set "PY312=C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"
set "PIPE=%~dp0pipeline_en_master.py"
rem Extra flags always passed to the pipeline (optional). Example:
rem   set "EXTRA_ARGS=--window 3 --langs en,es,zh"
set "EXTRA_ARGS="
rem Hugging Face mirror (needed in China / when huggingface.co is blocked)
set "HF_ENDPOINT=https://hf-mirror.com"
rem =============================================================================

rem  subtitelar_en_master.bat
rem  Double-click: asks for video/audio path (quotes OK).
rem  CLI: subtitelar_en_master.bat "C:\ruta\video.mp4" [--langs en,es] ...

title Subtitler - pipeline EN master + ES/ZH
cd /d "%~dp0"

if not exist "%PY312%" (
  echo [ERROR] Python 3.12 not found: %PY312%
  goto :end_fail
)
if not exist "%PIPE%" (
  echo [ERROR] Missing %PIPE%
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
echo  Pipeline EN master
echo  3x WhisperX -^> arbitro -^> resegment -^> NLLB
echo  Salida: [nombre] subs.en.srt + subs.es.srt
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
rem Strip ALL double quotes (handles pasted "path")
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

echo.
echo Entrada: !INPUT!
echo.

"%PY312%" "%PIPE%" "!INPUT!" !EXTRA_ARGS! !REST!
set "ERR=!ERRORLEVEL!"

echo.
if not "!ERR!"=="0" (
  echo [ERROR] Pipeline termino con codigo !ERR!
  goto :end_fail
)

echo Listo.
goto :end_ok

:end_fail
echo.
pause
exit /b 1

:end_ok
echo.
pause
exit /b 0
