@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem CONFIG
rem ============================================================
set "PY312=C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"
set "PIPE=%~dp0pipeline_aligned.py"
set "EXTRA_ARGS="
set "HF_ENDPOINT=https://hf-mirror.com"
rem ============================================================

if not exist "%PY312%" (
  echo [ERROR] Python 3.12 no encontrado: %PY312%
  pause
  exit /b 1
)

set "MEDIA=%~1"
if "%MEDIA%"=="" (
  set /p "MEDIA=Arrastra el video aqui y pulsa Enter: "
)
rem quitar comillas si se pego desde el Explorador
set "MEDIA=%MEDIA:"=%"

if not exist "%MEDIA%" (
  echo [ERROR] no existe: %MEDIA%
  pause
  exit /b 1
)

shift
set "REST="
:collect
if "%~1"=="" goto run
set "REST=%REST% %1"
shift
goto collect

:run
"%PY312%" "%PIPE%" "%MEDIA%" %EXTRA_ARGS% %REST%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] el pipeline termino con codigo %RC%
pause
exit /b %RC%
