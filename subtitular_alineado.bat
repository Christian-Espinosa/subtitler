@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem CONFIG
rem ============================================================
rem Python and mirrors: _python.bat finds Python 3.12; set PY312 / HF_ENDPOINT in
rem subtitler.local.bat (see subtitler.local.example.bat) if you need to.
call "%~dp0_python.bat"
set "PIPE=%~dp0pipeline_aligned.py"
set "EXTRA_ARGS="
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
