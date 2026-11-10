@echo off
rem Sets PY312 to the full path of the Python that runs the subtitler, with no path to
rem a particular PC in the repo. Order: PY312 from subtitler.local.bat > .venv next to
rem this file > "py -3.12" > the first python on PATH. Called by the other .bat files.
if exist "%~dp0subtitler.local.bat" call "%~dp0subtitler.local.bat"
if defined PY312 if exist "%PY312%" goto :eof
set "PY312="
if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY312=%~dp0.venv\Scripts\python.exe"
  goto :eof
)
for /f "delims=" %%P in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PY312=%%P"
if defined PY312 goto :eof
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY312 set "PY312=%%P"
goto :eof
