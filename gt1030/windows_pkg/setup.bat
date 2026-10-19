@echo off
rem ===========================================================================
rem CPU-forge iGPU trainer - one-time setup (Windows)
rem   setup.bat          install into .venv next to this file (PyPI)
rem   setup.bat mirror   same, but use the Tsinghua PyPI mirror (China networks)
rem   setup.bat --check-only   only run doctor.bat with the current Python
rem ===========================================================================
setlocal EnableExtensions
set "DIR=%~dp0"
cd /d "%DIR%"

set "NOPAUSE="
set "USEMIRROR="
set "CHECKONLY="
for %%A in (%*) do (
    if /I "%%~A"=="--no-pause" set "NOPAUSE=1"
    if /I "%%~A"=="mirror" set "USEMIRROR=1"
    if /I "%%~A"=="--check-only" set "CHECKONLY=1"
)
if defined CHECKONLY goto :check_only

echo === CPU-forge iGPU trainer setup ===
echo.
echo [1/3] finding Python 3.10+ ...
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; assert sys.version_info >= (3,10); assert sys.maxsize > 2**32" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    python -c "import sys; assert sys.version_info >= (3,10); assert sys.maxsize > 2**32" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo   !! 64-bit Python 3.10-3.12 was not found.
    echo   Install it from https://www.python.org/downloads/windows/ and tick
    echo   "Add python.exe to PATH", then run setup.bat again.
    goto :fail
)
echo   using: %PY%

echo.
echo [2/3] creating .venv and installing packages (torch is the big one) ...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
set "PIPARGS="
if defined USEMIRROR set "PIPARGS=-i https://pypi.tuna.tsinghua.edu.cn/simple"
".venv\Scripts\python.exe" -m pip install %PIPARGS% torch numpy pyopencl transformers safetensors psutil
if errorlevel 1 goto :fail

echo.
echo [3/3] checking the environment ...
".venv\Scripts\python.exe" "check_env.py" --bench
if errorlevel 1 (
    echo.
    echo Setup finished, but the environment check did not pass.
    echo Read the messages above; you can still try:  train.bat --backend cpu ...
) else (
    echo.
    echo Setup complete. Next:
    echo   doctor.bat
    echo   train.bat --model ^<local model folder or HF repo id^> --steps 20
)
if not defined NOPAUSE pause
exit /b 0

:check_only
call "doctor.bat" %*
if not defined NOPAUSE pause
exit /b %errorlevel%

:fail
echo.
echo SETUP FAILED - fix the error above and run setup.bat again.
if not defined NOPAUSE pause
exit /b 1
