@echo off
rem ===========================================================================
rem Run iGPU training. All arguments are passed to igpu_train.py.
rem
rem Examples:
rem   train.bat --model sshleifer/tiny-gpt2 --steps 5      (small downloadable smoke test)
rem   train.bat --model D:\models\qwen3.5-0.8B --steps 20
rem   train.bat --model Qwen/Qwen2-0.5B --steps 50
rem   train.bat --model D:\models\x --data corpus.txt --tokens 128 --steps 200 --save out\lora
rem   train.bat --backend cpu --model D:\models\x --steps 20   (no iGPU / fallback)
rem ===========================================================================
setlocal EnableExtensions
set "DIR=%~dp0"
cd /d "%DIR%"
if not defined CLBLAST_PATH set "CLBLAST_PATH=%DIR%clblast.dll"
set "NOPAUSE="
set "SHOWHELP="
for %%A in (%*) do (
    if /I "%%~A"=="--no-pause" set "NOPAUSE=1"
    if /I "%%~A"=="--help" set "SHOWHELP=1"
)
rem strip our own --no-pause before forwarding the rest to python
set "ARGS=%*"
set "ARGS=%ARGS:--no-pause=%"
if defined SHOWHELP goto :help

echo === CPU-forge iGPU trainer ===
echo CLBlast: %CLBLAST_PATH%
echo.
if exist "%DIR%.venv\Scripts\python.exe" (
    "%DIR%.venv\Scripts\python.exe" "%DIR%igpu_train.py" %ARGS%
    set "RC=%errorlevel%"
) else (
    python "%DIR%igpu_train.py" %ARGS%
    set "RC=%errorlevel%"
)
if not "%RC%"=="0" (
    echo.
    echo trainer exited with code %RC% - run doctor.bat to diagnose the environment.
    if not defined NOPAUSE pause
)
exit /b %RC%

:help
echo Usage: train.bat --model ^<local folder or HF repo id^> [options]
echo.
if exist "%DIR%.venv\Scripts\python.exe" (
    "%DIR%.venv\Scripts\python.exe" "%DIR%igpu_train.py" --help
) else (
    python "%DIR%igpu_train.py" --help
)
exit /b 0
