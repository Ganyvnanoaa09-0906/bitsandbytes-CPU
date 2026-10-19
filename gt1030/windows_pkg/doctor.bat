@echo off
rem Environment doctor: Python / torch / pyopencl / CLBlast / OpenCL devices.
rem Usage: doctor.bat [--bench]
setlocal EnableExtensions
set "DIR=%~dp0"
cd /d "%DIR%"
if not defined CLBLAST_PATH set "CLBLAST_PATH=%DIR%clblast.dll"
if exist "%DIR%.venv\Scripts\python.exe" (
    "%DIR%.venv\Scripts\python.exe" "%DIR%check_env.py" %*
    exit /b %errorlevel%
)
python "%DIR%check_env.py" %*
exit /b %errorlevel%
