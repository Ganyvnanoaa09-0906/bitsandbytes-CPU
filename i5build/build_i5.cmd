@echo off
REM ============================================================
REM build_i5.cmd -- build the CPU DLL on the i5 with its own vcvars
REM ------------------------------------------------------------
REM Points at D:\vs2022bt (the BuildTools install that actually has a Windows
REM SDK), NOT D:\vs (the pre-existing VS tree on this box, which has no SDK and
REM therefore cannot compile anything).
REM
REM Passes its argument straight through, so:
REM   build_i5.cmd            -> auto-detect vendor
REM   build_i5.cmd amd        -> force /favor:AMD64
REM   build_i5.cmd intel      -> force /favor:INTEL64
REM ============================================================
setlocal
set VCVARS=D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat
set REPO=C:\Users\GanYv\bnb_repo

if not exist "%VCVARS%" (
    echo [ERROR] vcvars not found: %VCVARS%
    exit /b 2
)
if not exist "%REPO%\csrc\cpu_ops.cpp" (
    echo [ERROR] repo not found at %REPO%
    exit /b 2
)

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed
    exit /b 2
)
echo [env] VCToolsInstallDir=%VCToolsInstallDir%
echo [env] WindowsSdkDir=%WindowsSdkDir%

cd /d "%REPO%"
echo.
echo [run] build_manual\build_manual.bat %*
echo ----------------------------------------
call build_manual\build_manual.bat %*
set RC=%errorlevel%
echo ----------------------------------------
echo [done] build_manual.bat exit code = %RC%
exit /b %RC%
