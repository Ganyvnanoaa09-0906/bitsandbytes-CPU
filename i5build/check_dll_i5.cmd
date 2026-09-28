@echo off
REM ============================================================
REM check_dll_i5.cmd -- confirm the DLL on the i5 is real and exports kernels
REM ------------------------------------------------------------
REM Why a wrapper: building a "call vcvars && dumpbin | find" chain inline over
REM ssh kept failing on quoting and `&` handling. A batch file has no such layers.
REM ============================================================
setlocal enabledelayedexpansion
set VCVARS=D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat
set DLL=C:\Users\GanYv\bnb_repo\bitsandbytes\libbitsandbytes_cpu.dll
set OUT=%TEMP%\dll_exports.txt

echo === DLL file ===
if not exist "%DLL%" (
    echo [ERROR] DLL not found: %DLL%
    exit /b 1
)
for %%F in ("%DLL%") do echo   size: %%~zF bytes   time: %%~tF

echo.
echo === exports ===
call "%VCVARS%" >nul 2>&1
dumpbin /nologo /exports "%DLL%" > "%OUT%" 2>&1
if not exist "%OUT%" (
    echo   [ERROR] dumpbin produced nothing
    exit /b 1
)
for %%K in (cquantize cdequantize cgemm cgemv coptimizer cgdn) do (
    set FOUND=0
    for /f "delims=" %%L in ('findstr /c:"%%K" "%OUT%"') do set FOUND=1
    if "!FOUND!"=="1" (echo   [OK] %%K found) else (echo   [--] %%K MISSING)
)

echo.
echo === total exported names ===
set N=0
for /f %%C in ('type "%OUT%" ^| find /c /v ""') do set N=%%C
echo   dumpbin output lines: !N!
del "%OUT%" >nul 2>&1
exit /b 0
