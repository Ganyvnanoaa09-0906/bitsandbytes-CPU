@echo off
REM ============================================================================
REM accept_i5.cmd -- ONE uninterrupted Intel acceptance run, from a clean build
REM                  to the 1000-step training verdict.
REM
REM Run this on the i5 (DESKTOP-2J5D1P7) from C:\Users\GanYv\i5build:
REM     accept_i5.cmd > accept_i5.log 2>&1
REM
REM Everything it prints goes into that one log, so the whole chain (clean build
REM -> DLL check -> C selftest -> python suite -> 1000-step training -> cross-
REM architecture verdict) is a single artefact instead of a scatter of session
REM transcripts.
REM
REM Pure ASCII on purpose: cmd decodes .cmd with the system ANSI codepage, and
REM non-ASCII bytes break the parser rather than merely garbling.
REM
REM THE i5 LAYOUT, read off a directory listing after three failed runs guessed
REM at it. There is exactly ONE checkout root here, and the package is a
REM subdirectory of it -- unlike the R5, where the checkout root and the package
REM directory have the same name:
REM
REM   checkout root  C:\Users\GanYv\bnb_repo
REM                  bitsandbytes\  <- the package (__init__.py, gdn_cpu.py,
REM                                   optim\, nn\, cextension.py)
REM                  bitsandbytes\libbitsandbytes_cpu.dll  <- build output
REM                  csrc\ build_manual\ ...              <- build inputs
REM
REM   There is NO bnb_repo\bitsandbytes\bitsandbytes. Running python from
REM   bnb_repo\bitsandbytes fails `import bitsandbytes`, because that directory
REM   IS the package and `import bitsandbytes` looks for a `bitsandbytes` entry
REM   on sys.path. That mistake cost a full acceptance run: 3 tests failed with
REM   ModuleNotFoundError while the DLL and kernels were perfectly fine.
REM
REM Optional overrides, so the same chain can be dry-run elsewhere:
REM     accept_i5.cmd [BUILD_ROOT] [PY_ROOT] [SCRIPTS] [STEPS] [LABEL]
REM ============================================================================
setlocal
set REPO=%~1
if "%REPO%"=="" set REPO=C:\Users\GanYv\bnb_repo
set PYROOT=%~2
if "%PYROOT%"=="" set PYROOT=C:\Users\GanYv\bnb_repo
set HERE=%~3
if "%HERE%"=="" set HERE=C:\Users\GanYv\i5build
set STEPS=%~4
if "%STEPS%"=="" set STEPS=1000
set LABEL=%~5
if "%LABEL%"=="" set LABEL=train1000_i5.json

set DLL=%REPO%\bitsandbytes\libbitsandbytes_cpu.dll
set STAGEERR=0

REM ---------------------------------------------------------------------------
REM Force UTF-8 for every python process in this chain.
REM
REM Without this, a redirected cp936 (GBK) console cannot encode the check marks
REM that the test scripts print, and the tests die with UnicodeEncodeError
REM exactly when they have a FAILURE to report -- so failures could not be
REM reported at all, and a run that verified nothing looked green. Measured
REM twice on this box. Individual scripts also guard themselves, but the
REM environment variable is the one place that covers all of them.
REM ---------------------------------------------------------------------------
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ############################################################################
echo # CPU-forge Intel acceptance run
echo #   build root  = %REPO%
echo #   python root = %PYROOT%
echo #   scripts     = %HERE%
echo #   steps       = %STEPS%
echo #   dll         = %DLL%
echo #   started     = %DATE% %TIME%
echo ############################################################################

echo.
echo ############################################################################
echo # STAGE 0 - environment
echo ############################################################################
hostname
powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name; 'cores=' + (Get-CimInstance Win32_Processor).NumberOfCores + ' threads=' + (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"
python -c "import sys, platform; print('python', sys.version.split()[0], platform.machine())"
python -c "import torch; print('torch', torch.__version__)"

echo.
echo ############################################################################
echo # STAGE 1 - clean rebuild (vendor auto-detect)
echo ############################################################################
if exist "%DLL%" (
    echo   removing old DLL
    del /q "%DLL%"
)
call "%HERE%\build_i5.cmd"
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed
if not exist "%DLL%" (
    echo [FAIL] build reported success but produced no DLL at %DLL%
    set STAGEERR=9
    goto failed
)
echo   [OK] DLL rebuilt: %DLL%

echo.
echo ############################################################################
echo # STAGE 2 - DLL present, exports the kernels
echo ############################################################################
call "%HERE%\check_dll_i5.cmd"
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed

echo.
echo ############################################################################
echo # STAGE 3 - C selftest (5 checks, incl. vector-vs-scalar bit identity)
echo ############################################################################
call "%HERE%\selftest_i5.cmd"
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed

echo.
echo ############################################################################
echo # STAGE 4 - python regression suite
echo ############################################################################
REM run_all_tests.py locates its child scripts relative to its OWN directory, and
REM the test scripts live next to the package, so run it from where it actually
REM is (PYROOT when what is deployed is just the package).
cd /d "%PYROOT%"
python run_all_tests.py
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed

echo.
echo ############################################################################
echo # STAGE 5 - %STEPS%-step training acceptance run
echo ############################################################################
REM train_1000_steps.py lives in the CHECKOUT ROOT, one level up from the
REM package, so switch back. Forgetting this made the run fail here with
REM "can't open file ...\\bitsandbytes\\train_1000_steps.py".
cd /d "%REPO%"
python train_1000_steps.py --steps %STEPS% --out "%HERE%\%LABEL%"
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed

echo.
echo ############################################################################
echo # STAGE 6 - cross-architecture trajectory verdict
echo ############################################################################
cd /d "%HERE%"
python compare_trajectories.py
set STAGEERR=%errorlevel%
if not "%STAGEERR%"=="0" goto failed

echo.
echo ############################################################################
echo # ACCEPTANCE PASSED - every stage above succeeded
echo #   finished = %DATE% %TIME%
echo ############################################################################
endlocal
exit /b 0

:failed
echo.
echo ############################################################################
echo # ACCEPTANCE FAILED - stage exit code %STAGEERR%
echo #   finished = %DATE% %TIME%
echo ############################################################################
endlocal
exit /b 1
