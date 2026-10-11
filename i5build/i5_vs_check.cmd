@echo off
set VS=D:\vs2022bt
echo === vcvars present? ===
if exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" (echo YES vcvars64.bat) else (echo NO vcvars64.bat)
if exist "%VS%\VC\Auxiliary\Build\vcvarsall.bat" (echo YES vcvarsall.bat) else (echo NO vcvarsall.bat)
echo === cl.exe anywhere under the VS dir ===
dir /s /b "%VS%\cl.exe" 2>nul
echo === VC Tools versions ===
if exist "%VS%\VC\Tools\MSVC" dir /b "%VS%\VC\Tools\MSVC"
echo === Windows SDK ===
if exist "C:\Program Files (x86)\Windows Kits\10\Include" dir /b "C:\Program Files (x86)\Windows Kits\10\Include"
echo === try compiling a hello world with vcvars ===
if exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" (
  call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
  echo --- after vcvars ---
  where cl
  echo int main(void){return 0;} > "%TEMP%\hw.c"
  cl /nologo /O2 "%TEMP%\hw.c" /Fe:"%TEMP%\hw.exe" /Fo:"%TEMP%\hw.obj"
  echo cl_exit=%ERRORLEVEL%
  if exist "%TEMP%\hw.exe" echo BUILD OK
)
