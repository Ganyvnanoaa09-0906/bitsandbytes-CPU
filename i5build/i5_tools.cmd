@echo off
echo === compilers on PATH ===
where cl 2>nul
where gcc 2>nul
where g++ 2>nul
where clang 2>nul
where cmake 2>nul
echo === Visual Studio ===
set VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe
if exist "%VSWHERE%" (
  "%VSWHERE%" -latest -property installationPath
  "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
) else (
  echo no vswhere
)
echo === any Visual Studio install directories ===
if exist "C:\Program Files\Microsoft Visual Studio" dir /b "C:\Program Files\Microsoft Visual Studio"
if exist "C:\Program Files (x86)\Microsoft Visual Studio" dir /b "C:\Program Files (x86)\Microsoft Visual Studio"
echo === scoop shims ===
if exist D:\scoop\shims dir /b D:\scoop\shims | findstr /i "gcc clang cmake ninja make"
