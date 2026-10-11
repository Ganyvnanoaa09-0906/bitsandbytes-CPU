@echo off
chcp 65001 >nul
echo ==================== wsl --status ====================
wsl --status
echo.
echo ==================== wsl --version ====================
wsl --version
echo.
echo ==================== is this an elevated shell? ====================
net session >nul 2>&1 && echo ADMIN || echo NOT ADMIN
echo.
echo ==================== WSL services ====================
sc query LxssManager 2>&1 | findstr /i "SERVICE_NAME STATE"
sc query WslService 2>&1 | findstr /i "SERVICE_NAME STATE"
sc query vmcompute 2>&1 | findstr /i "SERVICE_NAME STATE"
echo.
echo ==================== where is WSL installed ====================
if exist "C:\Windows\System32\lxss\tools\lxss.sys" (echo WSL1 driver: present) else (echo WSL1 driver: MISSING)
if exist "C:\Program Files\WSL\wsl.exe" (echo Store WSL: present) else (echo Store WSL: not in Program Files)
dir /b "C:\Windows\System32\wsl.exe" 2>nul
dir /b "C:\Windows\System32\wslapi.dll" 2>nul
echo.
echo ==================== installed distributions (registry) ====================
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss" /s 2>nul | findstr /i "DistributionName BasePath State" 
echo.
echo ==================== optional features (needs admin; may be denied) ====================
dism /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux 2>&1 | findstr /i "State Feature Name"
dism /online /get-featureinfo /featurename:VirtualMachinePlatform 2>&1 | findstr /i "State Feature Name"
echo.
echo ==================== virtualization support ====================
systeminfo 2>nul | findstr /i "Hyper-V Virtualization"
