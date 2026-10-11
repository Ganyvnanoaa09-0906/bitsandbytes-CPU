@echo off
chcp 65001 >nul
set SDIST=/mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz

echo ======================================================================
echo  The user's scenario: pip install, then type the command.
echo ======================================================================
wsl -u root -e bash -lc "/root/venv3/bin/pip install --force-reinstall --no-deps %SDIST% 2>&1 | grep -Ei 'build_linux|Successfully installed' | head -3"

echo.
echo === is the executable there ===
wsl -u root -e bash -lc "ls -la /root/venv3/bin/ | grep -i bitsandbytes"

echo.
echo === \$ bitsandbytes-cpu version ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu version"

echo.
echo === \$ bitsandbytes-cpu detect ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu detect"

echo.
echo === \$ bitsandbytes-cpu doctor ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu doctor"

echo.
echo === \$ bitsandbytes-cpu selftest ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu selftest"

echo.
echo === \$ bitsandbytes-cpu help (first 40 lines) ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu help | head -40"

echo.
echo === the moved tooling imports on Linux ===
wsl -u root -e bash -lc "cd /tmp && /root/venv3/bin/python3 -c \"
import bitsandbytes.detect_cpu, bitsandbytes.torch_cpu_kit, bitsandbytes.disk_balancer
print('  all three CPU tooling modules import OK')
print('  physical cores via torch_cpu_kit:', bitsandbytes.torch_cpu_kit.physical_cores())
\""
