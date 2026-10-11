@echo off
chcp 65001 >nul
echo ======================================================================
echo  What a Linux user gets TODAY from PyPI, without any of the fixes.
echo  Installing the published sdist into the venv that already has torch.
echo ======================================================================
wsl -u root -e bash -lc "cd /tmp && /root/venv2/bin/pip install --force-reinstall --no-deps 'bitsandbytes-cpu-fork' 2>&1 | grep -Ei 'Downloading|Building|Created wheel|Successfully|error' | head -8"

echo.
echo === what landed in site-packages ===
wsl -u root -e bash -lc "ls -la /root/venv2/lib/python3.14/site-packages/bitsandbytes/ | grep -Ei 'libbitsandbytes|total'"

echo.
echo === does it import? ===
wsl -u root -e bash -lc "cd /tmp && /root/venv2/bin/python3 -c 'import bitsandbytes as b; print(\"imported\", b.__version__)' 2>&1 | tail -12"

echo.
echo === and the check a user would run ===
wsl -u root -e bash -lc "cd /tmp && /root/venv2/bin/python3 /root/wsl_user_check.py 2>&1 | tail -8"
