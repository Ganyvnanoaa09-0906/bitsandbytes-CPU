@echo off
chcp 65001 >nul
set SDIST=/mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz

echo === fresh venv ===
wsl -u root -e bash -lc "rm -rf /root/venv2 && python3 -m venv /root/venv2 && /root/venv2/bin/pip install -q --upgrade pip 2>&1 | tail -1"

echo.
echo === install CPU torch (the README's first line; ~200 MB, not the 2.5 GB CUDA stack) ===
wsl -u root -e bash -lc "/root/venv2/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3; /root/venv2/bin/pip list 2>/dev/null | grep -Ei '^torch|nvidia' | head -8"

echo.
echo === install the fork (setup.py should build the .so itself) ===
wsl -u root -e bash -lc "/root/venv2/bin/pip install --quiet %SDIST% 2>&1 | tail -3; echo pip_exit=$?"

echo.
echo === did any nvidia package come along this time? ===
wsl -u root -e bash -lc "/root/venv2/bin/pip list 2>/dev/null | grep -ci nvidia || echo 0"

echo.
echo === the check a user would run ===
wsl -u root -e bash -lc "cp /mnt/d/work/bitsandbytes-CPU/i5build/wsl_user_check.py /root/ && cd /tmp && /root/venv2/bin/python3 /root/wsl_user_check.py"

echo.
echo === 'python -m bitsandbytes' (the command the README tells users to run) ===
wsl -u root -e bash -lc "cd /tmp && /root/venv2/bin/python3 -m bitsandbytes 2>&1 | head -12"
