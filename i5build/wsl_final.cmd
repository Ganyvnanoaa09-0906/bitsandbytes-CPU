@echo off
chcp 65001 >nul
set SDIST=/mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz

echo ======================================================================
echo  FINAL: the whole Linux path, from a clean venv, with all fixes in.
echo ======================================================================
wsl -u root -e bash -lc "rm -rf /root/venv3 && python3 -m venv /root/venv3 && /root/venv3/bin/pip install -q --upgrade pip 2>&1 | tail -1"

echo.
echo === 1. CPU torch (the README's first line) ===
wsl -u root -e bash -lc "/root/venv3/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2; /root/venv3/bin/pip list 2>/dev/null | grep -Ei '^torch '"

echo.
echo === 2. the fork: pip builds it, setup.py compiles the .so ===
wsl -u root -e bash -lc "/root/venv3/bin/pip install %SDIST% 2>&1 | grep -Ei 'build_linux|\.so|Successfully installed|error' | head -6"

echo.
echo === 3. nvidia packages pulled in ===
wsl -u root -e bash -lc "n=\$(/root/venv3/bin/pip list 2>/dev/null | grep -ci nvidia); echo \"  \$n\""

echo.
echo === 4. the glibc floor of what was just installed ===
wsl -u root -e bash -lc "objdump -T /root/venv3/lib/python3.14/site-packages/bitsandbytes/libbitsandbytes_cpu.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 | sed 's/^/  /'"

echo.
echo === 5. the user-facing check ===
wsl -u root -e bash -lc "cd /tmp && /root/venv3/bin/python3 /root/wsl_user_check.py"

echo.
echo === 6. python -m bitsandbytes ===
wsl -u root -e bash -lc "cd /tmp && /root/venv3/bin/python3 -m bitsandbytes 2>&1 | head -6"
