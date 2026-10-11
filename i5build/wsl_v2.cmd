@echo off
chcp 65001 >nul
wsl -u root -e bash -lc "/root/venv3/bin/pip install --force-reinstall /mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz 2>&1 | grep -Ei 'Successfully installed' | head -2"
echo.
echo === detect on Linux, now with psutil ===
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu detect"
echo.
echo === the three moved modules import ===
wsl -u root -e bash -lc "cd /tmp && /root/venv3/bin/python3 -c 'import bitsandbytes.detect_cpu, bitsandbytes.torch_cpu_kit, bitsandbytes.disk_balancer; print(\"  all three import OK\"); print(\"  physical cores:\", bitsandbytes.torch_cpu_kit.physical_cores())'"
