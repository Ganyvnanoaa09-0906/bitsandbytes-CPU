@echo off
chcp 65001 >nul
wsl -u root -e bash -lc "/root/venv3/bin/pip install --force-reinstall /mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz 2>&1 | grep -E 'Successfully installed' | head -1"
wsl -u root -e bash -lc "ls /root/venv3/bin/ | grep -iE 'bitsandbytes|bnb'"
wsl -u root -e bash -lc "/root/venv3/bin/bitsandbytes-cpu detect"
wsl -u root -e bash -lc "/root/venv3/bin/bnb-cpu selftest"
