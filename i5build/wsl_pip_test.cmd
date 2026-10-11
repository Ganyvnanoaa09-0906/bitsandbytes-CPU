@echo off
chcp 65001 >nul
set SDIST=/mnt/d/work/bitsandbytes-CPU/bitsandbytes/sdist_out/bitsandbytes_cpu_fork-0.50.2.dev0.tar.gz

echo === prerequisites for a venv on this distribution ===
wsl -u root -e bash -lc "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip 2>&1 | tail -2"
wsl -u root -e bash -lc "rm -rf /root/venv && python3 -m venv /root/venv && /root/venv/bin/pip install -q --upgrade pip 2>&1 | tail -2; /root/venv/bin/pip --version"

echo.
echo === THE TEST: pip install the sdist on Linux ===
echo     (setup.py should run build_linux.sh by itself; before today it produced a
echo      wheel containing only a Windows .dll)
wsl -u root -e bash -lc "/root/venv/bin/pip install -v %SDIST% 2>&1 | grep -Ei 'build_linux|libbitsandbytes|\.so|Running|error|Successfully|note:' | tail -25"

echo.
echo === what actually landed in site-packages ===
wsl -u root -e bash -lc "ls -la /root/venv/lib/python*/site-packages/bitsandbytes/ | grep -Ei 'libbitsandbytes|^total|\.so|\.dll'"

echo.
echo === does the installed package find and load its library ===
wsl -u root -e bash -lc "cd /tmp && /root/venv/bin/python3 -c \"
import bitsandbytes, os, ctypes
p = os.path.join(os.path.dirname(bitsandbytes.__file__), 'libbitsandbytes_cpu.so')
print('  package :', bitsandbytes.__file__)
print('  version :', bitsandbytes.__version__)
print('  library :', p, os.path.isfile(p))
lib = ctypes.CDLL(p)
print('  loaded  : OK')
for n in ('cgemv_4bit_inference_cpu_fp32','coptimizer_update_8bit_blockwise_cpu'):
    print('   ', n, hasattr(lib, n))
\""
