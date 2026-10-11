@echo off
chcp 65001 >nul
echo === install the toolchain (needs root; sudo would prompt) ===
wsl -u root -e bash -lc "apt-get update -qq 2>&1 | tail -2; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ libomp-dev 2>&1 | tail -3"
echo === versions ===
wsl -u root -e bash -lc "g++ --version | head -1; ls /usr/lib/gcc/x86_64-linux-gnu/*/libgomp.so"
echo === unpack the source ===
wsl -u root -e bash -lc "rm -rf /root/bnb_linux && mkdir -p /root/bnb_linux && cd /root/bnb_linux && python3 -m zipfile -e /mnt/d/work/bitsandbytes-CPU/i5build/bnb_linux_src2.zip . && ls"
echo === build + selftest ===
wsl -u root -e bash -lc "cd /root/bnb_linux && bash build_linux.sh --selftest 2>&1 | tail -30"
echo === load check ===
wsl -u root -e bash -lc "cd /root/bnb_linux && python3 linux_check.py"
echo === what glibc does THIS build need ===
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3"
