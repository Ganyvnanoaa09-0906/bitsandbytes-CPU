@echo off
chcp 65001 >nul
echo === rebuild in WSL with the two portability fixes ===
wsl -u root -e bash -lc "rm -rf /root/bnb_linux && mkdir -p /root/bnb_linux && cd /root/bnb_linux && python3 -m zipfile -e /mnt/d/work/bitsandbytes-CPU/i5build/bnb_linux_src2.zip . && bash build_linux.sh --selftest 2>&1 | tail -14"

echo.
echo === the glibc floor now ===
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3"

echo.
echo === every symbol still above 2.30, with the version it needs ===
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -E 'GLIBC_2\.(3[0-9]|4[0-9])' | sed 's/.*GLIBC/GLIBC/' | sort -u"

echo.
echo === did the __isoc23_ symbols go away ===
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -c isoc23 || echo '0  (gone)'"

echo.
echo === library still loads and self-tests ===
wsl -u root -e bash -lc "cd /root/bnb_linux && python3 linux_check.py | tail -4"
