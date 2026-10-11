@echo off
chcp 65001 >nul
echo === which symbols force the high glibc floor? ===
wsl -u root -e bash -lc "cd /root/bnb_linux 2>/dev/null || exit 1; objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -E 'GLIBC_2\.(2[89]|3[0-9]|4[0-9])' | awk '{print $NF, $(NF-1)}' | sort -u"
echo.
echo === the same, grouped by required version ===
wsl -u root -e bash -lc "cd /root/bnb_linux; objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -o 'GLIBC_[0-9.]*' | sort -uV"
echo.
echo === does the C++ runtime contribute? ===
wsl -u root -e bash -lc "cd /root/bnb_linux; objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -oE 'GLIBCXX_[0-9.]*|CXXABI_[0-9.]*' | sort -uV | tail -6"
echo.
echo === what does libstdc++ itself drag in ===
wsl -u root -e bash -lc "ldd /root/bnb_linux/bitsandbytes/libbitsandbytes_cpu.so"
