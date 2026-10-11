@echo off
echo === WSL distributions ===
wsl -l -v
echo === inside the default distribution ===
wsl -e bash -lc "uname -a; echo '---'; cat /etc/os-release | head -3; echo '--- tools ---'; for t in g++ clang++ cmake make; do printf '%-10s ' $t; command -v $t || echo MISSING; done; echo '--- openmp ---'; ls /usr/lib/gcc/x86_64-linux-gnu/*/libgomp.so 2>/dev/null || echo 'libgomp not at the usual path'; echo '--- python ---'; python3 --version 2>/dev/null || echo 'no python3'; echo '--- glibc ---'; ldd --version | head -1"
