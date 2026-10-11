@echo off
chcp 65001 >nul
echo === distribution ===
wsl -e bash -lc "cat /etc/os-release | head -2; uname -r; ldd --version | head -1"
echo === toolchain (a fresh Ubuntu has no compiler) ===
wsl -e bash -lc "for t in g++ clang++ make cmake; do printf '%-9s ' $t; command -v $t || echo MISSING; done"
echo === python ===
wsl -e bash -lc "python3 --version; python3 -c 'import torch' 2>&1 | tail -1"
echo === CPU features seen from inside WSL ===
wsl -e bash -lc "grep -o -m1 -E 'avx2|avx512f|avx512_vnni' /proc/cpuinfo | sort -u | tr '\n' ' '; echo"
