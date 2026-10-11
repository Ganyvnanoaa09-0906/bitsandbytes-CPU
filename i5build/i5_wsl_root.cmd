@echo off
echo === install toolchain as root (no sudo prompt) ===
wsl -u root -e bash -lc "apt-get update -qq 2>&1 | tail -3; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ libomp-dev 2>&1 | tail -6"
echo === verify ===
wsl -u root -e bash -lc "g++ --version | head -1; echo '--- libgomp ---'; ls /usr/lib/gcc/x86_64-linux-gnu/*/libgomp.so 2>/dev/null || echo 'libgomp missing'; echo '--- glibc ---'; ldd --version | head -1"
