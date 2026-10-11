@echo off
echo === docker available in WSL? ===
wsl -e bash -lc "command -v docker || echo 'no docker'"
echo === installing the toolchain (this takes a few minutes) ===
wsl -e bash -lc "sudo apt-get update -qq 2>&1 | tail -2; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ libomp-dev 2>&1 | tail -5"
echo === verify ===
wsl -e bash -lc "g++ --version | head -1; echo '--- libgomp ---'; ls /usr/lib/gcc/x86_64-linux-gnu/*/libgomp.so"
