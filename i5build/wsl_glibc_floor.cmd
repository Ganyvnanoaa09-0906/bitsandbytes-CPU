@echo off
chcp 65001 >nul
echo ======================================================================
echo  Can the glibc floor be lowered? Each candidate is compiled separately
echo  and its highest required GLIBC_ symbol reported.
echo ======================================================================

echo.
echo --- baseline (what build_linux.sh does today) ---
wsl -u root -e bash -lc "cd /root/bnb_linux && g++ -O2 -std=c++17 -fopenmp -march=native -D__AVX2__ -fPIC -shared -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp -o /tmp/base.so -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu.so 2>&1 | tail -2; objdump -T /tmp/base.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3"

echo.
echo --- + -fno-math-errno (GCC emits sqrtss instead of calling libm) ---
wsl -u root -e bash -lc "cd /root/bnb_linux && g++ -O2 -std=c++17 -fno-math-errno -fopenmp -march=native -D__AVX2__ -fPIC -shared -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp -o /tmp/nme.so -Wl,--no-undefined -Wl,--export-dynamic 2>&1 | tail -2; objdump -T /tmp/nme.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3"

echo.
echo --- + -U_GNU_SOURCE (avoids glibc's C23 strto*/scanf redirects) ---
wsl -u root -e bash -lc "cd /root/bnb_linux && g++ -O2 -std=c++17 -fno-math-errno -U_GNU_SOURCE -fopenmp -march=native -D__AVX2__ -fPIC -shared -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp -o /tmp/ngs.so -Wl,--no-undefined -Wl,--export-dynamic 2>&1 | tail -3; objdump -T /tmp/ngs.so 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -3"

echo.
echo --- which symbols still need 2.3x+ in each build ---
wsl -u root -e bash -lc "for f in /tmp/base.so /tmp/nme.so /tmp/ngs.so; do echo \"  \$f:\"; objdump -T \$f 2>/dev/null | grep -E 'GLIBC_2\.(2[89]|3[0-9]|4[0-9])' | awk '{print \"     \" \$NF, \$(NF-1)}' | sort -u; done"
