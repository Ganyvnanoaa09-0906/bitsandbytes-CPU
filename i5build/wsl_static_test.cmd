@echo off
chcp 65001 >nul
echo ======================================================================
echo  Can __libc_single_threaded (GLIBC_2.32) be removed too?
echo  It comes from the C++/OpenMP runtime, not from our code.
echo ======================================================================

echo.
echo --- A: static libstdc++ / libgcc ---
wsl -u root -e bash -lc "cd /root/bnb_linux && g++ -O2 -std=c++17 -fno-math-errno -fopenmp -march=native -D__AVX2__ -fPIC -shared -static-libstdc++ -static-libgcc -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp -o /tmp/a.so -Wl,--no-undefined -Wl,--export-dynamic 2>&1 | tail -2; echo '  floor:'; objdump -T /tmp/a.so 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1; objdump -T /tmp/a.so 2>/dev/null | grep -E 'GLIBC_2\.(3[0-9]|4[0-9])' | sed 's/.*GLIBC/    GLIBC/' | sort -u"

echo.
echo --- B: static runtime + statically linked libgomp ---
wsl -u root -e bash -lc "cd /root/bnb_linux && g++ -O2 -std=c++17 -fno-math-errno -fopenmp -march=native -D__AVX2__ -fPIC -shared -static-libstdc++ -static-libgcc -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp -o /tmp/b.so -Wl,--no-undefined -Wl,--export-dynamic -Wl,-Bstatic -lgomp -Wl,-Bdynamic -lpthread 2>&1 | tail -2; echo '  floor:'; objdump -T /tmp/b.so 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1"

echo.
echo --- C: does the build still work + selftest after static linking ---
wsl -u root -e bash -lc "cd /root/bnb_linux && ls -la /tmp/a.so /tmp/b.so 2>/dev/null; ldd /tmp/a.so 2>/dev/null | head -6"

echo.
echo --- D: what does the CURRENT build depend on for that symbol ---
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep single_threaded; echo '--- who defines it ---'; for l in /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libgomp.so.1; do printf '%s: ' \$l; objdump -T \$l 2>/dev/null | grep -c single_threaded; done"
