@echo off
REM Build libbitsandbytes_cpu.so inside WSL and test it. This is the Linux path a pip
REM user takes when pip falls back to the sdist: setup.py runs build_linux.sh, which
REM compiles csrc with the local compiler.
REM
REM Two WSL quirks this works around, both of which cost a round trip to find:
REM   * this distribution is WSL1 (kernel 4.4.0-19041-Microsoft), where tar cannot create
REM     files at all -- every entry fails with "Function not implemented". Python's
REM     zipfile module writes through a path WSL1 does handle.
REM   * the zip is built by a Python script rather than Compress-Archive, because the
REM     latter stores backslashes and Linux then extracts files literally named
REM     "csrc\cpu_ops.cpp", which do not compile and do not show up as missing in ls.
echo === unpack ===
wsl -u root -e bash -lc "rm -rf /root/bnb_linux && mkdir -p /root/bnb_linux && cd /root/bnb_linux && python3 -m zipfile -e /mnt/d/qwen21/bnb_linux_src2.zip . && echo unpacked && ls"
echo === build + selftest ===
wsl -u root -e bash -lc "cd /root/bnb_linux && bash build_linux.sh --selftest 2>&1 | tail -45"
echo === produced? ===
wsl -u root -e bash -lc "cd /root/bnb_linux && ls -la bitsandbytes/libbitsandbytes_cpu.so 2>/dev/null || echo 'NO .so PRODUCED'"
echo === load test ===
wsl -u root -e bash -lc "cd /root/bnb_linux && python3 linux_check.py"
