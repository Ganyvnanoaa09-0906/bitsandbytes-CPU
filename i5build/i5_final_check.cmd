@echo off
echo === re-extract (the check script was corrected) and verify the existing .so ===
wsl -u root -e bash -lc "cd /root/bnb_linux && python3 -m zipfile -e /mnt/d/qwen21/bnb_linux_src2.zip . && python3 linux_check.py"
echo === copy the built library back to the Windows side ===
wsl -u root -e bash -lc "cp -f /root/bnb_linux/bitsandbytes/libbitsandbytes_cpu.so /mnt/d/qwen21/libbitsandbytes_cpu.so && ls -la /mnt/d/qwen21/libbitsandbytes_cpu.so"
echo === what does it link against (the portability question) ===
wsl -u root -e bash -lc "cd /root/bnb_linux && objdump -p bitsandbytes/libbitsandbytes_cpu.so | grep NEEDED; echo '--- glibc symbols required ---'; objdump -T bitsandbytes/libbitsandbytes_cpu.so | grep -o 'GLIBC_[0-9.]*' | sort -u"
