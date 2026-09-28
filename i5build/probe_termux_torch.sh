#!/data/data/com.termux/files/usr/bin/bash
# probe_termux_torch.sh -- is torch installable on this Termux, and at what cost?
#
# Why: the C kernels, the ctypes boundary, 8-bit quantize and gemm_8bit are all
# verified on this device. What is NOT verified here is the python TRAINING path
# (disk_balancer / latent_chunk_store / the 1000-step run), because all of those
# import torch and Termux has none. PyPI cannot help: every aarch64 torch wheel
# is manylinux_2_28, i.e. glibc, and Android is bionic.
#
# Termux's own repo does carry a community `pytorch` package. This probe reports
# whether it exists, and how large it is, before committing several GB and a long
# download on a device with 5.8 GB free.
#
# Output goes to the terminal and to /data/local/tmp so the PC can read it.
set -u

OUT=/data/local/tmp/termux_torch_probe.txt
exec > >(tee "$OUT") 2>&1

echo "=== termux torch probe -- $(date) ==="
echo "python : $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
echo "free   : $(df -h /data | awk 'NR==2{print $4}') on /data"
echo

echo "--- pkg search (torch / pytorch / numpy) ---"
pkg search torch 2>&1 | head -20
echo
pkg search pytorch 2>&1 | head -10
echo

echo "--- is it already installed? ---"
python - <<'PY' 2>&1
for m in ("torch", "numpy", "psutil"):
    try:
        mod = __import__(m)
        print(f"  {m:8s} PRESENT {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:8s} missing ({type(e).__name__})")
PY
echo

echo "--- candidate package sizes (what an install would cost) ---"
for p in python-pytorch pytorch libtorch python-torch; do
  line=$(pkg show "$p" 2>/dev/null | grep -E '^(Package|Version|Installed-Size|Download-Size):' | tr '\n' ' ')
  if [ -n "$line" ]; then echo "  $p: $line"; else echo "  $p: not in repo"; fi
done
echo
echo "=== done -- $(date) ==="
cp "$OUT" /sdcard/Download/termux_torch_probe.txt 2>/dev/null && echo "copied to /sdcard/Download/"
