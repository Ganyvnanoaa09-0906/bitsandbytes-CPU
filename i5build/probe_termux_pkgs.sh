#!/data/data/com.termux/files/usr/bin/bash
OUT=/data/local/tmp/termux_pkg_probe.txt
exec > >(tee "$OUT") 2>&1
echo "=== termux pkg probe ==="
echo "python: $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
echo
for p in tokenizers safetensors transformers regex pyyaml sentencepiece protobuf rust cargo python-pillow python-scipy python-sympy python-networkx; do
  line=$(pkg show "$p" 2>/dev/null | grep -E '^(Package|Version|Installed-Size|Download-Size):' | tr '\n' ' ')
  if [ -n "$line" ]; then echo "  [REPO] $p: $line"; else echo "  [ -- ] $p: not in repo"; fi
done
echo
echo "--- what IS available matching these keywords ---"
for kw in tokeniz safetensor transformer sentencepiece protobuf rust; do
  echo "  search '$kw':"
  pkg search "$kw" 2>/dev/null | head -6 | sed 's/^/      /'
done
echo
echo "--- pip view ---"
pip --version 2>&1
pip config list 2>&1 | head -5
echo "=== done ==="