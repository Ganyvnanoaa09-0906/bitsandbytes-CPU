#!/usr/bin/env bash
# ISA compatibility matrix: static audits + runtime selftests for every CPU variant.
#
#   x86_64: AVX2 (zig, glibc 2.17), generic (zig), AVX-512 (zig) -- plus their static
#           instruction audits; AVX2/generic run on this CPU, AVX-512 is static-only here
#           (runtime is validated on the EPYC container; see docs_cpu/EPYC_VALIDATION.md).
#   aarch64: static musl selftest under qemu-aarch64 (arm64_selftest.sh), plus the gnu/musl
#           shared-library compile artifacts and NEON codegen audit.
#
# Usage (inside WSL): bash /mnt/d/work/bitsandbytes-CPU/isa_matrix.sh
set -uo pipefail
REPO=/mnt/d/work/bitsandbytes-CPU
PKG="$REPO/bitsandbytes"
MANY=/mnt/d/work/research/manylinux_so
ARM="$PKG/build_arm64"
FAIL=0

AARCH64_OBJDUMP=""
command -v aarch64-linux-gnu-objdump >/dev/null 2>&1 && AARCH64_OBJDUMP=aarch64-linux-gnu-objdump

section() { echo; echo "==============================================================================="; echo "$*"; echo "==============================================================================="; }

audit_x86() { # audit <path> <label>
  local f="$1" label="$2"
  [ -f "$f" ] || { echo "MISSING $label ($f)"; FAIL=$((FAIL+1)); return; }
  echo "-- $label"
  file "$f" | sed 's/.*: /   /'
  echo -n "   NEEDED: "
  objdump -p "$f" 2>/dev/null | awk '/NEEDED/{printf "%s ", $2}'; echo
  echo -n "   max GLIBC: "
  objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -u -V | tail -1 || echo none
  local avx zmm bf16
  avx=$(objdump -d "$f" 2>/dev/null | grep -cE '%[yz]mm|vfmadd|vzeroupper|vmovaps|vmulps|vaddps' || true)
  zmm=$(objdump -d "$f" 2>/dev/null | grep -c '%zmm' || true)
  bf16=$(objdump -d "$f" 2>/dev/null | grep -cE 'vdpbf16ps|vcvtneps2bf16|vcvtne2ps2bf16' || true)
  echo "   AVX-class insns: $avx | zmm: $zmm | bf16 insns: $bf16"
  echo -n "   exports has_avx512bf16_cpu: "
  nm -D --defined-only "$f" 2>/dev/null | grep -c has_avx512bf16_cpu || true
}

audit_aarch64() { # audit_aarch64 <path> <label>
  local f="$1" label="$2"
  [ -f "$f" ] || { echo "MISSING $label ($f)"; FAIL=$((FAIL+1)); return; }
  echo "-- $label"
  file "$f" | sed 's/.*: /   /'
  echo -n "   Machine: "
  readelf -h "$f" 2>/dev/null | awk '/Machine:/{print $2, $3}'
  echo -n "   NEEDED: "
  readelf -d "$f" 2>/dev/null | awk '/NEEDED/{gsub(/\[|\]/,"",$5); printf "%s ", $5}'; echo
  echo -n "   max GLIBC: "
  readelf -V "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -u -V | tail -1 || echo none
  if [ -n "$AARCH64_OBJDUMP" ]; then
    local neon
    neon=$("$AARCH64_OBJDUMP" -d "$f" 2>/dev/null | grep -cE '\b(fmla|fadd|fmul|ld1|st1|sqdmulh|scvtf|fcvt)\b' || true)
    echo "   NEON-class insns: $neon"
  else
    echo "   NEON-class insns: (install binutils-aarch64-linux-gnu to audit)"
  fi
  echo -n "   exports has_avx512bf16_cpu: "
  readelf -sW "$f" 2>/dev/null | grep -c has_avx512bf16_cpu || true
}

section "1. STATIC AUDITS: x86_64 manylinux variants (zig, glibc 2.17)"
audit_x86 "$MANY/libbitsandbytes_cpu.so"          "x86_64 AVX2"
audit_x86 "$MANY/libbitsandbytes_cpu_generic.so"  "x86_64 generic"
audit_x86 "$MANY/libbitsandbytes_cpu_avx512.so"   "x86_64 AVX-512"

section "2. STATIC AUDITS: aarch64 shared libraries (zig)"
audit_aarch64 "$ARM/libbitsandbytes_cpu.so"        "aarch64 gnu .so"
audit_aarch64 "$ARM/libbitsandbytes_cpu_musl.so"   "aarch64 musl .so"

section "3. X86_64 RUNTIME SELFTESTS (this CPU: AVX2 only)"
cd "$PKG" || exit 1
for pair in "libbitsandbytes_cpu.so:avx2" "libbitsandbytes_cpu_generic.so:generic"; do
  so="${pair%%:*}"; label="${pair##*:}"
  echo "-- selftest linked against $label ($so)"
  g++ -O2 -std=c++17 -DNOMINMAX -DNDEBUG -I csrc selftest_cpu.c \
      -L "$MANY" -l:"$so" -Wl,-rpath,"$MANY" -o "/tmp/isa_selftest_$label" 2>&1 | tail -3
  if [ -x "/tmp/isa_selftest_$label" ]; then
    LD_LIBRARY_PATH="$MANY" "/tmp/isa_selftest_$label" 2>&1 | tail -3
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || { echo "   SELFTEST FAILED rc=$rc"; FAIL=$((FAIL+1)); }
  else
    echo "   LINK FAILED"; FAIL=$((FAIL+1))
  fi
done
echo "-- AVX-512 runtime: SKIPPED on this AVX2-only CPU (static audits only;"
echo "   runtime validated on AMD EPYC 9Y24, see docs_cpu/EPYC_VALIDATION.md)"

section "4. AARCH64 RUNTIME: static musl selftest under qemu-aarch64"
bash "$PKG/../arm64_selftest.sh" >/tmp/arm64_isa.log 2>&1
ARM_RC=$?
tail -12 /tmp/arm64_isa.log
[ "$ARM_RC" -eq 0 ] || { echo "   ARM64 SELFTEST rc=$ARM_RC"; FAIL=$((FAIL+1)); }

section "RESULT"
if [ "$FAIL" -eq 0 ]; then
  echo "ISA_MATRIX_PASS"
  exit 0
fi
echo "ISA_MATRIX_FAIL ($FAIL check(s))"
exit 1
