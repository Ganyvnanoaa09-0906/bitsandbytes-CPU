The majority of bitsandbytes is licensed under MIT, however portions of the project are available under separate license terms: PyTorch is licensed under the BSD license.

## Contributions

This fork adds CPU training kernels on top of bitsandbytes v0.45.1 (MIT, Facebook):

- `csrc/cpu_gdn.cpp` — Gated DeltaNet fused forward/backward kernels (AVX2/FMA,
  OpenMP by head, chunk-checkpointed backward). New file.
- `csrc/cpu_ops.cpp` — fused blockwise 8-bit dequant GEMM (`gemm_8bit`),
  fused 8-bit optimizer single-pass kernels, AVX2 micro-kernels for
  quantize/dequantize with NEON fallback on ARM64. Extends the upstream file.
- `csrc/cpu_ops.h` / `csrc/pythonInterface.cpp` — declarations and
  `extern "C"` exports for the new CPU kernels.
- `bitsandbytes/gdn_cpu.py` — GDN Python wrapper + `patch_transformers()` /
  `patch_fla()` runtime replacement + native-library self-test. New file.
- `bitsandbytes/backends/cpu/ops.py` — CPU dispatch for `bitsandbytes::gemm_8bit`,
  plus a fix so the 4-bit GEMV fallback works on AVX2-only machines.
- `build_manual/` — Windows manual `cl` build (no CMake) + export .def.
- `build_linux.sh` / `selftest_cpu.c` / `wsl_verify.sh` —
  Linux build entry and a torch-free C-level kernel self-test.

These contributions are likewise released under the MIT license, same as the
upstream bitsandbytes project they extend.
