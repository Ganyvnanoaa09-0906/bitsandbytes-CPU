"""dev9: Windows native DLL, q3bit real-model validation, iGPU label fix."""
from __future__ import annotations

import io

ROOT = r'D:\work\bitsandbytes-CPU'
INNER = ROOT + r'\bitsandbytes\README.md'
EN = ROOT + r'\bitsandbytes\README_EN.md'
OUTER = ROOT + r'\README.md'
CLI = ROOT + r'\bitsandbytes\bitsandbytes_cpu_cli.py'
INIT = ROOT + r'\bitsandbytes\bitsandbytes\__init__.py'


def patch(path: str, pairs: list[tuple[str, str]]) -> None:
    t = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        assert t.count(old) == 1, (path, old[:70], t.count(old))
        t = t.replace(old, new, 1)
    io.open(path, 'w', encoding='utf-8', newline='').write(t)
    print('updated', path)


patch(INIT, [('__version__ = "0.50.2.dev8"', '__version__ = "0.50.2.dev9"')])

patch(INNER, [
    ('**Version: `0.50.2.dev8`**', '**Version: `0.50.2.dev9`**'),
    ('- **Windows wheel** ships a prebuilt DLL. That DLL predates the dev5/dev6 2/3-bit\n'
     '  kernels and the restored 4-bit optimizer kernel, so those run the pure-torch\n'
     '  fallback there (same numbers, lower speed).',
     '- **Windows wheel**: from dev9 on it ships the MSVC DLL built by CI from the current\n'
     '  `csrc`, which exports the 2/3-bit dequant/GEMM and 4-bit optimizer kernels; they run\n'
     '  natively on Windows (verified: dequant maxdiff 1.2e-6, 4-bit native vs torch 1.2e-7).\n'
     '  Wheels older than dev9 fall back to pure torch for those symbols.'),
    ('| 4-bit optimizer | `AdamW4bit`: ≈1.03 bytes of state per parameter (restored in dev8; scalar kernel + torch fallback) |',
     '| 4-bit optimizer | `AdamW4bit`: ≈1.03 bytes of state per parameter (scalar kernel; native on Linux and, since dev9, Windows) |'),
    ('| Fused lm_head 3-bit GEMM (research) | 14–22% faster than fp32 matmul, 89% less weight memory |',
     '| Fused lm_head 3-bit GEMM (research) | 14–22% faster than fp32 matmul, 89% less weight memory |\n'
     '| 3-bit frozen base + LoRA (Qwen3-1.7B, CPU) | 80 steps: eval loss 4.34 → **4.03**; RSS **2.9 GB vs 8.7 GB** fp32; step 7.7 s vs 5.5 s (native Windows) |'),
    ('| `0.50.2.dev8` | **restored `AdamW4bit`** (the kernel was dropped when dev4 vendored upstream; now scalar C kernel + torch fallback, 1.031 B/param); **new `bitsandbytes.q3bit` API** — `quantize_3bit`/`dequantize_3bit`/`gemm_3bit`/`gemm_3bit_transposed`, `QuantLinear3bit` (frozen 3-bit base + LoRA), `convert_linears_3bit`, native kernels on Linux with identical pure-torch fallback; CLI reference now includes 4bitopt/q3bit and all 43 examples execute in CI-style verification |',
     '| `0.50.2.dev8` | **restored `AdamW4bit`** (the kernel was dropped when dev4 vendored upstream; now scalar C kernel + torch fallback, 1.031 B/param); **new `bitsandbytes.q3bit` API** — `quantize_3bit`/`dequantize_3bit`/`gemm_3bit`/`gemm_3bit_transposed`, `QuantLinear3bit` (frozen 3-bit base + LoRA), `convert_linears_3bit`, native kernels on Linux with identical pure-torch fallback; CLI reference now includes 4bitopt/q3bit and all 43 examples execute in CI-style verification |\n'
     '| `0.50.2.dev9` | **Windows native kernels**: CI now compiles the MSVC DLL from the current csrc and the export list includes the 4-bit optimizer and 2/3-bit symbols (they were silently missing before); **q3bit real-model validation** on Qwen3-1.7B (frozen 3-bit base + LoRA, eval 4.34→4.03 over 80 steps, RSS −67%, native step 7.7 s vs fp32 5.5 s); fixed the causal-LM **double-shift** label bug in the iGPU training kit (Windows package v0.1.2) |'),
])

patch(EN, [
    ('(import name unchanged: `import bitsandbytes as bnb`). Current release: **`0.50.2.dev8`**.',
     '(import name unchanged: `import bitsandbytes as bnb`). Current release: **`0.50.2.dev9`**.'),
    ('dev8 restored `AdamW4bit` (1.031 B/param, scalar kernel + torch fallback) and added the\n'
     '`bitsandbytes.q3bit` API (3-bit quant/dequant/fused GEMM, `QuantLinear3bit` frozen base +\n'
     'LoRA, `convert_linears_3bit`).',
     'dev8 restored `AdamW4bit` and added the `bitsandbytes.q3bit` API ·\n'
     'dev9 ships the CI-built MSVC DLL that now exports the 4-bit optimizer and 2/3-bit kernels\n'
     '(native on Windows), validates the 3-bit frozen base + LoRA on Qwen3-1.7B, and fixes the\n'
     'causal-LM double-shift label bug in the iGPU training kit.'),
])

patch(OUTER, [
    ('current\nrelease `0.50.2.dev8`', 'current\nrelease `0.50.2.dev9`'),
    ('| `0.50.2.dev8` | restored `AdamW4bit` (1.031 B of state/param, scalar kernel + torch fallback) + new `bitsandbytes.q3bit` API (blockwise 3-bit quant/dequant/fused GEMM, `QuantLinear3bit` frozen 3-bit base + LoRA, `convert_linears_3bit`) |',
     '| `0.50.2.dev8` | restored `AdamW4bit` (1.031 B of state/param, scalar kernel + torch fallback) + new `bitsandbytes.q3bit` API (blockwise 3-bit quant/dequant/fused GEMM, `QuantLinear3bit` frozen 3-bit base + LoRA, `convert_linears_3bit`) |\n'
     '| `0.50.2.dev9` | Windows native kernels (CI-built MSVC DLL now exports the 4-bit optimizer + 2/3-bit symbols), q3bit real-model validation on Qwen3-1.7B, iGPU causal-LM double-shift fix (package v0.1.2) |'),
])

patch(CLI, [
    ('the Windows wheel predates the 2/3-bit C kernels, so it runs the pure-torch ',
     'since dev9 the Windows wheel ships a CI-built MSVC DLL; older wheels use the pure-torch '),
    ('fallback; the Linux wheels use the native kernels',
     'fallback, while the dev9 Windows and Linux wheels use the native kernels'),
    ('# 3. if the shared library does not export',
     '# 3. older Windows wheels (< dev9) do not export'),
    ('#    coptimizer_update_4bit_blockwise_cpu (the Windows wheel does not),',
     '#    coptimizer_update_4bit_blockwise_cpu (pre-dev9 Windows wheels),'),
    ('# the pure-torch fallback (Windows wheel) is slower still',
     '# older Windows wheels run the pure-torch fallback (dev9+ is native)'),
])
print('dev9 docs/version updated')
