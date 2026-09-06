<p align="center"><img src="https://avatars.githubusercontent.com/u/175231607?s=200&v=4" alt=""></p>
<h1 align="center">bitsandbytes</h1>
<p align="center">
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/bitsandbytes-foundation/bitsandbytes.svg?color=blue"></a>
    <a href="https://pepy.tech/project/bitsandbytes"><img alt="Downloads" src="https://static.pepy.tech/badge/bitsandbytes/month"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/actions/workflows/tests-nightly.yml"><img alt="Nightly Unit Tests" src="https://img.shields.io/github/actions/workflow/status/bitsandbytes-foundation/bitsandbytes/tests-nightly.yml?logo=github&label=Nightly%20Tests"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/bitsandbytes-foundation/bitsandbytes"></a>
    <a href="https://pypi.org/project/bitsandbytes/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bitsandbytes"></a>
</p>

`bitsandbytes` enables accessible large language models via k-bit quantization for PyTorch. We provide three main features for dramatically reducing memory consumption for inference and training:

* 8-bit optimizers uses block-wise quantization to maintain 32-bit performance at a small fraction of the memory cost.
* LLM.int8() or 8-bit quantization enables large language model inference with only half the required memory and without any performance degradation. This method is based on vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication.
* QLoRA or 4-bit quantization enables large language model training with several memory-saving techniques that don't compromise performance. This method quantizes a model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

## System Requirements
bitsandbytes has the following minimum requirements for all platforms:

* Python 3.10+
* [PyTorch](https://pytorch.org/get-started/locally/) 2.4+
  * _Note: While we aim to provide wide backwards compatibility, we recommend using the latest version of PyTorch for the best experience._

#### Accelerator support:

<small>Note: this table reflects the status of the current development branch. For the latest stable release, see the
[document in the 0.50.0 tag](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.0/README.md#accelerator-support).
</small>

##### Legend:
🚧 = Planned |
〰️ = Partially Supported |
✅ = Supported |
❌ = Not Supported

<table>
  <thead>
    <tr>
      <th>Platform</th>
      <th>Accelerator</th>
      <th>Hardware Requirements</th>
      <th>LLM.int8()</th>
      <th>QLoRA 4-bit</th>
      <th>8-bit Optimizers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6">🐧 <strong>Linux, glibc >= 2.24</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>Minimum: AVX2<br>Optimized: AVX512F, AVX512BF16</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM60+ minimum<br>SM75+ recommended</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟥 AMD GPU <br><code>cuda</code></td>
      <td>
        CDNA: gfx908, gfx90a, gfx942, gfx950, gfx1250<br>
        RDNA: gfx103X, gfx110X, gfx115X, gfx120X
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Data Center GPU Max Series<br>
        Arc A-Series (Alchemist)<br>
        Arc B-Series (Battlemage)
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟪 Intel Gaudi <br><code>hpu</code></td>
      <td>Gaudi2, Gaudi3</td>
      <td>✅</td>
      <td>〰️</td>
      <td>❌</td>
    </tr>
    <tr>
      <td align="right">aarch64</td>
      <td>◻️ CPU</td>
      <td></td>
      <td>✅ *</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM75+</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td colspan="6">🪟 <strong>Windows 11 / Windows Server 2022+</strong></td>
    </tr>
    <tr>
      <td align="right">x86-64</td>
      <td>◻️ CPU</td>
      <td>AVX2</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM60+ minimum<br>SM75+ recommended</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟥 AMD GPU <br><code>cuda</code></td>
      <td>
        RDNA: gfx103X, gfx110X, gfx115X, gfx120X
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟦 Intel GPU <br><code>xpu</code></td>
      <td>
        Arc A-Series (Alchemist) <br>
        Arc B-Series (Battlemage)
      </td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td align="right">arm64</td>
      <td>◻️ CPU</td>
      <td></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>🟩 NVIDIA GPU <br><code>cuda</code></td>
      <td>SM121</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td colspan="6">🍎 <strong>macOS 14+</strong></td>
    </tr>
    <tr>
      <td align="right">arm64</td>
      <td>◻️ CPU</td>
      <td>Apple M1+</td>
      <td>✅ *</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td></td>
      <td>⬜ Metal <br><code>mps</code></td>
      <td>Apple M1+</td>
      <td>✅ *</td>
      <td>✅</td>
      <td>🚧</td>
  </tbody>
</table>
<sup>* While supported, these marked features may lack in performance optimizations.</sup>

## :book: Documentation
* [Official Documentation](https://huggingface.co/docs/bitsandbytes/main)
* 🤗 [Transformers](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
* 🤗 [Diffusers](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)
* 🤗 [PEFT](https://huggingface.co/docs/peft/developer_guides/quantization#quantize-a-model)

## :computer: CPU Backend Extensions (this fork)

This fork adds **training-grade CPU kernels** to the bitsandbytes CPU backend, for
machines **without NVIDIA GPUs** (AVX2 only / ARM64 NEON, e.g. consumer laptops,
Mini-PCs). The upstream CPU backend is inference-oriented; these
extensions make pure-CPU LoRA / quantization / diffusion training practical.

> 📚 **Docs** (`docs_cpu/`):
> - `TECHNICAL_GUIDE.md` — **技术文档**：改了什么、每个内核的架构与细节（面向技术人员）
> - `QUICKSTART.md` — **快速入门手册**：命令、用途与参数（面向首次使用者）
> - `TECH_REPORT.md` — **技术报告**：完整方法论、实验数据与结论（含负面结论）
> - 灾难恢复指南已并入 `QUICKSTART.md` §9（英文版见该文件附录）。
> - 🌐 **English**: 英文版已合并至上述中文文档的「附录：English Reference」中（以中文为准）；灾难恢复指南已并入 `QUICKSTART.md` §9
>
> 🌐 **English README**: see [`README_EN.md`](README_EN.md).
>
> > **关于硬盘均衡负载**：它能在内存紧张时帮助维持训练，但**不能替代 Windows 虚拟内存**。
> > 训练期间仍**建议保留至少 1GB 虚拟内存**以应对突发内存峰值。

### What's added

| Kernel | Entry point | Purpose |
|---|---|---|
| **Gated DeltaNet fwd/bwd** | `gdn_fwd_cpu` / `gdn_bwd_cpu` (csrc/cpu_gdn.cpp) | Fused Gated DeltaNet (Qwen3-Next/3.5 linear attention) — ~35× vs per-step Python loop; chunk-checkpointed backward |
| **Fused 8-bit blockwise dequant GEMM** | `cgemm_8bit_inference_cpu_fp32` (`bitsandbytes::gemm_8bit`, `bnb.functional.fused_dequant_linear_8bit`) | `out = A @ dequant8(B)` with weights kept uint8 (¼ DRAM traffic), no fp32 temporary |
| **Fused 8-bit optimizer** | `coptimizer_update_8bit_blockwise_cpu` (`bnb.optim.AdamW8bit` etc.) | Single-pass dequant→update→requant; optimizer state memory ≈ 1/3.8 of fp32 |
| **Blockwise 8/4-bit quant & dequant** | `cquantize_blockwise_cpu_*` / `cdequantize_blockwise_cpu_*` | AVX2 LUT quantize + dequantize; NF4/FP4 kernels with NEON/AVX2 paths |
| **4-bit inference GEMV** | `cgemv_4bit_inference_cpu_*` | Fused 4-bit dequant GEMV/GEMM for AVX2 machines (symbol alias fixed) |
| **GDN runtime patch** | `bitsandbytes/gdn_cpu.py` | `patch_transformers()` / `patch_fla()` — route transformers GDN slow paths to the fused kernel |
| **iGPU (DirectML) block-level executor** | `gpu_scheduler.py` (`IgpuExecutor`, `train.py --igpu`) | Experimental: frozen weights resident on the DirectML iGPU, forward/backward on-device, **one sync per step**; **off by default** — measured **+30~35%** for GEMM-dense seq≤256 training, **negative** at long sequences (driver limits) |

### Build

```bash
# Linux (x86_64 AVX2 or aarch64 NEON, g++/clang + OpenMP)
# deps: sudo apt install g++ libomp-dev   (Debian/Ubuntu)
bash build_linux.sh            # -> bitsandbytes/libbitsandbytes_cpu.so
bash build_linux.sh --selftest # also build+run torch-free C self-test

# Windows (MSVC, manual cl build, no CMake)
build_manual\build_manual.bat amd   # (in VS x64 Native Tools prompt)
```

> `.gitattributes` 强制 C/C++ 源码与构建脚本用 LF 换行，clone 到 Windows/Linux 均可
> 直接编译（CRLF 会让 MSVC 吞行、Linux `g++` 报 `bad interpreter`）。
> `disk_balancer`（`--flash`）跨平台：Windows 用 ctypes，Linux/WSL 用 psutil。

### Prebuilt wheels (optional)

预编译的 wheel（`pip install bitsandbytes-<ver>-py3-none-<platform>.whl`）可从 **Release**
获取，无需自己编译（Windows `win_amd64` / Linux `linux_x86_64` / aarch64
`linux_aarch64`）。wheel 内含 `bitsandbytes/` 包 + 编译好的 native 库
（`.dll`/`.so`），安装即用。

### Self-test (no torch required)

```bash
# C-level kernel self-test: quantize roundtrip / gemm_8bit / 4bit GEMV / 8bit optimizer
bash build_linux.sh --selftest
```

### Notes

- This fork is based on bitsandbytes v0.45.1 (MIT, Facebook) and keeps the upstream
  MIT license. See `NOTICE.md` for contributions.
- The CPU backend keeps the upstream NEON/AVX paths for plain quant/dequant;
  the **fused training kernels** (GDN, gemm_8bit, 8-bit optimizer) are the
  addition of this fork.
- Not recommended for use without the compiled native library — build first
  (Windows: `build_manual\build_manual.bat amd`, Linux: `build_linux.sh`).

## :heart: Sponsors
The continued maintenance and development of `bitsandbytes` is made possible thanks to the generous support of our sponsors. Their contributions help ensure that we can keep improving the project and delivering valuable updates to the community.

<kbd><a href="https://hf.co" target="_blank"><img width="100" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" alt="Hugging Face"></a></kbd>

## License
`bitsandbytes` is MIT licensed.

## How to cite us
If you found this library useful, please consider citing our work:

### QLoRA

```bibtex
@article{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

### LLM.int8()

```bibtex
@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

### 8-bit Optimizers

```bibtex
@article{dettmers2022optimizers,
  title={8-bit Optimizers via Block-wise Quantization},
  author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke},
  journal={9th International Conference on Learning Representations, ICLR},
  year={2022}
}
```
