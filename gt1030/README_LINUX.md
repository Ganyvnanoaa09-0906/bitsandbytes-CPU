# Linux 侧做什么、怎么做（现状 + 清单）

本文件针对两件不同的事，先分清楚，别混：

| 目标 | Linux 可行性 | 现状 |
|---|---|---|
| **bnb CPU 后端**（量化/8bit 优化器/CPU 训练） | ✅ 已支持 | `build_linux.sh` 在 WSL Ubuntu 25.10（g++ 15.2）验证过；x86 AVX2 / aarch64 NEON / portable 三产物齐全；selftest 通过 |
| **iGPU 训练**（`gt1030/` 这套 OpenCL 核显训练） | ⚠️ 需要**真 Linux**，WSL 不行 | 代码已做可移植（`CLBLAST_PATH`），但本机 WSL 里没有 OpenCL 设备 |

## 一、WSL 实测结论（本机，2026-10-19）

```
/dev/dri                不存在        -> WSL 没有把 iGPU 的 DRM 设备暴露给 Linux
libOpenCL.so*           未安装/不存在
python3 -c import torch 未安装
g++ 15.2                ✓
```

WSL2 的 GPU 直通面向 D3D12（DirectML）/CUDA，**不提供 iGPU 的 OpenCL**。所以：

- 想在 Linux 上跑核显训练 → 需要一台**真 Linux**（或双系统/裸盘安装）；
- 在 WSL 里能做的：**编译与测试 bnb CPU 后端**（不需要 GPU）、跑 CPU 训练基线（装 CPU 版 torch）。

## 二、WSL 上把 bnb CPU 后端跑起来（无需 sudo）

```bash
cd /mnt/d/work/bitsandbytes-CPU/bitsandbytes
bash build_linux.sh --selftest        # g++ 编译 + C 层自检
# 端到端训练还需要 torch：
python3 -m venv ~/venv && . ~/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ~800MB 下载
PYTHONPATH=$PWD python -c "import bitsandbytes; bitsandbytes.gdn_cpu.load_native()"
```

注意：WSL 无 sudo 也可以用 `pip --user`/venv；`build_linux.sh` 只需要 g++（已装）。

## 三、真 Linux 上跑 iGPU 训练需要什么

1. **OpenCL 运行时**：AMD Vega 6（gfx902）没有官方 ROCm 支持，走 **Mesa OpenCL**：
   - Mesa **Clover**（`mesa-opencl-icd`）——支持老 GCN/Vega，OpenCL 1.2；我们的 kernel 与 CLBlast 都按
     1.2 写，理论可用；
   - 或较新的 **Rusticl**（Mesa 24+，需要 `RUSTICL_ENABLE=...`；对 gfx902 支持有限）。
   验证：`clinfo` 能看到 `gfx902`。
2. **CLBlast**：`apt install libclblast-dev` 或源码 cmake 编译；运行脚本时：
   ```bash
   export CLBLAST_PATH=/usr/lib/x86_64-linux-gnu/libclblast.so
   python3 gt1030/igpu_train.py --model /path/to/model --steps 20
   ```
3. **Python 依赖**：`pip install pyopencl numpy torch transformers safetensors`。
4. **模型**：本地目录即可；无外网时 HF 下载会失败（脚本只对本地目录可靠）。

## 四、预期与风险（诚实）

- 我们的性能数字全部来自 **Windows + AMD 官方 OpenCL 驱动**；Linux 的 Mesa Clover 性能通常更差，
  且 CLBlast kernel 选择不同，**不能假设 7.4 s/步 会在 Linux 上复现**；
- 精度逻辑（fp32 Sgemm、CPU/iGPU parity 对拍）是代码层的，换平台后**先跑 `--check`**：
  ```bash
  python3 gt1030/igpu_train.py --model <local> --check
  ```
  它会打印 CPU vs iGPU 首步 loss 差；差异大就说明该平台的 OpenCL/CLBlast 行为不同；
- 设备常驻引擎（`gt1030/llama_resident.py`）目前仍未完成（多层前向有 GPU 缓冲区异常，见报告 §7.122），
  **不要把它当生产路径**。

## 五、建议的推进顺序

1. WSL：先跑 bnb CPU 后端 + CPU 训练基线（这条现在就能做）；
2. 找一台真 Linux（有 AMD iGPU 更好）：装 Mesa OpenCL + CLBlast，`clinfo` 确认 gfx902；
3. 用 `gt1030/igpu_train.py --check` 对拍，通过后再做 500 步与性能对比；
4. 若目标是"Linux 上核显训练要快"，那必须先把设备常驻引擎修完——这是独立的一轮工作。
