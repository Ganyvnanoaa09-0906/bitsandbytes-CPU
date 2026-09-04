# -*- coding: utf-8 -*-
"""detect_cpu.py — 一键检测 CPU 能力 + 推荐配置（开源小白友好）。

自动检测并推荐：
  - CPU 型号 / 物理核 / 逻辑核 / AVX2 / AVX-512
  - 内存总量/可用
  - 推荐线程数（按 GEMM/conv 任务类型）
  - 是否建议用 bnb 8-bit 优化器 / 量化基座 / disk_balancer
输出一行可用作配置的推荐摘要，可被训练脚本直接使用。

用法：
    python detect_cpu.py                 # 打印全部检测 + 推荐
    python detect_cpu.py --json          # 输出 JSON（供脚本解析）
"""
import argparse
import json
import os
import platform


def _cpu_model() -> str:
    try:
        import subprocess
        if os.name == "nt":
            out = subprocess.run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/value"],
                                 capture_output=True, text=True, timeout=15).stdout
        else:
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=15).stdout
        return out.strip().splitlines()[0] if out.strip() else platform.processor()
    except Exception:
        return platform.processor() or "unknown-cpu"


def _physical_cores() -> int:
    try:
        import torch_cpu_kit as tck
        return tck.physical_cores()
    except Exception:
        return os.cpu_count() or 1


def _mem() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except Exception:
        return 0.0, 0.0


def _cpu_capability() -> str:
    try:
        import torch
        return torch.backends.cpu.get_cpu_capability()
    except Exception:
        return "unknown"


def _recommend_threads(logical: int, physical: int) -> dict:
    """按任务推荐线程数（文档实测：LLM GEMM 8，生图 conv 12，R5无超线程拉满）。"""
    if logical == physical:  # 无超线程（R5-4500U 6C6T）→ 拉满物理核
        return {"lm": physical, "diffusion": physical, "image": physical}
    return {"lm": max(1, int(logical * 2 // 3)),  # GEMM 超线程帮不到
            "diffusion": logical,                    # conv 受益超线程
            "image": logical}


def detect() -> dict:
    logical = os.cpu_count() or 1
    physical = _physical_cores()
    total_gb, avail_gb = _mem()
    cap = _cpu_capability()
    r = _recommend_threads(logical, physical)
    return {
        "cpu_model": _cpu_model(),
        "physical_cores": physical,
        "logical_cores": logical,
        "cpu_capability": cap,           # 如 AVX2 / AVX512
        "has_avx2": "AVX2" in cap or "AVX512" in cap or "AVXNEON" in cap,
        "has_avx512": "AVX512" in cap,
        "ram_total_gb": round(total_gb, 1),
        "ram_available_gb": round(avail_gb, 1),
        "recommend_threads": r,
        "recommend_bf16": "BF16" in cap,   # AVX512-BF16/AMX 才建议 bf16
        "recommend_8bit_optimizer": True,  # 纯 CPU 存储紧张时 8bit 优化器省内存
        "recommend_disk_balancer": avail_gb < 8,  # 内存 <8GB 建议开 flash 冷参数卸载
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = detect()
    if a.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return
    print("=== CPU 能力检测 ===")
    print(f"  CPU: {d['cpu_model']}")
    print(f"  物理核/逻辑核: {d['physical_cores']}/{d['logical_cores']}")
    print(f"  SIMD 能力: {d['cpu_capability']}  (AVX2={d['has_avx2']}, AVX-512={d['has_avx512']})")
    print(f"  内存: 总 {d['ram_total_gb']}GB / 可用 {d['ram_available_gb']}GB")
    print("\n=== 推荐配置（实测依据）===")
    print(f"  LLM 训练线程数 : {d['recommend_threads']['lm']}（GEMM 密集，超线程帮不到）")
    print(f"  生图训练线程数 : {d['recommend_threads']['diffusion']}（conv 受益超线程）")
    print(f"  推荐 bf16      : {d['recommend_bf16']}（AVX2-only 机器应用 fp32）")
    print(f"  推荐 8bit 优化器: {d['recommend_8bit_optimizer']}（内存紧张时省内存）")
    print(f"  推荐 disk_balancer: {d['recommend_disk_balancer']}（内存<8GB 建议开 --flash）")


if __name__ == "__main__":
    main()
