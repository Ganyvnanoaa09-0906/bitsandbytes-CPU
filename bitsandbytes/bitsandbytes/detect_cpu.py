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
    """The CPU's name, not the first line of whatever tool we asked.

    `lscpu` was run and its FIRST line taken, which on every Linux is
    "Architecture: x86_64" -- the report said the CPU was a machine architecture. The
    model name is on the "Model name:" line; /proc/cpuinfo carries it too, and does not
    depend on util-linux being installed.

    Windows keeps the wmic path but parses the value out rather than the first line,
    since `wmic ... /value` emits "Key=Value" lines whose order is not guaranteed.
    """
    try:
        import subprocess
        if os.name == "nt":
            # Registry first. `wmic` has been removed from current Windows builds, and
            # when it is missing the old code fell through to platform.processor(),
            # which reports "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD" -- a
            # description of the CPU's family, not its name. The registry key is where
            # the name actually lives and needs no subprocess.
            try:
                import winreg
                k = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
            except Exception:
                pass
            out = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=15).stdout
            for line in out.splitlines():
                if line.strip().lower().startswith("name="):
                    return line.split("=", 1)[1].strip()
        else:
            # /proc/cpuinfo first: no dependency on lscpu being present.
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
            out = subprocess.run(["lscpu"], capture_output=True, text=True,
                                 timeout=15).stdout
            for line in out.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown-cpu"


def _physical_cores() -> int:
    """Physical cores, not logical ones: an SMT machine reports twice as many threads
    and the recommendation below depends on the difference.

    Tries the package-relative import first, because this module now ships inside
    `bitsandbytes`. The bare import is kept for the checkout layout, where the two files
    sit next to each other at the repository root and there is no package to be relative
    to. Falling back to os.cpu_count() reports LOGICAL cores, which on an SMT machine
    makes the recommendation wrong -- so that is the last resort, not the first.
    """
    try:
        from . import torch_cpu_kit as tck
        return tck.physical_cores()
    except Exception:
        pass
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
    """Recommend a thread count per task, from measurements rather than from core count.

    The previous version returned `logical` for diffusion on any SMT machine, on the
    stated basis that "conv benefits from hyperthreading". Measured on the i5-10400
    (6 physical / 12 logical), that is wrong for the workload this project actually
    runs. The Qwen-Image-2.1 step is 82% dense GEMM, and on the four shapes that model
    uses, six threads beat twelve on every one:

        16384x4096x4096    6T 1.316 s    12T 1.449 s
        16384x4096x12288   6T 4.048 s    12T 4.325 s
         4096x4096x4096    6T 0.335 s    12T 0.390 s
        GFLOPS              6T 407-418    12T 352-381

    A whole 1024x1024 render agreed: 225 s/step at 12 threads against 203 at 6. The
    second thread on a physical core shares its FP units and halves its cache, which a
    compute-bound GEMM cannot absorb.

    So: physical cores everywhere. The `logical` value is still returned under a
    separate key for callers that know their workload is bandwidth- or latency-bound
    rather than arithmetic-bound.
    """
    return {"lm": physical,
            "diffusion": physical,
            "image": physical,
            "logical_if_bandwidth_bound": logical}


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
        # 内存读不到时（没装 psutil）不要给 True：avail_gb 是 0，< 8 会成立，
        # 于是这个推荐就变成了一个由缺失依赖制造的假警报。
        "recommend_disk_balancer": bool(avail_gb) and avail_gb < 8,
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
    t = d["recommend_threads"]
    print(f"  LLM 训练线程数 : {t['lm']}（GEMM 密集，超线程只会抢同一个核的 FP 单元）")
    print(f"  生图线程数     : {t['diffusion']}（Qwen-Image 一步 82% 是 GEMM；i5 实测 "
          f"6 线程 203 s/step vs 12 线程 225）")
    print(f"  带宽受限时可用 : {t['logical_if_bandwidth_bound']}（仅当负载不是算术受限）")
    print(f"  推荐 bf16      : {d['recommend_bf16']}（AVX2-only 机器应用 fp32）")
    print(f"  推荐 8bit 优化器: {d['recommend_8bit_optimizer']}（内存紧张时省内存）")
    print(f"  推荐 disk_balancer: {d['recommend_disk_balancer']}（内存<8GB 建议开 --flash）")


if __name__ == "__main__":
    main()
