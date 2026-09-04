"""torch_cpu_kit — 无 CUDA 环境（纯 CPU）训练压榨工具包。

适用场景：Windows 10 / Linux，无 NVIDIA 显卡，用 pip 装的官方 PyTorch
（>= 2.0），目标是把 i5-10400（6C12T, AVX2） / R5-4500U（6C6T, AVX2）
这类 AVX2-only CPU 的训练吞吐和内存占用压到最优。

用法（两行上手）::

    import torch_cpu_kit as tck          # 1) 先 import 本 kit
    tck.apply_env()                      # 2) 设置 OMP/MKL 环境变量（必须在 import torch 之前）
    import torch                         # 3) 再 import torch
    tck.init(verbose=True)               # 4) 线程数 / oneDNN / matmul 精度一键调优

设计原则：
  - 纯 Python，不重编 torch，对任何 pip 安装的 torch 生效；
  - 所有函数都可以反复调用；检测失败时安全回退，绝不抛异常打断训练；
  - 与 bitsandbytes CPU 版（GDN 内核 + 8-bit 优化器）完全兼容，
    配合 GDN_CPU_CHUNK 环境变量一起用效果最佳。
"""

from __future__ import annotations

import os
import subprocess
import threading
from contextlib import nullcontext
from typing import Any, Optional

__all__ = [
    "apply_env",
    "init",
    "report",
    "physical_cores",
    "recommended_autocast",
    "autocast",
    "fast_loader",
    "make_contiguous_",
    "mem_report",
    "start_mem_monitor",
    "suspend_mem_monitor",
]

_state: dict[str, Any] = {"env": None, "init": None}


# ---------------------------------------------------------------------------
# 物理核检测（关键：i5-10400 是 6C12T，线程数应为 6 而不是 12；
# R5-4500U 是 6C6T 无超线程，直接用逻辑核数 6）
# ---------------------------------------------------------------------------
_cores_cache: Optional[int] = None


def physical_cores() -> int:
    """返回物理核数（结果缓存）。Linux 读 sysfs，Windows 读 wmic/PowerShell，
    都失败则回退逻辑核数（os.cpu_count()）。"""
    global _cores_cache
    if _cores_cache is not None:
        return _cores_cache
    # Linux: 每个 (package, core) 组合计一个物理核
    try:
        import glob

        ids = []
        for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
            try:
                with open(os.path.join(d, "physical_package_id")) as f:
                    pkg = f.read().strip()
                with open(os.path.join(d, "core_id")) as f:
                    core = f.read().strip()
                ids.append((pkg, core))
            except OSError:
                continue
        if ids:
            _cores_cache = max(1, len(set(ids)))
            return _cores_cache
    except Exception:
        pass
    # Windows: wmic（Win10 自带）-> PowerShell 兜底
    if os.name == "nt":
        for cmd in (
            ["wmic", "cpu", "get", "NumberOfCores", "/value"],
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
        ):
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=15
                ).stdout
                nums = [
                    int(line.split("=", 1)[1])
                    for line in out.splitlines()
                    if "=" in line and line.split("=", 1)[1].strip().isdigit()
                ]
                if not nums and out.strip().isdigit():  # powershell 直接输出一个数
                    nums = [int(out.strip())]
                if nums:
                    _cores_cache = max(1, sum(nums))
                    return _cores_cache
            except Exception:
                continue
    _cores_cache = os.cpu_count() or 1
    return _cores_cache


# ---------------------------------------------------------------------------
# 环境变量（必须在 import torch 之前调用才生效）
# ---------------------------------------------------------------------------
def apply_env(
    num_threads: Optional[int] = None,
    wait_policy: str = "active",
    verbose: bool = False,
) -> dict[str, str]:
    """设置 OpenMP/MKL 环境变量。**必须在 import torch 之前调用**（这些
    变量在 OpenMP 运行库加载时读取，之后改无效）。

    - OMP_NUM_THREADS / MKL_NUM_THREADS = 物理核数（见 physical_cores()）。
      超线程对 GEMM 帮助很小，却会和 bitsandbytes 内核自己的 OpenMP 线程
      池打架（12 逻辑核 x 2 套线程池 = 24 线程在 6 个物理核上空转）。
    - OMP_WAIT_POLICY=active：自旋等待，短生命周期并行区（比如每层都跑
      的 GDN kernel）唤醒延迟最低。
    - OMP_DYNAMIC=FALSE / MKL_DYNAMIC=FALSE：禁止运行库偷懒减线程。

    已存在的环境变量不会被覆盖；显式传 num_threads 则强制覆盖。
    返回实际生效的键值 dict。
    """
    n = num_threads or physical_cores()
    want = {
        "OMP_NUM_THREADS": str(n),
        "MKL_NUM_THREADS": str(n),
        "OMP_WAIT_POLICY": wait_policy,
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    set_vals: dict[str, str] = {}
    for k, v in want.items():
        if num_threads is not None or k not in os.environ or not os.environ[k]:
            os.environ[k] = v
        set_vals[k] = os.environ[k]
    _state["env"] = set_vals
    if verbose:
        print("[torch_cpu_kit] env:", set_vals)
    return set_vals


# ---------------------------------------------------------------------------
# torch 侧一键调优（import torch 之后调用）
# ---------------------------------------------------------------------------
def init(
    num_threads: Optional[int] = None,
    interop_threads: int = 1,
    verbose: bool = False,
    model_kind: Optional[str] = None,
) -> dict[str, Any]:
    """import torch 之后调用：线程池 / oneDNN / matmul 精度一键调优。

    - torch.set_num_threads(模型类型最优线程)：见下；
    - torch.set_num_interop_threads(1)：训练是单进程单图的串行图，interop
      池只会抢核（若 torch 已启动并行工作会抛 RuntimeError，内部吞掉）；
    - oneDNN 打开（AVX2 GEMM/conv 走它最快）；
    - 仅当 CPU 支持 AVX512-BF16/AMX 时才放宽 float32 matmul 精度到
      "high"；AVX2-only 的 CPU 保持 fp32 不降精度。

    ``model_kind`` 按任务选线程（i5-10400 6C12T 实测，GFLOP/s）：

    - ``'lm'``（LLM 微调，GEMM 密集）：**8 线程**最优（277→8线程313→12线程273，
      超线程帮不到 GEMM 还拖后腿）；
    - ``'vision'`` / ``'diffusion'``（生图 U-Net / 视觉 tower，**conv 密集**）：
      **12 线程**最优（conv 6线程329→8线程489→12线程618 GFLOP/s，超线程被 conv
      的 im2col/分块吃满）；attention 小块反而 6 线程好，但整体 conv 主导；
    - ``None``：回退物理核数。

    返回调优报告 dict（同 report()）。
    """
    import torch

    if num_threads is None:
        logical = os.cpu_count() or 1
        physical = physical_cores()
        if model_kind in ("vision", "diffusion", "image", "img"):
            _n = logical          # conv 受益超线程：用逻辑核（i5-10400 = 12）
        elif model_kind == "lm":
            # GEMM 密集：有超线程时逻辑核×2/3（i5-10400 12→8）；无超线程
            # （R5-4500U 6C6T）直接拉满物理核，别打折——实测 6 线程比
            # 4/5 线程快 16%/8%（GEMM 125 vs 101/108 GFLOP/s）
            _n = physical if physical == logical else max(1, int(logical * 2 // 3))
        else:
            _n = physical
        n = _n
    else:
        n = num_threads
    torch.set_num_threads(max(1, n))
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        pass  # 已经有并行工作启动了，保持原值
    torch.backends.mkldnn.enabled = True
    if _bf16_capable():
        torch.set_float32_matmul_precision("high")

    _state["init"] = {"threads": n, "interop": interop_threads, "model_kind": model_kind}
    rep = report()
    if verbose:
        print("[torch_cpu_kit] init:\n" + rep)
    return rep


def _capability() -> str:
    try:
        import torch

        return torch.backends.cpu.get_cpu_capability()
    except Exception:
        return "DEFAULT"


def _bf16_capable() -> bool:
    cap = _capability()
    return "AVX512" in cap and "BF16" in cap or cap == "AMX"


def report() -> str:
    """人类可读的当前 CPU/线程/精度状态报告（不修改任何东西）。"""
    import torch

    lines = [
        f"  CPU capability : {_capability()}",
        f"  logical cores  : {os.cpu_count()}",
        f"  physical cores : {physical_cores()}",
        f"  torch threads  : {torch.get_num_threads()} "
        f"(interop {torch.get_num_interop_threads()})",
        f"  oneDNN (mkldnn): {torch.backends.mkldnn.enabled}",
        f"  f32 matmul     : {torch.get_float32_matmul_precision()}",
        f"  autocast advice: {recommended_autocast()}",
    ]
    rss, avail = mem_report()
    lines.append(f"  memory         : RSS {rss:.0f} MB, available {avail:.0f} MB")
    env = _state.get("env")
    if env:
        lines.append(f"  env applied    : {env}")
    else:
        lines.append("  env applied    : (none - call apply_env() BEFORE import torch)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 混合精度决策：AVX2-only 的 CPU 上 bf16 反而更慢（没有硬件 bf16 指令，
# 走软件模拟），所以只有 AVX512-BF16 / AMX 才推荐 bf16
# ---------------------------------------------------------------------------
def recommended_autocast():
    """返回推荐的 CPU autocast dtype：AVX512-BF16/AMX -> torch.bfloat16，
    否则 -> torch.float32（i5-10400 / R5-4500U 都会是 fp32）。"""
    import torch

    return torch.bfloat16 if _bf16_capable() else torch.float32


def autocast(dtype=None, enabled: bool = True):
    """返回 CPU autocast 上下文管理器::

        with tck.autocast():
            out = model(x)

    dtype 省略时用 recommended_autocast() 的建议（AVX2-only 机器上返回
    nullcontext，即 fp32 原样训练——这本身就是这类 CPU 的最优解）。
    """
    import torch

    dt = dtype or recommended_autocast()
    if dt == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cpu", dtype=dt, enabled=enabled)


# ---------------------------------------------------------------------------
# DataLoader / 模型布局
# ---------------------------------------------------------------------------
def fast_loader(dataset, batch_size: int = 32, **kwargs):
    """带 CPU 最优默认值的 DataLoader：

    - pin_memory=False（没有 GPU，pin 只是无谓的锁定内存页）；
    - num_workers 默认 2（训练循环里 fwd/bwd 已经吃满物理核，worker 太多
      反而和主进程抢核；再靠 prefetch 提前备好 batch）；
    - num_workers > 0 时自动 persistent_workers=True（省掉每 epoch 重启
      worker 的进程创建开销——Windows 上进程创建尤其贵）。

    其余参数原样透传给 torch.utils.data.DataLoader。
    """
    import torch

    kw = dict(kwargs)
    workers = kw.pop("num_workers", 2)
    kw.setdefault("pin_memory", False)
    if workers > 0:
        kw.setdefault("persistent_workers", True)
        kw.setdefault("prefetch_factor", 2)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       num_workers=workers, **kw)


def make_contiguous_(module):
    """把 module 的所有参数/缓冲区改成 C 连续布局（原地）。

    非连续参数（transpose 出来的等）每次前向都要隐式拷贝；训练前一次性
    转正，配合 bitsandbytes 的 8-bit 优化器（C 内核只走连续内存）尤其
    重要——非连续梯度会被自动回退到慢路径。返回 module 本身。
    """
    for p in list(module.parameters()) + list(module.buffers()):
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
    return module


# ---------------------------------------------------------------------------
# 内存监控（防 swap：available 快耗尽时就该减 batch / 序列长了）
# ---------------------------------------------------------------------------
def mem_report() -> tuple[float, float]:
    """返回 (进程RSS_MB, 系统可用内存_MB)。优先 psutil（跨平台且不依赖
    ctypes 的 GlobalMemoryStatusEx，后者在部分环境下返回 INVALID_PARAMETER），
    否则 Linux 走 /proc、Windows 走 GlobalMemoryStatusEx。训练循环里周期性
    打点，available 逼近 0 就是在 swap 拷打 SSD 的前兆。"""
    rss = avail = -1.0
    try:  # psutil：跨平台最稳（Windows/Linux 都是 C 扩展实现）
        import psutil
        rss = float(psutil.Process().memory_info().rss) / (1 << 20)
        avail = float(psutil.virtual_memory().available) / (1 << 20)
        return rss, avail
    except Exception:
        pass
    try:  # Linux RSS
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024
                    break
    except OSError:
        pass
    try:  # Linux available
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024
                    break
    except OSError:
        pass
    if rss < 0 or avail < 0:  # Windows / 兜底
        try:
            import ctypes

            class _MS(ctypes.Structure):
                # 完整 MEMORYSTATUSEX（缺 ullTotalVirtual/ullAvailVirtual 会让
                # GlobalMemoryStatusEx 因 dwLength 校验失败返回 FALSE）
                _fields_ = [
                    ("dwLength", ctypes.c_uint), ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ]

            ms = _MS(dwLength=ctypes.sizeof(_MS))
            k32 = ctypes.windll.kernel32
            k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MS)]
            k32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                if avail < 0:
                    avail = float(ms.ullAvailPhys) / (1 << 20)
                if rss < 0:  # Windows 无 /proc，用工作集近似
                    import ctypes.wintypes as wt

                    class _PMC(ctypes.Structure):
                        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                                    ("PeakWorkingSetSize", ctypes.c_size_t),
                                    ("WorkingSetSize", ctypes.c_size_t)] + \
                                   [(n, ctypes.c_size_t) for n in (
                                       "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                       "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                       "PagefileUsage", "PeakPagefileUsage")]

                    pmc = _PMC(cb=ctypes.sizeof(_PMC))
                    psapi = ctypes.windll.psapi
                    # 显式签名：默认 c_int 会截断 64 位 HANDLE/指针，导致调用静默失败
                    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
                    psapi.GetProcessMemoryInfo.restype = wt.BOOL
                    h = ctypes.windll.kernel32.GetCurrentProcess()
                    if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                        rss = float(pmc.WorkingSetSize) / (1 << 20)
        except Exception:
            pass
    return rss, avail


# ---------------------------------------------------------------------------
# 后台内存/swap 监视器（防 SSD 拷打）
# ---------------------------------------------------------------------------
_MON_STOP = threading.Event()


def start_mem_monitor(interval: float = 20.0) -> threading.Event:
    """启动后台线程，每 ``interval`` 秒打印一次物理内存占用 + 提交量。

    提交量（commit）逼近物理内存即进入 swap（页面文件）——"拷打 SSD"的信号。
    返回一个 Event，调用 ``suspend_mem_monitor(ev)`` 停止。

    用法::

        from torch_cpu_kit import start_mem_monitor, suspend_mem_monitor
        _ev = start_mem_monitor(interval=30)
        ... 训练循环 ...
        suspend_mem_monitor(_ev)
    """
    _MON_STOP.clear()

    def loop():
        while not _MON_STOP.is_set():
            try:
                try:  # psutil 优先（ctypes 的 GlobalMemoryStatusEx 部分环境失败）
                    import psutil
                    vm = psutil.virtual_memory()
                    tg, ug = vm.total / (1 << 30), (vm.total - vm.available) / (1 << 30)
                    sm = psutil.swap_memory()
                    cg = sm.used / (1 << 30) if sm.total > 0 else ug  # 实际换页用量
                    flag = "  <-- SWAP! 降低batch/序列或检查页面文件" if cg > tg * 0.97 else ""
                    print(f"[tck.mem] RAM {ug:5.2f}/{tg:.2f} GB   swap {cg:6.2f} GB{flag}")
                except Exception:
                    import ctypes

                    class _MS(ctypes.Structure):
                        # 完整 MEMORYSTATUSEX（缺字段 -> GlobalMemoryStatusEx 返回 FALSE）
                        _fields_ = [("l", ctypes.c_uint), ("m", ctypes.c_uint),
                                    ("t", ctypes.c_uint64), ("a", ctypes.c_uint64),
                                    ("tp", ctypes.c_uint64), ("ap", ctypes.c_uint64),
                                    ("tv", ctypes.c_uint64), ("av", ctypes.c_uint64)]

                    ms = _MS(l=ctypes.sizeof(_MS))
                    k32 = ctypes.windll.kernel32
                    k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MS)]
                    k32.GlobalMemoryStatusEx.restype = ctypes.c_int
                    if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                        tg, ug = ms.t / (1 << 30), (ms.t - ms.a) / (1 << 30)
                        cg = (ms.tp - ms.ap) / (1 << 30)
                        flag = "  <-- SWAP! 降低batch/序列或检查页面文件" if cg > tg * 0.97 else ""
                        print(f"[tck.mem] RAM {ug:5.2f}/{tg:.2f} GB   commit {cg:6.2f} GB{flag}")
            except Exception:
                pass
            _MON_STOP.wait(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return _MON_STOP


def suspend_mem_monitor(ev: Optional[threading.Event] = None) -> None:
    """停止 ``start_mem_monitor`` 启动的监视线程。"""
    (ev or _MON_STOP).set()


if __name__ == "__main__":  # 自检：python torch_cpu_kit.py
    apply_env(verbose=True)
    import torch

    init(verbose=True)
    ds = torch.utils.data.TensorDataset(torch.randn(64, 8), torch.randn(64))
    dl = fast_loader(ds, batch_size=16)
    for xb, yb in dl:
        pass
    with autocast():
        y = torch.nn.Linear(8, 8)(torch.randn(4, 8))
    assert y.shape == (4, 8)
    print("[torch_cpu_kit] selftest OK")
