# -*- coding: utf-8 -*-
"""gpu_scheduler.py — 核显(DirectML) + CPU 智能算子调度（实验性，默认不启用）。

本模块并不"把模型搬到核显"，而是判断**哪些算子搬去核显算值得、哪些留 CPU**，
只在净收益为正时往返，规避共享内存带宽瓶颈。

原理：核显用系统共享内存（无独立显存），GPU<->CPU 搬运数据走同一条内存通路。
  - 大矩阵 GEMM：搬一次算很久，搬运成本被摊薄 -> 核显正收益；
  - 小矩阵/逐元素算子：搬运成本 >= 计算量 -> 无收益，必须留 CPU。

调度判据（GEMM）：以「计算量/搬运量」之比 M*N/(M+N)（M=批量维度、N=输出维度）
衡量一次搬运的最大摊薄程度。默认阈值 1024 等价于 2048² 大 GEMM；批量小、权重大的
GEMV（如 batch=1 的注意力投影）远低于该值，自动留在 CPU。

环境变量：
  GPU_SCHED_RATIO     覆盖 GEMM 调度比率阈值（默认 1024）
  GPU_SCHED_CONV_MIN  覆盖卷积调度元素数阈值（默认 8,000,000）
  GPU_SCHED_MAX_TOKENS 核显执行器每步最大 token 数（batch*seq，默认 256）
  GPU_SCHED_MEM_MB    核显执行器允许的最大 fp32 基座内存（默认按整机内存自适应：
                      16GB→7000MB、12GB→5250MB、8GB→3500MB，显式设置则覆盖）
  GPU_SCHED_CALIB=0   跳过执行器启动时的核显 GEMM 收益校准（默认 1 = 校准）
  GPU_SCHED_CALIB_MIN 校准门槛：核显/CPU 吞吐比低于该值自动回退纯 CPU（默认 1.10）
  GPU_SCHED_DISABLE=1 整体关闭调度（等价于不启用本模块）

环境隔离（重要）：主训练环境（torch ≥2.6）与 torch_directml（钉死 torch==2.4.1）
彼此不兼容，同环境安装会互相破坏（降级 torch 打断 diffusers）。跑 --igpu 请用
独立 DML venv（.venv_dml，torch 2.4.1 + torch_directml），主环境保持纯净；
在主环境执行 --igpu 会得到 venv 引导提示并自动回退纯 CPU，无害。

换机自适应（Intel UHD 630 等）：不同核显算力差异极大（AMD Vega 6 实测 GEMM 1.30x，
i5 的 R5 M240 实测慢 3.4×），因此执行器在 prepare() 时用一次流水线 GEMM 校准实测
本机收益（calibrate_gpu()），低于门槛自动回退并打印实测值，无需逐台手调参数。

实测边界（重要，详见 docs_cpu/TECH_REPORT.md §3.7）：单次大 GEMM 调度有 1.1~1.4×
加速；但完整训练循环（前向+反向每层的 CPU<->核显搬运）在无独立显存的共享内存
机型上为**负优化**。默认关闭，仅在明确测试过本机收益时使用。
"""

import os
import time

import torch
import torch.nn as nn

# 检测 DirectML 是否可用
_dml = None
_dml_device = None
_dml_name = None
_dml_import_err = None
try:
    import torch_directml as _dml
    if _dml.device_count() > 0:
        _dml_device = _dml.device(0)
        try:
            _dml_name = str(_dml.device_name(0)).strip("\x00 \t\r\n")
        except Exception:
            _dml_name = "DirectML 设备"
        _dml = True
except Exception as _e:
    _dml = None
    _dml_import_err = _e


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


_DISABLE = os.environ.get("GPU_SCHED_DISABLE", "") == "1"
# GEMM：M*N/(M+N) >= 阈值才值得往返（2048²x2048² 纯 GEMM 该值约为 1024）
_GEMM_MIN_RATIO = _env_float("GPU_SCHED_RATIO", 1024.0)
# 卷积：总元素数 >= 阈值才考虑核显（保守，小卷积一律留 CPU）
_CONV_MIN_ELEMS = _env_int("GPU_SCHED_CONV_MIN", 8_000_000)


def iGPU_available() -> bool:
    """DirectML 核显是否可用。"""
    return _dml is not None and _dml_device is not None


def iGPU_name() -> str:
    """DirectML 设备名称（如 AMD Radeon(TM) Graphics / Intel UHD 630）。"""
    return _dml_name if _dml_name else ""


def _is_software_adapter() -> bool:
    """是否为软件适配器（Microsoft Basic Render Driver，无硬件加速）。"""
    name = iGPU_name().lower()
    return "basic render" in name or "microsoft basic" in name


def _worth_gemm(m: int, n: int) -> bool:
    """按「计算量/搬运量」之比判断某 GEMM 是否值得往返核显。

    m = 批量维度（x 的 M），n = 输出维度（w 的 rows）。比值越大，搬运成本越被摊薄。
    """
    if m <= 0 or n <= 0:
        return False
    return (m * n) / (m + n) >= _GEMM_MIN_RATIO


def _to_dml(t: torch.Tensor) -> torch.Tensor:
    """搬到核显并保证连续（DirectML 算子要求连续张量）。"""
    d = t.to(_dml_device)
    return d.contiguous() if not d.is_contiguous() else d


def big_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM：a @ b。大矩阵走核显，小矩阵留 CPU（自动，无需改调用方）。"""
    if ((not _DISABLE) and iGPU_available() and a.dtype == torch.float32
            and a.dim() == 2 and b.dim() == 2):
        m, _k = a.shape
        n = b.shape[0]
        if _worth_gemm(m, n):
            try:
                return (_to_dml(a) @ _to_dml(b)).to("cpu")
            except Exception:
                pass  # 任何 DirectML 失败都回退 CPU
    return a @ b


def big_linear(x: torch.Tensor, w: torch.Tensor, bias=None) -> torch.Tensor:
    """Linear：x @ w^T + bias。批量×输出维度足够大才走核显。"""
    if ((not _DISABLE) and iGPU_available() and x.dtype == torch.float32
            and x.dim() >= 2 and w.dim() == 2):
        m = x.numel() // x.shape[-1]
        n = w.shape[0]
        if _worth_gemm(m, n):
            try:
                out = _to_dml(x) @ _to_dml(w).t()
                if bias is not None:
                    out = out + _to_dml(bias)
                return out.to("cpu")
            except Exception:
                pass
    return nn.functional.linear(x, w, bias)


def big_conv2d(x: torch.Tensor, w: torch.Tensor, bias=None,
               stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor:
    """Conv2d。仅当输入+权重总元素数足够大（计算密集、摊薄搬移）才走核显；
    小卷积（如生图 UNet 的 128² 层）搬移成本>计算量，核显负收益，回退 CPU。"""
    if ((not _DISABLE) and iGPU_available() and x.dtype == torch.float32
            and (x.numel() + w.numel()) >= _CONV_MIN_ELEMS):
        try:
            return nn.functional.conv2d(
                _to_dml(x), _to_dml(w), None if bias is None else _to_dml(bias),
                stride=stride, padding=padding, dilation=dilation, groups=groups).to("cpu")
        except Exception:
            pass
    return nn.functional.conv2d(x, w, bias, stride=stride, padding=padding,
                                dilation=dilation, groups=groups)


def auto(fn_cpu, fn_gpu, m: int, n: int):
    """通用调度：M*N/(M+N)>=阈值走 fn_gpu，否则 fn_cpu。返回 CPU 张量。"""
    if (not _DISABLE) and iGPU_available() and _worth_gemm(m, n):
        try:
            return fn_gpu()
        except Exception:
            pass
    return fn_cpu()


# ---------------------------------------------------------------------------
# 模型级接入：把 nn.Linear / nn.Conv2d 替换为调度子类（保留参数与 state_dict）
# ---------------------------------------------------------------------------

class ScheduledLinear(nn.Linear):
    """nn.Linear 子类：前向自动走 big_linear（大矩阵核显、小矩阵 CPU）。

    参数对象与 state_dict 键与 nn.Linear 完全一致，PEFT 的 isinstance 检测
    与 LoRA 注入不受影响。
    """

    def forward(self, x):
        return big_linear(x, self.weight, self.bias)


class ScheduledConv2d(nn.Conv2d):
    """nn.Conv2d 子类：前向自动走 big_conv2d。非 'zeros' padding 模式保持原生行为。"""

    def forward(self, x):
        if self.padding_mode != "zeros":
            return super().forward(x)
        return big_conv2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)


def patch_igpu(model) -> int:
    """递归替换模块树中的 nn.Linear / nn.Conv2d 为核显调度子类。

    只替换**精确类型**为 nn.Linear / nn.Conv2d 的层（自定义子类不受影响）。
    权重/偏置保持原 Parameter 对象，state_dict 键不变，模型可正常保存/加载。
    无核显或 GPU_SCHED_DISABLE=1 时空操作。返回替换层数。
    """
    if not iGPU_available() or _DISABLE:
        return 0
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                new = ScheduledLinear(child.in_features, child.out_features,
                                      bias=child.bias is not None)
                new.weight = child.weight
                new.bias = child.bias
                setattr(parent, name, new)
                count += 1
            elif type(child) is nn.Conv2d:
                new = ScheduledConv2d(child.in_channels, child.out_channels,
                                      child.kernel_size, child.stride, child.padding,
                                      child.dilation, child.groups,
                                      child.bias is not None, child.padding_mode)
                new.weight = child.weight
                new.bias = child.bias
                setattr(parent, name, new)
                count += 1
    return count


# ---------------------------------------------------------------------------
# 块级 DML 常驻执行器（train.py --igpu 的实际路径）
# ---------------------------------------------------------------------------


def _total_ram_mb():
    """整机物理内存（MB）；查询失败返回 None（默认值回退 7000MB）。"""
    try:
        import psutil
        return int(psutil.virtual_memory().total // (1 << 20))
    except Exception:
        pass
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MemStatus()
        ms.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return int(ms.ullTotalPhys // (1 << 20))
    except Exception:
        pass
    return None


_RAM_MB = _total_ram_mb()
_MAX_TOKENS = _env_int("GPU_SCHED_MAX_TOKENS", 256)
# 基座内存上限默认按整机内存等比缩放（16GB→7000，12GB→5250，8GB→3500）；
# 核显为共享内存，DML 运行时另有约 0.5~1GB 开销，小内存机型必须收紧。
_MAX_MEM_MB = _env_int("GPU_SCHED_MEM_MB", 0) or (
    min(7000, _RAM_MB * 7 // 16) if _RAM_MB else 7000)
_CALIB_DEFAULT = os.environ.get("GPU_SCHED_CALIB", "1") != "0"
_CALIB_MIN = _env_float("GPU_SCHED_CALIB_MIN", 1.10)


def calibrate_gpu(size=2048, iters=5, warmup=4) -> float:
    """实测本机「核显流水线 GEMM vs CPU GEMM」吞吐比（换机自适应判据）。

    方法与 TECH_REPORT §3.7 的实测纪律一致：CPU 侧直接计时；DML 侧流水线入队，
    仅在末尾把结果拉回 CPU 一次（torch_directml 每次 .to("cpu") 都是整队列排空，
    必须只做一次，否则测出来的是排空开销）。返回 CPU 耗时 / DML 耗时，>1 表示
    核显更快。任何失败返回 0.0（视为无收益）。

    稳定性：单轮校准实测有 ±0.1 抖动（核显空闲降频、CPU 睿频状态都会影响，
    同机曾测得 1.20x 与 1.06x 的波动），故预热加长让核显爬频，并跑两轮取
    较好值——代表长时间训练能达到的稳态，避免本机有收益却被一次冷启动误拒。
    """
    if not iGPU_available():
        return 0.0
    try:
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        ad = a.to(_dml_device)
        bd = b.to(_dml_device)
        best = 0.0
        for _round in range(2):
            for _ in range(warmup):
                _ = a @ b
                _ = ad @ bd
            r = ad @ bd
            _ = r.cpu()  # 预热后排空一次
            t0 = time.perf_counter()
            for _ in range(iters):
                r = a @ b
            t_cpu = (time.perf_counter() - t0) / iters
            t0 = time.perf_counter()
            for _ in range(iters):
                r = ad @ bd
            _ = r.cpu()  # 唯一一次排空
            t_dml = (time.perf_counter() - t0) / iters
            best = max(best, t_cpu / max(t_dml, 1e-9))
        return best
    except Exception:
        return 0.0


class IgpuExecutor:
    """块级核显常驻执行器：全部权重搬入 DirectML 常驻，每步只同步一次。

    实测基础（R5-4500U，详见 docs_cpu/TECH_REPORT.md §3.7）：
      - 大 GEMM 流水线（只末尾同步一次）DML 294.5 GFLOPS vs CPU 226.7 = 1.30x；
      - 真实 Qwen 结构全链（8 层 1024 宽）seq=128：1.35x；seq=512：0.70x
        （DML 的 F.sdpa 在长序列明显慢于 CPU，且 multi_head_attention 专用内核
        需要较新驱动，本机 27.20.11032 不支持）；
      - 关键：torch_directml 每次 .to("cpu")/item() 都是一次整队列排空（~10-15ms），
        所以必须「每步一次同步」，逐层/逐算子同步会负收益。

    适用条件（不满足时 prepare() 返回原因，调用方回退纯 CPU）：
      - 模型全部参数为 fp32（量化基座不支持）；
      - 基座内存 <= 上限（默认按整机内存自适应：16GB→7000、12GB→5250、8GB→3500）；
      - 每步 token 数（batch*seq）<= GPU_SCHED_MAX_TOKENS（默认 256）；
      - 启动时 GEMM 校准核显/CPU 吞吐比 >= GPU_SCHED_CALIB_MIN（默认 1.10；
        i5 的 R5 M240 实测慢 3.4× 会被拒，弱核显开 --igpu 无害只是不生效；
        GPU_SCHED_CALIB=0 可跳过）。

    可训练参数（LoRA 适配器等）：梯度每步拷回 CPU 供 bnb 8bit 优化器使用，
    更新后的权重再拷回核显（量小，开销可忽略）。
    """

    def __init__(self, model, max_tokens=None, max_mem_mb=None,
                 calibrate=None, calib_min=None):
        self.model = model
        self.max_tokens = _MAX_TOKENS if max_tokens is None else max_tokens
        self.max_mem_mb = _MAX_MEM_MB if max_mem_mb is None else max_mem_mb
        self.calibrate = _CALIB_DEFAULT if calibrate is None else bool(calibrate)
        self.calib_min = _CALIB_MIN if calib_min is None else float(calib_min)
        self.calib_ratio = None
        self.dev = None
        self.ok = False
        self.reason = ""
        self.trainable = []
        self.cpu_mirror = []

    def prepare(self, tokens=None) -> str:
        """执行前置检查并迁移模型；返回状态描述（"OK" 表示成功）。"""
        if not iGPU_available():
            if _dml_import_err is not None and "torch_directml_native" in str(_dml_import_err):
                _here = os.path.dirname(os.path.abspath(__file__))
                _wroot = os.path.dirname(_here)  # 拷贝根目录（bnbc 仓库的上级）
                _venv_py = os.path.join(_wroot, ".venv_dml", "Scripts", "python.exe")
                self.reason = (
                    "torch_directml 无法导入（torch 版本不匹配）。主训练环境（torch "
                    "≥2.6）与 torch_directml（钉死 torch==2.4.1）彼此不兼容，存在降级风险。"
                    "请切换到独立 DML venv 跑 --igpu 实验，例如：\n"
                    f"  {_venv_py} train.py --igpu ..."
                )
            else:
                self.reason = ("未检测到 DirectML 核显（需安装 torch_directml 且 GPU "
                               "驱动支持 D3D12）")
            return self.reason
        if _is_software_adapter():
            self.reason = (f"检测到软件适配器 {iGPU_name()}（未安装/启用厂商图形驱动），"
                           "核显执行器不可用")
            return self.reason
        if tokens is not None and tokens > self.max_tokens:
            self.reason = (f"batch*seq={tokens} > GPU_SCHED_MAX_TOKENS={self.max_tokens}"
                           "（实测长序列 DML 注意力负收益，回退纯 CPU）")
            return self.reason
        mem_mb = sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024
        if mem_mb > self.max_mem_mb:
            self.reason = f"基座 fp32 内存约 {mem_mb:.0f}MB > {self.max_mem_mb}MB"
            return self.reason
        for p in self.model.parameters():
            if p.dtype != torch.float32:
                self.reason = "存在非 fp32 参数（量化基座暂不支持核显执行器）"
                return self.reason
        if self.calibrate:
            # 换机自适应：不同核显算力差异极大（Vega 6 实测 1.30x，R5 M240 慢 3.4×），
            # 用一次流水线 GEMM 校准实测本机收益，不达标直接拒绝，避免负优化。
            try:
                self.calib_ratio = calibrate_gpu()
            except Exception as e:
                self.reason = f"核显校准失败（{type(e).__name__}: {e}），无法确认收益"
                return self.reason
            if self.calib_ratio < self.calib_min:
                if self.calib_ratio <= 0.01:
                    self.reason = "校准失败（核显 GEMM 无法执行，驱动可能异常），回退纯 CPU"
                else:
                    self.reason = (f"校准：核显 GEMM 仅为 CPU 的 {self.calib_ratio:.2f}x "
                                   f"< {self.calib_min:.2f}（本机核显无加速收益，回退纯 CPU；"
                                   f"如确需启用可调低 GPU_SCHED_CALIB_MIN 或 GPU_SCHED_CALIB=0）")
                return self.reason
        try:
            self.dev = _dml_device
            self.trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.cpu_mirror = [p.detach().clone().requires_grad_(True)
                               for p in self.trainable]
            self.model.to(self.dev)
            self.ok = True
            return "OK"
        except Exception as e:
            # 迁移中途失败（如大模型搬运 OOM）：model.to() 可能已把一部分参数搬到核显，
            # 若只设 reason 就返回，调用方拿到的模型是半 CPU 半核显的异构设备，后续纯
            # CPU 前向会崩。逐参数检查并把非 CPU 的搬回 CPU，确保回退后模型是纯 CPU。
            try:
                for p in self.model.parameters():
                    if p.device.type != "cpu":
                        p.data = p.data.cpu()
                # 若还有在核显上的可训练镜像/梯度，一并清理
                self.cpu_mirror = []
                self.trainable = []
            except Exception as rollback_e:
                print(f"[gpu_scheduler] 迁移失败回滚时又出错: {rollback_e}", flush=True)
            self.reason = f"迁移到核显失败: {type(e).__name__}: {e}（已回滚到纯 CPU）"
            return self.reason

    def grad_to_cpu(self):
        """把可训练参数梯度拷回 CPU 镜像（供优化器使用），并清空核显侧梯度。"""
        for p, pc in zip(self.trainable, self.cpu_mirror):
            pc.grad = None
            if p.grad is not None:
                pc.grad = p.grad.detach().to("cpu")
                p.grad = None

    def weights_from_cpu(self):
        """优化器更新后，把 CPU 镜像权重拷贝回核显。"""
        for p, pc in zip(self.trainable, self.cpu_mirror):
            p.data.copy_(pc.data.to(self.dev))

    def description(self) -> str:
        n_param = sum(p.numel() for p in self.model.parameters()) / 1e6
        gain = f"，校准 {self.calib_ratio:.2f}x" if self.calib_ratio else ""
        return (f"{iGPU_name() or 'DirectML'} | {n_param:.0f}M 参数常驻核显，"
                f"可训练 {len(self.trainable)} 组，每步 token <= {self.max_tokens}，"
                f"每步同步一次{gain}")


if __name__ == "__main__":
    print(f"DirectML iGPU 可用: {iGPU_available()}  [{iGPU_name() or '未知设备'}]")

    def bench(fn, iters=5):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000

    if iGPU_available():
        # 大 GEMM：调度后应走核显（净赚）
        a = torch.randn(2048, 2048)
        b = torch.randn(2048, 2048)
        ts = bench(lambda: big_gemm(a, b))
        tc = bench(lambda: a @ b)
        print(f"  大GEMM 2048: 调度={ts:.1f}ms vs CPU={tc:.1f}ms  "
              f"({'核显快' + f'{tc/ts:.2f}x' if ts < tc else '核显无优势,已回退'})")
        # 小卷积：应回退 CPU（核显负收益）
        x = torch.randn(1, 64, 128, 128)
        w = torch.randn(64, 64, 3, 3)
        ts = bench(lambda: big_conv2d(x, w, padding=1))
        tc = bench(lambda: nn.functional.conv2d(x, w, padding=1))
        print(f"  小卷积128^2: 调度={ts:.1f}ms vs CPU={tc:.1f}ms  "
              f"({'已回退CPU' if abs(ts - tc) < 1 else '核显'})")
        # 正确性：调度结果 vs 纯 CPU
        a2 = torch.randn(4096, 4096)
        b2 = torch.randn(4096, 4096)
        out_s = big_gemm(a2, b2)
        out_c = a2 @ b2
        print(f"  调度正确性: max|Δ|={(out_s - out_c).abs().max().item():.2e}")

    # 模型级 patch：替换计数 / state_dict 键一致 / 前向反向正常
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 32)
            self.cv1 = nn.Conv2d(3, 8, 3, padding=1)

    m0 = _Tiny()
    keys0 = list(m0.state_dict().keys())
    n = patch_igpu(m0)
    keys1 = list(m0.state_dict().keys())
    ok = keys0 == keys1
    x_in = torch.randn(2, 64)
    y = m0.fc1(x_in)
    y.sum().backward()
    grad_ok = m0.fc1.weight.grad is not None
    print(f"  patch_igpu: 替换 {n} 层; state_dict 键一致={ok}; "
          f"fc1 类型={type(m0.fc1).__name__}; 前向/反向梯度={grad_ok}")
    if iGPU_available():
        ratio = calibrate_gpu()
        verdict = ("达标，--igpu 执行器可用" if ratio >= _CALIB_MIN
                   else f"低于门槛 {_CALIB_MIN:.2f}，--igpu 将自动回退纯 CPU")
        print(f"  执行器校准: 核显/CPU GEMM = {ratio:.2f}x（{verdict}）")
    print("  提示：换机自查直接运行 `python gpu_scheduler.py`；DirectML 接口通用，"
          "收益由校准自动判定；主环境无 torch_directml 时请用 .venv_dml 运行本自检；"
          "若显示软件适配器请先安装厂商图形驱动。")
