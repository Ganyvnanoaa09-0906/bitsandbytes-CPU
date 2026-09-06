"""disk_balancer - 硬盘均衡负载 / 冷参数卸载（CPU 训练用）。

目标：
- 对抗 Windows 虚拟内存在 AI 训练时频繁擦写导致硬盘活动时间 100% 的问题；
- 异步写盘线程（queue.Queue），不阻塞训练循环；
- 内存紧张时自动检测冻结层（冷参数），卸载到磁盘释放 RAM；
- 多盘分散存储：SSD 优先放热数据，HDD 放冷数据；
- 读回使用 mmap，零拷贝；
- 优先保护硬盘寿命，同时不拖累 CPU 训练速度；
- 与 EFST / GDN / torch_cpu_kit / lowmem 配合。

典型用法::

    from disk_balancer import DiskBalancerConfig, DiskLoadBalancer

    # 自动模式（推荐）
    balancer = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))
    balancer.attach_model(model)  # 注册模型，自动检测冷参数
    balancer.start()

    # 训练循环中
    for step, batch in enumerate(loader):
        balancer.update_step()   # 检查内存，自动迁移冷参数到磁盘
        loss = model(**batch).loss
        loss.backward()
        opt.step()

    balancer.cleanup()   # 训练结束删除缓存

    # 手动模式
    balancer = DiskLoadBalancer(DiskBalancerConfig(
        mode="manual",
        sizes={"C:": 2048, "D:": 4096},  # 每盘缓存 MB
    ))

    # 速度优先模式（不管硬盘死活）
    balancer = DiskLoadBalancer(DiskBalancerConfig(
        mode="auto",
        speed_first=True,
    ))

API 触发方式（在训练脚本中）:
    --flash          # 启用硬盘均衡负载（自动模式，默认保护硬盘策略）
    --flash a        # 同 --flash，自动模式
    --flash manual   # 手动模式，需配合 --flash_sizes
    --flash true     # 同 --flash，启用
    --flash_sizes C:2048,D:4096,E:8192   # 手动指定每盘缓存大小(MB)
    --flash_paths C:\cache,D:\cache,E:\cache   # 指定缓存路径（默认脚本同目录）
    --flash_speed    # 速度优先模式（不管硬盘死活）
    --flash_keep     # 训练结束后保留缓存文件
"""

from __future__ import annotations

import ctypes
import mmap
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

__all__ = [
    "DiskBalancerConfig",
    "DiskLoadBalancer",
    "DiskInfo",
    "detect_disks",
    "get_disk_activity",
    "add_flash_args",
    "parse_flash_args",
]

# ---------------------------------------------------------------------------
# 磁盘信息
# ---------------------------------------------------------------------------


@dataclass
class DiskInfo:
    """单块磁盘的信息。"""

    mount: str            # 盘符，如 "C:" / "D:"
    drive_type: str       # "ssd" / "hdd" / "nvme" / "unknown"
    total_bytes: int = 0
    free_bytes: int = 0
    max_cache_bytes: int = 0  # 该盘允许的最大缓存字节数（0=自动）
    cache_dir: str = ""       # 缓存目录路径
    _io_counters: Optional[Tuple[int, int, float]] = None  # (read, write, ts)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1 << 30)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1 << 30)

    @property
    def max_cache_gb(self) -> float:
        return self.max_cache_bytes / (1 << 30)

    def __repr__(self) -> str:
        return (f"DiskInfo({self.mount}, {self.drive_type}, "
                f"total={self.total_gb:.1f}GB, free={self.free_gb:.1f}GB, "
                f"cache={self.max_cache_gb:.1f}GB)")


# ---------------------------------------------------------------------------
# Windows 磁盘检测（纯 ctypes，零额外依赖）
# ---------------------------------------------------------------------------

def _win_drive_type(mount: str) -> str:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetDriveTypeW.restype = ctypes.c_uint
    dt = kernel32.GetDriveTypeW(mount + "\\")
    return "fixed" if dt == 3 else "removable" if dt == 2 else "remote" if dt == 4 else "other"


def _win_disk_free(mount: str) -> Tuple[int, int]:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetDiskFreeSpaceExW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    kernel32.GetDiskFreeSpaceExW.restype = ctypes.c_bool
    free = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    if kernel32.GetDiskFreeSpaceExW(mount + "\\", None, ctypes.byref(total), ctypes.byref(free)):
        return total.value, free.value
    return 0, 0


def _win_is_ssd(mount: str) -> str:
    """通过 DeviceIoControl 查询磁盘类型。返回 "nvme" / "ssd" / "hdd"。

    方法：先查 STORAGE_ADAPTER_DESCRIPTOR.BusType==17(NVMe)，
    再查 DEVICE_TRIM_DESCRIPTOR.TrimEnabled(SSD)，否则 HDD。
    """
    path = f"\\\\.\\{mount[0]}:"
    # 先查 NVMe
    try:
        handle = ctypes.windll.kernel32.CreateFileW(path, 0, 3, None, 3, 0, None)
        if handle is not None and handle != -1:
            class STORAGE_PROPERTY_QUERY(ctypes.Structure):
                _fields_ = [("PropertyId", ctypes.c_int), ("QueryType", ctypes.c_int),
                            ("AdditionalParameters", ctypes.c_byte * 4)]

            class STORAGE_ADAPTER_DESCRIPTOR(ctypes.Structure):
                _fields_ = [
                    ("Version", ctypes.c_uint), ("Size", ctypes.c_uint),
                    ("MaximumTransferLength", ctypes.c_uint),
                    ("MaximumPhysicalPages", ctypes.c_uint),
                    ("AlignmentMask", ctypes.c_uint),
                    ("AdapterUsesPio", ctypes.c_bool), ("AdapterScansDown", ctypes.c_bool),
                    ("CommandQueueing", ctypes.c_bool), ("AcceleratedTransfer", ctypes.c_bool),
                    ("BusType", ctypes.c_byte), ("BusMajorVersion", ctypes.c_ushort),
                    ("BusMinorVersion", ctypes.c_ushort),
                    ("CommandQueueingMax", ctypes.c_longlong),
                ]

            query = STORAGE_PROPERTY_QUERY(0, 0, (0, 0, 0, 0))
            out = ctypes.create_string_buffer(1024)
            ret = ctypes.c_ulong(0)
            try:
                if ctypes.windll.kernel32.DeviceIoControl(
                        handle, 0x002D1400, ctypes.byref(query), ctypes.sizeof(query),
                        out, ctypes.sizeof(out), ctypes.byref(ret), None):
                    desc = STORAGE_ADAPTER_DESCRIPTOR.from_buffer_copy(out)
                    if desc.BusType == 17:  # NVMe
                        return "nvme"
            finally:
                # 确保句柄被关闭，即使 from_buffer_copy/DeviceIoControl 抛异常也不泄漏
                ctypes.windll.kernel32.CloseHandle(handle)
    except OSError:
        pass

    # 查 Trim（SSD 标志）
    try:
        handle = ctypes.windll.kernel32.CreateFileW(path, 0, 3, None, 3, 0, None)
        if handle is not None and handle != -1:
            class STORAGE_PROPERTY_QUERY(ctypes.Structure):
                _fields_ = [("PropertyId", ctypes.c_int), ("QueryType", ctypes.c_int),
                            ("AdditionalParameters", ctypes.c_byte * 4)]

            class DEVICE_TRIM_DESCRIPTOR(ctypes.Structure):
                _fields_ = [("Version", ctypes.c_uint), ("Size", ctypes.c_uint),
                            ("TrimEnabled", ctypes.c_bool)]

            query = STORAGE_PROPERTY_QUERY(8, 0, (0, 0, 0, 0))
            buf_sz = ctypes.sizeof(DEVICE_TRIM_DESCRIPTOR)
            out = ctypes.create_string_buffer(buf_sz)
            ret = ctypes.c_ulong(0)
            if ctypes.windll.kernel32.DeviceIoControl(
                    handle, 0x002D1400, ctypes.byref(query), ctypes.sizeof(query),
                    out, buf_sz, ctypes.byref(ret), None):
                trim = DEVICE_TRIM_DESCRIPTOR.from_buffer_copy(out)
                if trim.TrimEnabled:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return "ssd"
            ctypes.windll.kernel32.CloseHandle(handle)
    except OSError:
        pass

    return "hdd"


def detect_disks() -> List[DiskInfo]:
    """检测系统所有可用固定磁盘，返回 DiskInfo 列表。

    自动识别 SSD/NVMe/HDD，获取容量和剩余空间。
    只返回固定磁盘（排除网络盘、光驱、可移动磁盘）。
    跨平台：Windows 用 ctypes（GetDriveTypeW），Linux/WSL 用 psutil.disk_partitions。
    """
    disks: List[DiskInfo] = []
    if os.name != "nt":  # Linux / WSL / macOS：psutil 跨平台路径
        import psutil as _ps
        for part in _ps.disk_partitions(all=False):
            # 【防炸机】WSL 会把宿主 NTFS 经 9P 挂成 /mnt/c、/mnt/d——9P（网络文件
            # 系统语义）在高负载下对 NTFS 做冷参数卸载会扰乱宿主 MFT（已实测炸机）。
            # 必须排除 9p/fuse/宿主 NTFS 挂载，绝不对它们卸载冷参数。
            if _is_wsl_host_mount(part):
                continue
            try:
                usg = _ps.disk_usage(part.mountpoint)
            except Exception:
                continue
            mount = part.mountpoint
            # Linux 不区分 removable/remote（psutil 的 fstype 判 ssd/hdd 不准确，
            # 退化按 SSD 处理——Linux 上卸载保护磁盘寿命的策略以 SSD 为准）
            drive_type = "ssd" if _fs_ssd_like(part.fstype) else "hdd"
            disks.append(DiskInfo(
                mount=mount,
                drive_type=drive_type,
                total_bytes=usg.total,
                free_bytes=usg.free,
            ))
        return disks

    # Windows：ctypes 路径（原逻辑）
    for letter in range(ord("A"), ord("Z") + 1):
        mount = f"{chr(letter)}:"
        if _win_drive_type(mount) != "fixed":
            continue
        total, free = _win_disk_free(mount)
        if total <= 0:
            continue
        disks.append(DiskInfo(
            mount=mount,
            drive_type=_win_is_ssd(mount),
            total_bytes=total,
            free_bytes=free,
        ))
    return disks


def _is_wsl_host_mount(part) -> bool:
    """判断一个 psutil.disk_partitions 项是否为「WSL 宿主 NTFS / 网络文件系统挂载」。

    在 WSL 里，宿主 Windows 的 NTFS 盘经 9P 协议挂在 /mnt/c、/mnt/d 等。9P 是网络
    文件系统语义，在高负载/极端内存压力下对 NTFS 做冷参数卸载（SSD/HDD 高频读写/删除）
    会扰乱宿主 MFT、导致数据无法落盘（已实测炸机）。disk_balancer 绝不该卸载它们。

    True = 是该排除的挂载（9P / fuse / 宿主 NTFS / 网络盘）→ detect_disks 跳过。
    """
    try:
        fs = (part.fstype or "").lower()
        mount = (part.mountpoint or "").lower()
        # 9P / fuse / NFS / CIFS（网络文件系统）—— 一律排除
        if any(t in fs for t in ("9p", "9pfs", "fuse", "nfs", "cifs", "smb", "vfat", "ntfs", "exfat", "fat")):
            return True
        # WSL 宿主挂载：/mnt/c、/mnt/d 等 —— 只匹配【单字母盘符】的 /mnt/[a-z]
        # （WSL 用 /mnt/c、/mnt/d 挂宿主 Windows 盘）。真实 Linux 用户常把数据盘挂到
        # /mnt/data、/mnt/sda1 这类【多字符】路径，那不是 WSL 宿主挂载，不应排除。
        if mount.startswith("/mnt/"):
            import re
            tail = mount[len("/mnt/"):]
            # 单字母盘符（如 c、d）→ WSL 宿主；多字符（data、sda1、wsl 等）→ 保留
            if re.match(r"^[a-z][a-z]?$", tail):
                return True
        elif mount == "/mnt":
            return True
        # 明显的网络/虚拟挂载
        if fs in ("fuseblk", "drvfs", "9p", "cif"):
            return True
    except Exception:
        pass
    return False


def _in_wsl() -> bool:
    """是否运行在 WSL（Windows Subsystem for Linux）宿主环境——此时均衡负载必须整体禁用。

    disk_balancer 只应在两处运行：真 Windows（os.name == "nt"，ctypes）或纯 Linux
    （非 WSL 宿主挂载，psutil）。WSL 是"从 Windows 挂进来的"——宿主 NTFS 经 9P/drvfs
    挂在 /mnt/c、/mnt/d 等，9P（网络文件系统语义）在高负载/极端内存压力下对 NTFS
    卸载冷参数会扰乱宿主 MFT、导致数据无法落盘（已实测炸机）。所以 WSL 环境一律跳过。

    识别 WSL（三重确认，防止误判）：
      1) /proc/version 含 "microsoft"/"WSL"（WSL 内核版本特征，最可靠）；
      2) psutil.disk_partitions 里存在 9P/宿主 NTFS 挂载（/mnt/c、/mnt/d、drvfs 等）；
      3) 环境变量 WSL_DISTRO_NAME / WSL_INTEROP 存在（WSL 特有）。

    Windows（os.name == "nt"）永远返回 False（不在 WSL）。
    """
    if os.name == "nt":
        return False
    # 1) WSL 内核版本标记（最可靠）
    try:
        with open("/proc/version", "r", errors="ignore") as f:
            ver = f.read().lower()
        if "microsoft" in ver or "wsl" in ver:
            return True
    except OSError:
        pass
    # 2) WSL 环境变量
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    # 3) psutil 检测到 9P/宿主 NTFS 挂载
    try:
        import psutil as _ps
        for part in _ps.disk_partitions(all=False):
            if _is_wsl_host_mount(part):
                return True
    except Exception:
        pass
    return False


def _fs_ssd_like(fstype: str) -> bool:
    """粗略判断文件系统/挂载点是否更像 SSD（Linux 无设备级 SSD 检测，用 psutil
    disk_io_counters 无法区分 ssd/hdd，这里用 fstype 兜底：NVMe/常见 SSD 挂载）。

    Linux 上无法 100% 识别设备类型，这里保守返回 True（按 SSD 策略走，
    卸载以保护磁盘寿命为优先）。psutil 也提供 disk_io_counters 按磁盘（perdisk），
    但拿不到物理介质类型。保持简单、安全。
    """
    return True


def _get_disk_io(disk: DiskInfo) -> Tuple[int, int, float]:
    """获取磁盘累计读写字节数。返回 (read_bytes, write_bytes, timestamp)。
    跨平台：Windows 用 DeviceIoControl（按盘符）；Linux/WSL 用 psutil.disk_io_counters
    （全局累计，足够 disk_balancer 估算活动率）。
    """
    if os.name != "nt":  # Linux / WSL：psutil 跨平台
        try:
            io = psutil.disk_io_counters(perdisk=False)  # 全局累计
            if io is not None:
                return io.read_bytes, io.write_bytes, time.time()
        except Exception:
            pass
        return 0, 0, time.time()

    # Windows：DeviceIoControl（原逻辑）
    class DISK_PERFORMANCE(ctypes.Structure):
        _fields_ = [
            ("BytesRead", ctypes.c_longlong), ("BytesWritten", ctypes.c_longlong),
            ("ReadTime", ctypes.c_longlong), ("WriteTime", ctypes.c_longlong),
            ("IdleTime", ctypes.c_longlong), ("ReadCount", ctypes.c_uint),
            ("WriteCount", ctypes.c_uint), ("QueueDepth", ctypes.c_uint),
            ("SplitCount", ctypes.c_uint), ("QueryTime", ctypes.c_longlong),
            ("StorageDeviceNumber", ctypes.c_int),
            ("StorageManagerName", ctypes.c_wchar * 8),
        ]

    try:
        path = f"\\\\.\\{disk.mount[0]}:"
        handle = ctypes.windll.kernel32.CreateFileW(path, 0, 3, None, 3, 0, None)
        if handle is None or handle == -1:
            return 0, 0, time.time()
        buf = ctypes.create_string_buffer(ctypes.sizeof(DISK_PERFORMANCE))
        ret = ctypes.c_ulong(0)
        if ctypes.windll.kernel32.DeviceIoControl(
                handle, 0x00070020, None, 0, buf, ctypes.sizeof(buf),
                ctypes.byref(ret), None):
            perf = DISK_PERFORMANCE.from_buffer_copy(buf)
            ctypes.windll.kernel32.CloseHandle(handle)
            return perf.BytesRead, perf.BytesWritten, time.time()
        ctypes.windll.kernel32.CloseHandle(handle)
    except OSError:
        pass
    return 0, 0, time.time()


def get_disk_activity(disk: DiskInfo) -> float:
    """获取磁盘当前活动率（估算值，0.0~1.0）。

    通过两次采样间的 I/O 增量估算。
    """
    r, w, now = _get_disk_io(disk)
    if disk._io_counters is None:
        disk._io_counters = (r, w, now)
        return 0.0
    prev_r, prev_w, prev_t = disk._io_counters
    disk._io_counters = (r, w, now)
    dt = now - prev_t
    if dt <= 0:
        return 0.0
    total = (r - prev_r) + (w - prev_w)
    if total <= 0:
        return 0.0
    rate = total / dt
    limit = 500_000_000 if disk.drive_type in ("ssd", "nvme") else 100_000_000
    return min(1.0, rate / limit)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class DiskBalancerConfig:
    """硬盘均衡负载配置。

    Attributes:
        mode: "auto" 自动检测磁盘并分配；"manual" 手动指定。
        sizes: 手动模式下每盘缓存大小，如 ``{"C:": 2048, "D:": 4096}``（MB）。
        paths: 指定缓存路径列表，如 ``["D:\\cache", "E:\\cache"]``。
            不指定则默认在脚本所在目录下创建 ``.flash_cache``。
        speed_first: True=速度优先（不管硬盘寿命），False=保护硬盘优先。
        keep_cache: True=训练结束后保留缓存文件，False=默认删除。
        memory_threshold: 内存使用率阈值（0.0~1.0），超过此值触发冷参数卸载。
            默认 0.8（可用内存 < 20% 时触发）。
        min_param_size: 最小的参数大小（元素数），低于此值的参数不卸载。
            默认 100000（约 400KB fp32）。
        throttle_threshold: 磁盘活动率阈值，超过此值暂停写入（0.0~1.0），默认 0.85。
        min_free_ratio: 磁盘剩余空间低于此比例时禁止写入，默认 0.05（5%）。
    """

    mode: str = "auto"
    sizes: Dict[str, int] = field(default_factory=dict)   # {盘符: MB}
    paths: List[str] = field(default_factory=list)         # 缓存路径列表
    speed_first: bool = False
    keep_cache: bool = False
    memory_threshold: float = 0.8       # 内存使用率 > 80% 触发卸载
    min_param_size: int = 100000        # 最小参数量（元素数）
    throttle_threshold: float = 0.85
    min_free_ratio: float = 0.05
    offload_prefix: str = ""            # 只卸载匹配前缀的冷参数（空=全部），如 "_cold_"

    def __post_init__(self):
        """校验关键配置，避免非法值导致静默禁用 / 负缓存 / None 崩。"""
        if self.memory_threshold is None:
            raise ValueError("memory_threshold 不能为 None（应为 0.0~1.0）")
        if not (0.0 <= self.memory_threshold <= 1.0):
            raise ValueError(
                f"memory_threshold 必须在 [0,1] 内，得到 {self.memory_threshold}"
                f"（>1 会永久禁用冷参数卸载，<0 会误判）")
        if self.min_param_size is None or self.min_param_size < 0:
            raise ValueError(
                f"min_param_size 必须是非负整数，得到 {self.min_param_size}")
        if self.min_free_ratio is None or not (0.0 <= self.min_free_ratio <= 1.0):
            raise ValueError(
                f"min_free_ratio 必须在 [0,1] 内，得到 {self.min_free_ratio}"
                f"（>1 会让可用缓存为负）")
        if self.throttle_threshold is None or not (0.0 <= self.throttle_threshold <= 1.0):
            raise ValueError(
                f"throttle_threshold 必须在 [0,1] 内，得到 {self.throttle_threshold}")


# ---------------------------------------------------------------------------
# 核心：DiskLoadBalancer
# ---------------------------------------------------------------------------


class DiskLoadBalancer:
    """硬盘均衡负载管理器。

    核心机制：
    1. **异步写盘线程**：后台 daemon 线程从 queue.Queue 取任务写盘，不阻塞训练循环。
    2. **冷参数卸载**：``update_step()`` 通过 psutil 监控内存，当使用率超过阈值时，
       自动遍历模型参数，找到 ``requires_grad=False`` 且大于 ``min_param_size`` 的
       冻结层，调用 ``put_cold()`` 写入磁盘并释放内存。
    3. **mmap 回读**：``get_cold()`` 使用 mmap 零拷贝读取，避免额外内存分配。
    4. **多盘分层**：SSD 路径在前（热数据），HDD 路径在后（冷数据）。

    自动模式策略：
    - 检测所有磁盘，按 SSD→HDD 排序路径列表
    - 按容量自动分配缓存配额（保留 min_free_ratio 安全空间）
    - 单一磁盘时：优先不拖慢训练速度，其次是保护硬盘

    用法::

        balancer = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))
        balancer.attach_model(model)
        balancer.start()

        for step in range(total_steps):
            balancer.update_step()        # 自动检测内存，迁移冷参数
            loss = model(**batch).loss
            loss.backward()
            opt.step()

        balancer.cleanup()
    """

    def __init__(self, config: Optional[DiskBalancerConfig] = None):
        self._cfg = config or DiskBalancerConfig()
        # 【WSL 环境整体禁用】在 WSL 里，宿主 NTFS 经 9P/drvfs 挂在 /mnt/c、/mnt/d 等。
        # 9P（网络文件系统语义）在高负载下对 NTFS 卸载冷参数会扰乱宿主 MFT（已实测炸机）。
        # 所以：当处于 WSL / 9P 宿主环境时，均衡负载整体跳过、忽略用户参数，不卸载任何冷参数。
        self._9p_skip = _in_wsl()
        self._disks: List[DiskInfo] = []
        self._paths: List[str] = []        # 排序后的缓存路径（SSD→HDD）
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._model: Optional[torch.nn.Module] = None
        self._cold_index: Dict[str, Tuple[str, int, torch.dtype]] = {}  # name -> (path, numel, dtype)
        self._migrated_count: int = 0
        self._started = False
        self._lock = threading.Lock()
        self._script_dir: str = ""
        self._ssd_paths: List[str] = []
        self._hdd_paths: List[str] = []

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def attach_model(self, model: torch.nn.Module) -> "DiskLoadBalancer":
        """注册模型，用于 update_step() 自动检测冷参数。

        必须在 start() 之前调用。
        """
        self._model = model
        return self

    def start(self, script_dir: Optional[str] = None) -> "DiskLoadBalancer":
        """初始化磁盘、创建缓存目录、启动异步写盘线程。

        Args:
            script_dir: 训练脚本所在目录。缓存路径默认在此目录下。
        """
        if getattr(self, "_9p_skip", False):
            print("[disk_balancer] 检测到 WSL 9P/宿主 NTFS 挂载，均衡负载已整体禁用——"
                  "为避免 9P 写宿主 NTFS 扰乱 MFT（会炸机），本次不卸载任何冷参数"
                  "（用户 --flash 参数已忽略）。建议在纯 Linux 分区使用才对磁盘做均衡。")
            return self

        if self._started:
            return self

        self._script_dir = script_dir or os.path.dirname(os.path.abspath(
            __import__("sys").argv[0] if __import__("sys").argv else __file__))

        if self._cfg.mode == "auto":
            self._start_auto()
        elif self._cfg.mode == "manual":
            self._start_manual()
        else:
            raise ValueError(f"Unknown mode: {self._cfg.mode}")

        # 启动异步写盘线程
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._started = True
        self._print_summary()
        return self

    def _start_auto(self):
        all_disks = detect_disks()
        if not all_disks:
            raise RuntimeError("disk_balancer: 未检测到任何可用磁盘")

        # 按 SSD→HDD 排序路径
        for d in all_disks:
            usable = int(d.free_bytes * (1.0 - self._cfg.min_free_ratio))
            d.max_cache_bytes = min(usable, 4 * (1 << 30))  # 上限 4GB

            if self._cfg.paths:
                for cp in self._cfg.paths:
                    if os.path.splitdrive(cp)[0].upper() == d.mount.upper():
                        d.cache_dir = cp
                        break
            if not d.cache_dir:
                d.cache_dir = os.path.join(self._script_dir, ".flash_cache", d.mount[0])

            if d.drive_type in ("ssd", "nvme"):
                self._ssd_paths.append(d.cache_dir)
            else:
                self._hdd_paths.append(d.cache_dir)

        self._disks = all_disks
        # SSD 在前，HDD 在后
        self._paths = self._ssd_paths + self._hdd_paths

    def _start_manual(self):
        if not self._cfg.sizes:
            raise ValueError("disk_balancer: manual 模式需要指定 sizes")

        all_disks = detect_disks()
        disk_map = {d.mount: d for d in all_disks}

        for mount, size_mb in self._cfg.sizes.items():
            m = mount.upper()
            if m not in disk_map:
                raise ValueError(f"disk_balancer: 磁盘 {m} 不存在")

            d = disk_map[m]
            d.max_cache_bytes = size_mb * (1 << 20)
            if d.max_cache_bytes > d.free_bytes * (1.0 - self._cfg.min_free_ratio):
                raise ValueError(
                    f"disk_balancer: {m} 缓存 {size_mb}MB 超过可用空间 ({d.free_gb:.1f}GB)")

            if self._cfg.paths:
                for cp in self._cfg.paths:
                    if os.path.splitdrive(cp)[0].upper() == m:
                        d.cache_dir = cp
                        break
            if not d.cache_dir:
                d.cache_dir = os.path.join(self._script_dir, ".flash_cache", m[0])

            if d.drive_type in ("ssd", "nvme"):
                self._ssd_paths.append(d.cache_dir)
            else:
                self._hdd_paths.append(d.cache_dir)

            self._disks.append(d)

        if not self._disks:
            raise RuntimeError("disk_balancer: manual 模式下没有有效的磁盘配置")

        self._paths = self._ssd_paths + self._hdd_paths

    def _ensure_dirs(self):
        for p in self._paths:
            os.makedirs(p, exist_ok=True)

    def _print_summary(self):
        lines = ["[disk_balancer] 启动:"]
        lines.append(f"  模式: {self._cfg.mode}, 速度优先: {self._cfg.speed_first}")
        lines.append(f"  内存阈值: {self._cfg.memory_threshold * 100:.0f}%")
        for d in self._disks:
            m = "SSD" if d.drive_type in ("ssd", "nvme") else "HDD"
            lines.append(f"  [{d.mount}] {m} 缓存 {d.max_cache_gb:.1f}GB @ {d.cache_dir}")
        lines.append(f"  路径优先级: {self._paths}")
        if self._cfg.speed_first:
            lines.append("  ⚠ 速度优先模式：不保护硬盘，不节流")
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------
    # 异步写盘线程
    # ------------------------------------------------------------------

    def _writer_loop(self):
        """后台写盘线程：从 queue 取任务，写入磁盘。"""
        while True:
            try:
                key, data, path = self._write_queue.get()
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(data)
                except Exception as e:
                    # 写盘失败：不能静默丢数据 —— 从索引摘除该 key（内存已释放，必须显式
                    # 摘除并报警，否则调用方拿回一个"已登记但文件不存在/未被占用"的参数，
                    # 冻结层前向会悄悄变垃圾）。
                    with self._lock:
                        self._cold_index.pop(key, None)
                    print(f"[disk_balancer] 写盘失败 {key}: {e} —— 已从索引摘除，该参数将被丢弃（无法读回）", flush=True)
                finally:
                    self._write_queue.task_done()
            except Exception:
                # queue 关闭或线程退出
                break

    # ------------------------------------------------------------------
    # 冷参数检测与卸载
    # ------------------------------------------------------------------

    def _is_cold_param(self, name: str, param: torch.nn.Parameter) -> bool:
        """判断参数是否为"冷参数"（可卸载到磁盘的冻结层）。"""
        if self._cfg.offload_prefix and not name.startswith(self._cfg.offload_prefix):
            return False
        return (not param.requires_grad
                and param.numel() >= self._cfg.min_param_size
                and name not in self._cold_index)

    def _pick_path(self, tier: str = "auto", param_size: int = 0) -> str:
        """选择目标缓存路径。

        - "auto": 内存压力大时用 HDD，否则用 SSD
        - "ssd": 优先 SSD 路径
        - "hdd": 优先 HDD 路径
        """
        if tier == "hdd" and self._hdd_paths:
            # 轮询选 HDD
            return self._hdd_paths[self._migrated_count % len(self._hdd_paths)]
        if tier == "ssd" and self._ssd_paths:
            return self._ssd_paths[self._migrated_count % len(self._ssd_paths)]
        if tier == "auto":
            # 检查内存压力
            mem = psutil.virtual_memory()
            if mem.percent > 85 and self._hdd_paths:
                # 内存很紧张，用 HDD 放冷数据
                return self._hdd_paths[self._migrated_count % len(self._hdd_paths)]
            # 默认用 SSD
            if self._ssd_paths:
                return self._ssd_paths[self._migrated_count % len(self._ssd_paths)]
        # 兜底：第一个路径
        return self._paths[0] if self._paths else ""

    def _check_disk_activity(self, cache_dir: str) -> bool:
        """检查磁盘是否过载。"""
        if self._cfg.speed_first:
            return True
        for d in self._disks:
            if d.cache_dir == cache_dir:
                return get_disk_activity(d) < self._cfg.throttle_threshold
        return True

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def update_step(self) -> int:
        """每步调用：检查内存压力，自动迁移冷参数到磁盘。

        通过 psutil 监控内存使用率，当超过 memory_threshold 时：
        1. 遍历模型参数，找到冻结的大参数（requires_grad=False, numel > min_param_size）
        2. 调用 put_cold() 写入磁盘并释放内存
        3. 每次最多迁移 1 个参数（避免单步开销过大）

        Returns:
            本次迁移的参数数量（0 或 1）。
        """
        # 【9P 环境整体禁用】WSL 的 9P/宿主 NTFS 挂载下，卸载冷参数会扰宿主 MFT（炸机）。
        # 直接跳过、忽略用户参数（不迁移任何冷参数），宁可不做也要安全。
        if getattr(self, "_9p_skip", False):
            return 0
        if self._model is None:
            return 0
        if not self._started:
            return 0

        mem = psutil.virtual_memory()
        if mem.available / mem.total >= (1.0 - self._cfg.memory_threshold):
            return 0  # 内存充足，不迁移

        # 找第一个冷参数
        for name, param in self._model.named_parameters():
            if self._is_cold_param(name, param):
                # 自动选 tier：内存很紧张时用 HDD
                if mem.percent > 85:
                    tier = "hdd"
                else:
                    tier = "auto"
                self.put_cold(name, param, tier=tier)
                return 1

        return 0

    def put_cold(self, key: str, tensor: torch.Tensor, tier: str = "auto") -> str:
        """将冷参数写入磁盘并释放内存。

        1. 将 tensor 转为 bytes 放入异步写盘队列
        2. 记录到 _cold_index（路径、numel、dtype）
        3. 释放内存：``tensor.data = torch.empty(0)``

        Args:
            key: 参数名（如 "model.layers.0.self_attn.q_proj.weight"）
            tensor: 要卸载的参数 tensor
            tier: "auto" / "ssd" / "hdd"

        Returns:
            写入的磁盘路径。
        """
        if not self._started:
            return ""

        self._ensure_dirs()
        path = os.path.join(self._pick_path(tier, tensor.numel()), f"{key}.bin")

        # 记录元数据
        dtype = tensor.dtype
        numel = tensor.numel()

        with self._lock:
            self._cold_index[key] = (path, numel, dtype)

        # 异步写盘。bf16 的 numpy() 不支持（torch numpy 不支持 bf16），需先 view 成
        # uint16（bf16 原始字节）再 tobytes；其余 dtype 直接 numpy().tobytes()。
        if dtype == torch.bfloat16:
            data = tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
        else:
            data = tensor.detach().cpu().contiguous().numpy().tobytes()
        self._write_queue.put((key, data, path))

        # 释放内存
        tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
        self._migrated_count += 1

        return path

    def get_cold(self, key: str) -> Optional[torch.Tensor]:
        """从磁盘缓存读回冷参数。

        使用 mmap 零拷贝读取，然后 clone 回内存。

        Args:
            key: 参数名

        Returns:
            恢复的 tensor，如果不存在则返回 None。
        """
        if not self._started:
            return None

        with self._lock:
            entry = self._cold_index.get(key)
            if entry is None:
                return None
            path, numel, dtype = entry

        # 若该 key 仍在异步写盘队列中（在途），先等写完，避免把"待写"误判为"不存在"。
        self.wait_writes()

        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                import numpy as np
                # bf16 没有 numpy 原生类型（位布局 ≠ fp16），用 uint16 视图读再 view 成
                # bf16，否则按 fp16 解释会得到完全错误的数值（静默错值）。
                if dtype == torch.bfloat16:
                    arr = np.frombuffer(mm, dtype=np.uint16).copy()
                    mm.close()
                    tensor = torch.from_numpy(arr).view(torch.bfloat16)
                    return tensor[:numel].reshape(-1)
                _DTYPE_MAP = {
                    torch.float32: np.float32,
                    torch.float16: np.float16,
                    torch.float64: np.float64,
                    torch.int64: np.int64,
                    torch.int32: np.int32,
                    torch.int16: np.int16,
                    torch.int8: np.int8,
                    torch.uint8: np.uint8,
                    torch.bool: np.bool_,
                }
                np_dtype = _DTYPE_MAP.get(dtype, np.float32)
                # 先 copy 再关闭 mmap，避免 numpy 持有 mmap 引用导致关闭失败
                arr = np.frombuffer(mm, dtype=np_dtype).copy()
                mm.close()
            tensor = torch.from_numpy(arr).to(dtype)
            return tensor[:numel].reshape(-1)
        except Exception as e:
            print(f"[disk_balancer] 读盘失败 {key}: {e}", flush=True)
            return None

    def get_cold_shape(self, key: str, shape: torch.Size) -> Optional[torch.Tensor]:
        """从磁盘缓存读回并 reshape 到指定形状。

        Args:
            key: 参数名
            shape: 目标形状

        Returns:
            恢复的 tensor，如果不存在则返回 None。
        """
        t = self.get_cold(key)
        if t is None:
            return None
        return t.reshape(shape)

    def contains(self, key: str) -> bool:
        """检查 key 是否已被卸载到磁盘。"""
        with self._lock:
            return key in self._cold_index

    def drop(self, key: str):
        """删除指定 key 的磁盘缓存。"""
        with self._lock:
            entry = self._cold_index.pop(key, None)
        if entry and os.path.exists(entry[0]):
            try:
                os.remove(entry[0])
            except OSError:
                pass

    def wait_writes(self, timeout: float = 60.0):
        """等待所有异步写盘完成。

        加超时：若写盘线程意外死亡（如 print 到已关闭管道抛异常、解释器关闭），
        queue.join() 会永久阻塞（未消费的 item 永远不 task_done）。超时后放弃等待，
        避免 cleanup()/读盘 挂死，并打印警告。
        """
        deadline = time.time() + timeout
        while self._write_queue.unfinished_tasks > 0:
            if time.time() > deadline:
                print("[disk_balancer] wait_writes 超时：写盘线程可能已死亡，放弃等待", flush=True)
                return
            time.sleep(0.05)

    def cleanup(self):
        """清理所有缓存文件。

        如果 keep_cache=True，只清空索引，保留磁盘文件。
        如果 keep_cache=False（默认），删除所有磁盘缓存文件。
        """
        self.wait_writes()

        with self._lock:
            if not self._cfg.keep_cache:
                for d in self._disks:
                    if d.cache_dir and os.path.isdir(d.cache_dir):
                        shutil.rmtree(d.cache_dir, ignore_errors=True)
            self._cold_index.clear()

        if not self._cfg.keep_cache:
            print("[disk_balancer] 缓存已清理", flush=True)
        else:
            print("[disk_balancer] 磁盘缓存已保留", flush=True)

    def stats(self) -> Dict[str, Any]:
        """返回当前状态统计。"""
        with self._lock:
            cold_keys = list(self._cold_index.keys())
        mem = psutil.virtual_memory()
        return {
            "migrated_count": self._migrated_count,
            "cold_keys": len(cold_keys),
            "cold_keys_list": cold_keys[:10],  # 只显示前 10 个
            "memory_percent": mem.percent,
            "memory_available_mb": mem.available / (1 << 20),
            "queue_size": self._write_queue.qsize(),
            "disks": [
                {
                    "mount": d.mount,
                    "type": d.drive_type,
                    "cache_dir": d.cache_dir,
                    "cache_max_mb": d.max_cache_bytes / (1 << 20),
                }
                for d in self._disks
            ],
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"DiskLoadBalancer(migrated={s['migrated_count']}, "
                f"cold_keys={s['cold_keys']}, mem={s['memory_percent']:.0f}%)")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.cleanup()


# ---------------------------------------------------------------------------
# argparse 集成
# ---------------------------------------------------------------------------


def add_flash_args(parser) -> None:
    """给 argparse.ArgumentParser 添加 --flash 相关参数。"""
    parser.add_argument(
        "--flash", nargs="?", const="auto", default=None,
        help="启用硬盘均衡负载：auto/a=自动模式, manual=手动模式, true=启用自动模式",
    )
    parser.add_argument(
        "--flash_sizes", type=str, default=None,
        help="手动模式每盘缓存大小(MB)，格式: C:2048,D:4096,E:8192",
    )
    parser.add_argument(
        "--flash_paths", type=str, default=None,
        help="缓存路径，逗号分隔: D:\\cache,E:\\cache",
    )
    parser.add_argument(
        "--flash_speed", action="store_true", default=False,
        help="速度优先模式（不管硬盘死活）",
    )
    parser.add_argument(
        "--flash_keep", action="store_true", default=False,
        help="训练结束后保留缓存文件",
    )
    parser.add_argument(
        "--flash_threshold", type=float, default=0.8,
        help="内存使用率阈值（0.0~1.0），默认 0.8",
    )


def parse_flash_args(args) -> Optional[DiskBalancerConfig]:
    """从 argparse Namespace 解析出 DiskBalancerConfig。

    如果没有启用 --flash，返回 None。
    """
    flash_val = getattr(args, "flash", None)
    if flash_val is None:
        return None

    raw_mode = str(flash_val).lower() if flash_val else "auto"
    if raw_mode in ("true", "1", "yes", "auto", "a"):
        mode = "auto"
    elif raw_mode in ("manual", "m"):
        mode = "manual"
    else:
        mode = "auto"

    sizes: Dict[str, int] = {}
    if getattr(args, "flash_sizes", None):
        for part in args.flash_sizes.split(","):
            mount, size = part.split(":")
            m = mount.strip().upper()
            if not m.endswith(":"):          # "c" -> "C:"，补上盘符冒号（_start_manual 用 "C:"）
                m += ":"
            sizes[m] = int(size.strip())

    paths: List[str] = []
    if getattr(args, "flash_paths", None):
        paths = [p.strip() for p in args.flash_paths.split(",") if p.strip()]

    return DiskBalancerConfig(
        mode=mode,
        sizes=sizes,
        paths=paths,
        speed_first=getattr(args, "flash_speed", False),
        keep_cache=getattr(args, "flash_keep", False),
        memory_threshold=getattr(args, "flash_threshold", 0.8),
    )


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== disk_balancer v2 自检 ===\n")

    # 1. 磁盘检测
    print("1. 磁盘检测:")
    disks = detect_disks()
    for d in disks:
        print(f"  {d}")

    # 2. 自动模式启动
    print("\n2. 自动模式启动:")
    try:
        balancer = DiskLoadBalancer(DiskBalancerConfig(
            mode="auto",
            memory_threshold=0.95,  # 测试用高阈值，不触发自动迁移
        ))
        balancer.start()
        print(f"  {balancer}")

        # 3. 冷参数写入
        print("\n3. 冷参数写入:")
        t = torch.randn(100, 100)
        t_orig = t.clone()  # 保存原始值用于对比
        path = balancer.put_cold("test_param", t, tier="auto")
        print(f"  put_cold('test_param') -> {path}")
        print(f"  tensor.data 已释放: {t.data.shape}")

        # 4. 等待异步写入完成，然后读取
        balancer.wait_writes()
        print("\n4. 冷参数读取:")
        t2 = balancer.get_cold("test_param")
        if t2 is not None:
            print(f"  get_cold('test_param') -> shape={t2.shape}")
            # 对比：reshape 回原始形状
            t2_reshaped = t2.reshape(100, 100)
            match = torch.allclose(t_orig, t2_reshaped)
            print(f"  数值一致: {match}")
        else:
            print("  get_cold('test_param') -> None (读取失败)")

        # 5. 统计
        print(f"\n5. 统计: {balancer.stats()}")

        # 6. 清理
        print("\n6. 清理:")
        balancer.cleanup()
        print("  完成")

    except Exception as e:
        print(f"  自检出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== disk_balancer v2 自检结束 ===")