# -*- coding: utf-8 -*-
"""igpu_ctx.py — 核显设备侧 **context / arena 封装**（给 L3 Stack 用）。

背景（本项目实测的驱动缺陷）：
    同一个 OpenCL context 里「大小混杂」的 buffer 超过约 130 个之后，后续分配的内存会互相踩
    （读回来变成别的 buffer 的内容 / 全 0），**整个过程不报任何错误**。
因此 AR 引擎的做法是「每层一个 context，每层 ~40 个 buffer，层间用 host 传激活」。
本模块把这件事封装成可复用接口：

    gang = LayerGang(n_layers=4, arena_mb=900)      # 每层一个 context + 一个 arena
    L0 = gang[i]
    L0.activate()                                   # ★ 之后 A.ctx/A.queue/A.QP 指向该层
    w = L0.up(np_array)                             # 分配 + 显式 copy（禁用 COPY_HOST_PTR）
    buf = L0.buf(nbytes)                            # arena 里切一块
    L0.down(buf, shape)                             # 回读（建议 >=4KB，小读不可靠）
    K = L0.program(SRC, ['k1', 'k2'])               # 在该 context 内编译内核

    with gang.layer(i):                             # 或上下文管理器
        ...                                          # 内部自动 activate

自检：`python gt1030/igpu_ctx.py --check` —— 复现/证伪「>130 buffer 踩坏」：
  · 一个 context 里放 N 个大小混杂的 buffer，逐个写入自己的签名并回读校验；
  · 同样的 buffer 总量，拆到 4 个 context（每层一个），再校验；
  · 输出两种布局的失败率对比。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

A = MM.A
ALIGN = 4096

# 累计分配的硬上限：超过它一定失败，而且实测会**静默返回全零**（T=4096 分块反向踩过）。
# 只拦"累计超过物理上限"，不按 arena_mb 判，保留多块自动扩容的正常路径。
# 可用环境变量 DSH_ARENA_HARD_CAP_GB 覆盖（默认 8 GB，本机实测可写到约 8 GB）。
HARD_CAP_BYTES = int(os.environ.get("DSH_ARENA_HARD_CAP_GB", "8")) * 2 ** 30


CHUNK_MB = 3900      # 单块上限（驱动 4.74 GB，留余量）


class DevCtx(object):
    """一个 context + 一条 queue + 一个 arena；所有张量都是 arena 的 sub-buffer。"""

    def __init__(self, arena_mb=900, name=""):
        self.name = name
        dev = A.ctx.devices[0]
        self.ctx = cl.Context(devices=[dev])
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        # 多 arena：单块 ≤ CHUNK_MB，按需申请（驱动单 buffer 上限 4.74 GB）
        self._mb = int(arena_mb)
        self._chunk_bytes = min(int(arena_mb) << 20, int(CHUNK_MB) << 20)
        self.arenas = []          # [cl.Buffer, ...]
        self._chunk_off = 0       # 当前块内偏移
        self._chunk_idx = -1
        self._off = 0             # **活跃**字节（release 会减）
        self._peak = 0            # 历史峰值
        self._n = 0
        self._free_lists = {}     # chunk_idx -> [(offset, size), ...]  显式释放的空闲块
        self._map = {}            # cl.Buffer -> (chunk_idx, offset, size, tag)
        self._dead = {}           # id(Buffer) -> tag（用于 double release 报两处 owner）
        self._stats = {"reused": 0, "freed": 0, "double": 0}
        self._ensure_chunk(0)

    # ---- 生命周期 ----
    def activate(self):
        """把 A.* 的全局句柄指到本 context（CLBlast hgemm_* 走 A.QP / A.queue / A.ctx）。"""
        A.ctx, A.queue, A.mf = self.ctx, self.q, self.mf
        A.QP = cl.c_void_p(int(self.q.int_ptr)) if hasattr(cl, "c_void_p") else A.QP
        if not hasattr(cl, "c_void_p"):
            import ctypes
            A.QP = ctypes.c_void_p(int(self.q.int_ptr))
        return self

    # ---- 内存 ----
    # ---- 多 arena ----
    def _ensure_chunk(self, idx):
        """确保第 idx 块已申请；返回该块。"""
        while len(self.arenas) <= idx:
            if len(self.arenas) * self._chunk_bytes >= (self._mb << 20) + self._chunk_bytes:
                raise RuntimeError("%s arena 不够：已申请 %d 块 × %.0f MB，总量超过 %d MB"
                                   % (self.name, len(self.arenas),
                                      self._chunk_bytes / 1048576.0, self._mb))
            try:
                self.arenas.append(cl.Buffer(self.ctx, self.mf.READ_WRITE, self._chunk_bytes))
            except Exception as error:
                raise RuntimeError("%s arena 申请失败（第 %d 块，%.0f MB）：%r"
                                   % (self.name, len(self.arenas),
                                      self._chunk_bytes / 1048576.0, error))
        return self.arenas[idx]

    def release(self, b):
        """显式归还：把 b 放回所属 chunk 的空闲表（权重不要调用它）。"""
        info = self._map.pop(b, None)
        if info is None:
            # 未登记 = 已经 release 过（或不是本 arena 的缓冲）。打印**两处 owner** 帮助定位别名。
            self._stats["double"] = self._stats.get("double", 0) + 1
            if self._stats["double"] <= 8:      # 只打印前 8 次，避免刷屏
                import traceback
                first = self._dead.get(id(b), "?")
                now = " > ".join("%s:%d" % (f.name, f.lineno)
                                 for f in traceback.extract_stack()[-4:-1])
                print("[double-release] %s | 首次 owner=%s | 本次调用=%s"
                      % (self.name, first, now))
            return False
        idx, off, size, tag = info
        self._dead[id(b)] = tag
        self._free_lists.setdefault(idx, []).append((off, size))
        self._off -= size
        self._stats["freed"] += size
        return True

    def _try_reuse(self, n, tag=""):
        """best-fit：找 >= n 的最小空闲块，切掉用掉的部分，余量放回。"""
        best = None
        for idx, lst in self._free_lists.items():
            for i, (off, size) in enumerate(lst):
                if size >= n and (best is None or size < best[2]):
                    best = (idx, i, size, off)
        if best is None:
            return None
        idx, i, size, off = best
        lst = self._free_lists[idx]
        lst.pop(i)
        if size - n >= ALIGN:
            lst.append((off + n, size - n))
        b = self.arenas[idx].get_sub_region(off, n)
        self._map[b] = (idx, off, n, "reuse:" + tag)
        self._off += n
        self._stats["reused"] += n
        return b

    def buf(self, nbytes, tag=""):
        n = (int(nbytes) + ALIGN - 1) // ALIGN * ALIGN
        if n > self._chunk_bytes:
            raise RuntimeError("%s 单次申请 %d 字节超过分块大小 %.0f MB"
                               % (self.name, n, self._chunk_bytes / 1048576.0))
        # 硬上限：累计分配超过 HARD_CAP 直接报错。
        # 为什么需要：实测 T=4096 的分块注意力反向在 arena 不足时**返回全零梯度却不报错**
        # （静默失效），排查成本极高。这里只拦"累计超过物理上限"这种必然失败的申请，
        # 不按 arena_mb 判（正常的多块自动扩容路径必须保留）。
        if self._off + n > HARD_CAP_BYTES:
            raise RuntimeError(
                "%s 累计申请将达 %.2f GB，超过硬上限 %.2f GB（已申请 %d 块）；"
                "请检查是否漏回收或 arena_mb 过小"
                % (self.name, (self._off + n) / 2 ** 30, HARD_CAP_BYTES / 2 ** 30,
                   max(self._chunk_idx + 1, 1)))
        # 当前块放不下就换块（块内不跨界，避免跨 buffer 造 sub_region）
        if self._chunk_idx < 0 or self._chunk_off + n > self._chunk_bytes:
            self._chunk_idx += 1
            self._chunk_off = 0
        reuse = self._try_reuse(n, tag)
        if reuse is not None:
            self._n += 1
            return reuse
        arena = self._ensure_chunk(self._chunk_idx)
        b = arena.get_sub_region(self._chunk_off, n)
        self._map[b] = (self._chunk_idx, self._chunk_off, n, tag)
        self._chunk_off += n
        self._off += n
        self._peak = max(self._peak, self._off)
        self._n += 1
        return b

    def up(self, arr):
        a = np.ascontiguousarray(arr)          # ★ 不用 COPY_HOST_PTR（已知会传错）
        b = self.buf(a.nbytes)
        cl.enqueue_copy(self.q, b, a)
        self.q.finish()
        return b

    def down(self, buf, shape, dtype=np.float16):
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.q, out, buf, is_blocking=True)
        return out

    def program(self, src, names, options=("-cl-mad-enable", "-cl-fast-relaxed-math")):
        """在**本 context** 里编译内核（内核与 context 绑定，不能跨层复用）。"""
        prog = cl.Program(self.ctx, src).build(options=list(options))
        return {n: cl.Kernel(prog, n) for n in names}

    # ---- 统计 ----
    def stats(self):
        return {"name": self.name, "buffers": self._n,
                "arena_mb": self._mb, "used_mb": round(self._off / 1048576.0, 1),
        "peak_mb": round(self._peak / 1048576.0, 1),
        "reused_mb": round(self._stats["reused"] / 1048576.0, 1),
            "chunks": len(self.arenas)}

    def __repr__(self):
        return "DevCtx(%s, buffers=%d, used=%.1f/%d MB)" % (
            self.name, self._n, self._off / 1048576.0, self._mb)


class LayerGang(object):
    """每层一个 DevCtx（把单 context 的 buffer 数压到 ~130 以下）。"""

    def __init__(self, n_layers, arena_mb=900, prefix="layer"):
        self.layers = [DevCtx(arena_mb=arena_mb, name="%s%d" % (prefix, i))
                       for i in range(n_layers)]

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    @contextlib.contextmanager
    def layer(self, i):
        self.layers[i].activate()
        yield self.layers[i]

    def stats(self):
        return [d.stats() for d in self.layers]


# ------------------------------------------------------------------ 自检
def _check_single(n_bufs=200, mb=300):
    """一个 context 里放 n_bufs 个大小混杂的 buffer，逐个写签名并回读。"""
    d = DevCtx(arena_mb=mb, name="single")
    rng = np.random.default_rng(0)
    bufs, sigs = [], []
    sizes = [(7 + (i * 13) % 97) * 1024 for i in range(n_bufs)]    # 7KB~100KB 混杂
    bad = 0
    for i, nb in enumerate(sizes):
        b = d.buf(nb)
        sig = np.full(nb // 2, np.float16(i % 1000))
        cl.enqueue_copy(d.q, b, sig)
        bufs.append(b)
        sigs.append(sig)
    d.q.finish()
    for i, (b, sig) in enumerate(zip(bufs, sigs)):
        out = np.empty_like(sig)
        cl.enqueue_copy(d.q, out, b, is_blocking=True)
        if not np.array_equal(out, sig):
            bad += 1
    return bad, n_bufs, d


def _check_gang(n_layers=4, per_layer=75, mb=300):
    """把同样的 buffer 总量拆到多个 context（每层一个），再校验。"""
    gang = LayerGang(n_layers=n_layers, arena_mb=mb)
    errs = 0
    total = 0
    for li, d in enumerate(gang.layers):
        bufs, sigs = [], []
        for i in range(per_layer):
            nb = (7 + ((li * 31 + i * 13) % 97)) * 1024
            b = d.buf(nb)
            sig = np.full(nb // 2, np.float16((i + li * 100) % 1000))
            cl.enqueue_copy(d.q, b, sig)
            bufs.append(b)
            sigs.append(sig)
        d.q.finish()
        for b, sig in zip(bufs, sigs):
            out = np.empty_like(sig)
            cl.enqueue_copy(d.q, out, b, is_blocking=True)
            if not np.array_equal(out, sig):
                errs += 1
        total += per_layer
    return errs, total, gang


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if not a.check:
        print("  用 --check 运行自检")
        return 0
    bad, n, d = _check_single(a.n)
    print("  单 context: %d/%d 个 buffer 回读错误  (%s)" % (bad, n, d))
    errs, total, gang = _check_gang()
    print("  %d 个 context（每层一个）: %d/%d 个 buffer 回读错误"
          % (len(gang), errs, total))
    for st in gang.stats():
        print("    %s" % st)
    ok = (errs == 0)
    print("RESULT " + ("PASS（分层有效）" if ok else "FAIL（分层仍被踩坏）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
