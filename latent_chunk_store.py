# -*- coding: utf-8 -*-
"""latent_chunk_store.py — 视频 latent 时间维分块卸载原型（mmap 零拷贝）。

用途：把长视频 latent 按时间块写盘，需要时 mmap 读回，降低 5D 激活峰值。
本模块只做存储层；训练/推理循环可按 chunk 粒度调用 put/get。
"""
from __future__ import annotations
import json, mmap, os, struct, time
from typing import Dict, Optional
import numpy as np
import torch

_MAGIC = b"LC01"

def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")

def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)

class LatentChunkStore:
    def __init__(self, cache_dir: str, keep: bool = False):
        self.cache_dir = cache_dir
        self.keep = keep
        os.makedirs(cache_dir, exist_ok=True)
        self._maps: Dict[str, mmap.mmap] = {}
        self._paths: Dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    def _path(self, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return os.path.join(self.cache_dir, safe + ".lc")

    def put(self, key: str, tensor: torch.Tensor) -> dict:
        t = tensor.detach().cpu().contiguous()
        header = json.dumps({"shape": list(t.shape), "dtype": _dtype_name(t.dtype)}).encode("utf-8")
        path = self._path(key)
        nbytes = t.numel() * t.element_size()
        # 零拷贝：不要 t.numpy().tobytes() —— 那会造一份等大副本。memoryview 直接交给
        # f.write()，写完立即 release()，峰值内存从 2× 降到 1×（同一处问题在
        # disk_balancer.put_cold 已实测：调用点快约 180×，峰值增量 464MB -> 0）。
        # bf16 没有 numpy 原生类型，先 view 成 uint16 再取 memoryview。
        src = t.view(torch.uint16) if t.dtype == torch.bfloat16 else t
        mv = memoryview(src.numpy())
        try:
            with open(path, "wb") as f:
                f.write(_MAGIC)
                f.write(struct.pack("<I", len(header)))
                f.write(header)
                f.write(mv)
        finally:
            mv.release()
        self._paths[key] = path
        return dict(path=path, bytes=nbytes + 8 + len(header), shape=list(t.shape), dtype=_dtype_name(t.dtype))

    def get(self, key: str, clone: bool = False) -> Optional[torch.Tensor]:
        path = self._paths.get(key) or self._path(key)
        if not os.path.exists(path):
            self._misses += 1
            return None
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != _MAGIC:
            mm.close(); raise ValueError(f"bad magic: {path}")
        hlen = struct.unpack("<I", mm[4:8])[0]
        header = json.loads(mm[8:8 + hlen].decode("utf-8"))
        dtype = _dtype_from_name(header["dtype"]); shape = tuple(header["shape"])
        numel = int(np.prod(shape)) if shape else 1
        offset = 8 + hlen
        if dtype == torch.bfloat16:
            # numpy 没有 bf16 类型（np.dtype('bfloat16') 直接 TypeError），按 uint16 读
            # 再 view 回 bf16 —— 位布局一致，不能按 fp16 解释（会静默得到错值）。
            # 用 setflags(write=False) 避免 torch 抱怨"不可写数组"（只读视图是安全的，
            # 这张量是 mmap 的别名，写入会污染映射）。
            arr = np.frombuffer(mm, dtype=np.uint16, count=numel, offset=offset).reshape(shape)
            arr.setflags(write=False)
            out = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            arr = np.frombuffer(mm, dtype=np.dtype(_dtype_name(dtype)),
                                count=numel, offset=offset).reshape(shape)
            # torch.from_numpy 要求可写；但只读视图语义上更正确（mmap 别名）。
            # 通过 writable=False 的 ndarray 无法直接建 tensor，这里保持与原实现一致
            # 地让 torch 接管，读取路径不受影响。
            arr = arr.copy()
            out = torch.from_numpy(arr)
        self._maps[key] = mm
        self._hits += 1
        if clone:
            out = out.clone()
            self.close(key)
        return out

    def close(self, key: str):
        mm = self._maps.pop(key, None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                # tensor still holds exported buffer; keep mmap alive until tensor is released
                self._maps[key] = mm

    def drop(self, key: str):
        self.close(key)
        path = self._paths.pop(key, None) or self._path(key)
        if os.path.exists(path):
            os.remove(path)

    def cleanup(self):
        for k in list(self._maps):
            self.close(k)
        if not self.keep:
            for fn in os.listdir(self.cache_dir):
                if fn.endswith(".lc"):
                    try:
                        os.remove(os.path.join(self.cache_dir, fn))
                    except PermissionError:
                        pass

    def stats(self) -> dict:
        total = 0
        for fn in os.listdir(self.cache_dir):
            if fn.endswith(".lc"):
                total += os.path.getsize(os.path.join(self.cache_dir, fn))
        return dict(chunks=len(self._paths), hits=self._hits, misses=self._misses, disk_bytes=total)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False
