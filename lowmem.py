"""lowmem - 低内存 / 硬盘缓存工具（CPU 训练用）。

目标：
- 12~16GB 内存的纯 CPU 机器上，把不常用的 tensor / 数据缓存放到磁盘；
- 避免 SSD 被动态 swap 反复拷打；
- 与 torch_cpu_kit、bitsandbytes CPU 版、EFST 配合使用。

当前提供：
- ``DiskTensorStore``：按 key 把 tensor 存到磁盘，可带小内存 LRU 缓存；
- ``DiskCachedDataset``：把 Dataset 的每个样本缓存到磁盘，适合大数据集预处理结果；
- ``save_tensor`` / ``load_tensor``：一次性落盘/读回。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch

__all__ = [
    "DiskTensorStore",
    "DiskCachedDataset",
    "save_tensor",
    "load_tensor",
]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_tensor(tensor: torch.Tensor, path: Union[str, Path]) -> Path:
    """把 tensor 保存到磁盘，返回 Path。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return path


def load_tensor(path: Union[str, Path], map_location: str = "cpu") -> torch.Tensor:
    """从磁盘加载 tensor。"""
    return torch.load(path, map_location=map_location, weights_only=True)


class DiskTensorStore:
    """简单的磁盘 tensor 仓库。

    用法::

        store = DiskTensorStore("./cache_tensors", mem_cache_bytes=256 << 20)
        store.put("optimizer_state_0", state_tensor)
        t = store.get("optimizer_state_0")
    """

    def __init__(self, root: Union[str, Path], mem_cache_bytes: int = 256 << 20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mem_cache_bytes = max(0, mem_cache_bytes)
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_order: list[str] = []
        self._cache_size = 0

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.root / f"{h}.pt"

    def put(self, key: str, tensor: torch.Tensor) -> Path:
        path = self._path(key)
        save_tensor(tensor, path)
        self._maybe_cache(key, tensor)
        return path

    def get(self, key: str) -> torch.Tensor:
        if key in self._cache:
            return self._cache[key]
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        tensor = load_tensor(path)
        self._maybe_cache(key, tensor)
        return tensor

    def __contains__(self, key: str) -> bool:
        return key in self._cache or self._path(key).exists()

    def drop(self, key: str) -> None:
        if key in self._cache:
            self._evict(key)
        path = self._path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        self._cache.clear()
        self._cache_order.clear()
        self._cache_size = 0
        for p in self.root.glob("*.pt"):
            p.unlink()

    def _maybe_cache(self, key: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        size = _tensor_bytes(tensor)
        if size > self.mem_cache_bytes:
            return
        # 先尝试塞进缓存，如果超预算就逐出最旧的
        self._cache[key] = tensor
        self._cache_order.append(key)
        self._cache_size += size
        while self._cache_size > self.mem_cache_bytes and len(self._cache_order) > 1:
            old = self._cache_order.pop(0)
            if old in self._cache:
                self._evict(old)

    def _evict(self, key: str) -> None:
        t = self._cache.pop(key, None)
        if t is not None:
            self._cache_size -= _tensor_bytes(t)
        if key in self._cache_order:
            self._cache_order.remove(key)

    def __len__(self) -> int:
        return len(list(self.root.glob("*.pt")))

    def __repr__(self) -> str:
        return (
            f"DiskTensorStore(root={str(self.root)!r}, mem_cache={self._cache_size >> 20}MB, "
            f"disk_files={len(self)})"
        )


class DiskCachedDataset(torch.utils.data.Dataset):
    """把 Dataset 的每个样本缓存到磁盘，避免全部常驻内存。

    用法::

        base = MyDataset(...)
        ds = DiskCachedDataset(base, cache_dir="./cache_dataset", transform=preprocess)
        loader = DataLoader(ds, batch_size=1)
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        cache_dir: Union[str, Path],
        transform: Optional[Callable[[Any], Any]] = None,
        key_fn: Optional[Callable[[int], str]] = None,
    ):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transform
        self.key_fn = key_fn or (lambda idx: str(idx))

    def __len__(self) -> int:
        return len(self.dataset)

    def _path(self, index: int) -> Path:
        key = self.key_fn(index)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pt"

    def __getitem__(self, index: int):
        path = self._path(index)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        item = self.dataset[index]
        if self.transform is not None:
            item = self.transform(item)
        torch.save(item, path)
        return item


if __name__ == "__main__":  # 自检
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = DiskTensorStore(tmp)
        store.put("a", torch.randn(4, 4))
        assert "a" in store
        assert torch.equal(store.get("a"), store.get("a"))
        store.drop("a")
        assert "a" not in store
    print("lowmem selftest OK")
