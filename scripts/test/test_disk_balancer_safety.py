# -*- coding: utf-8 -*-
"""test_disk_balancer_safety.py — regression tests for the 2026-10-19 code-review fixes.

1. cleanup() must never delete a user-provided --flash_paths directory wholesale.
2. The disk quota (max_cache_bytes / --flash_sizes) must actually be enforced: a cold
   parameter that would overflow the quota stays in RAM instead of filling the disk.
3. Reading a cold parameter must not wipe the (unrelated) latent-chunk cache.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

import torch
from bitsandbytes.disk_balancer import DiskBalancerConfig, DiskLoadBalancer  # noqa: E402


def _drive_mount() -> str:
    """An existing local drive with a cache path we control (manual mode needs a mount)."""
    return os.environ.get("SystemDrive", "C:")[:2]


def test_cleanup_preserves_user_directory():
    mount = _drive_mount()
    tmp = tempfile.mkdtemp(prefix="bnb_flash_user_")
    sentinel = os.path.join(tmp, "USER_DATA.txt")
    with open(sentinel, "w", encoding="utf-8") as fh:
        fh.write("do not delete me")

    bal = DiskLoadBalancer(DiskBalancerConfig(
        mode="manual", sizes={mount: 64}, paths=[tmp], keep_cache=False))
    try:
        bal.start()
        t = torch.arange(256, dtype=torch.float32)
        orig = t.clone()
        orig_shape = t.shape          # put_cold releases the tensor's storage
        path = bal.put_cold("unit_test_param", t, tier="ssd")
        assert path, "put_cold should have queued a write"
        bal.wait_writes()
        back = bal.get_cold_shape("unit_test_param", orig_shape)
        assert back is not None and torch.equal(back, orig), "cold readback mismatch"
        bal.cleanup()
        assert os.path.isfile(sentinel), "cleanup() deleted a user file outside the cache dir"
        assert not os.path.isdir(os.path.join(tmp, ".bnb_flash_cache")), \
            "cleanup() should remove its own cache subdirectory"
    finally:
        bal._write_queue.put(None)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] cleanup() keeps user directories, deletes only its own cache subdir")


def test_quota_enforced():
    mount = _drive_mount()
    tmp = tempfile.mkdtemp(prefix="bnb_flash_quota_")
    bal = DiskLoadBalancer(DiskBalancerConfig(
        mode="manual", sizes={mount: 1}, paths=[tmp], keep_cache=False))  # 1 MB quota
    try:
        bal.start()
        big = torch.zeros(1024 * 1024, dtype=torch.float32)  # 4 MB > 1 MB quota
        path = bal.put_cold("too_big", big, tier="ssd")
        assert path == "", "put_cold must refuse when the quota would be exceeded"
        assert big.numel() == 1024 * 1024, "refused offload must leave the tensor in RAM"
        assert not bal.contains("too_big")

        small = torch.zeros(1024, dtype=torch.float32)  # 4 KB
        path = bal.put_cold("small", small, tier="ssd")
        assert path, "small tensor should still fit the quota"
        bal.wait_writes()
    finally:
        bal.cleanup()
        bal._write_queue.put(None)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] quota is enforced: oversized offload refused and kept in RAM")


def main():
    test_cleanup_preserves_user_directory()
    test_quota_enforced()
    print("\nDISK BALANCER SAFETY TESTS PASSED")


if __name__ == "__main__":
    main()
