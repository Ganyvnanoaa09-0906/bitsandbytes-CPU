# -*- coding: utf-8 -*-
"""test_disk_balancer_wsl_integration.py — real WSL/Linux integration for disk_balancer.

Skips on Windows. Run inside WSL:

    PYTHONPATH=/mnt/d/work/bitsandbytes-CPU/bitsandbytes \
        "$HOME/bnbvenv/bin/python" test_disk_balancer_wsl_integration.py

Covers:
  1. mount safety verdicts on the real /proc/mounts (ext4 ok; /mnt/[c-z] 9p rejected; / root rejected)
  2. auto mode: picks a cache on the WSL ext4, never under /mnt, offload/readback exact,
     cleanup removes only its own cache dir
  3. explicit unsafe --flash_paths is rejected with a clear error
  4. manual mode with sizes={"/": MB} + a $HOME cache path works and leaves the parent intact
  5. bounded in-flight write bytes (backpressure) does not deadlock or corrupt data
  6. throughput of write + readback for a 256 MB cold tensor on ext4 only.

SAFETY: this test NEVER writes to /mnt/* (WSL's 9P host drives). The host machine has no
backup and 9P write/delete churn once disturbed the host NTFS MFT; all file writes stay on the
WSL ext4 filesystem ($HOME / temp dirs there).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

import torch  # noqa: E402

from bitsandbytes import disk_balancer as db  # noqa: E402

MB = 1 << 20


_FORBIDDEN_WRITE_PREFIXES = ("/mnt/",)


def _assert_safe_write_path(path: str) -> None:
    """Refuse any test write outside the WSL ext4 filesystem (never touch /mnt/* 9P)."""
    ap = os.path.abspath(path)
    assert not any(ap.startswith(p) for p in _FORBIDDEN_WRITE_PREFIXES), (
        f"SAFETY: refusing to write to 9P host path {ap}; main machine has no backup")


def _safe_tempdir(prefix: str, base: str) -> str:
    _assert_safe_write_path(base)
    path = tempfile.mkdtemp(prefix=prefix, dir=base)
    _assert_safe_write_path(path)
    return path


def log(key: str, **kw) -> None:
    parts = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"RESULT {key} {parts}", flush=True)


def test_mount_verdicts():
    ok, reason = db._cache_dir_verdict("/")
    assert not ok, "root / must be rejected"
    home = os.path.expanduser("~")
    ok, reason = db._cache_dir_verdict(home)
    assert ok, f"$HOME should be safe: {reason}"
    if os.path.isdir("/mnt/d"):
        ok_d, reason_d = db._cache_dir_verdict("/mnt/d")
        assert not ok_d, f"/mnt/d (9P) must be rejected: {reason_d}"
        log("verdict_mnt_d", ok=ok_d, reason=reason_d.replace(" ", "_")[:60])
    log("verdict_home", ok=ok, reason=reason.replace(" ", "_")[:60])


def _put_get(bal, key, tensor):
    # put_cold releases the tensor's storage, so keep a reference copy for comparison.
    shape = tensor.shape
    reference = tensor.clone()
    bal.put_cold(key, tensor, tier="ssd")
    bal.wait_writes()
    back = bal.get_cold_shape(key, shape)
    assert back is not None, f"readback of {key} failed"
    assert torch.equal(back, reference), f"readback of {key} differs"
    return back


def test_auto_mode():
    bal = db.DiskLoadBalancer(db.DiskBalancerConfig(mode="auto"))
    try:
        bal.start()
        assert bal._started, "balancer failed to start on WSL"
        log("auto_paths", n=len(bal._paths), first=bal._paths[0].replace("/", "_"))
        assert bal._paths, "no cache path selected"
        for p in bal._paths:
            ok, reason = db._cache_dir_verdict(p)
            assert ok, (p, reason)
            assert not p.startswith("/mnt/c"), p
            assert not p.startswith("/mnt/d"), p
        t = torch.arange(8 * 1024 * 1024, dtype=torch.float32)  # 32 MB
        before = bal.stats()
        _put_get(bal, "it_auto_param", t)
        after = bal.stats()
        log("auto_offload", before_cold=before["cold_keys"], after_cold=after["cold_keys"],
            inflight_mb=round(after["inflight_mb"], 2))
        assert after["cold_keys"] == 1
    finally:
        bal.cleanup()
    assert not os.path.exists(bal._paths[0]), "cleanup should remove its own cache dir"
    log("auto_cleanup", ok=True)


def test_unsafe_path_rejected():
    try:
        bal = db.DiskLoadBalancer(db.DiskBalancerConfig(mode="auto", paths=["/mnt/d/bnb_flash"]))
        bal.start()
    except ValueError as exc:
        assert "不安全" in str(exc), str(exc)
        log("unsafe_path_rejected", ok=True)
    else:
        bal.cleanup()
        raise AssertionError("unsafe --flash_paths must be rejected")


def test_manual_mode():
    parent = _safe_tempdir("bnb_wsl_manual_", os.path.expanduser("~"))
    sentinel = os.path.join(parent, "KEEP_ME.txt")
    with open(sentinel, "w", encoding="utf-8") as fh:
        fh.write("keep")
    bal = db.DiskLoadBalancer(db.DiskBalancerConfig(
        mode="manual", sizes={"/": 256}, paths=[parent], keep_cache=False))
    try:
        bal.start()
        cache_dir = bal._paths[0]
        assert cache_dir == os.path.join(parent, ".bnb_flash_cache"), cache_dir
        t = torch.linspace(-1, 1, 4 * 1024 * 1024, dtype=torch.float32)  # 16 MB
        _put_get(bal, "it_manual_param", t)
        log("manual_ok", cache_dir=cache_dir.replace("/", "_"))
    finally:
        bal.cleanup()
    assert os.path.isfile(sentinel), "cleanup deleted the user parent directory contents"
    assert not os.path.isdir(os.path.join(parent, ".bnb_flash_cache")), "cache subdir should be gone"
    shutil.rmtree(parent, ignore_errors=True)
    log("manual_cleanup", ok=True)


def test_backpressure():
    parent = _safe_tempdir("bnb_wsl_bp_", os.path.expanduser("~"))
    bal = db.DiskLoadBalancer(db.DiskBalancerConfig(
        mode="manual", sizes={"/": 1024}, paths=[parent],
        max_inflight_bytes=32 * MB, write_queue_depth=1, keep_cache=False))
    try:
        bal.start()
        accepted = 0
        for i in range(6):
            t = torch.full((4 * 1024 * 1024,), float(i), dtype=torch.float32)  # 16 MB
            if bal.put_cold(f"it_bp_{i}", t, tier="ssd"):
                accepted += 1
        bal.wait_writes()
        st = bal.stats()
        log("backpressure", accepted=accepted, inflight_mb=round(st["inflight_mb"], 2),
            write_total_mb=round(st["write_total_mb"], 1))
        assert accepted >= 1, "backpressure must not reject everything"
        assert st["inflight_mb"] == 0, "in-flight bytes must drain to zero"
        for i in range(6):
            back = bal.get_cold_shape(f"it_bp_{i}", torch.Size([4 * 1024 * 1024]))
            if back is not None:
                assert torch.equal(back, torch.full((4 * 1024 * 1024,), float(i),
                                                    dtype=torch.float32)), i
    finally:
        bal.cleanup()
        shutil.rmtree(parent, ignore_errors=True)
    log("backpressure_cleanup", ok=True)


def test_plain_linux_emulation():
    """Plain Linux (non-WSL) goes through the same code path but without the /mnt guard.

    Emulate by patching _in_wsl to False: auto mode must still pick the ext4 root and reject
    the read-only /mnt/wslg/distro mount.
    """
    original = db._in_wsl
    db._in_wsl = lambda: False
    try:
        bal = db.DiskLoadBalancer(db.DiskBalancerConfig(mode="auto"))
        try:
            bal.start()
            assert bal._started
            assert bal._paths
            for p in bal._paths:
                ok, reason = db._cache_dir_verdict(p)
                assert ok, (p, reason)
                assert not p.startswith("/mnt/"), p
            log("plain_linux_auto", paths=len(bal._paths))
        finally:
            bal.cleanup()
    finally:
        db._in_wsl = original
    log("plain_linux_emulation", ok=True)


def test_training_style_offload():
    """update_step() offloads frozen params under memory pressure; forward still works."""
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 64),
    )
    for p in model.parameters():
        p.requires_grad_(False)
    parent = _safe_tempdir("bnb_wsl_train_", os.path.expanduser("~"))
    bal = db.DiskLoadBalancer(db.DiskBalancerConfig(
        mode="manual", sizes={"/": 512}, paths=[parent],
        min_param_size=100_000, memory_threshold=0.99))
    try:
        bal.start()
        bal.attach_model(model)
        bal._mem_pressure = lambda: True  # force the offload decision
        x = torch.randn(2, 64)
        ref = model(x).detach().clone()
        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        moved = bal.update_step()
        bal.wait_writes()
        names = list(bal._cold_index.keys())
        assert moved > 0 and names, "no cold parameter was offloaded"
        # put_cold releases storage, so remember numel/dtype from the index and restore.
        for name, p in list(model.named_parameters()):
            if not bal.contains(name):
                continue
            path, numel, dtype = bal._cold_index[name]
            elem = torch.empty(0, dtype=dtype).element_size()
            assert os.path.getsize(path) == numel * elem
            flat = bal.get_cold(name)
            assert flat is not None and flat.numel() == numel
            p.data = flat.reshape(shapes[name])
        out = model(x)
        assert torch.allclose(out, ref, atol=1e-6), "forward changed after offload/restore"
        log("training_style_offload", moved=moved, restored=len(names))
    finally:
        bal.cleanup()
        shutil.rmtree(parent, ignore_errors=True)
    log("training_style_cleanup", ok=True)


def test_throughput():
    parent = _safe_tempdir("bnb_wsl_tp_", os.path.expanduser("~"))
    bal = db.DiskLoadBalancer(db.DiskBalancerConfig(
        mode="manual", sizes={"/": 4096}, paths=[parent], max_inflight_bytes=8192 * MB))
    try:
        bal.start()
        n = 64 * 1024 * 1024  # 64M float32 = 256 MB
        t = torch.arange(n, dtype=torch.float32)
        shape = t.shape
        t0 = time.perf_counter()
        bal.put_cold("it_tp_param", t, tier="ssd")
        t_put = time.perf_counter() - t0
        t0 = time.perf_counter()
        bal.wait_writes()
        t_write = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = bal.get_cold_shape("it_tp_param", shape)
        t_read = time.perf_counter() - t0
        assert back is not None and torch.equal(back, torch.arange(n, dtype=torch.float32))
        log("throughput_ext4", size_mb=256, put_ms=round(t_put * 1e3, 1),
            write_ms=round(t_write * 1e3, 1), read_ms=round(t_read * 1e3, 1),
            write_mbps=round(256 / max(t_write, 1e-9), 1),
            read_mbps=round(256 / max(t_read, 1e-9), 1))
    finally:
        bal.cleanup()
        shutil.rmtree(parent, ignore_errors=True)

    # NOTE: deliberately no raw writes to /mnt/* (9P host drive): that is the exact
    # workload that disturbed the host MFT before. All writes in this test stay on ext4.


def main():
    if os.name == "nt":
        print("SKIP: this integration test runs inside WSL/Linux")
        return 0
    if not db._in_wsl():
        print("NOTE: not WSL, but running Linux paths anyway")
    _assert_safe_write_path(os.path.expanduser("~"))
    test_mount_verdicts()
    test_auto_mode()
    test_unsafe_path_rejected()
    test_manual_mode()
    test_plain_linux_emulation()
    test_training_style_offload()
    test_backpressure()
    test_throughput()
    print("WSL DISK BALANCER INTEGRATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
