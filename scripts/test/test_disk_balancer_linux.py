# -*- coding: utf-8 -*-
"""test_disk_balancer_linux.py — unit tests for the WSL/Linux mount-safety logic.

These run on any OS (the mount table is injected as a temp file), so the dangerous cases
are covered even on Windows: 9P host drives, network filesystems, read-only mounts, tmpfs,
and the "/" root directory must all be rejected as cache locations.
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

from bitsandbytes import disk_balancer as db  # noqa: E402

FAKE_MOUNTS = """\
/dev/sdd / ext4 rw,relatime 0 0
D:\\040drive /mnt/d 9p rw,relatime,aname=drvfs 0 0
C:\\040drive /mnt/c 9p rw,relatime,aname=drvfs 0 0
//server/share /mnt/share cifs rw 0 0
/dev/sdb1 /mnt/data ext4 rw,relatime 0 0
tmpfs /tmp tmpfs rw,nosuid 0 0
/dev/loop0 /ro ext4 ro,relatime 0 0
"""


def _write_mounts() -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix="_mounts", delete=False, encoding="utf-8")
    fh.write(FAKE_MOUNTS)
    fh.close()
    return fh.name


def test_mount_lookup():
    p = _write_mounts()
    try:
        rows = db._read_mounts(p)
        assert rows, "mount table should parse"
        assert rows[0][0] == "/mnt/share", "longest mountpoint must come first"
        assert db._mount_for_path("/mnt/data/sub/dir", p)[0] == "/mnt/data"
        # /mnt/data2 must NOT match /mnt/data (needs a path separator)
        assert db._mount_for_path("/mnt/data2/x", p)[0] == "/"
    finally:
        os.unlink(p)


def test_cache_verdicts():
    p = _write_mounts()
    try:
        ok, reason = db._cache_dir_verdict("/", p)
        assert not ok, "root directory must be rejected (cleanup would rmtree it)"
        for bad, needle in (
            ("/mnt/d/flash", "9P"),        # WSL host drive
            ("/mnt/share/flash", "网络"),  # cifs
            ("/tmp/flash", "内存盘"),       # tmpfs
            ("/ro/flash", "只读"),          # read-only
        ):
            ok, reason = db._cache_dir_verdict(bad, p)
            assert not ok, (bad, reason)
            assert needle in reason or "9P" in reason or "网络" in reason or "内存" in reason \
                or "只读" in reason, (bad, reason)
        ok, reason = db._cache_dir_verdict("/mnt/data/flash", p)
        assert ok, reason
        ok, reason = db._cache_dir_verdict("/home/user/.cache/bnb", p)
        assert ok, reason
    finally:
        os.unlink(p)


def test_wsl_host_mount_rule():
    # Even if a host mount somehow reports a safe fstype, WSL's /mnt/<letter> is rejected.
    ok, reason = db._fs_safety("ext4", "/mnt/c", "rw", is_wsl=True)
    assert not ok and "WSL" in reason, reason
    ok, reason = db._fs_safety("ext4", "/", "rw", is_wsl=True)
    assert ok, reason
    ok, reason = db._fs_safety("ext4", "/mnt/wslg/distro", "ro", is_wsl=True)
    assert not ok, reason


def test_cache_name():
    assert db._cache_name_for_mount("/") == "root"
    assert db._cache_name_for_mount("/mnt/data") == "mnt_data"
    assert db._cache_name_for_mount("C:") == "C"


def main():
    test_mount_lookup()
    test_cache_verdicts()
    test_wsl_host_mount_rule()
    test_cache_name()
    print("DISK BALANCER LINUX/WSL SAFETY UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
