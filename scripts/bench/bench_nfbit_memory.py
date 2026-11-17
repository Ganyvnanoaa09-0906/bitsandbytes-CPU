# -*- coding: utf-8 -*-
"""Measured RSS vs the nfbit storage arithmetic (closes LEAD's gap #3).

Reports, with GC between phases:
  * fp32 state dict loaded in memory;
  * packed NF2 payload in memory (pack_state_dict) and on disk;
  * unpacked fp32 (the temporary a forward would need);
  * one 4096x4096 QuantLinearNF forward: storage vs measured RSS delta vs
    forward_peak_bytes().
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bitsandbytes"))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from bitsandbytes import nfbit  # noqa: E402

CKPT = r"D:\work\cloud_salvage\oWAZY8\ckpt_3000.pt"


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / 2**20


def settle():
    gc.collect()
    return rss_mb()


def main():
    torch.set_num_threads(4)
    base = settle()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if torch.is_tensor(v)}
    fp32_mb = sum(v.numel() * v.element_size() for v in sd.values()) / 2**20
    after_sd = settle()
    print(
        "RSS base %.1f MB -> after fp32 state dict %.1f MB (delta %.1f, tensors %.1f)"
        % (base, after_sd, after_sd - base, fp32_mb)
    )

    packed = nfbit.pack_state_dict(sd, bits=2, block=64)
    after_pack = settle()
    theory = nfbit.packed_state_bytes(packed) / 2**20
    print(
        "after pack_state_dict %.1f MB (delta %.1f, arithmetic %.1f)"
        % (after_pack, after_pack - after_sd, theory)
    )

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.nf2.pt")
        torch.save(packed, p)
        file_mb = os.path.getsize(p) / 2**20
        disk = torch.load(p, map_location="cpu", weights_only=False)
        after_load = settle()
        print(
            "on disk %.1f MB; after torch.load %.1f MB (delta %.1f)"
            % (file_mb, after_load, after_load - after_pack)
        )
        del packed
        un = nfbit.unpack_state_dict(disk)
        after_unpack = settle()
        del disk
        print(
            "after unpack_state_dict %.1f MB (delta %.1f, fp32 expected %.1f)"
            % (after_unpack, after_unpack - after_load, fp32_mb)
        )
        del un

    # one big layer: storage vs measured RSS delta vs forward_peak_bytes()
    lin = torch.nn.Linear(4096, 4096)
    ql = nfbit.QuantLinearNF.from_linear(lin, bits=2, block=64, lora_r=8)
    del lin
    before = settle()
    x = torch.randn(4, 4096)
    with torch.no_grad():
        y = ql(x)
    after = settle()
    print(
        "layer 4096x4096: storage %.1f MB | forward_peak_bytes %.1f MB | "
        "measured RSS delta %.1f MB | out %s"
        % (
            ql.storage_bytes() / 2**20,
            ql.forward_peak_bytes(batch=4) / 2**20,
            after - before,
            tuple(y.shape),
        )
    )
    assert y.shape == (4, 4096)


if __name__ == "__main__":
    main()
