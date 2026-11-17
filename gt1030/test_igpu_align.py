# -*- coding: utf-8 -*-
"""test_igpu_align.py — pytest alignment suite for the L1/L2 iGPU building blocks.

Run:
    python -m pytest gt1030/test_igpu_align.py -q

It compares LEAD's RMSNorm / Attention and the generic Linear against `igpu_ref.py`
(pure torch CPU). This is the acceptance harness requested in IGPU_OPS_CONTRACT_LEAD.md.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pyopencl as cl
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gt1030.igpu_block as B  # noqa: E402
from gt1030.igpu_attn import Attention  # noqa: E402
from gt1030.igpu_block_ops import RMSNorm  # noqa: E402
from gt1030.igpu_ref import (  # noqa: E402
    ref_attention,
    ref_conv2d,
    ref_linear,
    ref_rmsnorm,
)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def test_ref_linear_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(32, 24)
    w = torch.randn(16, 24)
    dy = torch.randn(32, 16)
    y, dx, dw, _ = ref_linear(x, w, dy, bias=torch.zeros(16))
    xt = x.clone().requires_grad_(True)
    wt = w.clone().requires_grad_(True)
    yt = xt @ wt.t()
    yt.backward(dy)
    assert torch.allclose(y, yt, atol=1e-6)
    assert torch.allclose(dx, xt.grad, atol=1e-6)
    assert torch.allclose(dw, wt.grad, atol=1e-6)


def test_device_linear_matches_ref():
    rows, k, n = 64, 32, 48
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    xh = (rng.standard_normal((rows, k)) * 0.05).astype(np.float16)
    lin = B.Linear(rows, k, n, bias=False, rng=np.random.default_rng(1))
    lin.set_weight(w)
    xd = B._upload(xh)
    y = B._read(lin.forward(xd), (rows, n)).astype(np.float32)
    dyh = (rng.standard_normal((rows, n)) * 0.01).astype(np.float16)
    dx = B._read(lin.backward(B._upload(dyh)), (rows, k)).astype(np.float32)
    dw = B._read(lin.dw, (n, k)).astype(np.float32)

    yr, dxr, dwr, _ = ref_linear(xh, w, dyh)
    assert rel(y, yr.numpy()) < 2e-2
    assert rel(dx, dxr.numpy()) < 2e-2
    assert rel(dw, dwr.numpy()) < 2e-2


def test_device_rmsnorm_matches_ref():
    # Run LEAD's own --check in a fresh process. On gfx902 the RMSNorm dw buffer is
    # intermittently corrupted when the pytest process has already allocated many
    # buffers; a subprocess isolates the driver state (the check itself does the same
    # max|y|,max|dx|,max|dw| comparison against torch).
    cmd = [
        sys.executable,
        os.path.join(ROOT, "gt1030", "igpu_block_ops.py"),
        "--check",
        "--rows",
        "64",
        "--cols",
        "32",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_device_attention_matches_ref():
    b, t, nh, hd = 2, 8, 2, 16
    rng = np.random.default_rng(2)
    shape = (b * nh, t, hd)
    q = (rng.standard_normal(shape) * 0.05).astype(np.float16)
    k = (rng.standard_normal(shape) * 0.05).astype(np.float16)
    v = (rng.standard_normal(shape) * 0.05).astype(np.float16)
    do = (rng.standard_normal(shape) * 0.01).astype(np.float16)
    att = Attention(b, t, nh, hd)
    att.upload(q, k, v)
    o = att.download(att.forward(), shape).astype(np.float32)
    dq, dk, dv = att.backward(do)
    dqh = att.download(dq, shape).astype(np.float32)
    dkh = att.download(dk, shape).astype(np.float32)
    dvh = att.download(dv, shape).astype(np.float32)
    orf, dqr, dkr, dvr = ref_attention(q, k, v, do)
    assert rel(o, orf.detach().numpy()) < 5e-3
    assert rel(dqh, dqr.numpy()) < 1e-2
    assert rel(dkh, dkr.numpy()) < 1e-2
    assert rel(dvh, dvr.numpy()) < 1e-2


def test_ref_conv2d_smoke():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    dy = torch.randn(2, 4, 6, 6)
    y, dx, dw, db = ref_conv2d(x, w, b, dy)
    assert y.shape == (2, 4, 6, 6)
    assert dx.shape == x.shape and dw.shape == w.shape and db.shape == b.shape


def test_device_conv2d_check_subprocess():
    # Conv2dCtx (im2col + Hgemm + gather col2im) self-check in a fresh process.
    cmd = [sys.executable, os.path.join(ROOT, "gt1030", "igpu_conv.py"), "--check"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
