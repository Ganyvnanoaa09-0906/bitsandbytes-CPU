"""10-layer AR Hgemm trainer with device AdamW: fp16 GEMM chain + fp32 host master/AdamW, host head/CE.

First correctness vehicle for the Hgemm switch:
  - 10 layers: fp16 CLBlast Hgemm (full_gpu_model Hgemm layer), device checkpoints, half activations
  - weights: host fp32 master -> cast fp16 for device upload; dW downloaded half -> host AdamW fp32
  - embedding/final RMSNorm/lm_head/CE on host (fp32)
  - CPU fp32 reference with frozen per-layer RMSNorm (device layer uses shared ones) for isolation
Goal: 5-step loss vs CPU fp32 reference; then measure step time.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import pyopencl as cl
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM  # noqa: E402  (Hgemm)
import gt1030.full_gpu_layer as FMeta  # noqa: E402
from gt1030.full_gpu_train_sgemm import CpuRef, rms, TOKENS  # noqa: E402


def adamw(param, grad, state, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, t=1):
    m, v = state
    m *= b1; m += (1 - b1) * grad
    v *= b2; v += (1 - b2) * grad * grad
    param -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps) + lr * wd * param

M, MD, FF, WROW = FM.M, FM.MD, FM.FF, FM.WROW
B, T, NH, NKV, HD = FMeta.B, FMeta.T, FMeta.NH, FMeta.NKV, FMeta.HD
V = 1024
A = FMeta.A
ROWS = 8 * 256


def main():
    torch.set_num_threads(6)
    steps = int(__import__('os').environ.get('STEPS', '5'))
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ref = CpuRef()
    hosts = []
    with torch.no_grad():
        ref0 = CpuRef()
        ref.load_state_dict(ref0.state_dict())
        for b in ref.layers:
            b.ln1.requires_grad_(False); b.ln2.requires_grad_(False)
            hosts.append(dict(
                WQKV=np.ascontiguousarray(torch.cat([b.qi.weight, b.ki.weight, b.vi.weight], 0).T.detach().numpy().astype(np.float32)),
                WO=np.ascontiguousarray(b.o.weight.T.detach().numpy().astype(np.float32)),
                W1=np.ascontiguousarray(b.w1.weight.T.detach().numpy().astype(np.float32)),
                W3=np.ascontiguousarray(b.w3.weight.T.detach().numpy().astype(np.float32)),
                W2=np.ascontiguousarray(b.w2.weight.T.detach().numpy().astype(np.float32)),
            ))
        emb = ref.emb.weight.detach().numpy().astype(np.float32)
        norm = ref.norm.detach().numpy().astype(np.float32)
        head = ref.head.weight.detach().numpy().astype(np.float32)

    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-4, weight_decay=0.0)
    states = [[{k: [np.zeros_like(v), np.zeros_like(v)] for k, v in h.items()}][0] for h in hosts]
    st_emb = [np.zeros_like(emb), np.zeros_like(emb)]
    st_norm = [np.zeros_like(norm), np.zeros_like(norm)]
    st_head = [np.zeros_like(head), np.zeros_like(head)]

    tokens = torch.load(TOKENS, map_location='cpu')[:, :T + 1]
    data = tokens.numpy()
    dev = FM.Layer(rng, 0)
    input_buf = FM.buf(ROWS * MD * 2)
    grad_buf = FM.buf(ROWS * MD * 2)
    ckpt = FM.buf(11 * ROWS * MD * 2)
    ROWB = ROWS * MD * 2
    A.cl.enqueue_copy(A.queue, ckpt, np.zeros((11 * ROWS, MD), dtype=np.float16))
    A.queue.finish()

    # ---- device AdamW arena + master->fp16 weight conversion ----
    DEV_KERNEL = r"""
__kernel void f32_to_f16(__global const float* src, __global half* dst, int off, int n) {
    int i = get_global_id(0); if (i >= n) return; dst[i] = (half)src[off + i];
}
__kernel void adamw_h(__global float* master, __global float* mom, __global float* vel,
                      __global const half* grad, int off, int n,
                      float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int i = get_global_id(0); if (i >= n) return; int j = off + i;
    float g = (float)grad[i];
    float m = b1 * mom[j] + (1.f - b1) * g;
    float v = b2 * vel[j] + (1.f - b2) * g * g;
    mom[j] = m; vel[j] = v;
    master[j] -= lr * (m / bc1) / (sqrt(v / bc2) + eps) + lr * wd * master[j];
}
"""
    prog_dev = cl.Program(A.ctx, DEV_KERNEL).build(options=['-cl-mad-enable'])
    K_F2H = cl.Kernel(prog_dev, 'f32_to_f16')
    K_AH = cl.Kernel(prog_dev, 'adamw_h')
    keys = ('WQKV', 'WO', 'W1', 'W3', 'W2')
    layout = []; total = 0
    for h in hosts:
        d = {}
        for k in keys:
            d[k] = (total, h[k].size); total += h[k].size
        layout.append(d)
    master_host = np.concatenate([h[k].reshape(-1) for h in hosts for k in keys])
    master_buf = FM.buf(total * 4); mom_buf = FM.buf(total * 4); vel_buf = FM.buf(total * 4)
    A.cl.enqueue_copy(A.queue, master_buf, master_host)
    A.cl.enqueue_copy(A.queue, mom_buf, np.zeros(total, dtype=np.float32))
    A.cl.enqueue_copy(A.queue, vel_buf, np.zeros(total, dtype=np.float32))
    A.queue.finish()

    def upload_weights(i):
        for k in keys:
            off, n = layout[i][k]
            K_F2H(A.queue, ((n + 255) // 256 * 256,), (256,), master_buf,
                  dev.__dict__[k]['dev'], np.int32(off), np.int32(n))
        dev.b['wqkv'] = dev.WQKV['dev']; dev.b['wo'] = dev.WO['dev']
        dev.b['w1'] = dev.W1['dev']; dev.b['w3'] = dev.W3['dev']; dev.b['w2'] = dev.W2['dev']

    losses_gpu, losses_cpu = [], []
    wall0 = time.perf_counter()
    for step in range(1, steps + 1):
        tstep = time.perf_counter()
        idx = rng.integers(0, data.shape[0], B)
        inp = torch.from_numpy(data[idx, :-1].astype(np.int64))
        tgt = torch.from_numpy(data[idx, 1:].astype(np.int64))
        opt_ref.zero_grad()
        logits_cpu = ref(inp)
        loss_cpu = F.cross_entropy(logits_cpu, tgt.reshape(-1))
        loss_cpu.backward(); opt_ref.step()
        losses_cpu.append(float(loss_cpu))

        x0 = emb[inp.numpy().reshape(-1)].astype(np.float16)  # (ROWS, MD)
        A.cl.enqueue_copy(A.queue, ckpt, x0)
        for i in range(10):
            upload_weights(i)
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB, src_offset=i * ROWB, dst_offset=0)
            out = dev.forward(input_buf)
            A.cl.enqueue_copy(A.queue, ckpt, out, byte_count=ROWB, src_offset=0, dst_offset=(i + 1) * ROWB)
        hidden = np.empty((ROWS, MD), dtype=np.float16)
        A.cl.enqueue_copy(A.queue, hidden, ckpt, src_offset=10 * ROWB)
        A.queue.finish()
        hidden32 = hidden.astype(np.float32)
        ht = torch.from_numpy(hidden32).requires_grad_(True)
        norm_t = torch.from_numpy(norm.copy()).requires_grad_(True)
        head_t = torch.from_numpy(head.copy()).requires_grad_(True)
        logits_gpu = rms(ht, norm_t) @ head_t.T
        loss_gpu = F.cross_entropy(logits_gpu, tgt.reshape(-1))
        loss_gpu.backward()
        dhidden = ht.grad.numpy().astype(np.float16)
        gnorm = norm_t.grad.numpy().astype(np.float32)
        ghead = head_t.grad.numpy().astype(np.float32)
        A.cl.enqueue_copy(A.queue, grad_buf, dhidden)
        for i in reversed(range(10)):
            upload_weights(i)
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB, src_offset=i * ROWB, dst_offset=0)
            dev.forward(input_buf)
            dx = dev.backward(grad_buf)
            A.cl.enqueue_copy(A.queue, grad_buf, dx, byte_count=ROWB)
            bc1 = 1.0 - 0.9 ** step; bc2 = 1.0 - 0.999 ** step
            for k, dn in (('WQKV', 'dWqkv'), ('WO', 'dWo'), ('W1', 'dW1'), ('W3', 'dW3'), ('W2', 'dW2')):
                off, n = layout[i][k]
                K_AH(A.queue, ((n + 255) // 256 * 256,), (256,), master_buf, mom_buf, vel_buf,
                     dev.b[dn], np.int32(off), np.int32(n), np.float32(1e-4),
                     np.float32(0.9), np.float32(0.999), np.float32(1e-8), np.float32(0.0),
                     np.float32(bc1), np.float32(bc2))
        dxin = np.empty((ROWS, MD), dtype=np.float16)
        A.cl.enqueue_copy(A.queue, dxin, grad_buf)
        A.queue.finish()
        gemb = np.zeros_like(emb)
        flat = inp.numpy().reshape(-1)
        np.add.at(gemb, flat, dxin.astype(np.float32))
        adamw(emb, gemb, st_emb, t=step)
        adamw(norm, gnorm, st_norm, t=step)
        adamw(head, ghead, st_head, t=step)
        dt = time.perf_counter() - tstep
        losses_gpu.append(float(loss_gpu))
        if step % 1 == 0:
            print('step %2d loss_cpu %.6f loss_gpu %.6f diff %+.6f wall %.2fs' %
                  (step, losses_cpu[-1], losses_gpu[-1], losses_gpu[-1] - losses_cpu[-1], dt), flush=True)
    print('final loss diff %.6f (steps=%d) wall %.1fs' %
          (losses_gpu[-1] - losses_cpu[-1], steps, time.perf_counter() - wall0))


if __name__ == '__main__':
    main()
