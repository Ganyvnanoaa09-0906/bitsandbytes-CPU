"""50-step AR training: SGEMM device transformer + host embedding/head/CE/AdamW.

Interim correctness loop (not the final device-resident optimizer):
  - transformer 10 layers: fp32 SGEMM on iGPU, shared device weights, device checkpoint
  - embedding, final RMSNorm, lm_head, CE loss, AdamW: host fp32 (for the 50-step alignment)
  - real tokens from samples_step2000.tokens.pt (8 x 1024, vocab 1024)
  - CPU fp32 reference model with identical init/data/optimizer
Goal: loss trajectory difference < 0.005 after 50 steps; then move optimizer/head to device.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model_sgemm as FM  # noqa: E402
import gt1030.full_gpu_layer as FMeta  # noqa: E402

M, MD, FF, WROW = FM.M, FM.MD, FM.FF, FM.WROW
B, T, NH, NKV, HD = FMeta.B, FMeta.T, FMeta.NH, FMeta.NKV, FMeta.HD
V = 1024
A = FMeta.A
TOKENS = r'D:\work\bitsandbytes-CPU\i5build\ar_mgit\samples_step2000.tokens.pt'
ROWS = 8 * 256


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class CpuRef(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, MD)
        self.layers = nn.ModuleList()
        for _ in range(10):
            b = nn.Module()
            b.ln1 = nn.Parameter(torch.ones(MD)); b.ln2 = nn.Parameter(torch.ones(MD))
            b.qi = nn.Linear(MD, MD, bias=False)
            b.ki = nn.Linear(MD, NKV * HD, bias=False)
            b.vi = nn.Linear(MD, NKV * HD, bias=False)
            b.o = nn.Linear(MD, MD, bias=False)
            b.w1 = nn.Linear(MD, FF, bias=False)
            b.w3 = nn.Linear(MD, FF, bias=False)
            b.w2 = nn.Linear(FF, MD, bias=False)
            self.layers.append(b)
        self.norm = nn.Parameter(torch.ones(MD))
        self.head = nn.Linear(MD, V, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.layers:
            h = rms(x, b.ln1)
            q = b.qi(h).view(B, T, NH, HD).transpose(1, 2)
            k = b.ki(h).view(B, T, NKV, HD).transpose(1, 2).repeat_interleave(NH // NKV, dim=1)
            v = b.vi(h).view(B, T, NKV, HD).transpose(1, 2).repeat_interleave(NH // NKV, dim=1)
            s = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
            s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
            a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, MD)
            x = x + b.o(a)
            h = rms(x, b.ln2)
            x = x + b.w2(F.silu(b.w1(h)) * b.w3(h))
        return self.head(rms(x, self.norm)).reshape(-1, V)


def main():
    torch.set_num_threads(6)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ref = CpuRef()
    # The device layer currently uses one shared ones vector for both RMSNorm weights
    # (it does not yet upload per-layer norm weights). Freeze the CPU reference's layer
    # norms to ones so the 50-step comparison isolates the GEMM/optimizer path.
    for b in ref.layers:
        b.ln1.requires_grad_(False); b.ln2.requires_grad_(False)
    # host fp16 transformer weights (float32 copies for CPU reference are the same values)
    hosts = []
    with torch.no_grad():
        ref0 = CpuRef()
        ref.load_state_dict(ref0.state_dict())
        for b in ref.layers:
            hosts.append(dict(
                WQKV=torch.cat([b.qi.weight, b.ki.weight, b.vi.weight], 0).detach().numpy().astype(np.float32),
                WO=b.o.weight.detach().numpy().astype(np.float32),
                W1=b.w1.weight.detach().numpy().astype(np.float32),
                W3=b.w3.weight.detach().numpy().astype(np.float32),
                W2=b.w2.weight.detach().numpy().astype(np.float32),
            ))
        emb = ref.emb.weight.detach().numpy().astype(np.float32)
        norm = ref.norm.detach().numpy().astype(np.float32)
        head = ref.head.weight.detach().numpy().astype(np.float32)

    # CPU reference optimizer
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-4)
    # host optimizer states for device transformer + embedding/norm/head
    states = []
    for h in hosts:
        states.append({k: [np.zeros_like(v), np.zeros_like(v)] for k, v in h.items()})
    st_emb = [np.zeros_like(emb), np.zeros_like(emb)]
    st_norm = [np.zeros_like(norm), np.zeros_like(norm)]
    st_head = [np.zeros_like(head), np.zeros_like(head)]

    tokens = torch.load(TOKENS, map_location='cpu')[:, :T + 1]
    data = tokens.numpy()
    dev = FM.Layer(rng, 0)
    input_buf = FM.buf(ROWS * MD * 4)
    grad_buf = FM.buf(ROWS * MD * 4)
    ckpt = FM.buf(11 * ROWS * MD * 4)
    ROWB = ROWS * MD * 4
    A.cl.enqueue_copy(A.queue, ckpt, np.zeros((11 * ROWS, MD), dtype=np.float32))
    A.queue.finish()

    def upload_weights(h):
        for k in ('WQKV', 'WO', 'W1', 'W3', 'W2'):
            A.cl.enqueue_copy(A.queue, dev.__dict__[k]['dev'], h[k])
        dev.b['wqkv'] = dev.WQKV['dev']; dev.b['wo'] = dev.WO['dev']
        dev.b['w1'] = dev.W1['dev']; dev.b['w3'] = dev.W3['dev']; dev.b['w2'] = dev.W2['dev']

    def adamw(param, grad, state, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, t=1):
        m, v = state
        m *= b1; m += (1 - b1) * grad
        v *= b2; v += (1 - b2) * grad * grad
        param -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps) + lr * wd * param

    losses_gpu, losses_cpu = [], []
    t0 = time.perf_counter()
    for step in range(1, int(__import__('os').environ.get('STEPS', '50')) + 1):
        idx = rng.integers(0, data.shape[0], B)
        batch = torch.from_numpy(data[idx, :T + 1])
        inp, tgt = batch[:, :-1], batch[:, 1:]
        # CPU reference step
        opt_ref.zero_grad()
        logits_cpu = ref(inp)
        loss_cpu = F.cross_entropy(logits_cpu, tgt.reshape(-1))
        loss_cpu.backward(); opt_ref.step()
        losses_cpu.append(float(loss_cpu))

        # device path: embedding (host) -> transformer (iGPU) -> final norm/head/CE (host)
        x0 = emb[inp.numpy().reshape(-1)]  # (ROWS, MD)
        A.cl.enqueue_copy(A.queue, ckpt, x0)
        for i in range(10):
            upload_weights(hosts[i])
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB,
                              src_offset=i * ROWB, dst_offset=0)
            out = dev.forward(input_buf)
            A.cl.enqueue_copy(A.queue, ckpt, out, byte_count=ROWB,
                              src_offset=0, dst_offset=(i + 1) * ROWB)
        hidden = np.empty((ROWS, MD), dtype=np.float32)
        A.cl.enqueue_copy(A.queue, hidden, ckpt, src_offset=10 * ROWB)
        A.queue.finish()

        ht = torch.from_numpy(hidden).requires_grad_(True)
        norm_t = torch.from_numpy(norm.copy()).requires_grad_(True)
        head_t = torch.from_numpy(head.copy()).requires_grad_(True)
        hnorm = rms(ht, norm_t)
        logits_gpu = hnorm @ head_t.T
        loss_gpu = F.cross_entropy(logits_gpu, tgt.reshape(-1))
        loss_gpu.backward()
        dhidden = ht.grad.numpy().astype(np.float32)
        gnorm = norm_t.grad.numpy().astype(np.float32)
        ghead = head_t.grad.numpy().astype(np.float32)
        A.cl.enqueue_copy(A.queue, grad_buf, dhidden)
        for i in reversed(range(10)):
            upload_weights(hosts[i])
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB,
                              src_offset=i * ROWB, dst_offset=0)
            dev.forward(input_buf)
            dx = dev.backward(grad_buf)
            A.cl.enqueue_copy(A.queue, grad_buf, dx, byte_count=ROWB)
            dnames = {'WQKV': 'dWqkv', 'WO': 'dWo', 'W1': 'dW1', 'W3': 'dW3', 'W2': 'dW2'}
            for k, dn in dnames.items():
                g = np.empty_like(hosts[i][k])
                A.cl.enqueue_copy(A.queue, g, dev.b[dn])
                A.queue.finish()
                adamw(hosts[i][k], g, states[i][k], t=step)
        # embedding grad: dx at input buffer
        dxin = np.empty((ROWS, MD), dtype=np.float32)
        A.cl.enqueue_copy(A.queue, dxin, grad_buf)
        A.queue.finish()
        gemb = np.zeros_like(emb)
        flat = inp.numpy().reshape(-1)
        np.add.at(gemb, flat, dxin)
        adamw(emb, gemb, st_emb, t=step)
        adamw(norm, gnorm, st_norm, t=step)
        adamw(head, ghead, st_head, t=step)
        losses_gpu.append(float(loss_gpu))
        if step % 10 == 1 or step == 50:
            print('step %2d loss_cpu %.6f loss_gpu %.6f diff %+.6f' %
                  (step, losses_cpu[-1], losses_gpu[-1], losses_gpu[-1] - losses_cpu[-1]), flush=True)
    print('final loss diff %.6f (steps=%d)' % (losses_gpu[-1] - losses_cpu[-1], len(losses_gpu)))
    print('wall %.1fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
