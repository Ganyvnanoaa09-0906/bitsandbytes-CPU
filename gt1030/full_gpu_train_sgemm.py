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
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model_sgemm as FM  # noqa: E402
import gt1030.full_gpu_layer as FMeta  # noqa: E402
from gt1030.debug_full_gpu_model import cpu_layer  # noqa: E402

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
        self.last_hidden = x
        return self.head(rms(x, self.norm)).reshape(-1, V)


def main():
    torch.set_num_threads(6)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ref = CpuRef()
    # host fp16 transformer weights (float32 copies for CPU reference are the same values)
    hosts = []
    with torch.no_grad():
        ref0 = CpuRef()
        ref.load_state_dict(ref0.state_dict())
        for b in ref.layers:
            hosts.append(dict(
                WQKV=np.ascontiguousarray(torch.cat([b.qi.weight, b.ki.weight, b.vi.weight], 0).T.detach().numpy().astype(np.float32)),
                WO=np.ascontiguousarray(b.o.weight.T.detach().numpy().astype(np.float32)),
                W1=np.ascontiguousarray(b.w1.weight.T.detach().numpy().astype(np.float32)),
                W3=np.ascontiguousarray(b.w3.weight.T.detach().numpy().astype(np.float32)),
                W2=np.ascontiguousarray(b.w2.weight.T.detach().numpy().astype(np.float32)),
                LN1=np.ascontiguousarray(b.ln1.detach().numpy().astype(np.float32)),
                LN2=np.ascontiguousarray(b.ln2.detach().numpy().astype(np.float32)),
            ))
        emb = ref.emb.weight.detach().numpy().astype(np.float32)
        norm = ref.norm.detach().numpy().astype(np.float32)
        head = ref.head.weight.detach().numpy().astype(np.float32)

    # CPU reference optimizer
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-4, weight_decay=0.0)
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

    # ---- device optimizer: one fp32 master/m/v arena for all 10 layers ----
    OPT_KERNEL = r"""
__kernel void adamw_off(__global float* master, __global float* mom, __global float* vel,
                        __global const float* grad, int off, int n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    int j = off + i;
    float g = grad[i];
    float m = b1 * mom[j] + (1.f - b1) * g;
    float v = b2 * vel[j] + (1.f - b2) * g * g;
    mom[j] = m; vel[j] = v;
    master[j] -= lr * (m / bc1) / (sqrt(v / bc2) + eps) + lr * wd * master[j];
}
"""
    prog_opt = cl.Program(A.ctx, OPT_KERNEL).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
    K_OPT = cl.Kernel(prog_opt, 'adamw_off')
    keys = ('WQKV', 'WO', 'W1', 'W3', 'W2')
    layout = []
    total = 0
    for h in hosts:
        d = {}
        for k in keys:
            d[k] = (total, h[k].size)
            total += h[k].size
        layout.append(d)
    master_host = np.concatenate([h[k].reshape(-1) for h in hosts for k in keys])
    master_buf = FM.buf(total * 4)
    mom_buf = FM.buf(total * 4)
    vel_buf = FM.buf(total * 4)
    A.cl.enqueue_copy(A.queue, master_buf, master_host)
    A.cl.enqueue_copy(A.queue, mom_buf, np.zeros(total, dtype=np.float32))
    A.cl.enqueue_copy(A.queue, vel_buf, np.zeros(total, dtype=np.float32))
    A.queue.finish()

    def upload_weights(i):
        h = hosts[i]
        for k in keys:
            off, n = layout[i][k]
            A.cl.enqueue_copy(A.queue, dev.__dict__[k]['dev'], master_buf,
                              byte_count=n * 4, src_offset=off * 4, dst_offset=0)
        dev.b['wqkv'] = dev.WQKV['dev']; dev.b['wo'] = dev.WO['dev']
        dev.b['w1'] = dev.W1['dev']; dev.b['w3'] = dev.W3['dev']; dev.b['w2'] = dev.W2['dev']
        A.cl.enqueue_copy(A.queue, dev.b['wnorm1'], h['LN1'])
        A.cl.enqueue_copy(A.queue, dev.b['wnorm2'], h['LN2'])

    def adamw(param, grad, state, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, t=1):
        m, v = state
        m *= b1; m += (1 - b1) * grad
        v *= b2; v += (1 - b2) * grad * grad
        param -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps) + lr * wd * param

    losses_gpu, losses_cpu = [], []
    tm = {'cpu_ref': 0.0, 'head': 0.0, 'opt': 0.0}
    t0 = time.perf_counter()
    for step in range(1, int(__import__('os').environ.get('STEPS', '50')) + 1):
        idx = rng.integers(0, data.shape[0], B)
        batch = torch.from_numpy(data[idx, :T + 1])
        inp, tgt = batch[:, :-1], batch[:, 1:]
        tcr = time.perf_counter()
        # CPU reference step
        opt_ref.zero_grad()
        logits_cpu = ref(inp)
        loss_cpu = F.cross_entropy(logits_cpu, tgt.reshape(-1))
        loss_cpu.backward(); opt_ref.step()
        losses_cpu.append(float(loss_cpu))
        tm['cpu_ref'] += time.perf_counter() - tcr

        # device path: embedding (host) -> transformer (iGPU) -> final norm/head/CE (host)
        x0 = emb[inp.numpy().reshape(-1)]  # (ROWS, MD)
        A.cl.enqueue_copy(A.queue, ckpt, x0)
        for i in range(10):
            upload_weights(i)
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB,
                              src_offset=i * ROWB, dst_offset=0)
            out = dev.forward(input_buf)
            A.cl.enqueue_copy(A.queue, ckpt, out, byte_count=ROWB,
                              src_offset=0, dst_offset=(i + 1) * ROWB)
            if step == 1 and i == 0:
                h0 = np.empty((ROWS, MD), dtype=np.float32)
                A.cl.enqueue_copy(A.queue, h0, ckpt, src_offset=ROWB)
                A.queue.finish()
                hw = hosts[0]
                cref = cpu_layer(torch.from_numpy(x0), torch.from_numpy(hw['WQKV']),
                                 torch.from_numpy(hw['WO']), torch.from_numpy(hw['W1']),
                                 torch.from_numpy(hw['W3']), torch.from_numpy(hw['W2']))
                print('layer0 maxerr %.3e rel %.3e' % (
                    float(np.abs(h0 - cref.numpy()).max()),
                    float(np.abs(h0 - cref.numpy()).max()/(np.abs(cref.numpy()).max()+1e-9))), flush=True)
        hidden = np.empty((ROWS, MD), dtype=np.float32)
        A.cl.enqueue_copy(A.queue, hidden, ckpt, src_offset=10 * ROWB)
        A.queue.finish()

        if step == 1:
            hc = ref.last_hidden.detach().numpy().reshape(ROWS, MD)
            print('hidden maxerr %.3e rel %.3e' % (float(np.abs(hidden - hc).max()),
                  float(np.abs(hidden - hc).max() / (np.abs(hc).max() + 1e-9))), flush=True)
        tcm = time.perf_counter()
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
        tm['head'] += time.perf_counter() - tcm
        A.cl.enqueue_copy(A.queue, grad_buf, dhidden)
        for i in reversed(range(10)):
            upload_weights(i)
            A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROWB,
                              src_offset=i * ROWB, dst_offset=0)
            dev.forward(input_buf)
            dx = dev.backward(grad_buf)
            A.cl.enqueue_copy(A.queue, grad_buf, dx, byte_count=ROWB)
            dnames = {'WQKV': 'dWqkv', 'WO': 'dWo', 'W1': 'dW1', 'W3': 'dW3', 'W2': 'dW2'}
            top = time.perf_counter()
            bc1 = 1.0 - 0.9 ** step
            bc2 = 1.0 - 0.999 ** step
            for k, dn in dnames.items():
                off, n = layout[i][k]
                K_OPT(A.queue, (((n + 255) // 256) * 256,), (256,),
                      master_buf, mom_buf, vel_buf, dev.b[dn],
                      np.int32(off), np.int32(n), np.float32(1e-4),
                      np.float32(0.9), np.float32(0.999), np.float32(1e-8),
                      np.float32(0.0), np.float32(bc1), np.float32(bc2))
            for lname, dbuf in (('LN1', 'dw1'), ('LN2', 'dw2')):
                g = np.empty_like(hosts[i][lname])
                A.cl.enqueue_copy(A.queue, g, dev.b[dbuf])
                A.queue.finish()
                adamw(hosts[i][lname], g, states[i][lname], t=step)
            tm['opt'] += time.perf_counter() - top
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
        top2 = time.perf_counter()
        losses_gpu.append(float(loss_gpu))
        if step % 10 == 1 or step == 50:
            print('step %2d loss_cpu %.6f loss_gpu %.6f diff %+.6f' %
                  (step, losses_cpu[-1], losses_gpu[-1], losses_gpu[-1] - losses_cpu[-1]), flush=True)
    print('final loss diff %.6f (steps=%d)' % (losses_gpu[-1] - losses_cpu[-1], len(losses_gpu)))
    tot = time.perf_counter() - t0
    print('time/step ms: cpu_ref %.1f, head/loss/bwd %.1f, host_opt %.1f, other/device %.1f, wall %.1f' % (
        tm['cpu_ref']/len(losses_gpu)*1e3, tm['head']/len(losses_gpu)*1e3,
        tm['opt']/len(losses_gpu)*1e3,
        (tot - tm['cpu_ref'] - tm['head'] - tm['opt'])/len(losses_gpu)*1e3, tot/len(losses_gpu)*1e3))
    print('wall %.1fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
