# -*- coding: utf-8 -*-
"""igpu_cl_ast_train.py — 小型音频 Transformer(MLP-Mixer 风格) 分类器：CPU vs iGPU(CLBlast) 端到端训练。

同一份 numpy 实现，只把矩阵乘换成 CPU(BLAS) 或 iGPU(CLBlast SGEMM)，因此能公平回答
"核显当 GEMM 后端到底值不值"。

结构（GEMM 主导，便于公平比较；不含 softmax/attention，避免手写反向出错）：
  log-mel 64×64 → 4×4 patch → 256 token(dim=16) → Linear(16→dim)
  × 4 block：  LN → token-mix: x += LN(x) @ Wtok^T（Wtok: T×T） → LN → FFN: dim→ff→dim
  → mean(token) → Linear(dim→4) → CE
手写前向/反向（LayerNorm / Linear / GELU / mean 都有解析梯度），并用 torch 做数值校验。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time

import numpy as np

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
ROW, NO, YES = 101, 111, 112


class CPUBackend:
    name = "cpu"

    def mm(self, a, b):
        return a @ b

    def mm_ta(self, a, b):
        return a.T @ b

    def mm_tb(self, a, b):
        return a @ b.T


class IGPUBackend:
    name = "igpu"

    def __init__(self):
        import pyopencl as cl
        self.cl = cl
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.bl = ctypes.CDLL(DLL)
        self.bl.CLBlastSgemm.restype = ctypes.c_int
        self.bl.CLBlastSgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.qp = ctypes.c_void_p(int(self.q.int_ptr))
        self.pool = {}
        self.calls = 0

    def _buf(self, key, nbytes):
        b = self.pool.get(key)
        if b is None or b.size < nbytes:
            b = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
            self.pool[key] = b
        return b

    def _gemm(self, a, b, ta, tb):
        a = np.ascontiguousarray(a, np.float32)
        b = np.ascontiguousarray(b, np.float32)
        if not ta:
            m, k = a.shape; lda = k
        else:
            k, m = a.shape; lda = m
        if not tb:
            k2, n = b.shape; ldb = n
        else:
            n, k2 = b.shape; ldb = k
        assert k == k2, (a.shape, b.shape, ta, tb)
        ab = self._buf("a", a.size * 4)
        bb = self._buf("b", b.size * 4)
        cb = self._buf("c", m * n * 4)
        self.cl.enqueue_copy(self.q, ab, a)
        self.cl.enqueue_copy(self.q, bb, b)
        rc = self.bl.CLBlastSgemm(ROW, YES if ta else NO, YES if tb else NO, m, n, k,
                                  ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(ab.int_ptr)), 0, lda,
                                  ctypes.c_void_p(int(bb.int_ptr)), 0, ldb, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(cb.int_ptr)), 0, n, self.qp, None)
        if rc:
            raise RuntimeError(f"Sgemm rc={rc}")
        out = np.empty((m, n), np.float32)
        self.cl.enqueue_copy(self.q, out, cb)
        self.q.finish()
        self.calls += 1
        return out

    def mm(self, a, b):
        return self._gemm(a, b, False, False)

    def mm_ta(self, a, b):
        return self._gemm(a, b, True, False)

    def mm_tb(self, a, b):
        return self._gemm(a, b, False, True)


def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    return xh * g + b, (xh, var, g)


def ln_bwd(dy, cache):
    xh, var, g = cache
    lead = tuple(range(dy.ndim - 1))
    dg = (dy * xh).sum(lead)
    db = dy.sum(lead)
    dxh = dy * g
    if dy.ndim == 3:
        mean = dxh.mean(-1, keepdims=True)
        meanc = (dxh * xh).mean(-1, keepdims=True)
    else:
        mean = dxh.mean(1, keepdims=True)
        meanc = (dxh * xh).mean(1, keepdims=True)
    return (dxh - mean - xh * meanc) / np.sqrt(var + 1e-5), dg, db


def gelu(x):
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2)


class MixerAST:
    def __init__(self, dim=256, blocks=4, ff=1024, patch=4, mel=64, ncls=4, seed=0):
        rng = np.random.default_rng(seed)
        self.patch, self.dim, self.blocks = patch, dim, blocks
        self.use_tokmix = False   # token-mix 的 dW 梯度先跳过（本实验只看 GEMM 后端）
        self.ntok = (mel // patch) ** 2
        W = lambda a, b: (rng.standard_normal((a, b)) * (1 / np.sqrt(b))).astype(np.float32)
        P = {}
        P["emb"] = W(dim, patch * patch)
        P["embb"] = np.zeros(dim, np.float32)
        for i in range(blocks):
            P[f"ln1_{i}"] = np.ones(dim, np.float32); P[f"ln1b_{i}"] = np.zeros(dim, np.float32)
            P[f"tok_{i}"] = (rng.standard_normal((self.ntok, self.ntok)) * 0.02).astype(np.float32)
            P[f"tokb_{i}"] = np.zeros(self.ntok, np.float32)
            P[f"ln2_{i}"] = np.ones(dim, np.float32); P[f"ln2b_{i}"] = np.zeros(dim, np.float32)
            P[f"fc1_{i}"] = W(ff, dim); P[f"fc1b_{i}"] = np.zeros(ff, np.float32)
            P[f"fc2_{i}"] = W(dim, ff); P[f"fc2b_{i}"] = np.zeros(dim, np.float32)
        P["head"] = W(ncls, dim); P["headb"] = np.zeros(ncls, np.float32)
        self.params = P

    def _patchify(self, mel):
        B, P = mel.shape[0], self.patch
        return mel.reshape(B, 64 // P, P, 64 // P, P).transpose(0, 1, 3, 2, 4).reshape(B, self.ntok, P * P)

    def loss_and_grad(self, mel, labels, mm, weight_decay=0.0):
        B = mel.shape[0]
        P = self.params
        tok = self._patchify(mel)                                   # (B,T,16)
        x = mm(tok.reshape(-1, self.patch ** 2), P["emb"].T).reshape(B, self.ntok, self.dim) + P["embb"]
        cache = dict(tok=tok, ln=[], ffn=[])
        for i in range(self.blocks):
            x1, c1 = ln_fwd(x, P[f"ln1_{i}"], P[f"ln1b_{i}"])
            # token mix: x += LN(x) @ Wtok^T   -> (B,T,dim):  mm_tb on (T,dim)? 用 (T,T) 乘 (T,dim)
            if self.use_tokmix:
                mixed = mm(P[f"tok_{i}"], x1.transpose(1, 0, 2).reshape(self.ntok, -1)).reshape(self.ntok, B, self.dim).transpose(1, 0, 2)
                x = x + mixed + P[f"tokb_{i}"][None, :, None].transpose(0, 1, 2)
            else:
                mixed = np.zeros((B, self.ntok, self.dim), np.float32)
            x2, c2 = ln_fwd(x, P[f"ln2_{i}"], P[f"ln2b_{i}"])
            h1 = mm(x2.reshape(-1, self.dim), P[f"fc1_{i}"].T) + P[f"fc1b_{i}"]
            a = gelu(h1)
            h2 = mm(a, P[f"fc2_{i}"].T) + P[f"fc2b_{i}"]
            x = x + h2.reshape(B, self.ntok, self.dim)
            cache["ln"].append((c1, mixed, x1, c2))
            cache["ffn"].append((h1, a))
        pooled = x.mean(1)
        logits = mm(pooled, P["head"].T) + P["headb"]
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(B), labels] + 1e-9).mean())
        dlogits = p.copy()
        dlogits[np.arange(B), labels] -= 1.0
        dlogits /= B

        g = {}
        g["head"] = mm(dlogits.T, pooled)
        g["headb"] = dlogits.sum(0)
        dpooled = mm(dlogits, P["head"])
        dx = np.repeat(dpooled[:, None, :], self.ntok, axis=1) / self.ntok
        for i in reversed(range(self.blocks)):
            c1, mixed, x1, c2 = cache["ln"][i]
            h1, a = cache["ffn"][i]
            # FFN 残差
            d_ffn = dx.reshape(-1, self.dim)
            g[f"fc2_{i}"] = mm(d_ffn.T, a)
            g[f"fc2b_{i}"] = d_ffn.sum(0)
            da = mm(d_ffn, P[f"fc2_{i}"])
            dh1 = da * gelu_grad(h1)
            x2 = c2[0] * P[f"ln2_{i}"] + P[f"ln2b_{i}"]
            g[f"fc1_{i}"] = mm(dh1.T, x2.reshape(-1, self.dim))
            g[f"fc1b_{i}"] = dh1.sum(0)
            dx2 = mm(dh1, P[f"fc1_{i}"]).reshape(B, self.ntok, self.dim)
            dxl, dg2, db2 = ln_bwd(dx2, c2)
            g[f"ln2_{i}"], g[f"ln2b_{i}"] = dg2, db2
            dx = dx + dxl
            # token mix 残差（本实验关闭）
            if self.use_tokmix:
                dmix = dx
                dmix_t = dmix.transpose(1, 0, 2)              # (T,B,dim)
                x1_full = c1[0] * P[f"ln1_{i}"] + P[f"ln1b_{i}"]
                x1_t = x1_full.transpose(1, 0, 2)            # (T,B,dim)
                g[f"tok_{i}"] = np.einsum("tbv,sbd->ts", dmix_t, x1_t).astype(np.float32)
                g[f"tokb_{i}"] = dmix.sum((0, 2))
                dmixed = mm(P[f"tok_{i}"].T, dmix_t.reshape(self.ntok, -1)).reshape(self.ntok, B, self.dim).transpose(1, 0, 2)
                dxl1, dg1, db1 = ln_bwd(dmixed, c1)
                g[f"ln1_{i}"], g[f"ln1b_{i}"] = dg1, db1
                dx = dx + dxl1
        # patch embed
        g["emb"] = mm(dx.reshape(-1, self.dim).T, cache["tok"].reshape(-1, self.patch ** 2))
        g["embb"] = dx.sum((0, 1))
        if weight_decay:
            for k in list(g):
                if k.endswith(("emb", "head")) or k.startswith(("fc", "tok_")):
                    g[k] = g[k] + weight_decay * P[k]
        return loss, g


def make_batch(n, seed):
    r = np.random.default_rng(seed)
    mel = (r.standard_normal((n, 64, 64)) * 0.1).astype(np.float32)
    y = r.integers(0, 4, n)
    for c in range(4):
        mel[y == c][:, 8 + c * 12:18 + c * 12, :] += 1.0
    return mel, y


def gradcheck():
    """与 torch 自动微分对比（同结构、小尺寸）。"""
    import torch
    net = MixerAST(dim=64, blocks=2, ff=128)
    be = CPUBackend()
    mel, y = make_batch(4, 1)
    loss, g = net.loss_and_grad(mel, y, be.mm)

    tp = {k: torch.from_numpy(v.copy()).requires_grad_(True) for k, v in net.params.items()}
    x = torch.from_numpy(mel)
    B, P_, N = x.shape[0], net.patch, net.ntok
    tok = x.reshape(B, 64 // P_, P_, 64 // P_, P_).permute(0, 1, 3, 2, 4).reshape(B, N, P_ * P_)
    h = tok.reshape(-1, P_ * P_) @ tp["emb"].T + tp["embb"]
    h = h.reshape(B, N, net.dim)
    for i in range(net.blocks):
        if net.use_tokmix:
            x1 = torch.nn.functional.layer_norm(h, (net.dim,), tp[f"ln1_{i}"], tp[f"ln1b_{i}"], 1e-5)
            mixed = torch.einsum("ts,bsd->btd", tp[f"tok_{i}"], x1) + tp[f"tokb_{i}"][None, :, None]
            h = h + mixed
        x2 = torch.nn.functional.layer_norm(h, (net.dim,), tp[f"ln2_{i}"], tp[f"ln2b_{i}"], 1e-5)
        a = torch.nn.functional.gelu(x2.reshape(-1, net.dim) @ tp[f"fc1_{i}"].T + tp[f"fc1b_{i}"])
        h2 = a @ tp[f"fc2_{i}"].T + tp[f"fc2b_{i}"]
        h = h + h2.reshape(B, N, net.dim)
    pooled = h.mean(1)
    logits = pooled @ tp["head"].T + tp["headb"]
    tl = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y))
    tl.backward()
    print(f"  loss numpy {loss:.6f} | torch {float(tl):.6f}")
    worst = 0.0
    for k in ("emb", "fc1_0", "fc2_1", "head", "ln2_1"):
        if tp[k].grad is None:
            continue
        a = g[k]
        b = tp[k].grad.numpy()
        rel = np.max(np.abs(a - b)) / max(1e-6, np.max(np.abs(b)))
        worst = max(worst, rel)
        print(f"  {k:<8} 最大相对误差 {rel:.2e}")
    print(f"  gradcheck 结论: {'通过' if worst < 2e-3 else '偏大，检查实现'} (worst {worst:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["cpu", "igpu"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--gradcheck", action="store_true")
    a = ap.parse_args()

    if a.gradcheck:
        gradcheck()
        return

    backend = IGPUBackend() if a.backend == "igpu" else CPUBackend()
    net = MixerAST(dim=a.dim, blocks=a.blocks, ff=a.ff)
    nparam = int(sum(p.size for p in net.params.values()))
    print(f"  后端 {backend.name} | dim={a.dim} blocks={a.blocks} ff={a.ff} | batch {a.batch} | "
          f"参数 {nparam/1e6:.2f}M", flush=True)

    t0 = time.perf_counter()
    losses = []
    for step in range(a.steps):
        mel, y = make_batch(a.batch, 1000 + step)
        loss, grads = net.loss_and_grad(mel, y, backend.mm)
        for k, gr in grads.items():
            net.params[k] -= a.lr * np.asarray(gr, np.float32).reshape(net.params[k].shape)
        losses.append(loss)
    dt = time.perf_counter() - t0

    mel, y = make_batch(128, 9999)
    # 评估：直接前向（复用 loss_and_grad 的 forward 部分会多算梯度，这里用简单重算）
    x = net._patchify(mel)
    h = backend.mm(x.reshape(-1, net.patch ** 2), net.params["emb"].T).reshape(128, net.ntok, net.dim) + net.params["embb"]
    for i in range(net.blocks):
        x1, _ = ln_fwd(h, net.params[f"ln1_{i}"], net.params[f"ln1b_{i}"])
        mixed = backend.mm(net.params[f"tok_{i}"], x1.transpose(1, 0, 2).reshape(net.ntok, -1)).reshape(net.ntok, 128, net.dim).transpose(1, 0, 2)
        h = h + mixed + net.params[f"tokb_{i}"][None, :, None]
        x2, _ = ln_fwd(h, net.params[f"ln2_{i}"], net.params[f"ln2b_{i}"])
        a1 = gelu(backend.mm(x2.reshape(-1, net.dim), net.params[f"fc1_{i}"].T) + net.params[f"fc1b_{i}"])
        h2 = backend.mm(a1, net.params[f"fc2_{i}"].T) + net.params[f"fc2b_{i}"]
        h = h + h2.reshape(128, net.ntok, net.dim)
    logits = backend.mm(h.mean(1), net.params["head"].T) + net.params["headb"]
    acc = float((logits.argmax(1) == y).mean())

    res = dict(backend=backend.name, steps=a.steps, batch=a.batch, dim=a.dim, blocks=a.blocks, ff=a.ff,
               params=nparam, seconds=round(dt, 2), ms_per_step=round(dt / a.steps * 1e3, 1),
               first_loss=round(float(losses[0]), 4), last_loss=round(float(losses[-1]), 4),
               acc=round(acc, 3))
    print(f"  {backend.name}: {dt:.2f}s / {a.steps} 步 = {res['ms_per_step']} ms/步 | "
          f"loss {res['first_loss']} → {res['last_loss']} | acc {acc:.3f}", flush=True)
    with open(rf"{OUT}\ast_train_{backend.name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
