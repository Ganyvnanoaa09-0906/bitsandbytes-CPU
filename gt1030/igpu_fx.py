# -*- coding: utf-8 -*-
"""igpu_fx.py — dev13 P2：**FX 前端 + 自动反传**（任意 nn.Module → 设备计划 → 常驻训练）。

支持集（dev13 当前）：
    nn.Linear(bias=False) · nn.GELU · nn.SiLU · nn.LayerNorm · nn.Sequential（FX 自动内联）
    F.gelu · F.silu · torch.add / +（残差）

流程：
    symbolic_trace → 逐节点执行（同时记录 rec）→ 输出 → MSE 梯度 → 逆序反传（分叉累加）→ AdamW

所有张量（权重/梯度/优化器状态/激活）都在 one DevCtx 的 arena 里；host 只进 batch、出 loss。

    python gt1030/igpu_fx.py --check     # LayerNorm→Linear→GELU→Linear+残差，5 步对拍
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
import gt1030.igpu_graph as G  # noqa: E402
import gt1030.igpu_nn_ops as NN  # noqa: E402
import gt1030.igpu_ops2 as OPS2  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402

SILU_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void silu_fwd(__global const half* x, __global half* y, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    y[i] = (half)(v / (1.f + exp(-v)));
}
__kernel void silu_bwd(__global const half* x, __global const half* dy, __global half* dx, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    float s = 1.f / (1.f + exp(-v));
    float d = s * (1.f + v * (1.f - s));
    dx[i] = (half)(d * (float)dy[i]);
}
"""

_CACHE = {}


def _prog(dev, src, names, key):
    k = (int(dev.ctx.int_ptr), key)
    if k not in _CACHE:
        p = cl.Program(dev.ctx, src).build(options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _CACHE[k] = {n: cl.Kernel(p, n) for n in names}
    return _CACHE[k]


class T(object):
    __slots__ = ("buf", "shape")

    def __init__(self, buf, shape):
        self.buf, self.shape = buf, tuple(shape)

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


class P(object):
    """可训练参数：fp32 master + fp16 副本 + fp16 梯度 + Adam m/v（全部常驻）。"""

    def __init__(self, dev, w_np, name=""):
        self.dev, self.name = dev, name
        w = np.ascontiguousarray(w_np, dtype=np.float32)
        self.shape = w.shape
        self.n = w.size
        flat = w.reshape(-1)
        nb4 = max(4096, self.n * 4)
        self.master = dev.buf(nb4)
        self.mom = dev.buf(nb4)
        self.vel = dev.buf(nb4)
        self.buf = dev.buf(max(4096, self.n * 2))
        self.grad = dev.buf(max(4096, self.n * 2))
        cl.enqueue_copy(dev.q, self.master, flat)
        cl.enqueue_copy(dev.q, self.mom, np.zeros(self.n, np.float32))
        cl.enqueue_copy(dev.q, self.vel, np.zeros(self.n, np.float32))
        cl.enqueue_copy(dev.q, self.buf, flat.astype(np.float16))
        cl.enqueue_copy(dev.q, self.grad, np.zeros(self.n, np.float16))
        dev.q.finish()

    def zero_grad(self):
        cl.enqueue_copy(self.dev.q, self.grad, np.zeros(self.n, np.float16))


class Executor(object):
    def __init__(self, model, sample, arena_mb=900, name="fx"):
        import torch.fx as fx
        self.model = model
        self.sess = G.Session(arena_mb=arena_mb, name=name)
        self.dev = self.sess.dev
        self.gm = fx.symbolic_trace(model)
        self.params = {}       # key -> P
        self.recs = []         # 执行记录（fwd 时填）
        self.env = {}          # node name -> T
        self.placeholders = [n.name for n in self.gm.graph.nodes if n.op == "placeholder"]
        self.shapes = {}
        self._collect_params()
        self._infer_shapes(sample)

    # ---------------- 参数 ----------------
    def _collect_params(self):
        import torch.nn as nn
        for name, mod in self.gm.named_modules():
            if isinstance(mod, nn.Linear):
                self.params[name + ".weight"] = P(
                    self.dev, mod.weight.detach().cpu().numpy(), name + ".weight")
                if mod.bias is not None:
                    self.params[name + ".bias"] = P(
                        self.dev, mod.bias.detach().cpu().numpy(), name + ".bias")
            elif isinstance(mod, nn.Embedding):
                self.params[name + ".weight"] = P(
                    self.dev, mod.weight.detach().cpu().numpy(), name + ".weight")
            elif isinstance(mod, nn.LayerNorm):
                self.params[name + ".gamma"] = P(
                    self.dev, mod.weight.detach().cpu().numpy(), name + ".gamma")
                self.params[name + ".beta"] = P(
                    self.dev, mod.bias.detach().cpu().numpy(), name + ".beta")

    def sync_from_torch(self):
        """把 torch 当前参数灌进 device（保证同参对拍）。"""
        import torch.nn as nn
        for name, mod in self.gm.named_modules():
            if isinstance(mod, nn.Linear):
                p = self.params[name + ".weight"]
                w = mod.weight.detach().cpu().numpy().astype(np.float32)
                cl.enqueue_copy(self.dev.q, p.master, w.reshape(-1))
                cl.enqueue_copy(self.dev.q, p.buf, w.astype(np.float16).reshape(-1))
                if mod.bias is not None:
                    pb = self.params[name + ".bias"]
                    b = mod.bias.detach().cpu().numpy().astype(np.float32)
                    cl.enqueue_copy(self.dev.q, pb.master, b)
                    cl.enqueue_copy(self.dev.q, pb.buf, b.astype(np.float16))
            elif isinstance(mod, nn.Embedding):
                p = self.params[name + ".weight"]
                w = mod.weight.detach().cpu().numpy().astype(np.float32)
                cl.enqueue_copy(self.dev.q, p.master, w.reshape(-1))
                cl.enqueue_copy(self.dev.q, p.buf, w.astype(np.float16).reshape(-1))
            elif isinstance(mod, nn.LayerNorm):
                for suffix, src in ((".gamma", mod.weight), (".beta", mod.bias)):
                    p = self.params[name + suffix]
                    w = src.detach().cpu().numpy().astype(np.float32)
                    cl.enqueue_copy(self.dev.q, p.master, w.reshape(-1))
                    cl.enqueue_copy(self.dev.q, p.buf, w.astype(np.float16).reshape(-1))
        self.dev.q.finish()

    def _infer_shapes(self, sample):
        import torch.fx as fx
        shapes = {}

        class Rec(fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                if hasattr(r, "shape"):
                    shapes[n.name] = tuple(r.shape)
                return r

        Rec(self.gm).run(sample)
        self.shapes = shapes

    # ---------------- 内存 ----------------
    def _buf(self, shape):
        n = 1
        for d in shape:
            n *= d
        return self.dev.buf(max(4096, n * 2))

    def _new(self, shape):
        return T(self._buf(shape), shape)

    # ---------------- 前向 ----------------
    def forward(self, inputs=None, x_np=None):
        """inputs: {placeholder_name: np.ndarray}；也兼容单输入 forward(x_np)。"""
        if inputs is None:
            if x_np is None:
                raise SystemExit("forward 需要 inputs 或 x_np")
            inputs = {self.placeholders[0]: x_np}
        self.env = {}
        self.int_env = {}
        for name, arr in inputs.items():
            if arr.dtype in (np.int32, np.int64):
                buf = self.dev.buf(max(4096, arr.size * 4))
                cl.enqueue_copy(self.dev.q, buf, np.ascontiguousarray(arr, dtype=np.int32))
                self.env[name] = T(buf, arr.shape)
                self.int_env[name] = True
            else:
                self.env[name] = T(
                    self.dev.up(np.ascontiguousarray(arr, dtype=np.float16)), arr.shape)
        self.recs = []
        for node in self.gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            self._run(node)
        out_node = [n for n in self.gm.graph.nodes if n.op == "output"][0]
        out_name = out_node.args[0]
        return self.env[getattr(out_name, "name", out_name)]

    def _arg(self, a):
        if hasattr(a, "name") and a.name in self.env:
            return self.env[a.name]
        if isinstance(a, (int, float)):
            return a
        raise SystemExit("FX: 参数解析失败: %r" % (a,))

    def _run(self, node):
        import torch.nn as nn
        if node.op == "call_module":
            sub = self.gm.get_submodule(node.target)
            x = self._arg(node.args[0])
            if isinstance(sub, nn.Linear):
                out = self._new((x.shape[0], sub.out_features))
                p = self.params[node.target + ".weight"]
                GEMM.hgemm_nt(x.buf, p.buf, out.buf, x.shape[0], sub.out_features, sub.in_features)
                pb = self.params.get(node.target + ".bias")
                if pb is not None:
                    OPS2.bias_add(self.dev, out.buf, pb.buf, x.shape[0], sub.out_features)
                self.recs.append(("linear", node.name, x, p, pb))
            elif isinstance(sub, nn.GELU):
                out = self._new(x.shape)
                NN._kernels()["gelu_fwd"](self.dev.q, (x.numel,), None, x.buf, out.buf,
                                          np.int64(x.numel))
                self.recs.append(("gelu", node.name, x, None, None))
            elif isinstance(sub, nn.SiLU):
                out = self._new(x.shape)
                _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")["silu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel))
                self.recs.append(("silu", node.name, x, None, None))
            elif isinstance(sub, nn.Embedding):
                idx = self.env[node.args[0].name]
                out = self._new((idx.shape[0], sub.embedding_dim))
                p = self.params[node.target + ".weight"]
                NN._kernels()["emb_gather"](self.dev.q, (idx.shape[0], sub.embedding_dim), None,
                                            p.buf, idx.buf, out.buf,
                                            np.int32(idx.shape[0]), np.int32(sub.embedding_dim))
                self.recs.append(("emb", node.name, (idx, p), None, None))
            elif isinstance(sub, nn.LayerNorm):
                out = self._new(x.shape)
                rows, cols = int(np.prod(x.shape[:-1])), x.shape[-1]
                mean = self.dev.buf(max(4096, rows * 4))
                rstd = self.dev.buf(max(4096, rows * 4))
                gp = self.params[node.target + ".gamma"]
                bp = self.params[node.target + ".beta"]
                K = NN._kernels()
                K["ln_stats"](self.dev.q, (rows * 256,), (256,), x.buf, mean, rstd, np.int32(cols))
                K["ln_apply"](self.dev.q, (rows * 256,), (256,), x.buf, mean, rstd,
                              gp.buf, bp.buf, out.buf, np.int32(cols), np.float32(sub.eps))
                self.recs.append(("ln", node.name, (x, rows, cols, mean, rstd), (gp, bp), None))
            else:
                raise SystemExit("FX: 不支持的模块 %s（%s）" % (type(sub).__name__, node.target))
        elif node.op == "call_function":
            fname = getattr(node.target, "__name__", str(node.target))
            if fname in ("add", "__add__"):
                a, b = self._arg(node.args[0]), self._arg(node.args[1])
                out = self._new(a.shape)
                G._kernels(self.dev)["add_h"](self.dev.q, (a.numel,), None, a.buf, b.buf,
                                              out.buf, np.int64(a.numel))
                self.recs.append(("add", node.name, (a, b), None, None))
            elif fname == "gelu":
                x = self._arg(node.args[0])
                out = self._new(x.shape)
                NN._kernels()["gelu_fwd"](self.dev.q, (x.numel,), None, x.buf, out.buf,
                                          np.int64(x.numel))
                self.recs.append(("gelu", node.name, x, None, None))
            elif fname == "silu":
                x = self._arg(node.args[0])
                out = self._new(x.shape)
                _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")["silu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel))
                self.recs.append(("silu", node.name, x, None, None))
            else:
                raise SystemExit("FX: 不支持的函数 %s（支持集见文件头）" % fname)
        else:
            raise SystemExit("FX: 不支持的 node.op=%s" % node.op)
        self.dev.q.finish()
        self.env[node.name] = out

    # ---------------- 反传 ----------------
    def backward(self, dy_buf, dy_shape):
        """dy 是**已缩放**的输出梯度（fp16）；按 recs 逆序传播并累加到参数梯度。"""
        self.grads = {}
        cur = {self.recs[-1][1]: (dy_buf, dy_shape)}
        # 每个节点可能被多次消费：用 map 收集
        pending = {}
        pending[self.recs[-1][1]] = dy_buf
        K = G._kernels(self.dev)
        for rec in reversed(self.recs):
            kind, name = rec[0], rec[1]
            g = pending.pop(name, None)
            if g is None:
                continue
            if kind == "linear":
                x, p = rec[2], rec[3]
                M, Kd, N = x.shape[0], p.shape[1], p.shape[0]
                dx, dW = self._buf((M, Kd)), self._buf((N, Kd))
                GEMM.hgemm_nn(g, p.buf, dx, M, Kd, N)
                GEMM.hgemm_tn(g, x.buf, dW, N, Kd, M)
                self._acc_param(p, dW)
                pb = rec[4]
                if pb is not None:
                    db = self._buf((pb.n,))
                    OPS2.col_sum(self.dev, g, db, M, N)
                    self._acc_param(pb, db)
                self._propagate(pending, x, dx)
            elif kind in ("gelu", "silu"):
                x = rec[2]
                dx = self._buf(x.shape)
                kern = "gelu_bwd" if kind == "gelu" else "silu_bwd"
                f = (NN._kernels()["gelu_bwd"] if kind == "gelu"
                     else _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")["silu_bwd"])
                f(self.dev.q, (x.numel,), None, x.buf, g, dx, np.int64(x.numel))
                self._propagate(pending, x, dx)
            elif kind == "emb":
                idx, p = rec[2]
                # 表梯度：先清零（zero_grads 已做），再把 dy 按行散加进去
                NN._kernels()["emb_scatter"](self.dev.q, (p.shape[1],), None, g, idx.buf,
                                             p.grad, np.int32(idx.shape[0]),
                                             np.int32(p.shape[1]))
            elif kind == "add":
                a, b = rec[2]
                for t in (a, b):
                    self._add_pending(pending, t, g)
            elif kind == "ln":
                x, rows, cols, mean, rstd = rec[2]
                gp, bp = rec[3]
                dg = self._buf((cols,))
                db = self._buf((cols,))
                dx = self._buf(x.shape)
                dgf, dbf = self.dev.buf(max(4096, cols * 4)), self.dev.buf(max(4096, cols * 4))
                # 先用 stats 求 rowsum(dy*g)、rowsum(dy*g*xhat)
                K2 = NN._kernels()
                s1 = self.dev.buf(max(4096, rows * 4))
                s2 = self.dev.buf(max(4096, rows * 4))
                K2["ln_bwd_stats"](self.dev.q, (rows * 256,), (256,), g, x.buf, mean, rstd,
                                   gp.buf, s1, s2, np.int32(cols))
                K2["ln_bwd_apply"](self.dev.q, (rows * 256,), (256,), g, x.buf, mean, rstd,
                                   gp.buf, s1, s2, dx, np.int32(cols))
                K2["ln_param_grads"](self.dev.q, (cols,), None, g, x.buf, mean, rstd,
                                     dgf, dbf, np.int32(rows), np.int32(cols))
                # dγ/dβ 是 fp32 → 转 fp16 后累加
                cl.enqueue_copy(self.dev.q, dg, self.dev.down(dgf, (cols,), np.float32)
                                .astype(np.float16))
                cl.enqueue_copy(self.dev.q, db, self.dev.down(dbf, (cols,), np.float32)
                                .astype(np.float16))
                self._acc_param(gp, dg)
                self._acc_param(bp, db)
                self._propagate(pending, x, dx)
        self.dev.q.finish()

    def _propagate(self, pending, x, dx):
        """把 dx 交给产生 x 的节点（按需累加）。"""
        # x 是通过 self.env 存的对象；用对象身份反查节点名
        for nm, t in self.env.items():
            if t is x:
                self._add_pending(pending, t, dx, name=nm)
                return
        raise SystemExit("反传: 找不到张量的来源节点")

    def _add_pending(self, pending, t, g, name=None):
        if name is None:
            for nm, v in self.env.items():
                if v is t:
                    name = nm
                    break
        if name is None:
            return
        if name in pending:
            out = self.dev.buf(max(4096, t.numel * 2))
            G._kernels(self.dev)["add_h"](self.dev.q, (t.numel,), None, pending[name], g,
                                          out, np.int64(t.numel))
            pending[name] = out
        else:
            pending[name] = g

    def _acc_param(self, p, dW_buf):
        G._kernels(self.dev)["acc_h"](self.dev.q, (p.n,), None, p.grad, dW_buf, np.int64(p.n))

    # ---------------- 优化器 ----------------
    def zero_grads(self):
        for p in self.params.values():
            p.zero_grad()

    def step(self, lr, t, eps=1e-8):
        for p in self.params.values():
            self.sess.adam_single(self.dev.q, (p.n,), None, p.master, p.mom, p.vel, p.grad,
                                  p.buf, np.int32(p.n), np.float32(lr),
                                  np.float32(0.9), np.float32(0.999), np.float32(eps),
                                  np.float32(0.0), np.int32(t))
        self.dev.q.finish()


def check(steps=5, d=64, batch=32, lr=1e-3, scale=65536.0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.f1 = nn.Linear(d, d, bias=False)
            self.f2 = nn.Linear(d, d, bias=False)

        def forward(self, x):
            h = F.gelu(self.f1(self.n1(x)))
            return x + self.f2(h)

    torch.manual_seed(0)
    model = Blk()
    x = torch.randn(batch, d) * 0.3
    tgt = torch.randn(batch, d) * 0.1

    ex = Executor(model, x)
    ex.sync_from_torch()
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    ytgt = ex.dev.up(tgt.numpy().astype(np.float16))
    dy = ex.dev.buf(max(4096, batch * d * 2))
    losses_dev, losses_torch = [], []
    for step in range(1, steps + 1):
        ex.zero_grads()
        y = ex.forward(x.numpy())
        n = y.numel
        G._kernels(ex.dev)["mse_grad_h"](ex.dev.q, (n,), None, y.buf, ytgt, dy,
                                         np.int64(n), np.float32(1.0 / n), np.float32(scale))
        ex.dev.q.finish()
        losses_dev.append(float(((ex.dev.down(y.buf, y.shape).astype(np.float32)
                                  - tgt.numpy().astype(np.float32)) ** 2).mean()))
        ex.backward(dy, y.shape)
        ex.step(lr, step, eps=1e-8 * scale)

        opt.zero_grad()
        tl = F.mse_loss(model(x), tgt)
        tl.backward()
        losses_torch.append(float(tl.detach()))
        opt.step()

    print("  step   device      torch       rel")
    worst = 0.0
    for i, (a, b) in enumerate(zip(losses_dev, losses_torch), 1):
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        print("  %4d  %10.6f  %10.6f  %8.2e" % (i, a, b, rel))
    print("RESULT " + ("PASS" if worst < 5e-2 else "FAIL"))
    return worst < 5e-2


def check_lm(steps=5, vocab=64, d=32, batch=16, lr=1e-3, scale=65536.0):
    """迷你 LM：Embedding → Linear → GELU → Linear → CE，与 torch 同参对拍。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.f1 = nn.Linear(d, d * 2)
            self.f2 = nn.Linear(d * 2, vocab)

        def forward(self, tokens):
            h = F.gelu(self.f1(self.emb(tokens)))
            return self.f2(h)

    torch.manual_seed(0)
    model = LM()
    toks = torch.randint(0, vocab, (batch,))
    tgt = torch.randint(0, vocab, (batch,))

    ex = Executor(model, toks)
    ex.sync_from_torch()
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    tgt_dev = ex.dev.buf(max(4096, batch * 4))
    cl.enqueue_copy(ex.dev.q, tgt_dev, np.ascontiguousarray(tgt.numpy(), dtype=np.int32))
    loss_buf = ex.dev.buf(max(4096, batch * 4))
    grad_buf = ex.dev.buf(max(4096, batch * vocab * 2))
    losses_dev, losses_torch = [], []
    for step in range(1, steps + 1):
        ex.zero_grads()
        logits = ex.forward({"tokens": toks.numpy()})
        OPS2.ce_forward(ex.dev, logits.buf, tgt_dev, loss_buf, grad_buf,
                        batch, vocab, 1.0 / batch, scale)
        ex.dev.q.finish()
        lp = ex.dev.down(loss_buf, (batch,), np.float32)
        losses_dev.append(float(lp.mean()))
        ex.backward(grad_buf, logits.shape)
        ex.step(lr, step, eps=1e-8 * scale)

        opt.zero_grad()
        tl = F.cross_entropy(model(toks), tgt)
        tl.backward()
        losses_torch.append(float(tl.detach()))
        opt.step()

    print("  step   device(CE)  torch(CE)     rel")
    worst = 0.0
    for i, (a, b) in enumerate(zip(losses_dev, losses_torch), 1):
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        print("  %4d  %10.6f  %10.6f  %8.2e" % (i, a, b, rel))
    print("RESULT " + ("PASS" if worst < 5e-2 else "FAIL"))
    return worst < 5e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-lm", action="store_true")
    a = ap.parse_args()
    if a.check_lm:
        return 0 if check_lm() else 1
    if not a.check:
        print(__doc__)
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
