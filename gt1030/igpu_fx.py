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
        p = cl.Program(dev.ctx, src).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"]
        )
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

    def __init__(self, dev, w_np, name="", state_bits=32):
        self.dev, self.name = dev, name
        self.state_bits = state_bits
        w = np.ascontiguousarray(w_np, dtype=np.float32)
        self.shape = w.shape
        self.n = w.size
        flat = w.reshape(-1)
        nb4 = max(4096, self.n * 4)
        self.master = dev.buf(nb4)
        if state_bits in (8, 4):
            nblocks = (self.n + 255) // 256
            self.mom = dev.buf(max(4096, nblocks * 4))
            self.vel = dev.buf(max(4096, nblocks * 4))
            if state_bits == 8:
                self.mom8 = dev.buf(max(4096, self.n))
                self.vel8 = dev.buf(max(4096, self.n))
            else:
                nb = (self.n + 1) // 2
                self.mom8 = dev.buf(max(4096, nb))
                self.vel8 = dev.buf(max(4096, nb))
        else:
            self.mom = dev.buf(nb4)
            self.vel = dev.buf(nb4)
        self.buf = dev.buf(max(4096, self.n * 2))
        self.grad = dev.buf(max(4096, self.n * 2))
        cl.enqueue_copy(dev.q, self.master, flat)
        if state_bits in (8, 4):
            nblocks = (self.n + 255) // 256
            cl.enqueue_copy(dev.q, self.mom, np.zeros(nblocks, np.float32))
            cl.enqueue_copy(dev.q, self.vel, np.zeros(nblocks, np.float32))
            if state_bits == 8:
                cl.enqueue_copy(dev.q, self.mom8, np.full(self.n, 128, np.uint8))
                cl.enqueue_copy(dev.q, self.vel8, np.full(self.n, 128, np.uint8))
            else:
                nb = (self.n + 1) // 2
                cl.enqueue_copy(dev.q, self.mom8, np.full(nb, 0x88, np.uint8))
                cl.enqueue_copy(dev.q, self.vel8, np.full(nb, 0x88, np.uint8))
        else:
            cl.enqueue_copy(dev.q, self.mom, np.zeros(self.n, np.float32))
            cl.enqueue_copy(dev.q, self.vel, np.zeros(self.n, np.float32))
        cl.enqueue_copy(dev.q, self.buf, flat.astype(np.float16))
        cl.enqueue_copy(dev.q, self.grad, np.zeros(self.n, np.float16))
        dev.q.finish()

    def zero_grad(self):
        cl.enqueue_copy(self.dev.q, self.grad, np.zeros(self.n, np.float16))


import torch.nn as _nn


class AttnModule(_nn.Module):
    """对 FX 透明的**叶子模块**：内部是 4 个 Linear + 设备注意力。

    模型侧直接 `self.attn = igpu_fx.AttnModule(nh, hd)`；FX 把它当一个 op，
    执行器负责 qkv/out 投影 + 注意力前后向（全部在 device）。
    """

    def __init__(self, nh, hd):
        import torch.nn as nn

        super().__init__()
        self.nh, self.hd = int(nh), int(hd)
        self.q_proj = nn.Linear(nh * hd, nh * hd, bias=False)
        self.k_proj = nn.Linear(nh * hd, nh * hd, bias=False)
        self.v_proj = nn.Linear(nh * hd, nh * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, nh * hd, bias=False)

    def forward(self, x):  # FX 不应追踪到这里；仅作占位（参考实现用不到）
        return x


class AttnTracer(object):
    """让 symbolic_trace 把 AttnModule 当叶子（不展开内部 Linear）。"""

    @classmethod
    def trace(cls, model):
        import torch.fx as fx

        class T(fx.Tracer):
            def is_leaf_module(self, m, qualname):
                if isinstance(m, AttnModule):
                    return True
                return super().is_leaf_module(m, qualname)

        t = T()
        graph = t.trace(model)
        return fx.GraphModule(model, graph)


class Executor(object):
    def __init__(
        self,
        model,
        sample,
        arena_mb=900,
        name="fx",
        state_bits=32,
        muon=False,
        muon_lr=None,
        muon_wd=0.0,
        muon_momentum=0.95,
    ):
        self.state_bits = state_bits
        # auto_zero_grads：优化器 kernel 读完梯度顺手写 0，跳过每参数一次的清零 copy。
        # 注意：Muon 路径尚未在 kernel 内清零，启用 muon 时本开关自动失效。
        self.auto_zero_grads = False
        self.schedule = None  # set_schedule(...) 之后 step() 可以不传 lr
        self.use_muon = muon
        # 实测：muon_lr=0.02 在 64-vocab 小 LM 上 12 步内 NaN；安全区 ≤0.005
        self.muon_lr = muon_lr if muon_lr is not None else 0.005
        self.muon_wd = muon_wd
        self.muon_momentum = muon_momentum
        import torch.fx as fx

        self.model = model
        self.sess = G.Session(arena_mb=arena_mb, name=name)
        self.dev = self.sess.dev
        self.gm = AttnTracer.trace(model)
        self.params = {}  # key -> P
        self.recs = []  # 执行记录（fwd 时填）
        self.env = {}  # node name -> T
        self._node_bufs = {}  # (node.name, shape) -> T（每步复用，避免 arena 只增不减）
        self._bufs_raw = {}  # (key, shape) -> cl.Buffer（反传/临时块复用）
        self.placeholders = [
            n.name for n in self.gm.graph.nodes if n.op == "placeholder"
        ]
        self.shapes = {}
        self._collect_params()
        self._infer_shapes(sample)

    # ---------------- 参数 ----------------
    def _collect_params(self):
        import torch.nn as nn

        for name, mod in self.gm.named_modules():
            if isinstance(mod, nn.Conv2d):
                # 卷积权重 (OC,C,KH,KW) → (OC, C*KH*KW)，正好是 hgemm_nt 要的布局
                self.params[name + ".weight"] = P(
                    self.dev,
                    mod.weight.detach().cpu().numpy().reshape(mod.out_channels, -1),
                    name + ".weight",
                    state_bits=self.state_bits,
                )
                if not hasattr(self, "conv_meta"):
                    self.conv_meta = {}
                self.conv_meta[name] = dict(
                    oc=mod.out_channels, c=mod.in_channels, kh=mod.kernel_size[0],
                    kw=mod.kernel_size[1], sh=mod.stride[0], sw=mod.stride[1],
                    ph=mod.padding[0], pw=mod.padding[1],
                )
                if mod.bias is not None:
                    self.params[name + ".bias"] = P(
                        self.dev,
                        mod.bias.detach().cpu().numpy().reshape(-1),
                        name + ".bias",
                        state_bits=self.state_bits,
                    )
            elif isinstance(mod, nn.GroupNorm):
                self.params[name + ".weight"] = P(
                    self.dev, mod.weight.detach().cpu().numpy().reshape(-1),
                    name + ".weight", state_bits=self.state_bits)
                if mod.bias is not None:
                    self.params[name + ".bias"] = P(
                        self.dev, mod.bias.detach().cpu().numpy().reshape(-1),
                        name + ".bias", state_bits=self.state_bits)
                if not hasattr(self, "gn_meta"):
                    self.gn_meta = {}
                self.gn_meta[name] = dict(groups=mod.num_groups, eps=float(mod.eps))
            elif isinstance(mod, nn.Linear):
                self.params[name + ".weight"] = P(
                    self.dev,
                    mod.weight.detach().cpu().numpy(),
                    name + ".weight",
                    state_bits=self.state_bits,
                )
                if mod.bias is not None:
                    self.params[name + ".bias"] = P(
                        self.dev,
                        mod.bias.detach().cpu().numpy(),
                        name + ".bias",
                        state_bits=self.state_bits,
                    )
            elif isinstance(mod, nn.Embedding):
                self.params[name + ".weight"] = P(
                    self.dev,
                    mod.weight.detach().cpu().numpy(),
                    name + ".weight",
                    state_bits=self.state_bits,
                )
            elif isinstance(mod, nn.LayerNorm):
                self.params[name + ".gamma"] = P(
                    self.dev,
                    mod.weight.detach().cpu().numpy(),
                    name + ".gamma",
                    state_bits=self.state_bits,
                )
                self.params[name + ".beta"] = P(
                    self.dev,
                    mod.bias.detach().cpu().numpy(),
                    name + ".beta",
                    state_bits=self.state_bits,
                )

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
    def _attn_block(self, name, bt, nh, hd):
        import gt1030.igpu_attn_block as AB

        try:
            import gt1030.igpu_attn_fast as AF
        except Exception:
            AF = None
        if not hasattr(self, "_attn_cache"):
            self._attn_cache = {}
        key = (name, bt, nh, hd)
        if key not in self._attn_cache:
            if AF is not None:
                self._attn_cache[key] = AF.FastAttn(
                    self.dev, bt, nh, hd
                )  # head-batched（快）
            else:
                self._attn_cache[key] = AB.AttnBlock(self.dev, bt, nh, hd)
        return self._attn_cache[key]

    def _cached(self, key, shape):
        """按 key 缓存裸缓冲（反传临时块用）。"""
        k = (key, tuple(shape))
        if k not in self._bufs_raw:
            self._bufs_raw[k] = self._buf(shape)
        return self._bufs_raw[k]

    def _node_buf(self, name, shape):
        """按节点名缓存激活缓冲：同一个节点每步复用同一块显存。"""
        key = (name, tuple(shape))
        if key not in self._node_bufs:
            self._node_bufs[key] = T(self._buf(shape), shape)
        return self._node_bufs[key]

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
        elif isinstance(inputs, np.ndarray):
            inputs = {self.placeholders[0]: inputs}
        self.env = {}
        self.int_env = {}
        for name, arr in inputs.items():
            if arr.dtype in (np.int32, np.int64):
                if not hasattr(self, "_int_bufs"):
                    self._int_bufs = {}
                key = (name, arr.shape)
                if key not in self._int_bufs:
                    self._int_bufs[key] = self.dev.buf(max(4096, arr.size * 4))
                buf = self._int_bufs[key]
                cl.enqueue_copy(
                    self.dev.q, buf, np.ascontiguousarray(arr, dtype=np.int32)
                )
                self.env[name] = T(buf, arr.shape)
                self.int_env[name] = True
            else:
                if not hasattr(self, "input_bufs"):
                    self.input_bufs = {}
                key = (name, arr.shape)
                if key not in self.input_bufs:
                    self.input_bufs[key] = T(
                        self.dev.buf(max(4096, arr.size * 2)), arr.shape
                    )
                buf = self.input_bufs[key]
                cl.enqueue_copy(
                    self.dev.q,
                    buf.buf,
                    np.ascontiguousarray(arr, dtype=np.float16).reshape(-1),
                )
                self.env[name] = buf
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
        import torch.fx as fx

        if node.op == "call_function" and getattr(node.target, "__name__", "") == "pad":
            import gt1030.igpu_conv as CV
            x = self._arg(node.args[0])
            pads = tuple(int(v) for v in node.args[1])
            if pads != (0, 1, 0, 1):
                raise SystemExit("FX: 目前只支持 F.pad(x, (0,1,0,1))，收到 %r" % (pads,))
            N, C, H, W = x.shape
            out = self._node_buf(node.name, (N, C, H + 1, W + 1))
            CV.K(self.dev)["pad01"](self.dev.q, (N * C * (H + 1) * (W + 1),), None,
                                    x.buf, out.buf, np.int32(N), np.int32(C),
                                    np.int32(H), np.int32(W))
            self.recs.append(("pad01", node.name, x))
            self.env[node.name] = out
            return out
        if node.op == "call_method" and node.target in ("flatten", "view", "reshape"):
            x = self._arg(node.args[0])
            if node.target == "flatten":
                start = int(node.args[1]) if len(node.args) > 1 else 1
                head = 1
                for d in x.shape[:start]:
                    head *= int(d)
                tail = int(x.numel // max(head, 1))
                new_shape = (head, tail) if start > 0 else (1, int(x.numel))
            else:
                new_shape = tuple(int(d) if d != -1 else 0 for d in node.args[1:])
                if 0 in new_shape:
                    known = 1
                    for d in new_shape:
                        if d != 0:
                            known *= d
                    new_shape = tuple(int(x.numel) // known if d == 0 else d for d in new_shape)
            out = T(x.buf, tuple(int(d) for d in new_shape))
            self.env[node.name] = out
            # 登记 view 记录：反传到这个节点时要把梯度原样传回上游
            self.recs.append(("view", node.name, x, out))
            return out
        if node.op == "call_module":
            sub = self.gm.get_submodule(node.target)
            x = self._arg(node.args[0])
            if isinstance(sub, nn.GroupNorm):
                import gt1030.igpu_gn as GN
                meta = self.gn_meta[node.target]
                N, C, H, W = x.shape
                if not hasattr(self, "_gns"):
                    self._gns = {}
                key = (node.name, N, C, H, W)
                if key not in self._gns:
                    self._gns[key] = GN.GroupNorm(self.dev, N, C, H, W, meta["groups"],
                                                  meta["eps"])
                gn = self._gns[key]
                pg = self.params[node.target + ".weight"]
                pb = self.params.get(node.target + ".bias")
                cl.enqueue_copy(self.dev.q, gn.gamma, pg.buf)
                if pb is not None:
                    cl.enqueue_copy(self.dev.q, gn.beta, pb.buf)
                out = self._node_buf(node.name, x.shape)
                gn.forward(x.buf, out.buf)
                self.recs.append(("gn", node.name, x, pg, pb, gn))
                self.env[node.name] = out
            elif isinstance(sub, nn.Conv2d):
                import gt1030.igpu_conv as CV
                meta = self.conv_meta[node.target]
                N, C, H, W = x.shape
                OH = (H + 2 * meta["ph"] - meta["kh"]) // meta["sh"] + 1
                OW = (W + 2 * meta["pw"] - meta["kw"]) // meta["sw"] + 1
                if not hasattr(self, "_convs"):
                    self._convs = {}
                key = (node.name, N, C, H, W)
                if key not in self._convs:
                    self._convs[key] = CV.Conv2d(
                        self.dev, N, C, H, W, meta["oc"], meta["kh"], meta["kw"],
                        meta["sh"], meta["sw"], meta["ph"], meta["pw"])
                layer = self._convs[key]
                p = self.params[node.target + ".weight"]
                pb = self.params.get(node.target + ".bias")
                y_nhwc = layer.forward(x.buf, p.buf, pb.buf if pb is not None else None)
                out = self._node_buf(node.name, (N, meta["oc"], OH, OW))
                CV.K(self.dev)["nhwc2nchw"](
                    self.dev.q, (N * meta["oc"] * OH * OW,), None, y_nhwc, out.buf,
                    np.int32(N), np.int32(meta["oc"]), np.int32(OH), np.int32(OW))
                self.recs.append(("conv2d", node.name, x, p, pb, layer, (N, OH, OW)))
                self.env[node.name] = out
            elif isinstance(sub, nn.Linear):
                out = self._node_buf(node.name, (x.shape[0], sub.out_features))
                p = self.params[node.target + ".weight"]
                GEMM.hgemm_nt(
                    x.buf, p.buf, out.buf, x.shape[0], sub.out_features, sub.in_features
                )
                pb = self.params.get(node.target + ".bias")
                if pb is not None:
                    OPS2.bias_add(
                        self.dev, out.buf, pb.buf, x.shape[0], sub.out_features
                    )
                self.recs.append(("linear", node.name, x, p, pb))
            elif isinstance(sub, nn.GELU):
                out = self._node_buf(node.name, x.shape)
                NN._kernels()["gelu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel)
                )
                self.recs.append(("gelu", node.name, x, None, None))
            elif isinstance(sub, nn.SiLU):
                out = self._node_buf(node.name, x.shape)
                _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")["silu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel)
                )
                self.recs.append(("silu", node.name, x, None, None))
            elif isinstance(sub, AttnModule):
                D = sub.nh * sub.hd
                x = self._arg(node.args[0])
                pq = self.params[node.target + ".q_proj.weight"]
                pk = self.params[node.target + ".k_proj.weight"]
                pv = self.params[node.target + ".v_proj.weight"]
                po = self.params[node.target + ".o_proj.weight"]
                q = self._node_buf(node.name + ".q", (x.shape[0], D))
                k = self._node_buf(node.name + ".k", (x.shape[0], D))
                v = self._node_buf(node.name + ".v", (x.shape[0], D))
                GEMM.hgemm_nt(x.buf, pq.buf, q.buf, x.shape[0], D, D)
                GEMM.hgemm_nt(x.buf, pk.buf, k.buf, x.shape[0], D, D)
                GEMM.hgemm_nt(x.buf, pv.buf, v.buf, x.shape[0], D, D)
                blk = self._attn_block(node.target, x.shape[0], sub.nh, sub.hd)
                o = blk.forward(q.buf, k.buf, v.buf)
                out = self._node_buf(node.name, (x.shape[0], D))
                GEMM.hgemm_nt(o, po.buf, out.buf, x.shape[0], D, D)
                self.recs.append(
                    ("attn", node.name, (x, q, k, v, o, blk, (pq, pk, pv, po), D))
                )
            elif isinstance(sub, nn.Embedding):
                idx = self.env[node.args[0].name]
                out = self._node_buf(node.name, (idx.shape[0], sub.embedding_dim))
                p = self.params[node.target + ".weight"]
                NN._kernels()["emb_gather"](
                    self.dev.q,
                    (idx.shape[0], sub.embedding_dim),
                    None,
                    p.buf,
                    idx.buf,
                    out.buf,
                    np.int32(idx.shape[0]),
                    np.int32(sub.embedding_dim),
                )
                self.recs.append(("emb", node.name, (idx, p), None, None))
            elif isinstance(sub, nn.LayerNorm):
                # add + LN 前向融合：若输入正好是单消费者 add，则 add 已被执行，
                # 这里撤掉它的 rec，改用 ln_fwd_res 一次算 h=x+res 和 LN(h)。
                prev = node.args[0]
                fused = None
                pop_add = False
                if isinstance(prev, fx.Node) and prev.op == "call_function":
                    pname = getattr(prev.target, "__name__", str(prev.target))
                    if pname in ("add", "__add__"):
                        for ri, rec in enumerate(self.recs):
                            if rec[0] == "add" and rec[1] == prev.name:
                                fused = rec[2]
                                pop_add = len(prev.users) == 1
                                if pop_add:
                                    self.recs.pop(ri)
                                break
                rows, cols = int(np.prod(x.shape[:-1])), x.shape[-1]
                out = self._node_buf(node.name, x.shape)
                mean = self._node_buf(node.name + ".mean", (rows,)).buf
                rstd = self._node_buf(node.name + ".rstd", (rows,)).buf
                gp = self.params[node.target + ".gamma"]
                bp = self.params[node.target + ".beta"]
                K = NN._kernels()
                if fused is not None:
                    a, b = fused
                    # pop_add=True 时 add 已被撤掉，需要独立 h；
                    # 否则 add 节点仍保留，h 就写进它的输出 buffer（add kernel 已跳过）。
                    hbuf = self._node_buf(node.name + ".h", x.shape) if pop_add else x
                    K["ln_fwd_res"](
                        self.dev.q,
                        (rows * 256,),
                        (256,),
                        a.buf,
                        b.buf,
                        gp.buf,
                        bp.buf,
                        out.buf,
                        hbuf.buf,
                        mean,
                        rstd,
                        np.int32(cols),
                        np.float32(sub.eps),
                    )
                    if pop_add:
                        self.recs.append(
                            (
                                "ln_res",
                                node.name,
                                (a, b, rows, cols, mean, rstd, hbuf),
                                (gp, bp),
                                None,
                            )
                        )
                    else:
                        self.recs.append(
                            (
                                "ln",
                                node.name,
                                (x, rows, cols, mean, rstd),
                                (gp, bp),
                                None,
                            )
                        )
                else:
                    K["ln_fwd_fused"](
                        self.dev.q,
                        (rows * 256,),
                        (256,),
                        x.buf,
                        gp.buf,
                        bp.buf,
                        out.buf,
                        mean,
                        rstd,
                        np.int32(cols),
                        np.float32(sub.eps),
                    )
                    self.recs.append(
                        ("ln", node.name, (x, rows, cols, mean, rstd), (gp, bp), None)
                    )
            else:
                raise SystemExit(
                    "FX: 不支持的模块 %s（%s）" % (type(sub).__name__, node.target)
                )
        elif node.op == "call_function":
            fname = getattr(node.target, "__name__", str(node.target))
            if fname in ("add", "__add__"):
                a, b = self._arg(node.args[0]), self._arg(node.args[1])
                out = self._node_buf(node.name, a.shape)
                # 若 add 的输出被 LayerNorm 消费，add 的 fwd 可省：由 ln_fwd_res
                # 一次写 h=x+res 和 LN(h)；add 节点/反向 identity 仍保留。
                skip = False
                for u in node.users:
                    if u.op == "call_module":
                        try:
                            us = self.gm.get_submodule(u.target)
                        except Exception:
                            continue
                        if isinstance(us, nn.LayerNorm):
                            skip = True
                            break
                if not skip:
                    G._kernels(self.dev)["add_h"](
                        self.dev.q,
                        (a.numel,),
                        None,
                        a.buf,
                        b.buf,
                        out.buf,
                        np.int64(a.numel),
                    )
                self.recs.append(("add", node.name, (a, b), None, None))
            elif fname == "gelu":
                x = self._arg(node.args[0])
                out = self._node_buf(node.name, x.shape)
                NN._kernels()["gelu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel)
                )
                self.recs.append(("gelu", node.name, x, None, None))
            elif fname == "silu":
                x = self._arg(node.args[0])
                out = self._node_buf(node.name, x.shape)
                _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")["silu_fwd"](
                    self.dev.q, (x.numel,), None, x.buf, out.buf, np.int64(x.numel)
                )
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
            if kind == "pad01":
                import gt1030.igpu_conv as CV
                x = rec[2]
                Nx, Cx, Hx, Wx = x.shape
                dx = self._cached("bwd.pad01.dx." + name, x.shape)
                CV.K(self.dev)["crop01"](self.dev.q, (Nx * Cx * Hx * Wx,), None,
                                         g, dx, np.int32(Nx), np.int32(Cx),
                                         np.int32(Hx), np.int32(Wx))
                self._propagate(pending, x, dx)
            elif kind == "gn":
                import gt1030.igpu_gn as GN
                x, pg, pb, gn = rec[2], rec[3], rec[4], rec[5]
                dx = self._cached("bwd.gn.dx." + name, x.shape)
                gn.backward(g, x.buf, dx)
                NN._kernels()["acc_from_f32"](self.dev.q, (pg.n,), None, pg.grad,
                                              gn.dgamma, np.int64(pg.n))
                if pb is not None:
                    NN._kernels()["acc_from_f32"](self.dev.q, (pb.n,), None, pb.grad,
                                                  gn.dbeta, np.int64(pb.n))
                self._propagate(pending, x, dx)
            elif kind == "view":
                # 形状变了但 buffer 相同，梯度直接回传给源张量
                x_src = rec[2]
                g_src = self._cached("bwd.view.g." + name, x_src.shape)
                cl.enqueue_copy(self.dev.q, g_src, g)
                self._propagate(pending, x_src, g_src)
            elif kind == "conv2d":
                import gt1030.igpu_conv as CV
                x, p, pb, layer, (N, OH, OW) = rec[2], rec[3], rec[4], rec[5], rec[6]
                dy_nhwc = self._cached("bwd.conv.dy." + name, (N, OH, OW, p.shape[0]))
                CV.K(self.dev)["nchw2nhwc"](
                    self.dev.q, (N * p.shape[0] * OH * OW,), None, g, dy_nhwc,
                    np.int32(N), np.int32(p.shape[0]), np.int32(OH), np.int32(OW))
                dx, dw, db = layer.backward(dy_nhwc, x.buf, p.buf)
                self._acc_param(p, dw)
                if pb is not None and db is not None:
                    self._acc_param(pb, db)
                    cl.enqueue_fill_buffer(self.dev.q, layer.db, np.uint16(0), 0, layer.db.size)
                self._propagate(pending, x, dx)
            elif kind == "linear":
                x, p = rec[2], rec[3]
                M, Kd, N = x.shape[0], p.shape[1], p.shape[0]
                dx = self._cached("bwd.lin.dx." + name, (M, Kd))
                dW = self._cached("bwd.lin.dW." + name, (N, Kd))
                GEMM.hgemm_nn(g, p.buf, dx, M, Kd, N)
                GEMM.hgemm_tn(g, x.buf, dW, N, Kd, M)
                self._acc_param(p, dW)
                pb = rec[4]
                if pb is not None:
                    db = self._cached("bwd.lin.db." + name, (pb.n,))
                    OPS2.col_sum(self.dev, g, db, M, N)
                    self._acc_param(pb, db)
                self._propagate(pending, x, dx)
            elif kind in ("gelu", "silu"):
                x = rec[2]
                dx = self._cached("bwd.act.dx." + name, x.shape)
                kern = "gelu_bwd" if kind == "gelu" else "silu_bwd"
                f = (
                    NN._kernels()["gelu_bwd"]
                    if kind == "gelu"
                    else _prog(self.dev, SILU_SRC, ("silu_fwd", "silu_bwd"), "silu")[
                        "silu_bwd"
                    ]
                )
                f(self.dev.q, (x.numel,), None, x.buf, g, dx, np.int64(x.numel))
                self._propagate(pending, x, dx)
            elif kind == "attn":
                x, q, k, v, o, blk, (pq, pk, pv, po), D = rec[2]
                BT = x.shape[0]
                do_pre = self._cached("bwd.attn.dpre." + name, (BT, D))
                dW_o = self._cached("bwd.attn.dWo." + name, (D, D))
                GEMM.hgemm_nn(g, po.buf, do_pre, BT, D, D)  # dO_pre = dy W_o
                GEMM.hgemm_tn(g, o, dW_o, D, D, BT)  # dW_o = dyᵀ O
                self._acc_param(po, dW_o)
                dq, dk, dv = blk.backward(do_pre)
                # dW_q = dqᵀ x, 并累计 dx
                dx = self._cached("bwd.attn.dx." + name, (BT, D))
                for gi, (p_in, g_in) in enumerate(((pq, dq), (pk, dk), (pv, dv))):
                    dW = self._cached("bwd.attn.dW%d." % gi + name, (D, D))
                    GEMM.hgemm_tn(g_in, x.buf, dW, D, D, BT)
                    self._acc_param(p_in, dW)
                    tmp = self._cached("bwd.attn.tmp%d." % gi + name, (BT, D))
                    GEMM.hgemm_nn(g_in, p_in.buf, tmp, BT, D, D)
                    if gi == 0:
                        cl.enqueue_copy(self.dev.q, dx, tmp)
                    else:
                        G._kernels(self.dev)["add_h"](
                            self.dev.q, (BT * D,), None, dx, tmp, dx, np.int64(BT * D)
                        )
                self._propagate(pending, x, dx)
            elif kind == "emb":
                idx, p = rec[2]
                # 表梯度：先清零（zero_grads 已做），再把 dy 按行散加进去
                NN._kernels()["emb_scatter"](
                    self.dev.q,
                    (p.shape[1],),
                    None,
                    g,
                    idx.buf,
                    p.grad,
                    np.int32(idx.shape[0]),
                    np.int32(p.shape[1]),
                )
            elif kind == "add":
                a, b = rec[2]
                for t in (a, b):
                    self._add_pending(pending, t, g)
            elif kind == "ln_res":
                a, b, rows, cols, mean, rstd, hbuf = rec[2]
                gp, bp = rec[3]
                dg = self._cached("bwd.ln.dg." + name, (cols,))
                db = self._cached("bwd.ln.db." + name, (cols,))
                dx = self._cached("bwd.ln.dx." + name, hbuf.shape)
                dgf = self._cached("bwd.ln.dgf." + name, (cols * 1,))
                dbf = self._cached("bwd.ln.dbf." + name, (cols * 1,))
                K2 = NN._kernels()
                s1 = self._cached("bwd.ln.s1." + name, (rows,))
                s2 = self._cached("bwd.ln.s2." + name, (rows,))
                K2["ln_bwd_stats"](
                    self.dev.q,
                    (rows * 256,),
                    (256,),
                    g,
                    hbuf.buf,
                    mean,
                    rstd,
                    gp.buf,
                    s1,
                    s2,
                    np.int32(cols),
                )
                K2["ln_bwd_apply"](
                    self.dev.q,
                    (rows * 256,),
                    (256,),
                    g,
                    hbuf.buf,
                    mean,
                    rstd,
                    gp.buf,
                    s1,
                    s2,
                    dx,
                    np.int32(cols),
                )
                K2["ln_param_grads"](
                    self.dev.q,
                    (cols,),
                    None,
                    g,
                    hbuf.buf,
                    mean,
                    rstd,
                    dgf,
                    dbf,
                    np.int32(rows),
                    np.int32(cols),
                )
                K2["acc_from_f32"](
                    self.dev.q, (cols,), None, gp.grad, dgf, np.int64(cols)
                )
                K2["acc_from_f32"](
                    self.dev.q, (cols,), None, bp.grad, dbf, np.int64(cols)
                )
                self._add_pending(pending, a, dx)
                self._add_pending(pending, b, dx)
            elif kind == "ln":
                x, rows, cols, mean, rstd = rec[2]
                gp, bp = rec[3]
                dg = self._cached("bwd.ln.dg." + name, (cols,))
                db = self._cached("bwd.ln.db." + name, (cols,))
                dx = self._cached("bwd.ln.dx." + name, x.shape)
                dgf = self._cached("bwd.ln.dgf." + name, (cols * 1,))
                dbf = self._cached("bwd.ln.dbf." + name, (cols * 1,))
                # 先用 stats 求 rowsum(dy*g)、rowsum(dy*g*xhat)
                K2 = NN._kernels()
                s1 = self._cached("bwd.ln.s1." + name, (rows,))
                s2 = self._cached("bwd.ln.s2." + name, (rows,))
                K2["ln_bwd_fused"](
                    self.dev.q,
                    (rows * 256,),
                    (256,),
                    g,
                    x.buf,
                    mean,
                    rstd,
                    gp.buf,
                    dx,
                    np.int32(cols),
                )
                K2["ln_param_grads"](
                    self.dev.q,
                    (cols,),
                    None,
                    g,
                    x.buf,
                    mean,
                    rstd,
                    dgf,
                    dbf,
                    np.int32(rows),
                    np.int32(cols),
                )
                # dγ/dβ 留在设备上累加（原来每步 32 次 host 往返 copy）
                K2["acc_from_f32"](
                    self.dev.q, (cols,), None, gp.grad, dgf, np.int64(cols)
                )
                K2["acc_from_f32"](
                    self.dev.q, (cols,), None, bp.grad, dbf, np.int64(cols)
                )
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
            out = self._cached("acc." + name, (t.numel,))  # 复用，避免每步新申请
            G._kernels(self.dev)["add_h"](
                self.dev.q, (t.numel,), None, pending[name], g, out, np.int64(t.numel)
            )
            pending[name] = out
        else:
            pending[name] = g

    def _acc_param(self, p, dW_buf):
        G._kernels(self.dev)["acc_h"](
            self.dev.q, (p.n,), None, p.grad, dW_buf, np.int64(p.n)
        )

    # ---------------- 优化器 ----------------
    def zero_grads(self):
        """清空梯度。auto_zero_grads=True 时优化器 kernel 已代为清零，这里直接跳过。"""
        if self.auto_zero_grads and not self.use_muon:
            return
        for p in self.params.values():
            p.zero_grad()

    def set_schedule(
        self,
        kind="cosine",
        total=1000,
        lr_max=1e-3,
        lr_min=0.0,
        warmup=0,
        t_mult=2,
        t0=None,
    ):
        """kind: 'constant' | 'cosine' | 'sgdr'。设置后 step(lr=None) 自动按 step 号取 lr。"""
        self.schedule = dict(
            kind=kind,
            total=int(total),
            lr_max=float(lr_max),
            lr_min=float(lr_min),
            warmup=int(warmup),
            t_mult=int(t_mult),
            t0=int(t0 or max(1, total)),
        )
        return self.schedule

    def lr_at(self, t):
        if self.schedule is None:
            return None
        cfg = self.schedule
        if cfg["kind"] == "constant":
            return cfg["lr_max"]
        if cfg["kind"] == "sgdr":
            return G.sgdr_lr(
                t,
                cfg["t0"],
                cfg["lr_max"],
                cfg["lr_min"],
                t_mult=cfg["t_mult"],
                warmup=cfg["warmup"],
            )[0]
        return G.cosine_lr(
            t, cfg["total"], cfg["lr_max"], cfg["lr_min"], warmup=cfg["warmup"]
        )

    def _is_muon_param(self, name, p):
        """2D 权重且不是 embedding/head（与 CPU 侧 Muon 的分工一致）。"""
        if not self.use_muon or len(p.shape) != 2:
            return False
        # 量化状态（8/4-bit）下 Muon 用独立动量缓冲（igpu_muon._mom_buf），可以共存
        if "emb" in name or "head" in name:
            return False
        return name.endswith(".weight")

    def step(self, lr=None, t=1, eps=1e-8, wd=0.0):
        import math

        bc1r = 1.0 / (1.0 - 0.9 ** int(t))
        bc2r = 1.0 / math.sqrt(1.0 - 0.999 ** int(t))
        if lr is None:
            lr = self.lr_at(t)
            if lr is None:
                raise SystemExit("step(lr=None) 需要先 set_schedule(...)")
        if self.use_muon:
            import gt1030.igpu_muon as MU

            # Muon 跟随调度缩放：热重启把 Adam lr 拉回峰值时，Muon 不能还按常数走，
            # 否则两者叠加会把更新推爆（实测 SGDR + Muon 常数 lr 会在重启后 NaN）。
            muon_scale = 1.0
            if self.schedule is not None:
                peak = float(self.schedule.get("lr_max") or 0.0)
                if peak > 0:
                    muon_scale = float(lr) / peak
            for name, p in self.params.items():
                if self._is_muon_param(name, p):
                    MU.muon_step(
                        self.dev,
                        p,
                        p.grad,
                        lr=float(self.muon_lr or lr) * muon_scale,
                        wd=float(self.muon_wd),
                        momentum=float(self.muon_momentum),
                    )
        if not hasattr(self, "_k8"):
            self._k8 = None
        for name, p in self.params.items():
            if self.use_muon and self._is_muon_param(name, p):
                continue
            if getattr(p, "state_bits", 32) == 4:
                if getattr(self, "_k4", None) is None:
                    import pyopencl as _cl

                    prog = _cl.Program(self.dev.ctx, G.EXTRA).build(
                        options=["-cl-mad-enable", "-cl-fast-relaxed-math"]
                    )
                    self._k4 = _cl.Kernel(prog, "adamw1_q4")
                nblocks = (p.n + 255) // 256
                self._k4(
                    self.dev.q,
                    (nblocks * 256,),
                    (256,),
                    p.master,
                    p.mom8,
                    p.vel8,
                    p.mom,
                    p.vel,
                    p.grad,
                    p.buf,
                    np.int32(p.n),
                    np.float32(lr),
                    np.float32(0.9),
                    np.float32(0.999),
                    np.float32(eps),
                    np.float32(wd),
                    np.float32(bc1r),
                    np.float32(bc2r),
                )
                continue
            if getattr(p, "state_bits", 32) == 8:
                if self._k8 is None:
                    import pyopencl as _cl

                    prog = _cl.Program(self.dev.ctx, G.EXTRA).build(
                        options=["-cl-mad-enable", "-cl-fast-relaxed-math"]
                    )
                    self._k8 = _cl.Kernel(prog, "adamw1_q8")
                nblocks = (p.n + 255) // 256
                self._k8(
                    self.dev.q,
                    (nblocks * 256,),
                    (256,),
                    p.master,
                    p.mom8,
                    p.vel8,
                    p.mom,
                    p.vel,
                    p.grad,
                    p.buf,
                    np.int32(p.n),
                    np.float32(lr),
                    np.float32(0.9),
                    np.float32(0.999),
                    np.float32(eps),
                    np.float32(wd),
                    np.float32(bc1r),
                    np.float32(bc2r),
                )
            else:
                self.sess.adam_single(
                    self.dev.q,
                    (p.n,),
                    None,
                    p.master,
                    p.mom,
                    p.vel,
                    p.grad,
                    p.buf,
                    np.int32(p.n),
                    np.float32(lr),
                    np.float32(0.9),
                    np.float32(0.999),
                    np.float32(eps),
                    np.float32(wd),
                    np.int32(t),
                )
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
        G._kernels(ex.dev)["mse_grad_h"](
            ex.dev.q,
            (n,),
            None,
            y.buf,
            ytgt,
            dy,
            np.int64(n),
            np.float32(1.0 / n),
            np.float32(scale),
        )
        ex.dev.q.finish()
        losses_dev.append(
            float(
                (
                    (
                        ex.dev.down(y.buf, y.shape).astype(np.float32)
                        - tgt.numpy().astype(np.float32)
                    )
                    ** 2
                ).mean()
            )
        )
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


def check_stack(layers=4, steps=5, bt=16, d=32, nh=4, hd=8, lr=1e-3):
    """P3 验收：N 层 transformer stack 走通用路径，对拍 torch，并报步时。"""
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(layers):
                self.blocks.append(nn.Sequential())  # 占位，下面替换
            self.blocks = nn.ModuleList([_make_block() for _ in range(layers)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    def _make_block():
        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = nn.LayerNorm(d)
                self.attn = AttnModule(nh, hd)
                self.n2 = nn.LayerNorm(d)
                self.f1 = nn.Linear(d, d * 4, bias=False)
                self.f2 = nn.Linear(d * 4, d, bias=False)

            def forward(self, x):
                h = x + self.attn(self.n1(x))
                return h + self.f2(F.gelu(self.f1(self.n2(h))))

        return B()

    torch.manual_seed(0)
    model = Stack()
    x = torch.randn(bt, d) * 0.3
    tgt = torch.randn(bt, d) * 0.1

    ex = Executor(model, x)
    ex.sync_from_torch()
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scale = 65536.0
    yt = ex.dev.up(tgt.numpy().astype(np.float16))
    dy = ex.dev.buf(max(4096, bt * d * 2))
    losses_dev, losses_torch, times = [], [], []
    for step in range(1, steps + 1):
        t0 = time.time()
        ex.zero_grads()
        y = ex.forward(x.numpy())
        n = y.numel
        G._kernels(ex.dev)["mse_grad_h"](
            ex.dev.q,
            (n,),
            None,
            y.buf,
            yt,
            dy,
            np.int64(n),
            np.float32(1.0 / n),
            np.float32(scale),
        )
        ex.dev.q.finish()
        losses_dev.append(
            float(
                (
                    (
                        ex.dev.down(y.buf, y.shape).astype(np.float32)
                        - tgt.numpy().astype(np.float32)
                    )
                    ** 2
                ).mean()
            )
        )
        ex.backward(dy, y.shape)
        ex.step(lr, step, eps=1e-8 * scale)
        times.append(time.time() - t0)

        opt.zero_grad()
        tl = F.mse_loss(model(x), tgt)
        tl.backward()
        losses_torch.append(float(tl.detach()))
        opt.step()

    print("  stack layers=%d  d=%d  bt=%d" % (layers, d, bt))
    print("  step   device      torch       rel")
    worst = 0.0
    for i, (a, b) in enumerate(zip(losses_dev, losses_torch), 1):
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        print("  %4d  %10.6f  %10.6f  %8.2e" % (i, a, b, rel))
    print("  step time: mean %.1f ms" % (1000 * sum(times) / len(times)))
    print("RESULT " + ("PASS" if worst < 5e-2 else "FAIL"))
    return worst < 5e-2


def check_block(steps=5, bt=16, d=32, nh=4, hd=8, lr=1e-3):
    """一个完整 transformer block（LN→Attn→res→LN→FFN→res）+ CE，对拍 torch。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.attn = AttnModule(nh, hd)
            self.n2 = nn.LayerNorm(d)
            self.f1 = nn.Linear(d, d * 4, bias=False)
            self.f2 = nn.Linear(d * 4, d, bias=False)

        def forward(self, x):
            h = x + self.attn(self.n1(x))
            return h + self.f2(F.gelu(self.f1(self.n2(h))))

    torch.manual_seed(0)
    model = Block()
    x = torch.randn(bt, d) * 0.3

    ex = Executor(model, x)
    ex.sync_from_torch()

    def torch_attn(mod, h):
        B = 1
        q = mod.q_proj(h).view(bt, nh, hd).transpose(0, 1)
        k = mod.k_proj(h).view(bt, nh, hd).transpose(0, 1)
        v = mod.v_proj(h).view(bt, nh, hd).transpose(0, 1)
        s = (q @ k.transpose(-1, -2)) / (hd**0.5)
        mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        o = (p @ v).transpose(0, 1).reshape(bt, d)
        return mod.o_proj(o)

    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scale = 65536.0
    tgt = torch.randn(bt, d) * 0.1
    tgt_dev = ex.dev.up(tgt.numpy().astype(np.float16))
    dy = ex.dev.buf(max(4096, bt * d * 2))
    losses_dev, losses_torch = [], []
    for step in range(1, steps + 1):
        ex.zero_grads()
        y = ex.forward(x.numpy())
        n = y.numel
        G._kernels(ex.dev)["mse_grad_h"](
            ex.dev.q,
            (n,),
            None,
            y.buf,
            tgt_dev,
            dy,
            np.int64(n),
            np.float32(1.0 / n),
            np.float32(scale),
        )
        ex.dev.q.finish()
        losses_dev.append(
            float(
                (
                    (
                        ex.dev.down(y.buf, y.shape).astype(np.float32)
                        - tgt.numpy().astype(np.float32)
                    )
                    ** 2
                ).mean()
            )
        )
        ex.backward(dy, y.shape)
        ex.step(lr, step, eps=1e-8 * scale)

        opt.zero_grad()
        h = x + torch_attn(model.attn, model.n1(x))
        out = h + model.f2(F.gelu(model.f1(model.n2(h))))
        tl = F.mse_loss(out, tgt)
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
    cl.enqueue_copy(
        ex.dev.q, tgt_dev, np.ascontiguousarray(tgt.numpy(), dtype=np.int32)
    )
    loss_buf = ex.dev.buf(max(4096, batch * 4))
    grad_buf = ex.dev.buf(max(4096, batch * vocab * 2))
    losses_dev, losses_torch = [], []
    for step in range(1, steps + 1):
        ex.zero_grads()
        logits = ex.forward({"tokens": toks.numpy()})
        OPS2.ce_forward(
            ex.dev,
            logits.buf,
            tgt_dev,
            loss_buf,
            grad_buf,
            batch,
            vocab,
            1.0 / batch,
            scale,
        )
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
    ap.add_argument("--check-block", action="store_true")
    ap.add_argument("--check-stack", type=int, default=0, help="N 层 stack 验收")
    a = ap.parse_args()
    if a.check_stack:
        return 0 if check_stack(layers=a.check_stack) else 1
    if a.check_block:
        return 0 if check_block() else 1
    if a.check_lm:
        return 0 if check_lm() else 1
    if not a.check:
        print(__doc__)
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
