# -*- coding: utf-8 -*-
"""F1-M4：把 30M 的 SmallARImageModelV2 整机（10 层 + flat head + CE）搬到 iGPU。

覆盖配置（= 我们 weibo 30M ckpt 的实际配置）：
    cond_mode=none, pos_mode=2d, token_order=raster, head_mode=flat,
    use_bos=False, mask_mode=causal, token_scale=1, fuse_qkv=False, use_l3=False

位置编码在 host 侧做（token_emb + row_emb[r] + col_emb[c]），head / CE / AdamW 都在 device。

已冻结（CPU 训练里它们是可训的）：ln1/ln2 的 w、q_norm/k_norm 的 w、ls1/ls2。
它们需要额外的列归约 kernel（M4b）。实测 ckpt@12000：ls 已从 1e-2 长到 0.03~0.07、
ln/q_norm 长到 1.03~1.05 —— 也就是不能永久冻结，M6 之前必须补。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import pyopencl as cl

import gt1030.attention_gpu as A

BATCH = int(os.environ.get('AR_B', '2'))
SEQ = int(os.environ.get('AR_T', '1024'))
A.B, A.T = BATCH, SEQ
A.M = A.B * A.T
A.BH = A.B * A.NH
import gt1030.full_gpu_step_v2 as V2       # noqa: E402
V2.M = BATCH * SEQ                        # V2 里 M 硬编码 2048；B=1 时要用 1024
import gt1030.full_gpu_layer as L          # noqa: E402  (import 顺序不能动：它读 A.B/A.T/V2.M)

B, T, MD, FF, WROW, BH, HD, NH, NKV = L.B, L.T, L.MD, L.FF, L.WROW, L.BH, L.HD, L.NH, L.NKV
M = L.M
scale = L.scale

KSRC = r"""
// per-head RMSNorm 前向（w 是 float*，与 rms_apply 的 half* 不同）
__kernel void qk_norm_fwd(__global const half* x, __global const float* w,
                          __global half* y, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global half* out = y + (long)r * D;
    float ss = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) out[i] = (half)((float)row[i] * inv * w[i]);
}
// 反向：dx = dy*inv*w - x*(x·dy)*inv^3/D*w
__kernel void qk_norm_dx(__global const half* x, __global const float* w,
                         __global const half* dy, __global half* dx, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global const half* drow = dy + (long)r * D;
    __global half* ox = dx + (long)r * D;
    float ss = 0.f, dot = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) dot += (float)row[i] * (float)drow[i];
    float c = dot * inv * inv * inv / D;
    for (int i = 0; i < D; ++i)
        ox[i] = (half)((float)drow[i] * inv * w[i] - (float)row[i] * c * w[i]);
}
// 融合 w12 输出 (M,2FF) -> silu(gate)*up  (M,FF)
__kernel void split_silu(__global const half* a2, __global half* g, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    g[i] = (half)(a / (1.f + exp(-a)) * b);
}
__kernel void split_silu_bwd(__global const half* a2, __global const half* dg,
                             __global half* da2, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    float g = (float)dg[i];
    float sg = 1.f / (1.f + exp(-a));
    float silu_a = a * sg;
    float dsilu = sg * (1.f + a * (1.f - sg));
    da2[(long)r * 2 * FF + j] = (half)(g * b * dsilu);
    da2[(long)r * 2 * FF + FF + j] = (half)(g * silu_a);
}
// 逐元素乘 (d,) 向量（ls1/ls2）
__kernel void scale_vec(__global const half* x, __global const float* s, __global half* y,
                        int rows, int D) {
    int i = get_global_id(0); if (i >= rows * D) return;
    y[i] = (half)((float)x[i] * s[i % D]);
}
// 交叉熵 + dlogits：一行一个 workgroup，V<=1024。
//   loss_row[r] = -log p[target] * mask[r]，dlogits = mask*(p - onehot)*inv_cnt
__kernel void ce_softmax(__global const half* lg, __global const int* tgt,
                         __global const float* mask, __global half* dlg,
                         __global float* loss_row, int V, float inv_cnt) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float vals[1024];
    __local float red[256];
    __global const half* line = lg + (long)r * V;
    __global half* out = dlg + (long)r * V;
    float mx = -1e30f;
    for (int k = lid; k < V; k += lsz) mx = fmax(mx, (float)line[k]);
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.f;
    for (int k = lid; k < V; k += lsz) { float e = exp((float)line[k] - m); vals[k] = e; s += e; }
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    int t = tgt[r];
    float mk = mask[r];
    float ls = 0.f;
    for (int k = lid; k < V; k += lsz) {
        float p = vals[k] * inv;
        out[k] = (half)(mk * (p - ((k == t) ? 1.f : 0.f)) * inv_cnt);
        if (k == t) ls = -log(p + 1e-12f);
    }
    red[lid] = ls; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) loss_row[r] = red[0] * mk;
}
// 梯度全局 L2 范数（分块部分和，host 再汇总）
__kernel void ssq_part(__global const half* g, __global float* part, int n, int blk) {
    int b = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    long st = (long)b * blk, en = min((long)(b + 1) * blk, (long)n);
    float s = 0.f;
    for (long i = st + lid; i < en; i += lsz) { float v = (float)g[i]; s += v * v; }
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) part[b] = red[0];
}
__kernel void scale_half(__global half* g, float f, int n) {
    int i = get_global_id(0); if (i >= n) return;
    g[i] = (half)((float)g[i] * f);
}
"""

class DriverBroken(RuntimeError):
    """本机 AMD 驱动偶发：队列整体变哑（copy 静默 no-op）。抓到就换 context 重建。"""


def new_context():
    """建一个全新的 OpenCL context/queue，并把 A.* 的全局句柄换掉。

    hgemm/l.hgemm_* 全部通过 A.QP / A.queue / A.ctx 取值，所以换这三个就够。
    """
    import ctypes
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    A.ctx, A.queue, A.mf = ctx, queue, cl.mem_flags
    A.QP = ctypes.c_void_p(int(queue.int_ptr))
    return ctx, queue


KNAMES = ('rms_stats', 'rms_apply', 'rms_bwd_stats', 'rms_bwd_apply', 'split_q', 'split_kv',
          'softmax_causal', 'softmax_bwd_causal', 'merge_o', 'split_dout', 'merge_dqkv',
          'add_vec', 'adamw_single',
          'qk_norm_fwd', 'qk_norm_dx', 'split_silu', 'split_silu_bwd', 'scale_vec',
          'ce_softmax', 'ssq_part', 'scale_half')


class Dev(object):
    """一个 OpenCL context 上的一组 buffer / kernel / 上传下载小工具。"""

    def __init__(self):
        self.ctx, self.q, self.mf = A.ctx, A.queue, A.mf
        self.prog = cl.Program(self.ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + KSRC).build(
            options=['-cl-mad-enable'])
        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self._tick_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, 2048)
        self._health()

    def _tick(self):
        """每次分配后打一次：写一小块再读回并 finish。

        本机 AMD 驱动在『连续分配很多 buffer 但队列不 sync』时会随机把整条队列搞哑
        （之后所有 copy 静默 no-op：读回全 0 或别的 buffer 的内容）。探针实测：
        每分配一个 buffer 插一次 tick，431 个 buffer / 1.36GB 连续 3 次全正常；
        不插则 5 次里坏 3~4 次。代价 0.1ms/次。
        """
        a = np.arange(1024, dtype=np.float16)
        cl.enqueue_copy(self.q, self._tick_buf, a, is_blocking=True)
        out = np.empty(1024, np.float16)
        cl.enqueue_copy(self.q, out, self._tick_buf)
        self.q.finish()
        if not np.array_equal(out, a):
            raise DriverBroken('tick failed')

    def _health(self):
        """建 context 后立刻验一次『写入->回读』。本机驱动偶发整个队列哑掉，
        这种状态下面每一次 copy 都会静默 no-op，越早发现越好（否则会在构建到一半
        才炸，还得先释放 1.3GB 才能重建 context，而释放往往不干净）。

        为什么不在进程内重建 context：实测旧 buffer 没释放干净时
        cl.CommandQueue 直接 OUT_OF_HOST_MEMORY，重试 6 次全失败。
        正确做法是让进程退出、由调用方重启（见 selftest 的 reexec）。
        """
        a = np.array([0.5, -1.5, 2.25, 3.75], np.float32)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        out = np.empty(4, np.float32)
        cl.enqueue_copy(self.q, out, b)
        self.q.finish()
        if not np.array_equal(out, a):
            raise DriverBroken('dev tick failed: %r != %r' % (out, a))

    def up(self, a):
        """上传 host 数组。

        ⚠️ 不要用 mf.COPY_HOST_PTR：本机 AMD 驱动（gfx902 / Adrenalin）在 buffer 数量
        或总量上去之后会把数据串到**别的** buffer 里（实测 41 个 Param 有 22 个在创建时
        回读就是错的，症状是读出来的是另一个 buffer 的内容 / 全 0，整条链路静默失效）。
        显式分配 + enqueue_copy + finish 完全可靠（0 失败）。
        """
        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        self.q.finish()      # 见 _fix_model_up 注释：必须同步，否则批量 copy 会把队列搞哑
        # 上传即校验：本机驱动偶发『队列变哑』，copy 静默 no-op，不校验的话后面全是
        # 静默的错误结果（权重读回全 0）。只回读 8 字节，代价可忽略。
        # ⚠️ 本机驱动对『大 buffer 的小片段回读』不可靠（8 字节必错、4KB 也出错），
        # 全量回读一直是对的，所以这里按全量校验（每次构建总共多读 ~120MB，可接受）。
        back = np.empty(a.size, a.dtype)
        cl.enqueue_copy(self.q, back, b)
        self.q.finish()
        if not np.array_equal(back, a.ravel()):
            want = a.ravel()
            nbad = int((back != want).sum())
            print('[verify] FAIL %s dtype=%s n=%d bad=%d  back[:4]=%s want[:4]=%s '
                  'back_absmax=%.4g want_absmax=%.4g back_mean=%.4g'
                  % (a.shape, a.dtype, a.size, nbad, back[:4], want[:4],
                     float(np.abs(back).max()), float(np.abs(want).max()),
                     float(np.mean(back))), flush=True)
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))
        return b

    def buf(self, nbytes):
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
        self._tick()
        return b

    def buf_f(self, n):
        return self.buf(n * 4)

    def down(self, b, shape, dtype=np.float16):
        a = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.q, a, b)
        self.q.finish()
        return a

    def finish(self):
        self.q.finish()


class Param(object):
    """一组设备端权重：fp16 前向副本 + fp32 master/mom/vel（AdamW 用）。"""

    def __init__(self, dev, w_fp16, lr_scale=1.0):
        self.dev = dev
        self.n = int(np.prod(w_fp16.shape))
        self.w = dev.up(w_fp16)
        self.grad = dev.buf(self.n * 2)
        self.master = dev.up(w_fp16.astype(np.float32).ravel())
        self.mom = dev.up(np.zeros(self.n, np.float32))
        self.vel = dev.up(np.zeros(self.n, np.float32))
        self.lr_scale = lr_scale

    def step(self, lr, b1, b2, eps, wd, bc1, bc2):
        dev = self.dev
        dev.K['adamw_single'](
            dev.q, (self.n // 4,), None, self.master, self.mom, self.vel, self.grad, self.w,
            np.int32(self.n), np.float32(lr * self.lr_scale), np.float32(b1), np.float32(b2),
            np.float32(eps), np.float32(wd), np.float32(bc1), np.float32(bc2))


class Layer(object):
    """一层 V2 block 的设备权重 + 前向要保存的激活。"""

    KEYS = ('Wqkv', 'Wo', 'W12', 'W3')

    def __init__(self, dev, sd, idx):
        self.dev = dev
        p = 'blocks.%d.' % idx
        Wqkv = torch.cat([sd[p + 'attn.q.weight'], sd[p + 'attn.k.weight'],
                          sd[p + 'attn.v.weight']], 0).T.contiguous().numpy().astype(np.float16)
        Wo = sd[p + 'attn.o.weight'].T.contiguous().numpy().astype(np.float16)
        W12 = sd[p + 'mlp.w12.weight'].T.contiguous().numpy().astype(np.float16)
        W3 = sd[p + 'mlp.w3.weight'].T.contiguous().numpy().astype(np.float16)
        self.W = {k: Param(dev, v) for k, v in
                  (('Wqkv', Wqkv), ('Wo', Wo), ('W12', W12), ('W3', W3))}
        # 冻结的小向量
        self.ln1 = dev.up(sd[p + 'ln1.w'].numpy().astype(np.float16))
        self.ln2 = dev.up(sd[p + 'ln2.w'].numpy().astype(np.float16))
        self.qw = dev.up(sd[p + 'attn.q_norm.w'].numpy().astype(np.float32))
        self.kw = dev.up(sd[p + 'attn.k_norm.w'].numpy().astype(np.float32))
        self.ls1 = dev.up(sd[p + 'ls1'].numpy().astype(np.float32))
        self.ls2 = dev.up(sd[p + 'ls2'].numpy().astype(np.float32))
        # 前向保存的激活（反向要用）
        self.b = {}
        for k, nb in (('xin', M * MD), ('h1', M * MD), ('h2', M * MD), ('x1', M * MD),
                      ('o_tok', M * MD), ('P', BH * T * T), ('Qn', BH * T * HD),
                      ('Kn', BH * T * HD), ('Qraw', BH * T * HD), ('Kraw', BH * T * HD),
                      ('Vb', BH * T * HD), ('a2', M * 2 * FF), ('g', M * FF)):
            self.b[k] = dev.buf(nb * 2)


class V2ModelDev(object):
    def __init__(self, ckpt_path, dev=None, retries=1):
        last = None
        for attempt in range(1, retries + 1):
            try:
                self._build(ckpt_path, dev)
                return
            except DriverBroken as e:
                last = e
                if attempt == retries:
                    break
                print('[V2ModelDev] 驱动队列失效（%s），换 context 重建（第 %d 次）'
                      % (e, attempt + 1), flush=True)
                self._release()
        raise DriverBroken('重试 %d 次仍失败: %s' % (retries, last))

    def _release(self):
        import gc
        for attr in list(self.__dict__.keys()):
            if attr != 'cfg':
                self.__dict__.pop(attr, None)
        gc.collect()

    def _build(self, ckpt_path, dev=None):
        if dev is not None:
            # dev 里可能已经是坏 context；除非刚建好并校验过，否则重建
            pass
        if dev is None or getattr(dev, 'broken', False):
            new_context()
            dev = Dev()
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ck['cfg']
        assert cfg['cond_mode'] == 'none' and cfg['pos_mode'] == '2d', cfg['cond_mode']
        assert cfg['head_mode'] == 'flat' and not cfg['use_bos'] and cfg['token_order'] == 'raster'
        assert cfg['grid'] * cfg['grid'] == T, 'grid %d != T %d' % (cfg['grid'], T)
        sd = ck['model']
        self.n_layer = int(cfg['n_layer'])
        self.ck = ck
        self.cfg = cfg
        self.dev = dev or Dev()
        d = self.dev
        self.layers = [Layer(d, sd, i) for i in range(self.n_layer)]

        # ---- head (flat) ----
        Wh = sd['gen_head.lin.weight'].numpy().astype(np.float32)      # (V, MD)
        self.V = Wh.shape[0]
        self.head = Param(d, Wh.T.copy().astype(np.float16))            # (MD, V)

        # ---- embeddings（host 侧，AdamW 也用 numpy） ----
        self.tok_tab = sd['token_emb.weight'].numpy().astype(np.float32)   # (V, MD)
        self.row_tab = sd['row_emb'].numpy().astype(np.float32)            # (G, MD)
        self.col_tab = sd['col_emb'].numpy().astype(np.float32)
        self.tok_m = np.zeros_like(self.tok_tab); self.tok_v = np.zeros_like(self.tok_tab)
        self.row_m = np.zeros_like(self.row_tab); self.row_v = np.zeros_like(self.row_tab)
        self.col_m = np.zeros_like(self.col_tab); self.col_v = np.zeros_like(self.col_tab)
        g = cfg['grid']
        ordr = sd['order'].numpy() if 'order' in sd else np.arange(g * g)
        self.row_idx = (ordr // g).astype(np.int64)[:T]
        self.col_idx = (ordr % g).astype(np.int64)[:T]
        self.pos_tab = (self.row_tab[self.row_idx] + self.col_tab[self.col_idx])   # (T,MD)
        self.row_idx_full = np.tile(self.row_idx, B)
        self.col_idx_full = np.tile(self.col_idx, B)

        # ---- 共享 scratch ----
        self.sc = {}
        for k, nb in (('qkv', M * WROW), ('S', BH * T * T), ('O', BH * T * HD),
                      ('attn', M * MD), ('y', M * MD), ('logits', M * self.V),
                      ('dqkv', M * WROW), ('dy', M * MD), ('dg', M * FF), ('da2', M * 2 * FF),
                      ('dh2', M * MD), ('dx1mlp', M * MD), ('dattn', M * MD),
                      ('dO_tok', M * MD), ('dO', BH * T * HD), ('dP', BH * T * T),
                      ('dS', BH * T * T), ('dQ', BH * T * HD), ('dK', BH * T * HD),
                      ('dV', BH * T * HD), ('dQraw', BH * T * HD), ('dKraw', BH * T * HD),
                      ('dh1', M * MD), ('dxattn', M * MD), ('dtmp', M * MD),
                      ('dlogits', M * self.V)):
            self.sc[k] = d.buf(nb * 2)
        # 前向/反向各用两块 ping-pong buffer
        self.xa, self.xb2 = d.buf(M * MD * 2), d.buf(M * MD * 2)
        self.da, self.db = d.buf(M * MD * 2), d.buf(M * MD * 2)
        self.ss = d.buf_f(M)
        self.sco = d.buf_f(M)
        self.loss_row = d.buf_f(M)
        self.tgt = d.buf(M * 4)
        self.msk = d.buf_f(M)
        self.part = d.buf_f(4096)
        self.last_loss = None
        self.x_final = None
        self.dx_final = None

    # ------------------------------------------------------------------ 前向
    def embed(self, tokens):
        ids = tokens.reshape(-1)
        x = self.tok_tab[ids] + np.tile(self.pos_tab, (B, 1))
        cl.enqueue_copy(self.dev.q, self.xa, x.astype(np.float16), is_blocking=True)
        self.dev.finish()
        return x

    def forward(self, tokens):
        d, K = self.dev, self.dev.K
        assert tokens.shape[0] == B
        self.embed(tokens)
        xin, alt = self.xa, self.xb2
        for ly in self.layers:
            b = ly.b
            cl.enqueue_copy(d.q, b['xin'], xin, byte_count=M * MD * 2)
            K['rms_stats'](d.q, (M,), (64,), xin, self.ss, np.int32(M), np.int32(MD))
            K['rms_apply'](d.q, (M * MD,), (256,), xin, ly.ln1, self.ss, b['h1'],
                           np.int32(M), np.int32(MD))
            L.hgemm_nn(b['h1'], ly.W['Wqkv'].w, self.sc['qkv'], M, WROW, MD)
            K['split_q'](d.q, (BH * T * HD,), (256,), self.sc['qkv'], b['Qraw'],
                         np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
            K['split_kv'](d.q, (BH * T * HD,), (256,), self.sc['qkv'], b['Kraw'], b['Vb'],
                          np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
            K['qk_norm_fwd'](d.q, (BH * T,), None, b['Qraw'], ly.qw, b['Qn'],
                             np.int32(BH * T), np.int32(HD))
            K['qk_norm_fwd'](d.q, (BH * T,), None, b['Kraw'], ly.kw, b['Kn'],
                             np.int32(BH * T), np.int32(HD))
            A.hgemm_batched(b['Qn'], b['Kn'], self.sc['S'], T, T, HD, A.NO, A.YES,
                            HD, HD, T, T * HD, T * HD, T * T)
            K['softmax_causal'](d.q, (T * 256, BH), (256, 1), self.sc['S'], b['P'],
                                np.int32(T), np.float32(scale))
            A.hgemm_batched(b['P'], b['Vb'], self.sc['O'], T, HD, T, A.NO, A.NO,
                            T, HD, HD, T * T, T * HD, T * HD)
            K['merge_o'](d.q, (BH * T * HD,), (256,), self.sc['O'], b['o_tok'],
                         np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
            L.hgemm_nn(b['o_tok'], ly.W['Wo'].w, self.sc['attn'], M, MD, MD)
            K['scale_vec'](d.q, (M * MD,), (256,), self.sc['attn'], ly.ls1, b['x1'],
                           np.int32(M), np.int32(MD))
            K['add_vec'](d.q, (M * MD,), (256,), xin, b['x1'], b['x1'], np.int32(M * MD))
            K['rms_stats'](d.q, (M,), (64,), b['x1'], self.ss, np.int32(M), np.int32(MD))
            K['rms_apply'](d.q, (M * MD,), (256,), b['x1'], ly.ln2, self.ss, b['h2'],
                           np.int32(M), np.int32(MD))
            L.hgemm_nn(b['h2'], ly.W['W12'].w, b['a2'], M, 2 * FF, MD)
            K['split_silu'](d.q, (M * FF,), (256,), b['a2'], b['g'], np.int32(M), np.int32(FF))
            L.hgemm_nn(b['g'], ly.W['W3'].w, self.sc['y'], M, MD, FF)
            K['scale_vec'](d.q, (M * MD,), (256,), self.sc['y'], ly.ls2, self.sc['y'],
                           np.int32(M), np.int32(MD))
            K['add_vec'](d.q, (M * MD,), (256,), b['x1'], self.sc['y'], alt, np.int32(M * MD))
            xin, alt = alt, xin           # 下一层输入 = 本层输出
        self.x_final = xin
        L.hgemm_nn(xin, self.head.w, self.sc['logits'], M, self.V, MD)
        d.finish()
        return xin

    # ------------------------------------------------------------------ 损失
    def make_targets(self, tokens):
        tgt = np.zeros(M, np.int64)
        msk = np.zeros(M, np.float32)
        for b in range(B):
            r0 = b * T
            tgt[r0:r0 + T - 1] = tokens[b, 1:]
            msk[r0:r0 + T - 1] = 1.0
        inv_cnt = 1.0 / float(msk.sum())
        cl.enqueue_copy(self.dev.q, self.tgt, tgt.astype(np.int32))
        cl.enqueue_copy(self.dev.q, self.msk, msk)
        self.dev.finish()
        return inv_cnt

    def loss_and_dlogits(self, tokens):
        d, K = self.dev, self.dev.K
        inv_cnt = self.make_targets(tokens)
        # ⚠️ 必须 M*256：kernel 里用 get_group_id(0) 当行号，(M,) 只有 M/256 个工作组，
        # 也就是只会算前 8 行（loss 少了两个数量级，而且 dlogits 大部分是 0）。
        K['ce_softmax'](d.q, (M * 256,), (256,), self.sc['logits'], self.tgt, self.msk,
                        self.sc['dlogits'], self.loss_row, np.int32(self.V), np.float32(inv_cnt))
        d.finish()
        self.last_loss = float(d.down(self.loss_row, (M,), np.float32).sum())
        return self.last_loss, inv_cnt

    # ------------------------------------------------------------------ 反向
    def backward(self, inv_cnt):
        d, K = self.dev, self.dev.K
        L.hgemm_tn(self.x_final, self.sc['dlogits'], self.head.grad, MD, self.V, M)
        L.hgemm_nt(self.sc['dlogits'], self.head.w, self.sc['dtmp'], M, MD, self.V)
        din, dalt = self.sc['dtmp'], self.da
        for ly in reversed(self.layers):
            b = ly.b
            K['scale_vec'](d.q, (M * MD,), (256,), din, ly.ls2, self.sc['dy'],
                           np.int32(M), np.int32(MD))
            L.hgemm_tn(b['g'], self.sc['dy'], ly.W['W3'].grad, FF, MD, M)
            L.hgemm_nt(self.sc['dy'], ly.W['W3'].w, self.sc['dg'], M, FF, MD)
            K['split_silu_bwd'](d.q, (M * FF,), (256,), b['a2'], self.sc['dg'],
                                self.sc['da2'], np.int32(M), np.int32(FF))
            L.hgemm_tn(b['h2'], self.sc['da2'], ly.W['W12'].grad, MD, 2 * FF, M)
            L.hgemm_nt(self.sc['da2'], ly.W['W12'].w, self.sc['dh2'], M, MD, 2 * FF)
            K['rms_bwd_stats'](d.q, (M,), (64,), b['x1'], self.sc['dh2'], ly.ln2,
                               self.ss, self.sco, np.int32(M), np.int32(MD))
            K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['x1'], self.sc['dh2'], ly.ln2,
                               self.ss, self.sco, self.sc['dx1mlp'], np.int32(M), np.int32(MD))
            # attention 分支
            K['scale_vec'](d.q, (M * MD,), (256,), din, ly.ls1, self.sc['dattn'],
                           np.int32(M), np.int32(MD))
            L.hgemm_tn(b['o_tok'], self.sc['dattn'], ly.W['Wo'].grad, MD, MD, M)
            L.hgemm_nt(self.sc['dattn'], ly.W['Wo'].w, self.sc['dO_tok'], M, MD, MD)
            K['split_dout'](d.q, (BH * T * HD,), (256,), self.sc['dO_tok'], self.sc['dO'],
                            np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
            A.hgemm_batched(self.sc['dO'], b['Vb'], self.sc['dP'], T, T, HD, A.NO, A.YES,
                            HD, HD, T, T * HD, T * HD, T * T)
            K['softmax_bwd_causal'](d.q, (T * 256, BH), (256, 1), b['P'], self.sc['dP'],
                                    self.sc['dS'], np.int32(T), np.float32(scale))
            A.hgemm_batched(b['P'], self.sc['dO'], self.sc['dV'], T, HD, T, A.YES, A.NO,
                            T, HD, HD, T * T, T * HD, T * HD)
            A.hgemm_batched(self.sc['dS'], b['Kn'], self.sc['dQ'], T, HD, T, A.NO, A.NO,
                            T, HD, HD, T * T, T * HD, T * HD)
            A.hgemm_batched(self.sc['dS'], b['Qn'], self.sc['dK'], T, HD, T, A.YES, A.NO,
                            T, HD, HD, T * T, T * HD, T * HD)
            K['qk_norm_dx'](d.q, (BH * T,), None, b['Qraw'], ly.qw, self.sc['dQ'],
                            self.sc['dQraw'], np.int32(BH * T), np.int32(HD))
            K['qk_norm_dx'](d.q, (BH * T,), None, b['Kraw'], ly.kw, self.sc['dK'],
                            self.sc['dKraw'], np.int32(BH * T), np.int32(HD))
            K['merge_dqkv'](d.q, (M * WROW,), (256,), self.sc['dQraw'], self.sc['dKraw'],
                            self.sc['dV'], self.sc['dqkv'], np.int32(T), np.int32(HD),
                            np.int32(NH), np.int32(NKV), np.int32(B))
            L.hgemm_tn(b['h1'], self.sc['dqkv'], ly.W['Wqkv'].grad, MD, WROW, M)
            L.hgemm_nt(self.sc['dqkv'], ly.W['Wqkv'].w, self.sc['dh1'], M, MD, WROW)
            K['rms_bwd_stats'](d.q, (M,), (64,), b['xin'], self.sc['dh1'], ly.ln1,
                               self.ss, self.sco, np.int32(M), np.int32(MD))
            K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['xin'], self.sc['dh1'], ly.ln1,
                               self.ss, self.sco, self.sc['dxattn'], np.int32(M), np.int32(MD))
            K['add_vec'](d.q, (M * MD,), (256,), din, self.sc['dx1mlp'], dalt, np.int32(M * MD))
            K['add_vec'](d.q, (M * MD,), (256,), dalt, self.sc['dxattn'], dalt, np.int32(M * MD))
            din, dalt = dalt, din         # 本层 dx 成为下一层（更靠前）的 dout
        self.dx_final = din
        d.finish()
        return din

    def embed_backward(self, tokens):
        """d_embed -> token/row/col 表的梯度（host scatter）"""
        dx = self.dev.down(self.dx_final, (M, MD), np.float16).astype(np.float32)
        ids = tokens.reshape(-1)
        dtok = np.zeros_like(self.tok_tab)
        np.add.at(dtok, ids, dx)
        drow = np.zeros_like(self.row_tab)
        np.add.at(drow, self.row_idx_full, dx)
        dcol = np.zeros_like(self.col_tab)
        np.add.at(dcol, self.col_idx_full, dx)
        self.dtok_last, self.drow_last, self.dcol_last = dtok, drow, dcol
        return dtok, drow, dcol

    # ------------------------------------------------------------------ 优化器
    def grad_norm_and_clip(self, max_norm=1.0):
        d = self.dev
        params = [self.head] + [ly.W[k] for ly in self.layers for k in Layer.KEYS]
        tot, blk = 0.0, 1 << 20
        for p in params:
            nb = (p.n + blk - 1) // blk
            d.K['ssq_part'](d.q, (nb,), (256,), p.grad, self.part, np.int32(p.n), np.int32(blk))
            tot += float(d.down(self.part, (nb,), np.float32).sum())
        gn = float(np.sqrt(tot))
        self.last_gnorm = gn
        if max_norm and gn > max_norm:
            f = np.float32(max_norm / (gn + 1e-6))
            for p in params:
                d.K['scale_half'](d.q, (p.n,), None, p.grad, f, np.int32(p.n))
        d.finish()
        return gn

    def adamw_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, max_norm=1.0):
        # 先在 fp16 梯度上算范数/裁剪（与 CPU 的 clip_grad_norm_ 语义等价）
        self.grad_norm_and_clip(max_norm)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for p in [self.head] + [ly.W[k] for ly in self.layers for k in Layer.KEYS]:
            p.step(lr, b1, b2, eps, wd, bc1, bc2)
        # 小向量（host numpy AdamW）
        for tab, g, m, v in ((self.tok_tab, self.dtok_last, self.tok_m, self.tok_v),
                             (self.row_tab, self.drow_last, self.row_m, self.row_v),
                             (self.col_tab, self.dcol_last, self.col_m, self.col_v)):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            tab -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps) - lr * wd * tab
        # 位置表由 row/col 表推出，更新后必须重算（否则下一步嵌入用的是旧位置）
        self.pos_tab = self.row_tab[self.row_idx] + self.col_tab[self.col_idx]
        self.dev.finish()
