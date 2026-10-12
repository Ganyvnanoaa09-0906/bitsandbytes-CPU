# -*- coding: utf-8 -*-
"""train_ar_v2.py - 用 v2 架构训练 AR 图像模型。

相对 v1 训练脚本的差别
----------------------
- 模型换成 SmallARImageModelV2（2D 位置 / adaLN / 因子化头 / SwiGLU / RMSNorm / QK Norm / GQA）
- **因子化损失**：CE(cluster) + CE(code)；簇划分从 VQ 码本上 k-means 得到
- 可选 `--captions`：读 auto_caption.py 的 jsonl，用 CLIP 文本编码器编码成条件
- 目标仍然**移位一位**（v1 曾因没移位退化成恒等映射，loss 趋 0 却完全无法生成）

用法
----
  python train_ar_v2.py --tokens tokens.pt --vqvae vqvae_ckpt.pt --out run \\
      --stages "16:800,64:800,256:4000" --batch "16:64,64:16,256:8" \\
      --cond-mode none
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_image_model_v2 import (SmallImageConfigV2, SmallARImageModelV2, build_cluster_map)
from small_vqvae import VQVAE, vqvae_from_ckpt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes as bnb



# ============================================================================
# Muon + Newton-Schulz, lifted verbatim from ab_muon_v2.py.
#
# Both were measured before being brought here: Muon reaches the same loss in 3.19x fewer
# steps and 1.96x less wall clock than AdamW8bit (7.27), and dropping Newton-Schulz from 5
# iterations to 3 takes another 20.4% off the optimiser stage at a 0.9% loss difference over
# 2000 steps (7.31.4). Neither was in the real training path, which is what this does.
#
# Muon applies only to 2-D weight matrices; embeddings, the head and every 1-D parameter stay
# on AdamW, which is what the A/B harness did and therefore what the numbers describe.
_MUON_WIRED = True


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz quintic orthogonalisation, exactly as measured in 7.31.4."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    """Muon on 2-D weights. The NS iteration count is the H3 variable."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].get('buf')
                if buf is None:
                    buf = torch.zeros_like(p)
                    self.state[p]['buf'] = buf
                buf.mul_(g['momentum']).add_(p.grad)
                d = p.grad.add(buf, alpha=g['momentum'])
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d, g['ns_iters'])
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-g['lr'])


def build_optimizers(a, model):
    """AdamW8bit on everything, or Muon on 2-D weights plus AdamW8bit on the rest.

    Returns a list, so the training loop can iterate without caring which case it is. The
    split mirrors ab_muon_v2.py: 2-D parameters whose names contain neither 'emb' nor 'head'
    go to Muon, everything else to AdamW8bit.
    """
    import bitsandbytes as bnb
    name = (getattr(a, 'optimizer', 'adamw') or 'adamw').lower()
    if name != 'muon':
        return [bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)]
    muon_p, adam_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'emb' not in n and 'head' not in n:
            muon_p.append(p)
        else:
            adam_p.append(p)
    ns_it = int(getattr(a, 'ns_iters', 5) or 5)
    print('[muon] %d 个 2D 权重走 Muon（NS=%d）· %d 个其余走 AdamW8bit'
          % (len(muon_p), ns_it, len(adam_p)), flush=True)
    return [Muon(muon_p, lr=float(getattr(a, 'muon_lr', 0.02)), ns_iters=ns_it),
            bnb.optim.AdamW8bit(adam_p, lr=a.lr)]




def opt_zero_grad(opts):
    """zero_grad on one optimiser or a list of them (Muon mode holds two)."""
    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):
        o.zero_grad(set_to_none=True)


def opt_step(opts):
    """step on one optimiser or a list of them."""
    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):
        o.step()


def opt_lr(opts):
    """The learning rate to report. In Muon mode the first is Muon."""
    o = opts[0] if isinstance(opts, (list, tuple)) else opts
    return o.param_groups[0]['lr']


def opt_groups(opts):
    """Flatten param_groups so the LR schedule writes to every optimiser."""
    out = []
    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):
        out.extend(o.param_groups)
    return out

def rss():
    return psutil.Process().memory_info().rss / 1e9


class Peak:
    def __init__(self):
        self.p = rss(); self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            self.p = max(self.p, rss()); time.sleep(0.02)

    def start(self):
        self.t.start()

    def done(self):
        self.stop.set(); self.t.join(0.2); return self.p


def parse_batch(spec):
    m = {}
    for x in str(spec).split(','):
        x = x.strip()
        if not x:
            continue
        if ':' in x:
            k, v = x.split(':'); m[int(k)] = int(v)
        else:
            m[None] = int(x)
    return m


def load_caption_cond(cap_path, paths, text_dim, model_id='openai/clip-vit-base-patch32'):
    """把 caption 编码成 (N, text_dim) 的 pooled 条件。返回 None 表示不可用。"""
    import json as _json
    caps = [_json.loads(l) for l in open(cap_path, encoding='utf-8') if l.strip()]
    by = {os.path.abspath(c['path']): c.get('caption', '') for c in caps}
    texts = [by.get(os.path.abspath(p), 'a person') for p in paths]
    try:
        # ★ 必须用 _feat 解包：transformers 5.x 的 get_text_features 返回的是
        #   BaseModelOutputWithPooling 而不是张量。原来直接 .float() 会抛异常，
        #   而这里【被 try/except 吞掉并返回 None】⇒ 条件训练会静默退化成无条件！
        #   （这就是"条件注入太弱"的真正原因 —— 条件从来没生效过。）
        from clip_score_eval import load_clip, clip_feat
        m, p = load_clip(model_id)
        feats = []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                inp = p(text=texts[s:s + 64], return_tensors='pt', padding=True, truncation=True)
                f = clip_feat(m, m.get_text_features(**inp), 'text')
                feats.append(F.normalize(f.float(), dim=-1))
        return torch.cat(feats, 0)
    except Exception as e:
        print('[v2] caption 编码失败: %s' % e, flush=True)
        raise
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--vqvae', required=True, help='用于取码本做 k-means 簇划分')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', default='16:800,64:800,256:4000')
    ap.add_argument('--batch', default='16:64,64:16,256:8')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=0,
                    help='线性 warmup 步数：前 N 步学习率从 0 线性升到 --lr。长跑建议设 500~1000。')
    # ★ 带重启的余弦退火（SGDR, Loshchilov & Hutter 2016, arXiv:1608.03983）。
    # 动机来自本项目自己的曲线：验证 loss 每 500 步的改善是 −0.256 → −0.105 → −0.051
    # → −0.028 → −0.014，正好减半——单次余弦退火下 lr 降下来后步长变小，模型停在
    # 它恰好落进的盆地里。重启把 lr 拉回峰值，让它有机会换一个盆地。
    # ★ 默认 lr_restarts=1 与改动前【逐位相同】，因此不传这两个参数的长跑不受影响。
    ap.add_argument('--lr-restarts', type=int, default=1,
                    help='余弦退火的周期数。1 = 单次退火（原行为，默认）；'
                         '2~4 为 SGDR 式重启。直接决定总步数里出现几次 lr 峰值。')
    ap.add_argument('--lr-restart-decay', type=float, default=1.0,
                    help='每个周期的峰值 lr 乘数（默认 1.0 = 原文形式，每周期回到同一峰值；'
                         '0.5~0.8 则逐周期压低，更保守）。')
    ap.add_argument('--lr-min', type=float, default=0.0,
                    help='余弦退火的终点学习率。默认 0 = 退到 0；设成 --lr 的 10%% 更稳。')
    # ★ 多分辨率课程用。token_scale：一个 token 占 s x s 个坐标格，于是 16x16 的图
    # （256 token）能复用 grid=32 的位置表与词表 —— 同一个模型、同一个任务，只是分辨率不同。
    # 见 small_image_model_v2.py 的 coords()。默认 1，不传则与原行为逐位相同。
    ap.add_argument('--token_scale', type=int, default=1,
                    help='一个 token 占几个坐标格（1=原行为；2 配 16x16 的 token 文件）')
    # ★ total_tokens 覆盖：课程的【两个阶段必须是同一个模型】，因此第二阶段要沿用第一
    # 阶段的 total_tokens/grid，而不能从 token 文件的行宽推。默认 0 = 从文件推（原行为）。
    # ★ H3 实验用：Muon 的 Newton-Schulz 迭代次数。默认 5（原文与基线）。
    # 第二轮微基准：5→3 每步省 763 ms（优化器 2562→1799 ms），但 512×512 更新的条件数
    # 从 27 涨到 320。是否可接受只能在训练里测，见 research/ns_iterations.md。
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'],
                    help='adamw = 现有行为（默认）；muon = 2D 权重走 Muon，其余走 AdamW8bit')
    ap.add_argument('--muon-lr', type=float, default=0.02,
                    help='Muon 的学习率（A/B 用的是 0.02，未调优）')
    ap.add_argument('--ns-iters', type=int, default=5,
                    help='Muon 的 NS 迭代次数（默认 5；H3 实验用 3 或 4）')
    ap.add_argument('--total_tokens', type=int, default=0,
                    help='覆盖模型的总 token 数（0 = 从 token 文件的行宽推）')
    # ★ 每步实际喂给模型的序列长度（0 = 用 token 文件的行宽）。课程的第一阶段用 16x16
    # 的 token 文件（256 宽）配 --token_scale 2 与 --total_tokens 1024，于是模型与第二
    # 阶段完全一致，只是输入分辨率更粗。
    ap.add_argument('--seq_len', type=int, default=0,
                    help='每步的序列长度（0 = 用 token 文件的行宽）')
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--n_layer', type=int, default=10)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--loop_start', type=int, default=4,
                    help='循环块的起始层（权重共享反复跑的那段）')
    ap.add_argument('--loop_end', type=int, default=7)
    ap.add_argument('--loop_times', type=int, default=1,
                    help='循环段跑几遍。2 会让第 loop_start..loop_end 层重复执行，'
                         '实测 n_layer=10 + loop_times=2 每次前向跑 14 个 block pass，'
                         '比 loop_times=1 慢 38%%，参数完全相同（权重共享）。'
                         '调研（ELT 2604.09168）指出：没做 Intra-Loop Self Distillation 时'
                         '朴素循环收益很小，默认已设为 1。')
    # ---- 循环 Transformer（见 loop_transformer.py）----
    ap.add_argument('--loop_L', type=int, default=1,
                    help='循环 Transformer 的固定轮数。>1 时启用循环路径'
                         '（步数编码 + 跨轮门控残差）。与 --loop_times 的区别：'
                         'loop_times 是无状态重复，loop_L 带步数编码和跨轮连接。')
    ap.add_argument('--loop_L_max', type=int, default=3,
                    help='随机循环轮数的上界（配合 --loop_random）。')
    ap.add_argument('--loop_random', action='store_true',
                    help='训练时 L~U{1..loop_L_max} 随机采样，推理时可弹性选 L。')
    ap.add_argument('--loop_no_step_enc', action='store_true',
                    help='禁用循环步数编码（消融用）。')
    ap.add_argument('--loop_no_cross_res', action='store_true',
                    help='禁用跨轮门控残差（消融用）。')
    ap.add_argument('--loop_cache', default='shared', choices=['shared', 'perloop'],
                    help='生成时循环各轮的 KV cache 策略。shared = 共用一份、只有第 0 轮写入'
                         '（历史行为）；perloop = 每轮各自一份。训练路径不传 cache，每轮都是'
                         '用当轮隐状态重算 K/V，所以 perloop 才与训练语义一致 —— shared 下'
                         '第 t>0 轮 attend 的是第 0 轮的 K/V。见 arXiv:2607.22083 / 2607.15456。')
    ap.add_argument('--use_bos', action='store_true',
                    help='序列前置一个可学习 BOS（推荐开启）。'
                         '不开的话：训练用 [x0..x_{T-1}] 预测 [x1..x_T]，永远不预测第一个 token；'
                         '而采样时 generate() 从 token ID 0 起步 —— 那是个真实码本条目，'
                         '模型被迫在与目标无关的前提下生成整张图，之后每步都在分布外。'
                         '实测（sample_ar_cond.py）：给 75%% 真实前缀能完美延续、50%% 良好、'
                         '0%% 变抽象色块 —— 断点就在冷启动。开启后序列长度与算力都不变。')
    ap.add_argument('--loop_p', default='',
                    help='自定义 L 采样分布（逗号分隔，长度=L_max，给出 p(L=1..L_max)）。'
                         '如 "1,0,0"=强制 L=1（对照实验）；"0.8,0,0.2"=偏斜课程。'
                         '空=均匀。需与 --loop_random 同用（保证循环模块按 L_max 挂载）。')
    ap.add_argument('--n_kv_head', type=int, default=4)
    # ---- L³ 替换 FFN（arXiv:2601.21461）----
    # 本机全模型实测（B=8 T=256 n_layer=10 d=512）：
    #   SwiGLU 基线            4.70 s/步  1.00x  30.49M
    #   L³ 1gram V=1024 k=2    3.59 s/步  1.31x  26.03M
    #   L³ 1gram V=1024 k=4    3.87 s/步  1.22x  39.14M
    #   L³ 1gram V=1024 k=8    3.96 s/步  1.19x  65.35M   ← 容量 2.14x 还快 1.19x
    #   L³ 4gram V=4096 k=2    4.20 s/步  1.12x  65.35M
    # ⚠️ 踩过的坑：表做大到 V=4096 且 k=8 时，稠密梯度会慢到 0.78x。
    #    根因是 nn.Embedding 反向产生【全表大小】的稠密梯度。
    #    V=1024 小表 + 1gram 键（只用当前 token，访问模式对 cache 友好）才快。
    ap.add_argument('--use_l3', action='store_true', help='用 L³ 层替换 SwiGLU FFN')
    ap.add_argument('--l3_v', type=int, default=1024, help='查表行数（默认 1024 = 词表大小）')
    ap.add_argument('--l3_k', type=int, default=8, help='每个键分配几个 embedding')
    ap.add_argument('--l3_d_emb', type=int, default=128)
    ap.add_argument('--l3_d_up', type=int, default=128)
    ap.add_argument('--l3_key_mode', default='1gram',
                    choices=['1gram', '2gram', '4gram', '5gram'],
                    help='空间因果邻域键的模式；1gram=只用当前 token（最快）')
    ap.add_argument('--l3_from', type=int, default=0, help='从第几层开始用 L³')
    ap.add_argument('--l3_sparse', action='store_true',
                    help='表用 sparse=True（需配 SparseAdam，本项目未接）')
    ap.add_argument('--ffn_dim', type=int, default=2048)
    ap.add_argument('--n_cluster', type=int, default=32)
    ap.add_argument('--think_tokens', type=int, default=16)
    ap.add_argument('--think_weight', type=float, default=0.1)
    ap.add_argument('--cond-mode', default='none', choices=['none', 'add', 'adaln', 'cross'])
    ap.add_argument('--cond-dropout', type=float, default=0.1)
    ap.add_argument('--captions', default='', help='auto_caption.py 产出的 jsonl')
    ap.add_argument('--cond-pt', default='',
                    help='预计算的 (N,text_dim) 条件张量。给了就直接用，跳过 CLIP 编码'
                         '（云端没有 CLIP 权重，这是唯一可行的路径）')
    ap.add_argument('--text-dim', type=int, default=512,
                    help='条件维度。★两阶段 AR 用：把 Stage-1 的粗 token 嵌入（维度=码本 '
                         'embed_dim，如 64）作为条件喂给 Stage-2，需要把 text_dim 设成 64，'
                         '并把 --cond-pt 传成 (N, 粗token数, 64) 的三维张量，配合 '
                         '--cond-mode cross 使用。')
    ap.add_argument('--pos-mode', default='2d', choices=['1d', '2d', 'ms'],
                    help='1d = 全序列位置嵌入；2d = 行/列分解（grid² 内）；'
                         'ms = 多尺度专用（VAR 式）：全序列位置嵌入 + 尺度嵌入。'
                         '⚠️ 多尺度 token（T=341）必须用 ms —— 2d 走 self.order，'
                         '只有 grid²=256 项，切片长度根本不匹配')
    ap.add_argument('--token-order', default='raster', choices=['raster', 'zorder'])
    # 治 exposure bias：训练时把输入 token 按此概率替换成随机 token（目标仍是真值）。
    # 依据见报告 §10.28：AR 喂真前缀距天花板仅 1.6 dB，喂自己输出塌 6.8 dB。
    ap.add_argument('--token-dropout', type=float, default=0.0,
                    help='训练时随机替换输入 token 的比例（0=关闭）。因果掩码下不泄漏。')
    # 真正的 scheduled sampling：把输入换成【模型自己的预测】，按线性日程从 0 爬升到
    # sched_sample。比 token dropout 更贴近推理时的错误分布，代价是多一次前向（约 +45% 步时）。
    ap.add_argument('--sched-sample', type=float, default=0.0,
                    help='终值比例（0=关闭）。用模型自己的 top-k 预测替换输入 token。')
    ap.add_argument('--sched-ramp', type=float, default=0.5,
                    help='爬升占训练总步数的比例（默认 0.5）。')
    # ★ 治 GRN 的"照抄捷径"：论文的损失在全部位置上平均，而输入里 l_t 比例的位置
    #   直接含真值 ⇒ 梯度被"复制输入"主导，模型学不会"无物可抄时怎么生成"。
    #   实测证据：GRN 训练 loss 正常下降（7.10→4.72），但采样塌成全黑；
    #   且精化步数越多越塌（−20.8 → −33.3 dB）⇒ 收敛到退化的不动点。
    ap.add_argument('--grn-corrupt-only', action='store_true',
                    help='GRN 损失只算在【被腐化的位置】上（~S 加权），而非全部位置。需配 --grn-refine。')
    ap.add_argument('--grn-refine', action='store_true',
                    help='GRN(arXiv 2604.13030) 风格 global refinement 训练：'
                         '输入 token 按 l~U[0,1] 随机替换成随机 token，'
                         '目标仍是真值，注意力必须全双向（配合 --mask-mode full）。'
                         '论文硬数字：refine vs 冻结式 mask 是 FID 3.63 vs 185.62；'
                         'random vs confidence 是 3.63 vs 10.64。')
    ap.add_argument('--mask-mode', default='causal',
                    choices=['causal', 'block', 'full', 'msblock', 'checkerboard', 'msa'],
                    help='causal = 光栅序逐 token 因果（默认，向后兼容）；'
                         'block = 借 VAR(arXiv 2404.02905, NeurIPS 2024 Best Paper) '
                         '的掩码结构：Z-order 块内双向、块间因果。'
                         '⚠️ 【已证实泄漏】：实测 teacher-forcing top-1 = 88.1%%'
                         '（正常 ~5%%），因为它抄了 mask 却没抄 VAR 的【多尺度残差量化】'
                         '这个前提 —— 单尺度同块 token 是真实内容、强相关。'
                         'msblock = VAR 的正确用法：token 按尺度拼接（patch_nums 决定），'
                         '规则 scale(i)>=scale(j)。同尺度 token 是【残差】(实测条件熵'
                         '泄漏 0.208 vs 单尺度 0.369，依赖降 1.8 倍)。必须配 --patch-nums。'
                         'checkerboard = 保底方案：尺度内棋盘格两相，同相 token 不相邻'
                         '⇒【严格无泄漏】(实测同尺度相邻可见对 = 0)，步数 5→10。'
                         'msa = 多尺度压缩注意力（2026-10-04 加）：局部窗口 + 行摘要(×grid)'
                         '+ 四行摘要(×4·grid)，借 CSA/HCA(arXiv:2606.19348) 的序列轴压缩，'
                         '但组大小按二维网格重定。动机是 §8.4 测到的"空间联合结构弱"'
                         '（AR 的布局拖垮 SDXL）。⚠️ 必须整段前向，不支持增量 KV cache，'
                         '采样要用逐步全量重算的采样器。')
    ap.add_argument('--msa-groups', default='',
                    help="多尺度注意力的组大小，逗号分隔；默认 'grid,4*grid'。"
                         "不要放 ≥T 的值：全局摘要要整幅图生成完才存在，因果下用不上。")
    ap.add_argument('--msa-local', type=int, default=0,
                    help='多尺度注意力的未压缩局部窗口长度（默认 2*grid = 两行）')
    ap.add_argument('--patch-nums', default='',
                    help='多尺度尺度划分，如 1,2,4,8,16（总 token = Σ pn² = 341）。'
                         '留空则尝试从 <tokens>.meta.json 自动读取（tokenize_dataset.py 会写）。'
                         'msblock/checkerboard 必需；pos_mode=ms 也必需')
    ap.add_argument('--loss-scale-alpha', type=float, default=0.0,
                    help='NextFlow(arXiv 2601.02204, 字节) 的【逐尺度 loss 权重】：'
                         'k_s = 1/(h_s·w_s)^alpha，再归一化以保持总 loss 量级不变。'
                         '【零参数、零额外 FLOPs】。论文动机：切到 512² 后整体 loss 降'
                         '但低尺度 loss 反升，Geneval 0.67→0.57；论文用 alpha=0.9。'
                         '我们 341-token 布局(1/4/16/64/256)下 alpha=0.9 的权重是 '
                         '[50.7, 14.55, 4.18, 1.20, 0.345]，最粗尺度那 1 个 token 的'
                         '相对权重放大 51 倍。0 = 关闭（默认，向后兼容）')
    ap.add_argument('--head-mode', default='flat', choices=['flat', 'factorized'],
                    help='flat 是本规模下的正确选择：输出层只占总算力 1.7%%，'
                         '因子化只省 1.6%% 却把 1024 类压成 32x32 可分解近似（实测劣于 flat）')
    ap.add_argument('--hier', action='store_true',
                    help='coarse-to-fine：Think 头预测 4x4 粗 token，Gen 头条件于它生成 16x16。'
                         '序列变为 [BOS ; coarse(16) ; fine(256)]，目标**不需要移位**'
                         '（BOS 与粗前缀天然提供了 shift）。开启后 --masks 被忽略。')
    ap.add_argument('--masks', default='')
    ap.add_argument('--nbr-fuse', action='store_true',
                    help='邻居融合：把上/左邻居的 token 嵌入直接拼进输入。'
                         '依据（实测 29113 张图的 16x16 token 流条件熵）：'
                         'H(x|左)=8.674 bits、H(x|上)=8.486 bits，但 '
                         'H(x|左+上)=3.384 bits —— 【强超可加】，信息在两邻居的'
                         '联合构型里。光栅序下上邻距离 15 步、zorder 下 1~2 步，'
                         '模型都要靠注意力间接学这个交互；直接拼进来就变成一跳。'
                         '因果安全（上/左必然 < 自身位置，代码里有 assert 兜底），'
                         '成本 2*d² = 524K 参数（+1.7%%）。仅支持单尺度布局')
    ap.add_argument('--nbr-fuse-window', type=int, default=0,
                    help='邻居融合的窗口半径：0=只用上/左两个 tap（配 --nbr-fuse）；'
                         '1=因果半平面 4 tap {(0,-1),(-1,-1),(-1,0),(-1,1)}；'
                         '2=更宽（7 tap）。'
                         '⚠️ 窗口该多宽【目前量不准】：probe_ctx_gain.py 用计数式留出 CE '
                         '比较不同上下文集合，结果加更多上下文反而更差 —— 因为稀疏偏差随'
                         '条件变量个数指数增长（左+上 有 778k 个键、4 tap 有 3.28M 个键，'
                         '训练样本只有 335 万）。所以窗口宽度靠 A/B 实测决定')
    ap.add_argument('--train_rows', type=int, default=0,
                    help='只用 token 文件的前 N 行做训练（0=全部）。'
                         '⚠️ 关键设计：验证集【仍然是原文件尾部 val_frac】，不随 N 变。'
                         '为什么必须这样：若直接截断文件再 --val_frac 0.05，验证集会落在'
                         '截断后的尾部 ⇒ 与全量实验的验证集不是同一批图 ⇒ 两者的 val '
                         '【不可比】。这是本项目第 21 类"看起来在对比、其实不可比"的坑。'
                         '用途：在花几小时生成新数据之前，先花 2 小时证伪"数据是杠杆"——'
                         '同配置同 seed 只用一半数据，看 val 是否变差')
    ap.add_argument('--repa-feats', default='',
                    help='REPA 教师特征文件（precompute_clip_feats.py 产出）。'
                         '依据：Seedream 3.0 技术报告 (arXiv 2504.11346) §2.2.2 —— '
                         '损失 = 中间层特征与教师特征的【余弦距离】，教师 DINOv2-L，λ=0.5；'
                         'REPA 官方代码 (2410.06940) 也支持 clip 类教师、超参相同。'
                         '我们只有本地 CLIP ViT-B/32（HF 不可达，拿不到 DINOv2）。'
                         '⚠️ 必须做【patch 级】对齐而非 global-CLS：'
                         'iREPA (2512.10794) 的结论是"驱动收益的是空间结构，不是全局语义"。'
                         '另按 iREPA 加两个修正：投影用 Conv2d、教师特征做空间归一化。')
    ap.add_argument('--repa-weight', type=float, default=0.5, help='λ，论文用 0.5')
    ap.add_argument('--repa-layer', type=int, default=-1,
                    help='取第几个 block 的输出（默认 n_layer//2 = 中间层）')
    ap.add_argument('--repa-patch', type=int, default=8,
                    help='教师特征的空间网格边长（CLIP-B/32 在 256px 下是 8 -> 8x8=64）')
    ap.add_argument('--repa-until', type=int, default=0,
                    help='到第几步停止加 REPA（0=一直加）')
    ap.add_argument('--val-range', default='',
                    help='把验证集【钉死】为 token 文件的行区间 a:b（如 27658:29113）。'
                         '为什么必需：数据从 29113 涨到 ~69000 后，--val_frac 0.05 会取'
                         '【新文件尾部 5%%】= 全是新图，与 ar_flat256 的验证集不是同一批 ⇒ '
                         '两个 val 不可比（本项目第 21 类"看起来在对比其实不可比"的坑）。'
                         '钉死区间后，两边验证的是同一批图，可以直接比。'
                         '⚠️ 同时会把该区间从训练集中排除，避免污染。')
    ap.add_argument('--maskgit', action='store_true',
                    help='★用 MaskGIT(arXiv 2202.04200) 的【离散扩散/掩码预测】替代因果 AR。'
                         '动机：① 它按构造不可能泄漏（输入是 [MASK]，loss 只算被掩码的位置）'
                         '—— 而我们试过的全双向+GRN 因为 loss 算在所有位置，模型学会抄自己，'
                         'val 0.97/top-1 79.6%% 全是假数字；'
                         '② 采样只要 n_iter 次前向（而不是 T 次），且每次能看【整张图】的双向上下文。'
                         '要求：--mask-mode full（自动切换）、不用 --use-bos、--head-mode flat。'
                         'vocab 会自动 +1（多一个 MASK id）')
    ap.add_argument('--maskgit-iter', type=int, default=16,
                    help='解掩码的迭代步数（MaskGIT 常用 8~16）')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--log_every', type=int, default=20)
    # ---- 验证集：判断过拟合 vs 欠拟合（14k 图对 30M 参数是很少的数据）----
    ap.add_argument('--val_frac', type=float, default=0.0,
                    help='划出多少比例做验证集（0=不划）。推荐 0.05')
    ap.add_argument('--val_every', type=int, default=500,
                    help='每多少步算一次验证 loss')
    ap.add_argument('--hflip', action='store_true',
                    help='水平翻转增强（免费的 2 倍数据）。token 网格左右翻等于原图水平翻转，'
                         '是合法的图像增强，能缓解小数据上的过拟合')
    ap.add_argument('--ckpt_every', type=int, default=2000)
    ap.add_argument('--no-ckpt', dest='no_ckpt', action='store_true',
                    help='完全不写 checkpoint')
    ap.add_argument('--resume', default='',
                    help='从 ckpt_<step>.pt 继续训练：恢复模型权重与全局步数，'
                         '并跳过各阶段中已完成的步。优化器动量会重置，'
                         '前几十步 loss 可能有轻微抖动。')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    os.makedirs(a.out, exist_ok=True)

    tokens = torch.load(a.tokens, map_location='cpu')
    if tokens.ndim != 2:
        tokens = tokens.reshape(tokens.shape[0], -1)
    N = tokens.shape[0]

    # 网格尺寸必须从 token 文件推出来。
    # 绝不能吃 SmallImageConfigV2 的默认值 grid=16 ——
    # 否则 8x8（64 token）的数据集会被当成 16x16 处理，位置编码直接对不上。
    T_tok = int(tokens.shape[1])

    # ---- 多尺度（ms_vq）token 的识别与 patch_nums 解析 ----
    # 多尺度 token 是 (N, 341)，不是完全平方数，sqrt 推断会直接报错。
    # 优先从 tokenize_dataset.py 写的 sidecar 读 —— 和"ch 必须从 ckpt 读"
    # 是同一条原则：架构信息不能靠命令行手输。
    pns = []
    if a.patch_nums:
        pns = [int(x) for x in a.patch_nums.replace(' ', '').split(',') if x]
    else:
        meta_p = a.tokens + '.meta.json'
        if os.path.exists(meta_p):
            with open(meta_p, encoding='utf-8') as f:
                _meta = json.load(f)
            if _meta.get('ms_vq'):
                pns = [int(x) for x in _meta.get('patch_nums', [])]
                print('[v2] 从 %s 读到 patch_nums=%s'
                      % (os.path.basename(meta_p), pns), flush=True)
    is_ms = bool(pns) and sum(p * p for p in pns) == T_tok
    if pns and not is_ms:
        raise SystemExit('patch_nums=%s 的总长 %d 与 token 数 %d 不符'
                         % (pns, sum(p * p for p in pns), T_tok))
    if is_ms:
        grid = max(pns)
        print('[v2] ★多尺度 token: patch_nums=%s 总长 %d  max_grid=%d'
              % (pns, T_tok, grid), flush=True)
        if a.mask_mode not in ('msblock', 'checkerboard', 'causal', 'full'):
            raise SystemExit('多尺度 token 请用 --mask-mode msblock 或 checkerboard'
                             '（block 是 Z-order 版，对多尺度不适用）')
        if a.pos_mode == '2d':
            raise SystemExit('多尺度必须用 --pos-mode ms（2d 走 self.order，只有 '
                             'grid²=256 项，T=341 会长度不匹配）')
    else:
        grid = int(round(T_tok ** 0.5))
        if grid * grid != T_tok:
            raise SystemExit('token 数 %d 不是完全平方数，且没有可用的 patch_nums '
                             '（需要 --patch-nums 或 <tokens>.meta.json）' % T_tok)

    # ---- token 顺序：zorder 必须【真的重排数据】，不能只改位置嵌入 ----
    # 2026-09-18 发现的 bug：`--token-order zorder` 以前只把 order 喂给位置嵌入
    # （cfg.token_order -> self.order），token 序列本身仍是光栅序 ⇒ 白改了。
    #
    # 为什么 zorder 重要（有硬数字支撑）：
    #   实测 16x16 token 流的条件熵
    #     H(x_t | 左邻)       = 8.674 bits
    #     H(x_t | 左邻+上邻)  = 3.384 bits   ← 熵掉了 5.29 bits！
    #   说明 2D 邻居携带的信息是压倒性的。但光栅序下 (r-1,c) 在序列里距离 15 步，
    #   信息要跨 15 步间接传递；模型实测只学到相当于"左邻"级别的上下文。
    #   Z-order（Morton）保证 (r-1,c) 与 (r,c-1) 的序列距离是 O(1)~O(2)，
    #   且仍是合法的 2D 因果序（上/左邻居的 morton 码必然更小）。
    #   ⚠️ 严格说光栅序 causal 的【上下文更大】（前缀是整片区域），zorder 只是让
    #   最近的邻居更近 ⇒ 这是【归纳偏置/可优化性】的赌注，不是信息量的赌注。
    #   在欠拟合（实测 gap≈+0.08）的情形下，这个赌注值得下。
    seq_perm = None
    flip_seq = None
    if a.token_order == 'zorder':
        if is_ms:
            raise SystemExit('多尺度布局的尺度内是双向的，zorder 无意义也无害；'
                             '请直接用 raster（--token-order raster）')
        from small_image_model_v2 import make_token_order
        seq_perm = make_token_order(grid, 'zorder')
        assert sorted(seq_perm) == list(range(grid * grid))
        tokens = tokens[:, torch.tensor(seq_perm, dtype=torch.long)]
        # 水平翻转在 zorder 序列空间 = 一个固定的位置置换，预先算好（免费）
        inv = [0] * len(seq_perm)
        for p, cell in enumerate(seq_perm):
            inv[cell] = p
        flip = [0] * len(seq_perm)
        for p, cell in enumerate(seq_perm):
            r, c = divmod(cell, grid)
            flip[p] = inv[r * grid + (grid - 1 - c)]
        flip_seq = torch.tensor(flip, dtype=torch.long)
        print('[v2] ★token 顺序 = zorder：已重排 token 序列（%d 项置换）'
              % len(seq_perm), flush=True)
        print('[v2]   zorder 下 2D 邻居的序列距离 ~O(1)，光栅序是 O(grid)=%d'
              % grid, flush=True)
        print('[v2]   依据：H(x|左邻)=8.674 bits  vs  H(x|左邻+上邻)=3.384 bits',
              flush=True)

    # 簇划分：在 VQ 码本上做 k-means（只给 factorized 头用）
    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    # ch 也必须从 checkpoint 读：8x8 的 tokenizer 是 5 层，16x16 是 4 层
    vq_ch = tuple(vk['ch']) if vk.get('ch') else (64, 128, 256, 256)
    if vk.get('ms_vq'):
        # 多尺度码本没有"层级排序"，簇划分对它无意义；head_mode=flat 也不需要
        cluster_of = offset_of = None
        local_vocab = int(vk.get('n_codes', 1024))
        per = local_vocab
        n_cluster = local_vocab
        print('[v2] 多尺度码本：跳过 k-means 簇划分（head_mode 需为 flat）', flush=True)
    else:
        # ⚠️ 必须用 vqvae_from_ckpt：它从 ckpt 读【全部】架构字段
        # （ch / embed_dim / use_attn / two_3x3）。以前这里硬编码
        # `VQVAE(embed_dim=64, n_codes=..., ch=vq_ch)`，导致带 use_attn/two_3x3 的
        # 分词器（如 P_16x16_attn_small）直接 load_state_dict 失败 ——
        # 这就是报告里记过的"7 处硬编码"老坑之一。
        vq, _ = vqvae_from_ckpt(vk)
        cluster_of, offset_of, per = build_cluster_map(vq.vq.codebook.weight.data, a.n_cluster)
        n_cluster = int(cluster_of.max().item()) + 1
        local_vocab = vq.vq.codebook.num_embeddings
    print('[v2] token 文件: %d 张 x %d token -> 网格 %dx%d   VQ ch=%s'
          % (N, T_tok, grid, grid, vq_ch), flush=True)
    if cluster_of is not None:
        print('[v2] 码本簇划分: %d 簇 x %d 码 (vocab=%d)'
              % (n_cluster, per, local_vocab), flush=True)

    # ---- MaskGIT（离散扩散）：vocab 多一个 MASK id ----
    # 依据：用户提的"离散扩散/MaskGIT"思路 + 我们自己的判决（MaskGIT 的训练目标
    # 按构造不可能泄漏，正好修掉 ar_grn256 那个 val=0.97 的假数字）。
    if a.maskgit:
        if a.use_bos:
            raise SystemExit('--maskgit 与 --use-bos 冲突：MaskGIT 要位置 i 装 token i，'
                             'BOS 会前置并把序列移位')
        if a.mask_mode != 'full':
            a.mask_mode = 'full'
            print('[maskgit] 自动改用 --mask-mode full（双向是 MaskGIT 的前提）', flush=True)
        if a.head_mode != 'flat':
            raise SystemExit('--maskgit 目前只支持 --head-mode flat')

    # ★ 课程的两个阶段必须【同一个模型】：total_tokens/grid 由 --total_tokens 覆盖，
    # 不能从 token 文件的行宽推（16x16 的文件是 256 宽，会建出 grid=16 的模型）。
    _tt = int(getattr(a, 'total_tokens', 0) or 0)
    if _tt:
        T_tok = _tt
        grid = int(round(_tt ** 0.5))
        print('[curriculum] total_tokens 覆盖为 %d ⇒ grid %d' % (T_tok, grid), flush=True)

    cfg = SmallImageConfigV2(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,
                             n_kv_head=a.n_kv_head, ffn_dim=a.ffn_dim,
                             vocab_size=(local_vocab + 1) if a.maskgit else local_vocab,
                             n_cluster=n_cluster, think_tokens=a.think_tokens,
                             cond_mode=a.cond_mode, cond_dropout=a.cond_dropout,
                             pos_mode=a.pos_mode, token_order=a.token_order,
                             mask_mode=a.mask_mode,
                             msa_groups=a.msa_groups, msa_local=a.msa_local,
                             head_mode=a.head_mode, hier=a.hier,
                             total_tokens=T_tok, grid=grid,
                             patch_nums=tuple(pns),
                             nbr_fuse=bool(getattr(a, 'nbr_fuse', False)),
                             nbr_window=int(getattr(a, 'nbr_fuse_window', 0) or 0),
                             loop_start=a.loop_start, loop_end=a.loop_end,
                             loop_times=a.loop_times, use_bos=a.use_bos,
                             use_l3=a.use_l3, l3_v=a.l3_v, l3_k=a.l3_k,
                             l3_d_emb=a.l3_d_emb, l3_d_up=a.l3_d_up,
                             l3_key_mode=a.l3_key_mode, l3_from=a.l3_from,
                             l3_sparse=a.l3_sparse,
                             text_dim=a.text_dim,
                               token_scale=int(getattr(a, 'token_scale', 1) or 1))
    model = SmallARImageModelV2(cfg)
    # ---- 循环 Transformer（可选，默认关闭）----
    if a.loop_L > 1 or a.loop_random:
        from loop_transformer import LoopConfig
        _loop_p = None
        if getattr(a, 'loop_p', ''):
            _loop_p = [float(x) for x in a.loop_p.split(',')]
            if len(_loop_p) > a.loop_L_max:
                raise SystemExit('--loop_p 长度(%d) 超过 --loop_L_max(%d)'
                                 % (len(_loop_p), a.loop_L_max))
            print('[loop] 自定义 L 采样分布 p =', _loop_p, flush=True)
        lc = LoopConfig(loop_start=a.loop_start, loop_end=a.loop_end,
                        loop_L=a.loop_L, loop_L_max=a.loop_L_max,
                        loop_random=a.loop_random, loop_p=_loop_p,
                        step_enc=not a.loop_no_step_enc,
                        cross_res=not a.loop_no_cross_res,
                        loop_cache=a.loop_cache)
        model.set_loop(lc)
        npass = a.n_layer + (a.loop_end - a.loop_start + 1) * (a.loop_L_max if a.loop_random else a.loop_L - 1)
    nparam = sum(p.numel() for p in model.parameters())
    npass = a.n_layer + (a.loop_end - a.loop_start + 1) * (a.loop_times - 1) if a.loop_times > 1 else a.n_layer

    # ---- NextFlow 逐尺度 loss 权重（零参数、零额外 FLOPs）----
    # k_s = 1/(h_s·w_s)^alpha，再归一化使 Σ_s (k_s · n_s) = N（保持总 loss 量级不变）。
    # 动机（arXiv 2601.02204）：不加权时 loss 被最细尺度主导 —— 我们 341 token 里
    # 256 个（75%）属于尺度 16，而最粗尺度只有 1 个 token（0.29%）。
    # 实测论文里切到 512² 后整体 loss 降但低尺度 loss 反升，Geneval 0.67→0.57。
    pos_weight = None
    if a.loss_scale_alpha and a.loss_scale_alpha > 0:
        if not pns:
            raise SystemExit('--loss-scale-alpha 需要 --patch-nums（或 .meta.json）')
        ns = [pn * pn for pn in pns]
        ks = [1.0 / (n ** a.loss_scale_alpha) for n in ns]
        tot = sum(k * n for k, n in zip(ks, ns))
        scale_f = float(T_tok) / tot
        ks = [k * scale_f for k in ks]
        w = []
        for k, n in zip(ks, ns):
            w.extend([k] * n)
        pos_weight = torch.tensor(w, dtype=torch.float32)
        # 只对尺度 16x16 的那一批 token 报一个对照值，方便和论文对齐
        print('[v2] ★NextFlow 逐尺度 loss 权重 alpha=%.2f  各尺度权重 = [%s]'
              % (a.loss_scale_alpha, ', '.join('%.3f' % k for k in ks)), flush=True)
        print('[v2]   最粗尺度那 %d 个 token 的相对权重 %.1fx（平均权重 = 1.0）；'
              '加权后总权重和 %.1f (= T=%d)'
              % (ns[0], ks[0], float(pos_weight.sum()), T_tok), flush=True)

    if getattr(a, 'mask_mode', 'causal') == 'block':
        print('[mask] VAR 风格 block-causal：Z-order 块内双向/块间因果  '
              'token_order=%s  自回归步数 = log4(T)+1' % a.token_order, flush=True)
        if a.token_order != 'zorder':
            print('[mask] ⚠️ block mask 要求 token_order=zorder，当前是 %s'
                  % a.token_order, flush=True)
    if getattr(a, 'grn_refine', False):
        print('[grn] global refinement 训练：输入按 l~U[0,1] 随机替换，'
              '目标为真值；mask_mode 应为 full（全双向）', flush=True)
        if getattr(a, 'mask_mode', 'causal') != 'full':
            print('[grn] ⚠️ grn_refine 要求 mask_mode=full', flush=True)
    print('[v2] params=%.2fM tokens=%s' % (nparam / 1e6, tuple(tokens.shape)), flush=True)
    print('[v2] n_layer=%d loop=[%d..%d]x%d -> 实际 %d 个 block pass/前向'
          % (a.n_layer, a.loop_start, a.loop_end, a.loop_times, npass), flush=True)
    print('[v2] use_bos=%s  (True 时 logits[:,i] 直接预测 tokens[:,i]，损失不移位)'
          % bool(getattr(cfg, 'use_bos', False)), flush=True)
    if a.use_l3:
        print('[v2] L³ 替换 FFN: V=%d k=%d d_emb=%d d_up=%d key=%s 从第%d层起'
              % (a.l3_v, a.l3_k, a.l3_d_emb, a.l3_d_up, a.l3_key_mode, a.l3_from), flush=True)

    start_gstep = 0
    if a.resume:
        rk = torch.load(a.resume, map_location='cpu', weights_only=False)
        # strict=False：允许【跨配置热启动】。
        # 典型用法：从没有 BOS 的旧 checkpoint 热启动带 BOS 的新训练 ——
        # 旧参数全部同名同形，只有新增的 bos_emb/bos_pos 缺失（它们本就该从头学）。
        # 其余缺失/多余键仍然打印出来，避免静默错配。
        r = model.load_state_dict(rk['model'], strict=False)
        miss = [k for k in r.missing_keys if 'bos' not in k]
        if miss or r.unexpected_keys:
            print('[v2] ⚠️ resume 键不匹配: missing=%s unexpected=%s'
                  % (miss[:6], list(r.unexpected_keys)[:6]), flush=True)
        if any('bos' in k for k in r.missing_keys):
            print('[v2] resume: 新增的 BOS 参数从初始化开始学（其余参数已载入）', flush=True)
        start_gstep = int(rk.get('step', 0))
        print('[v2] resume: %s -> 全局步 %d' % (os.path.basename(a.resume), start_gstep),
              flush=True)

    # 条件
    cond_all = None
    # ---- 优先用【预计算好的条件张量】----
    # 为什么需要：条件来自 CLIP 文本编码器，而云端 HF 不可达、也没有 CLIP 权重。
    # 与其上传 605MB 的 CLIP 快照，不如本地编好 (N,512) 张量（约 60MB）传上去。
    # 顺带把"训练时依赖 CLIP"这件事彻底去掉，云端更不容易出岔子。
    if a.cond_mode != 'none' and a.cond_pt:
        if not os.path.exists(a.cond_pt):
            raise SystemExit('--cond-pt 不存在: %s' % a.cond_pt)
        ct = torch.load(a.cond_pt, map_location='cpu', weights_only=False)
        ct = ct.float() if torch.is_tensor(ct) else torch.as_tensor(ct).float()
        # 2 维 = (N, text_dim) 的 pooled 条件（text/adaLN）；
        # 3 维 = (N, L, text_dim) 的【序列】条件（cross-attn，用于两阶段 AR 的粗 token 前缀）
        if ct.dim() == 2:
            ok = (ct.shape[0] == N and ct.shape[1] == cfg.text_dim)
        elif ct.dim() == 3:
            ok = (ct.shape[0] == N and ct.shape[2] == cfg.text_dim)
            if cfg.cond_mode != 'cross':
                raise SystemExit('--cond-pt 是三维序列条件，但 cond_mode=%s；'
                                 '序列条件必须用 --cond-mode cross' % cfg.cond_mode)
        else:
            ok = False
        if not ok:
            raise SystemExit('--cond-pt 形状 %s 与 tokens(%d)/text_dim(%d) 不符'
                             % (tuple(ct.shape), N, cfg.text_dim))
        cond_all = ct
        print('[v2] 条件就绪(预计算张量): %s  dim=%d  范数均值 %.4f'
              % (tuple(cond_all.shape), cond_all.dim(),
                 float(cond_all.reshape(-1, cond_all.shape[-1]).norm(dim=1).mean())), flush=True)
    if a.cond_mode != 'none' and cond_all is None and a.captions:
        pj = a.tokens + '.paths.json'
        if os.path.exists(pj):
            paths = json.load(open(pj, encoding='utf-8'))
            cond_all = load_caption_cond(a.captions, paths, cfg.text_dim)
            if cond_all is not None:
                print('[v2] 条件就绪: %s' % tuple(cond_all.shape), flush=True)
    if a.cond_mode != 'none' and cond_all is None:
        # ⚠️ 这里【必须炸】而不是静默降级：传了条件却在无条件训练，
        #    是本项目最隐蔽的一类失效（§10.22）。
        raise SystemExit('[v2] cond_mode=%s 但没有拿到任何条件 —— 拒绝静默退化为无条件训练'
                         % a.cond_mode)

    # mask
    masks = None
    if a.masks:
        z = np.load(a.masks, allow_pickle=False)
        m = z['masks']
        if m.shape[0] != N:
            raise SystemExit('masks 张数 %d != tokens %d' % (m.shape[0], N))
        masks = torch.from_numpy(m.reshape(m.shape[0], -1).astype(np.bool_))
        if seq_perm is not None:
            # token 被重排了，mask 必须用同一个置换，否则屏蔽区和 token 对不上
            masks = masks[:, torch.tensor(seq_perm, dtype=torch.long)]
            print('[v2] masks 已按 zorder 同步重排', flush=True)
        print('[v2] masks 屏蔽率 %.2f%%' % (masks.float().mean() * 100), flush=True)

    # ---- REPA（表示对齐，Seedream 3.0 §2.2.2 / REPA 2410.06940 / iREPA 2512.10794）----
    # 为什么做这个：判决实验证明容量是边际（200M 只领先 0.04）、分词器近似中性
    # （H(x|左邻) 三者只差 1%）、数据是杠杆（+0.18/翻倍，正在清洗 4 万张）。
    # REPA 是唯一还没试过的、与"表示/数据/容量"都正交的杠杆：
    # 用一个预训练视觉编码器的特征去约束中间层表示，把视觉先验注入一个数据受限的模型。
    repa = None
    if a.repa_feats:
        if not os.path.exists(a.repa_feats):
            raise SystemExit('--repa-feats 不存在: %s' % a.repa_feats)
        _F = torch.load(a.repa_feats, map_location='cpu')
        if isinstance(_F, dict):
            _F = _F['feats']
        _F = _F.float()
        if _F.shape[0] != N:
            raise SystemExit('教师特征 %d 行 != token %d 行（必须逐行对齐！）'
                             % (_F.shape[0], N))
        n_patch = a.repa_patch
        assert _F.shape[1] == n_patch * n_patch, \
            '教师 patch 数 %d != %d²（--repa-patch 设错？）' % (_F.shape[1], n_patch)
        if n_patch * n_patch != grid * grid and (grid % n_patch) != 0:
            raise SystemExit('网格 %d 不能被教师 patch %d 整除' % (grid, n_patch))
        repa = {
            'feats': _F, 'n_patch': n_patch,
            'layer': a.repa_layer if a.repa_layer >= 0 else a.n_layer // 2,
            'proj': nn.Conv2d(a.d_model, _F.shape[2], 3, padding=1),   # iREPA：用 Conv2d
            'buf': {},
        }
        # 前向钩子抓中间层输出（不改模型代码）
        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            repa['buf']['h'] = h
        repa['handle'] = model.blocks[repa['layer']].register_forward_hook(_hook)
        print('[repa] ★开启：教师 %s  %s  patch=%dx%d  dim=%d  层=%d  λ=%.2f'
              % (os.path.basename(a.repa_feats), tuple(_F.shape), n_patch, n_patch,
                 _F.shape[2], repa['layer'], a.repa_weight), flush=True)
        print('[repa]   iREPA 两个修正：投影=Conv2d(3x3)  +  教师特征【空间】归一化',
              flush=True)

    _OPTS = build_optimizers(a, model)

    opt = _OPTS[0] if len(_OPTS) == 1 else _OPTS
    if repa is not None:
        opt.add_param_group({'params': list(repa['proj'].parameters()), 'lr': a.lr})

    def repa_loss(batch_idx, x):
        """REPA 余弦对齐损失。返回标量（已平均）。"""
        Fh = repa['buf'].get('h')
        if Fh is None:
            return None
        B = Fh.shape[0]
        h = Fh[:, :T]                                  # 位置 i 的隐状态预测 token i
        hg = h.reshape(B, grid, grid, -1).permute(0, 3, 1, 2)      # (B,d,16,16)
        pj = repa['proj'](hg)                                      # (B,768,16,16)
        tf = repa['feats'][batch_idx].to(pj.device).view(B, repa['n_patch'],
                                                         repa['n_patch'], -1)
        tf = tf.permute(0, 3, 1, 2)                                # (B,768,8,8)
        # iREPA 的空间归一化（对 CLIP 这种弱教师尤其重要）
        tf = tf - tf.mean(dim=(2, 3), keepdim=True)
        tf = tf / (tf.std(dim=(2, 3), keepdim=True) + 1e-6)
        if repa['n_patch'] != grid:
            r = grid // repa['n_patch']
            tf = tf.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)
        cos = F.cosine_similarity(pj, tf, dim=1)                   # (B,16,16)
        return -cos.mean()

    # ---- 训练/验证划分 ----
    # 为什么要这个：14,433 张图对 30M 参数的无条件生成是很少的数据，
    # 必须知道是在【过拟合】还是【欠拟合】才能决定该加数据还是加模型。
    # 之前完全没有验证集，只能看训练 loss 一路降，无法判断。
    val_frac = max(0.0, min(0.5, getattr(a, 'val_frac', 0.0)))
    _vr = getattr(a, 'val_range', '') or ''
    if _vr:
        # ---- 把验证集钉死为行区间 [va, vb)，并把该区间从训练集中排除 ----
        # 为什么要这样：数据涨到 ~69000 行后，--val_frac 0.05 取的是【新文件尾部】，
        # 与 ar_flat256（29113 行文件的尾部）不是同一批图 ⇒ 两个 val 不可比。
        _a, _b = [int(x) for x in _vr.split(':')]
        assert 0 <= _a < _b <= N, '--val-range %s 超出 [0,%d)' % (_vr, N)
        va_sel = torch.arange(_a, _b)
        tr_idx_all = torch.cat([torch.arange(0, _a), torch.arange(_b, N)])
        n_val = _b - _a
        n_train = len(tr_idx_all)
        print('[v2] ★--val-range=%s：验证集钉死为行 [%d,%d) 共 %d 张，'
              '该区间已从训练集排除；训练 %d 行'
              % (_vr, _a, _b, n_val, n_train), flush=True)
        print('[v2]   （与 ar_flat256 验证的是同一批图 ⇒ val 可直接比）', flush=True)
    elif val_frac > 0:
        n_val = max(1, int(N * val_frac))
        n_train = N - n_val
        # ---- --train_rows：只用前 N 行训练，但验证集【仍是原文件尾部 5%】----
        # ⚠️ 不能直接截断文件再 --val_frac 0.05：那样验证集会落在截断后的尾部,
        # 与全量实验的验证集不是同一批图 ⇒ 两个 val 不可比。
        _tr = int(getattr(a, 'train_rows', 0) or 0)
        if _tr and _tr < n_train:
            tr_idx_all = torch.arange(0, _tr)
            print('[v2] ⚠️ --train_rows=%d：训练集只有前 %d 行，'
                  '验证集仍是原文件尾部 %d 张（与全量实验同一批图，可比）'
                  % (_tr, _tr, n_val), flush=True)
        else:
            tr_idx_all = torch.arange(0, n_train)
        # 验证集固定采样，保证 loss 曲线可比
        g = torch.Generator().manual_seed(1234)
        va_sel = torch.randperm(n_val, generator=g)[:min(512, n_val)] + n_train
        print('[v2] 训练/验证划分: train=%d val=%d (验证取固定 %d 张)'
              % (len(tr_idx_all), n_val, len(va_sel)), flush=True)
    else:
        n_train = N
        tr_idx_all = torch.arange(0, N)
        va_sel = None

    def val_loss(T, B):
        """在固定验证子集上算交叉熵（不训练）。

        ⚠️ 2026-09-18 判决（第 16 个度量陷阱）：`mask_mode='full'` 时【不能用干净输入
        评估】。全双向下位置 i 的注意力能看到自己的输入嵌入 token_emb(x[i])，
        输出头只要学会"抄自己的输入"，val 就崩到极低。实测 ar_grn256 的
        val=0.9716 nats / top-1=79.60%，而把输入换成随机后 CE=7.24 / top-1=0.17%
        —— **val 是抄出来的**。所以 full 模式必须用【和训练完全一样的随机替换】
        来评估；否则 val 与生成质量毫无关系，还会误导训练决策。
        """
        if va_sel is None:
            return None
        was = model.training
        model.eval()
        tot, cnt = 0.0, 0
        use_grn = getattr(a, 'grn_refine', False) or getattr(cfg, 'mask_mode', '') == 'full'
        with torch.no_grad():
            for s in range(0, len(va_sel), B):
                sel = va_sel[s:s + B]
                if len(sel) == 0:
                    continue
                xv = tokens[sel, :T].long()
                if a.maskgit:
                    # ★ MaskGIT 的验证【必须用同样的掩码方式】算 loss（只在掩码位）。
                    # 这是本项目最严重的一次度量失效的修复：
                    #   ar_grn256（全双向 + GRN 随机替换、loss 算在所有位置）报出
                    #   val=0.9716、top-1=79.6%，把输入换成随机后 CE=7.24/top-1=0.17%
                    #   ⇒ 那个 val 是"抄自己的输入"。MaskGIT 的输入是 [MASK]，
                    #   不含答案信息，所以按构造不可能抄。固定种子保证 val 可比。
                    _V = cfg.vocab_size - 1
                    _mid = cfg.vocab_size - 1
                    _g = torch.Generator().manual_seed(20240918)
                    rr = torch.rand(xv.shape[0], 1, generator=_g)
                    mk = torch.rand(xv.shape, generator=_g) < rr
                    mk = mk.to(xv.device)
                    if not bool(mk.any()):
                        continue
                    xin = torch.where(mk, torch.full_like(xv, _mid), xv)
                    lg = model.forward_masked(xin, None)[..., :_V]
                    l = F.cross_entropy(lg[mk], xv[mk])
                    tot += float(l) * len(sel); cnt += len(sel)
                    continue
                xin = xv
                if use_grn:
                    from grn_refine import grn_make_input
                    # 固定种子，保证 val 可比（否则每次抽样的 l 不同，曲线会抖）
                    _g = torch.Generator().manual_seed(20240918)
                    xin, _l, _S = grn_make_input(xv, cfg.vocab_size, generator=_g)
                o = model.forward_logits(xin, None)
                gg = o[1]
                lg = gg[0] if isinstance(gg, tuple) else gg
                if getattr(cfg, 'use_bos', False):
                    src, tgt = lg, xv
                    wsrc = None
                else:
                    src, tgt = lg[:, :-1], xv[:, 1:]
                    wsrc = pos_weight[1:].to(xv.device) if pos_weight is not None else None
                if wsrc is None:
                    l = F.cross_entropy(src.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
                else:
                    # ⚠️ 验证 loss 必须用【同一套】逐尺度权重，否则和训练 loss 不可比
                    # （训练 loss 会被权重改成加权平均，未加权的验证 loss 会系统性偏低）
                    le = F.cross_entropy(src.reshape(-1, cfg.vocab_size), tgt.reshape(-1),
                                         reduction='none').view(tgt.shape[0], tgt.shape[1])
                    l = (le * wsrc[None, :]).sum() / wsrc.sum().clamp(min=1e-6)
                tot += float(l) * len(sel); cnt += len(sel)
        if was:
            model.train()
        return tot / max(cnt, 1)


    # ---- 学习率调度：线性 warmup + 余弦退火 ----
    _total_steps = sum(int(s.split(':')[1]) for s in a.stages.split(','))
    _warm = max(0, a.warmup)

    _cycles = max(1, int(getattr(a, 'lr_restarts', 1) or 1))
    _decay = float(getattr(a, 'lr_restart_decay', 1.0) or 1.0)

    def lr_at(step):
        if _warm > 0 and step < _warm:
            return a.lr * float(step + 1) / float(_warm)
        span = max(1, _total_steps - _warm)
        prog = min(1.0, max(0.0, (step - _warm) / float(span)))
        if _cycles > 1:
            # 锯齿：把整段进度折进 _cycles 个周期。prog 会在每个周期边界从 1 回到 0，
            # 于是 lr 从 lr_min 跳回峰值——这就是"重启"。
            cyc = prog * _cycles
            idx = min(_cycles - 1, int(cyc))
            prog = cyc - idx
            peak = a.lr * (_decay ** idx)
        else:
            peak = a.lr
        return a.lr_min + (peak - a.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    _lr_schedule = ('cos' if _cycles == 1 else
                    'cos_restart%d%s' % (_cycles,
                                         '' if _decay == 1.0 else '_d%g' % _decay))

    def set_lr(step):
        v = lr_at(step)
        for g in opt_groups(opt):
            g['lr'] = v
        return v

    set_lr(start_gstep)
    bmap = parse_batch(a.batch)
    peak = Peak(); peak.start()
    log = open(os.path.join(a.out, 'train.jsonl'), 'a' if a.resume else 'w', encoding='utf-8')
    stats = []
    gstep = start_gstep
    # 腐化开关的实况统计（写进 summary，见下方自证逻辑）
    _corrupt_stat = dict(seen=0, td=0.0, ss=0.0, ss_rate=0.0)

    def tf(x):
        """token -> (cluster, code) 目标；越界则落到最后一个簇内位置。"""
        c = cluster_of[x]
        o = offset_of[x]
        return c, o.clamp(max=per - 1)

    resume_to = gstep
    seen = 0
    vl = None
    _last_val = [0]        # 上次算验证 loss 的全局步（用列表以便闭包内修改）
    for stage in a.stages.split(','):
        T, steps = stage.split(':'); T = int(T); steps = int(steps)
        if a.hier:
            # 4x4 粗图要从完整 16x16 采样导出，短序列拿不到 → hier 强制用满序列。
            # 课程仍然有意义：不同阶段的步数与学习率行为可以不同。
            T = cfg.grid ** 2
        stage_lo, stage_hi = seen, seen + steps
        seen = stage_hi
        # 断点续跑：整段已完成就跳过，否则从段内偏移接着跑
        if stage_hi <= resume_to:
            print('[v2] stage T=%d 已完成，跳过' % T, flush=True)
            continue
        skip = max(0, resume_to - stage_lo)
        B = bmap.get(T, bmap.get(None, 1))
        cond = torch.zeros(B, cfg.text_dim)
        print('[v2] stage T=%d steps=%d batch=%d%s' % (
            T, steps, B, (' (跳过前 %d 步)' % skip) if skip else ''), flush=True)
        ts, losses = [], []
        t0 = time.time()
        for i in range(steps):
            if i < skip:
                continue
            idx = tr_idx_all[torch.randint(0, len(tr_idx_all), (B,))]
            x = tokens[idx, :T].long()
            # 水平翻转增强：token 网格左右翻等于原图水平翻转，是【免费】的 2 倍数据。
            # 14,433 张图对 30M 参数的无条件生成是很少的数据，这一下能缓解过拟合。
            # 光栅序下翻转只是把每行的列序倒过来，不改变序列长度与生成顺序语义。
            if getattr(a, 'hflip', False):
                if flip_seq is not None:
                    # zorder：水平翻转 = 预计算的序列位置置换（比重建 2D 视图更省）
                    sel = torch.rand(B) < 0.5
                    if sel.any():
                        x = x.clone()
                        x[sel] = x[sel][:, flip_seq]
                else:
                    side = int(round(T ** 0.5))
                    if side * side == T:
                        xf = x.view(B, side, side)
                        sel = torch.rand(B) < 0.5
                        if sel.any():
                            xf = xf.clone()
                            xf[sel] = xf[sel].flip(-1)
                        x = xf.view(B, T)
            c = cond_all[idx] if cond_all is not None else cond
            t1 = time.perf_counter()

            if a.maskgit:
                # ---- MaskGIT 离散扩散（掩码预测）----
                # 每张图抽一个掩码比例 r ~ U[0,1]，逐位置独立掩码。
                # 为什么不按位置抽 r 而按图抽：MaskGIT 原文用的是 per-sample
                # 的调度（每个样本处于自己的"噪声水平"），这样一批里同时有
                # 从几乎全掩码到几乎全清的样本，梯度信号更丰富。
                #
                # ★★ 2026-09-19 修 BUG（本分支以前【没有 continue】）★★
                #   老代码在这里算完 loss_gen 后直接【掉进下面的因果分支】：
                #     第 864 行 x_in = x   ← 干净输入，把刚算好的掩码输入覆盖掉
                #     第 868 行 model.forward_logits(x_in, c)
                #     第 933 行 loss = loss_gen + ...  ← loss_gen 被重新赋值
                #   而 --maskgit 会把 mask_mode 强制成 full（双向），于是
                #   "位置 i 的双向注意力能看到位置 i+1 的真实 token" ⇒ 抄自己
                #   ⇒ 训练 loss 掉到 0.03，而 MaskGIT 的梯度【一次都没进过网络】。
                #   实测症状：train=0.0375 / val=16.1501 / gap=+16.09；
                #   独立用同一 ckpt 复算 masked CE ≈18（与 val 一致），
                #   训练循环报 0.03 ⇒ 两者矛盾，根因就在这个 fall-through。
                #   新版把整个 step（backward/step/log/ckpt/val）搬进本分支并 continue。
                _V = cfg.vocab_size - 1
                _mid = cfg.vocab_size - 1               # MASK id = 最后一个
                rr = torch.rand(B, 1, device=x.device)
                mk = torch.rand(B, T, device=x.device) < rr
                x_in = torch.where(mk, torch.full_like(x, _mid), x)
                gl = model.forward_masked(x_in, c)[..., :_V]
                if bool(mk.any()):
                    loss_gen = F.cross_entropy(gl[mk], x[mk])
                else:
                    loss_gen = torch.zeros((), device=x.device)
                loss_think = torch.zeros((), device=x.device)
                # REPA 不在这里做：REPA 对齐的是"干净图过编码器的表征"，
                # 掩码输入的表征与教师特征不同源，硬加会变成噪声项。
                loss = loss_gen
                opt_zero_grad(opt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_step(opt)
                ts.append(time.perf_counter() - t1)
                losses.append(float(loss.detach()))
                gstep += 1
                set_lr(gstep)
                # 不在这里重置 vl：外层第 745 行已有 vl = None，
                # 且主分支的约定是"沿用上一次算出的 val"，直到下次真正计算。
                # （原先在这里把 vl 赋成 NaN 会污染 jsonl —— NaN 在 Python 里是
                #   truthy，分析脚本 `if r.get('val')` 会把无效值当成有效值。）
                if i % a.log_every == 0 or i == steps - 1:
                    ve = getattr(a, 'val_every', 0)
                    if va_sel is not None and ve and (gstep - _last_val[0]) >= ve:
                        _last_val[0] = gstep
                        vl = val_loss(T, B)
                        print('[v2 T=%d] step %d  train=%.4f  val=%.4f  gap=%+.4f'
                              % (T, gstep, losses[-1], vl, vl - losses[-1]), flush=True)
                    log.write(json.dumps(dict(step=gstep, T=T, stage_step=i + 1, batch=B,
                                              lr=opt_lr(opt),
                                              loss=losses[-1], gen=float(loss_gen.detach()),
                                              think=0.0, repa=0.0, val=vl,
                                              mask_frac=float(mk.float().mean()),
                                              step_s=ts[-1],
                                              tokens_per_s=B * T / max(ts[-1], 1e-9),
                                              elapsed_s=time.time() - t0, rss_gb=rss()),
                                         ensure_ascii=False) + '\n')
                    log.flush()
                if a.ckpt_every and gstep % a.ckpt_every == 0 and not a.no_ckpt:
                    torch.save(dict(model=model.state_dict(), step=gstep, cfg=cfg.__dict__,
                                    cluster_of=cluster_of, offset_of=offset_of),
                               os.path.join(a.out, 'ckpt_%d.pt' % gstep))
                if i % max(1, steps // 5) == 0 or i == steps - 1:
                    print('[v2 maskgit T=%d] %d/%d loss=%.4f mask=%.3f %.3fs' % (
                        T, i + 1, steps, losses[-1], float(mk.float().mean()), ts[-1]),
                        flush=True)
                continue
            elif a.hier:
                # ---- coarse-to-fine ----
                # 序列 [BOS ; coarse(16) ; fine(256)]
                #   think 头取位置 0..15  -> 预测 coarse[0..15]   **无需移位**
                #   gen   头取位置 16..271 -> 预测 fine[0..255]    **无需移位**
                # BOS 与粗前缀本身就是 shift 的来源；再手动移位会错位。
                tho, geo = model.forward_hier(x, c)
                thl = tho[0] if isinstance(tho, tuple) else tho
                gel = geo[0] if isinstance(geo, tuple) else geo
                coarse = model.coarse_of(x)
                loss_think = F.cross_entropy(thl.reshape(-1, cfg.vocab_size), coarse.reshape(-1))
                loss_gen = F.cross_entropy(gel.reshape(-1, cfg.vocab_size), x.reshape(-1))
                gen = loss_gen
                think = loss_think
                loss = loss_gen + a.think_weight * loss_think
                # ---- REPA 项（表示对齐）----
                l_repa = 0.0
                if repa is not None:
                    if a.repa_until and gstep >= a.repa_until:
                        l_repa = 0.0
                    else:
                        _rl = repa_loss(idx, x)
                        if _rl is not None:
                            loss = loss + a.repa_weight * _rl
                            l_repa = float(_rl.detach())
                opt_zero_grad(opt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_step(opt)
                ts.append(time.perf_counter() - t1)
                losses.append(float(loss.detach()))
                gstep += 1
                set_lr(gstep)
                if i % a.log_every == 0 or i == steps - 1:
                    log.write(json.dumps(dict(step=gstep, T=T, stage_step=i + 1, batch=B,
                                              lr=opt_lr(opt),
                                              loss=losses[-1], gen=float(loss_gen.detach()),
                                              think=float(loss_think.detach()),
                                              step_s=ts[-1], tokens_per_s=B * T / max(ts[-1], 1e-9),
                                              elapsed_s=time.time() - t0, rss_gb=rss()),
                                         ensure_ascii=False) + '\n')
                    log.flush()
                if a.ckpt_every and gstep % a.ckpt_every == 0 and not a.no_ckpt:
                    torch.save(dict(model=model.state_dict(), step=gstep, cfg=cfg.__dict__),
                               os.path.join(a.out, 'ckpt_%d.pt' % gstep))
                if i % max(1, steps // 5) == 0 or i == steps - 1:
                    print('[v2 hier T=%d] %d/%d loss=%.4f gen=%.4f think=%.4f %.3fs' % (
                        T, i + 1, steps, losses[-1], float(loss_gen), float(loss_think), ts[-1]),
                        flush=True)
                continue

            # ---- GRN global refinement（arXiv 2604.13030）----
            # 输入按 l~U[0,1] 随机替换成随机 token，【目标仍是真值 x】
            # 论文硬数字：refine vs 冻结式 mask = FID 3.63 vs 185.62
            x_in = x
            _grn_S = None
            if getattr(a, 'grn_refine', False):
                from grn_refine import grn_make_input
                x_in, _grn_l, _grn_S = grn_make_input(x, cfg.vocab_size)
                # ★ 让 GRN 也自证：记录保留比例 l 与实际被替换的比例。
                #   否则和一个"根本没生效的 --grn-refine"从输出上无法区分
                #   （冒烟时就发现 corrupt 统计里 GRN 是空白）。
                _corrupt_stat['grn_n'] = _corrupt_stat.get('grn_n', 0) + 1
                _corrupt_stat['grn_l'] = _corrupt_stat.get('grn_l', 0.0) + float(_grn_l)
                _corrupt_stat['grn_repl'] = _corrupt_stat.get('grn_repl', 0.0) + \
                    (1.0 - float(_grn_S.float().mean()))
            # ---- token dropout / scheduled sampling（治 exposure bias）----
            # 动机（本项目的实测判决，见报告 §10.28）：
            #   喂真前缀时 AR 的高频缺口 −9.25 dB（距天花板 −7.66 只差 1.6），
            #   喂自己的输出时塌到 −16.08 dB、低频从 +0.32 崩到 −5.66（回归均值）。
            #   ⇒ 模型会预测，但不会从自己的错误里恢复；1024 步连续自我消费把误差滚成均值。
            # 做法：把输入里一部分 token 换成随机 token，【目标仍是干净的真值 x】。
            # 因果掩码下位置 i 看不到 x[i] ⇒ 不泄漏（区别于 ar_grn256 那次
            # 因为 mask_mode=full 让模型看到目标而造成的假收敛）。
            _td = float(getattr(a, 'token_dropout', 0.0) or 0.0)
            _corrupt_stat['seen'] += 1
            if _td > 0.0:
                _drop = torch.rand(x.shape, device=x.device) < _td
                _rand = torch.randint(0, cfg.vocab_size, x.shape,
                                      device=x.device, dtype=x.dtype)
                x_in = torch.where(_drop, _rand, x_in)
                _corrupt_stat['td'] += float(_drop.float().mean())
            # ---- scheduled sampling（真正的"喂模型自己的错误"）----
            # 为什么不能只用 token dropout：它替换成的是【均匀随机 token】，
            # 而推理时模型见到的是【它自己犯的错】—— 两者分布不同。
            # 这里改成把输入的一部分换成模型自己的预测（用与推理一致的 top-k 采样）。
            #
            # ⚠️ 必须【从 0 线性爬升】：训练早期模型输出近似随机，一上来就喂自己的预测
            #    等于喂结构化噪声，会直接毁掉训练（这也是 scheduled sampling 原始论文
            #    强调 inverse sigmoid 日程的原因）。
            # 泄漏检查：_pred 来自对【干净 x】的一次 teacher-forced 前向，
            #    _pred[i] 只依赖 x[:i]；把它放回输入位置 i 后，位置 i 仍看不到 x[i]
            #    （因果掩码）⇒ 不泄漏。
            _ss_max = float(getattr(a, 'sched_sample', 0.0) or 0.0)
            if _ss_max > 0.0:
                _ramp = float(getattr(a, 'sched_ramp', 0.5) or 0.5)
                _frac = min(1.0, gstep / max(1.0, _ramp * _total_steps))
                _ss = _ss_max * _frac
                _corrupt_stat['ss_rate'] = _ss
                if _ss > 1e-6:
                    with torch.no_grad():
                        _t2, _g2 = model.forward_logits(x, c)
                        _g2 = model.merge(_g2[0], _g2[1], None) if isinstance(
                            _g2, (tuple, list)) else _g2
                        _k = min(100, _g2.shape[-1])
                        _v, _ = torch.topk(_g2, _k, dim=-1)
                        _thr = _v[..., -1, None]
                        _g2 = torch.where(_g2 < _thr,
                                          torch.full_like(_g2, float('-inf')), _g2)
                        _pr = torch.softmax(_g2, dim=-1)
                        _pred = torch.multinomial(
                            _pr.reshape(-1, _pr.shape[-1]), 1).reshape(_pr.shape[:-1])
                    _m = torch.rand(x.shape, device=x.device) < _ss
                    x_in = torch.where(_m, _pred.to(x_in.dtype), x_in)
                    _corrupt_stat['ss'] += float(_m.float().mean())
            (thc, thd), (gec, ged) = model.forward_logits(x_in, c)

            keep = None
            if masks is not None:
                keep = (~masks[idx][:, :T])
            if a.head_mode == 'factorized':
                tgt_c, tgt_o = tf(x)
                # gen：预测 x[i+1]，所以 logits 取 [:-1]、目标取 [1:]
                gc, go = gec[:, :-1], ged[:, :-1]
                tcg, tog = tgt_c[:, 1:], tgt_o[:, 1:]
                if keep is None:
                    loss_gen = (F.cross_entropy(gc.reshape(-1, n_cluster), tcg.reshape(-1))
                                + F.cross_entropy(go.reshape(-1, per), tog.reshape(-1)))
                else:
                    kk = keep[:, 1:]
                    lc = F.cross_entropy(gc.reshape(-1, n_cluster), tcg.reshape(-1),
                                         reduction='none').view(B, T - 1)
                    lo = F.cross_entropy(go.reshape(-1, per), tog.reshape(-1),
                                         reduction='none').view(B, T - 1)
                    den = kk.sum().clamp(min=1.0)
                    loss_gen = ((lc + lo) * kk).sum() / den
                tt = min(cfg.think_tokens, T)
                if tt > 1:
                    thc2, thd2 = thc[:, :tt - 1], thd[:, :tt - 1]
                    tct, tot = tgt_c[:, 1:tt], tgt_o[:, 1:tt]
                    if keep is None:
                        loss_think = (F.cross_entropy(thc2.reshape(-1, n_cluster), tct.reshape(-1))
                                      + F.cross_entropy(thd2.reshape(-1, per), tot.reshape(-1)))
                    else:
                        kt = keep[:, 1:tt]
                        lc2 = F.cross_entropy(thc2.reshape(-1, n_cluster), tct.reshape(-1),
                                              reduction='none').view(B, tt - 1)
                        lo2 = F.cross_entropy(thd2.reshape(-1, per), tot.reshape(-1),
                                              reduction='none').view(B, tt - 1)
                        loss_think = ((lc2 + lo2) * kt).sum() / kt.sum().clamp(min=1.0)
                else:
                    loss_think = torch.zeros((), device=x.device)
            else:
                flat = model.merge(gec, ged, None)
                # use_bos=True 时 BOS 已经提供了 shift，logits[:, i] 直接预测 x[:, i]，
                # 再手动移位就会错位一格。
                if getattr(cfg, 'use_bos', False):
                    src, tgt = flat, x
                    ksrc = keep if keep is not None else None
                    wsrc = None
                else:
                    src, tgt = flat[:, :-1], x[:, 1:]
                    ksrc = keep[:, 1:] if keep is not None else None
                    # 同步移位：目标 x[1:] 对应的位置权重是 pos_weight[1:]
                    wsrc = pos_weight[1:].to(x.device) if pos_weight is not None else None
                if ksrc is None and wsrc is None and not (
                        getattr(a, 'grn_corrupt_only', False) and _grn_S is not None):
                    loss_gen = F.cross_entropy(src.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
                else:
                    # NextFlow 逐尺度加权：加权平均而不是算术平均。
                    # 权重已归一化到 Σw = T，所以除以 (Σ ksrc·wsrc) 保持量级。
                    lg = F.cross_entropy(src.reshape(-1, cfg.vocab_size), tgt.reshape(-1),
                                         reduction='none').view(B, tgt.shape[1])
                    ww = torch.ones_like(lg)
                    # ★ GRN "只算被腐化位置"：GRN Eq.7 的损失在【全部位置】上平均，
                    #   而输入里 l_t 比例的位置直接含着真值 ⇒ 梯度被"照抄输入"主导。
                    #   论文用 2B 参数能扛过这个捷径；我们在 30M 规模下模型学会了抄、
                    #   没学会从随机生成（推理从纯随机起步就只能吐边缘分布 ⇒ 全黑）。
                    #   ⇒ 只对【被替换掉的位置】算损失（~S），逼它学"无物可抄时怎么生成"。
                    if getattr(a, 'grn_corrupt_only', False) and _grn_S is not None:
                        _m = (~_grn_S).float()
                        if not getattr(cfg, 'use_bos', False):
                            _m = _m[:, 1:]
                        ww = ww * _m
                        # ★ 自证：记录实际参与损失的位置占比。
                        #   若这个数 ≈0.5（= 1−l 的期望）说明掩码在生效；
                        #   若 ≈1.0 说明 --grn-corrupt-only 静默没接上。
                        _corrupt_stat['gco_n'] = _corrupt_stat.get('gco_n', 0) + 1
                        _corrupt_stat['gco_frac'] = _corrupt_stat.get('gco_frac', 0.0) + \
                            float(_m.mean())
                    if wsrc is not None:
                        ww = ww * wsrc[None, :]
                    if ksrc is not None:
                        ww = ww * ksrc
                    loss_gen = (lg * ww).sum() / ww.sum().clamp(min=1e-6)
                loss_think = torch.zeros((), device=x.device)

            loss = loss_gen + a.think_weight * loss_think
            # ---- REPA 项（表示对齐）----
            l_repa = 0.0
            if repa is not None and not (a.repa_until and gstep >= a.repa_until):
                _rl = repa_loss(idx, x)
                if _rl is not None:
                    loss = loss + a.repa_weight * _rl
                    l_repa = float(_rl.detach())
            opt_zero_grad(opt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_step(opt)
            ts.append(time.perf_counter() - t1)
            losses.append(float(loss.detach()))
            gstep += 1
            set_lr(gstep)
            if i % a.log_every == 0 or i == steps - 1:
                ve = getattr(a, 'val_every', 0)
                if va_sel is not None and ve and (gstep - _last_val[0]) >= ve:
                    _last_val[0] = gstep
                    vl = val_loss(T, B)
                    # 过拟合判据：训练 loss 继续降但验证 loss 开始升
                    print('[v2 T=%d] step %d  train=%.4f  val=%.4f  gap=%+.4f'
                          % (T, gstep, losses[-1], vl, vl - losses[-1]), flush=True)
                log.write(json.dumps(dict(step=gstep, T=T, stage_step=i + 1, batch=B,
                                          lr=opt_lr(opt),
                                          loss=losses[-1], gen=float(loss_gen.detach()),
                                          think=float(loss_think.detach()),
                                          repa=l_repa,
                                          val=vl,
                                          step_s=ts[-1], tokens_per_s=B * T / max(ts[-1], 1e-9),
                                          elapsed_s=time.time() - t0, rss_gb=rss()),
                                     ensure_ascii=False) + '\n')
                log.flush()
            if a.ckpt_every and gstep % a.ckpt_every == 0 and not a.no_ckpt:
                torch.save(dict(model=model.state_dict(), step=gstep, cfg=cfg.__dict__,
                                cluster_of=cluster_of, offset_of=offset_of),
                           os.path.join(a.out, 'ckpt_%d.pt' % gstep))
            if i % max(1, steps // 5) == 0 or i == steps - 1:
                print('[v2 T=%d] %d/%d loss=%.4f %.3fs' % (T, i + 1, steps, losses[-1], ts[-1]),
                      flush=True)
        rec = dict(T=T, steps=steps, batch=B, mean_step_s=sum(ts) / len(ts),
                   first_loss=losses[0], last_loss=losses[-1], elapsed_s=time.time() - t0,
                   tokens_per_s=B * T / max(1e-9, sum(ts) / len(ts)))
        # ★ 自证开关真的生效：本项目的头号失效模式是"参数传了但静默没作用"，
        #   所以把两个腐化开关的【实际替换比例】写进 summary —— 每次运行都自报家门。
        if _corrupt_stat['seen'] > 0:
            rec['corrupt'] = dict(
                token_dropout=float(getattr(a, 'token_dropout', 0.0) or 0.0),
                sched_sample_max=float(getattr(a, 'sched_sample', 0.0) or 0.0),
                repl_token_dropout=round(_corrupt_stat['td'] / _corrupt_stat['seen'], 4),
                repl_sched=round(_corrupt_stat['ss'] / _corrupt_stat['seen'], 4),
                ss_rate_end=round(_corrupt_stat['ss_rate'], 4),
                n_steps_counted=_corrupt_stat['seen'])
            _gn = _corrupt_stat.get('grn_n', 0)
            if _gn > 0:
                rec['corrupt']['grn_refine'] = True
                rec['corrupt']['grn_mean_retention_l'] = round(_corrupt_stat['grn_l'] / _gn, 4)
                rec['corrupt']['grn_mean_replaced'] = round(_corrupt_stat['grn_repl'] / _gn, 4)
                rec['corrupt']['grn_n_steps'] = _gn
            _gc = _corrupt_stat.get('gco_n', 0)
            if _gc > 0:
                rec['corrupt']['grn_corrupt_only'] = True
                rec['corrupt']['grn_loss_weight_frac'] = round(_corrupt_stat['gco_frac'] / _gc, 4)
        stats.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    log.close()
    if not a.no_ckpt:
        torch.save(dict(model=model.state_dict(), step=gstep, cfg=cfg.__dict__,
                        cluster_of=cluster_of, offset_of=offset_of),
                   os.path.join(a.out, 'ckpt_last.pt'))
    summ = dict(kind='ar_v2', params=nparam, cfg=cfg.__dict__, stages=stats,
                peak_rss_gb=float(peak.done()), cond_used=cond_all is not None)
    with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
