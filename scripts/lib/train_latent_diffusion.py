# -*- coding: utf-8 -*-
"""train_latent_diffusion.py — 在【同一套 VQ 潜变量】上训一个小扩散模型，作为 AR 的对照。

对比设计（严格对齐，只换"生成范式"）：
  | 维度        | AR (ar_flat256)              | 扩散 (本脚本)                |
  | 数据        | 29113 图, 行 [27658,29113) 为验证 | 同                         |
  | 目标表示    | 离散 idx -> z_q              | 连续 z_q（同一个量化潜变量）  |
  | 解码器      | C_mse VQVAE dec              | 同一个 dec                  |
  | 模型规模    | 30.49M                       | UNet2D 25.98M               |
  | batch / lr  | 8 / 2e-4 warmup1000 cosine   | 完全相同                    |
  | 生成过程    | 256 次自回归（因果）          | N 次双向去噪（DDIM 50）      |

⚠️ 训练损失【不可比】（AR 是离散 CE，扩散是连续 MSE）。
   判决必须靠【采样出图】+ 采样耗时。

用法:
  python train_latent_diffusion.py --latents latents.pt --out .autopilot\\diff_cmse --steps 60000
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_unet(ch=(128, 256, 256), sample_size=16, in_ch=64):
    """潜变量空间上的 UNet。

    sample_size 必须来自【潜变量张量】而不是写死：
    16x16 网格 => 16；若换成 32x32 分词器 => 32（那时每个 token 只覆盖 8x8 像素）。
    """
    from diffusers import UNet2DModel
    return UNet2DModel(
        sample_size=sample_size, in_channels=in_ch, out_channels=in_ch,
        layers_per_block=2, block_out_channels=tuple(ch),
        down_block_types=tuple(['DownBlock2D'] * (len(ch) - 1) + ['AttnDownBlock2D']),
        up_block_types=tuple(['AttnUpBlock2D'] + ['UpBlock2D'] * (len(ch) - 1)),
        norm_num_groups=32,
    )


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        # warmup: 早期用 1/n 的滑动平均，避免 EMA 被随机初始权重拖住
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lr-min', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--train-rows', type=int, default=27658)
    ap.add_argument('--val-range', default='27658:29113')
    ap.add_argument('--val-every', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=0,
                    help='0=自动实测选优（推荐，可移植）；>0=手动指定。'
                         '★ 注意不能只按核数取最大值：本机实测【稳态】下 5 线程比 6 快，'
                         '因为持续负载会撞功耗墙。短基准测的是爆发峰值，会选错。')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--ch', default='128,256,256')
    # ---- Looped-DiT 分支（2026-09-22 加，默认 unet 保持原行为不变）----
    # 接口与 UNet2DModel 一致：x (B,C,G,G) + t -> (B,C,G,G)，所以扩散调度/val/日志全部复用。
    # ⚠️ LoopedDiT 签名默认 d=512 会建出 48.3M（文档目标 ≈25.9M 对应 d=384），故这里默认 384。
    ap.add_argument('--arch', default='unet', choices=['unet', 'looped'])
    ap.add_argument('--d', type=int, default=384, help='Looped-DiT 宽度 384=>27.2M  512=>48.3M')
    # ★ n_heads 必须写进 ckpt —— 它【不改变任何参数形状】（qkv 是 d->3d 的 Linear，
    #   与头数无关），所以「用 4 头训练、评测时按默认 8 头重建」会
    #   【静默加载成功但算错注意力】。这是报告里那类"参数没接线"的失效模式。
    ap.add_argument('--n-heads', type=int, default=8,
                    help='注意力头数；d/n_heads = d_head。实测 T=1024 下 d_head=96(n=4) '
                         '比 d_head=48(n=8) 的 SDPA 快约 13 个百分点。'
                         '注意 argparse 会对 help 做百分号格式化，此处不能出现裸百分号。')
    ap.add_argument('--n-unique', type=int, default=8)
    ap.add_argument('--loop-start', type=int, default=2)
    ap.add_argument('--loop-end', type=int, default=3)
    ap.add_argument('--n-loops', type=int, default=1, help='窗口重复次数；=1 即普通 8 块 DiT')
    ap.add_argument('--loop-t-emb', action='store_true')
    ap.add_argument('--loop-damp', default='1/n')
    ap.add_argument('--beta-schedule', default='scaled_linear',
                    help='★ 必须是 scaled_linear（SD 用的），不要用 squaredcos_cap_v2。'
                         '实测：squaredcos_cap_v2 的 ac[999]=2.4e-9 ⇒ sqrt=5e-5 ⇒ '
                         'DDIM 的 pred_x0=(x-√(1-ᾱ)ε)/√ᾱ 把 ε 误差放大 20291 倍；'
                         '而 scaled_linear 的 ac[999]=7.3e-4 ⇒ 只放大 37 倍。'
                         '后果：cosine 日程下模型在高 t 学到的就是"输出≈xt"'
                         '（因为那里 xt≈ε，这是损失最优），于是 DDIM 更新量 '
                         'x_prev=(a·a\'+b·b\')·x ≈ x，链几乎不动 ⇒ 纯噪声起步生成出彩虹噪点。'
                         '证据：按 t 分箱的 ε-MSE 在 900-1000 箱反而最好(0.0378)，'
                         '而 0-100 箱最差(0.286) —— 这不是"学得好"，是"高 t 只需复制输入"。')
    ap.add_argument('--beta-start', type=float, default=0.00085)
    ap.add_argument('--beta-end', type=float, default=0.012)
    # ---- 推理设备（2026-09-23 加，默认 cpu 保持原行为不变）----
    # 实测：自家 UNet 在 DirectML 上前向快 1.55~2.41x，端到端 DDIM 20 NFE 快 1.72x，
    # 误差 2.5e-06。但【训练】不能用核显（group_norm 反向回退 CPU，整步慢 1.45x）。
    # 所以这里只把 val_loss 的前向放到核显，训练照旧在 CPU。
    ap.add_argument('--infer-device', default='cpu', choices=['cpu', 'dml', 'auto'],
                    help='验证/推理用哪个设备跑；cpu=原行为，dml=核显，auto=有就用')
    # ---- 文本条件化（2026-09-23 加，默认关闭保持原行为）----
    # 为什么重要：无条件生成要把整个数据分布压进 27M 参数，必然模式平均 -> 出图是色块。
    # 加条件后模型只学 p(x|c)，难度是质的差别，而且可以直接用提示词控制出图。
    # .autopilot\cond_clip_29113.pt 是字幕的 CLIP 文本嵌入 (29113,512) fp16，
    # 用的路径清单与 latents 完全相同 => 行号一一对应。
    ap.add_argument('--cond-pt', default=None,
                    help='条件张量 .pt，形状 (N, cond_dim)，N 必须与 latents 一致')
    ap.add_argument('--cond-drop', type=float, default=0.1,
                    help='训练时把条件置零的概率（classifier-free guidance 用）')
    # ---- 逐通道标准化（2026-09-23 加）----
    # ★ 为什么必须做：实测这份 VQ 潜变量的逐通道 std 相差 12.5×（0.167~2.085），
    #   而【单标量归一化 + MSE】的损失权重是按方差走的 ⇒ 小方差通道的权重小 156 倍。
    #   但消融实验证明：通道 49-64（方差仅占 3.7%）删掉会让重建 PSNR 掉 10.23 dB，
    #   与删掉方差占 38% 的前 8 个通道一样严重；只留前 32 通道 PSNR 只有 11.58。
    #   ⇒ 方差排序和重要性排序不一致，MSE 的权重分配是错的，小方差通道学不到。
    #   这是 AR 与扩散两条路共同栽掉的一个原因（两者用同一份 latents + MSE 类损失）。
    ap.add_argument('--chan-norm', action='store_true', default=True,
                    help='逐通道标准化（默认开）。关掉可复现旧的单标量行为。')
    ap.add_argument('--no-chan-norm', dest='chan_norm', action='store_false')
    # ---- 预测目标 eps vs v（2026-09-23 加）----
    # ★ 为什么需要 v-prediction（实测诊断）：
    #   用 eps 目标时，高 t 的损失只有 0.0014 —— 因为那里 xt≈ε，"直接复制输入"就几乎最优，
    #   【结构信息在损失里占比极小，几乎没有梯度去学"从噪声推断布局"】。
    #   后果实测：单步 x0 在 t=50 能解码出能认的人（中位误差 0.013），
    #   但 t=950 的中位误差 0.26；完整采样（真噪声起跳去噪）出来只有【全局色调、零空间结构】，
    #   像素空间重建 PSNR 11.14 dB，而 VQVAE 天花板是 21.23 dB，且 1000→2000 步毫无改善。
    #   v = sqrt(ac)*eps - sqrt(1-ac)*x0，高 t 时 v ≈ -x0 ⇒ 损失【直接监督结构推断】。
    ap.add_argument('--pred', default='eps', choices=['eps', 'v'],
                    help='预测目标：eps（原行为）或 v（推荐，见注释）')
    a = ap.parse_args()
    if a.threads and a.threads > 0:
        nthreads = a.threads
        print('[diff] 线程数 = %d（手动指定）' % nthreads, flush=True)
    else:
        from autotune_threads import best_threads
        nthreads = best_threads(verbose=True)
        a.threads = nthreads
    torch.set_num_threads(nthreads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    from diffusers import DDPMScheduler
    PTYPE = 'v_prediction' if a.pred == 'v' else 'epsilon'
    sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule=a.beta_schedule,
                        beta_start=a.beta_start, beta_end=a.beta_end,
                        clip_sample=False, prediction_type=PTYPE)
    ac = sch.alphas_cumprod
    print('[diff] beta_schedule=%s (ac[999]=%.3e, sqrt=%.5f => 放大 %.0fx)  预测目标=%s'
          % (a.beta_schedule, ac[999], ac[999].sqrt(), 1.0 / float(ac[999].sqrt()), a.pred),
          flush=True)

    # ★ 保持存储精度、按批上转 fp32。
    #   为什么：16x16 latents 是 954 MB（fp16），`.float()` 后 1.9 GB 尚可；
    #   但 32x32 是 3.6 GB，一次性上转会变 7.3 GB，本机 15 GB 内存扛不住。
    #   按批上转每批只有几 MB，开销可忽略。
    L = torch.load(a.latents, map_location='cpu', weights_only=False)
    if L.dim() != 4:
        raise SystemExit('[diff] latents 必须是 (N,C,H,W)，得到 %s' % (tuple(L.shape),))
    _Ldtype = L.dtype
    N, C, H, W = L.shape
    lo, hi = (int(x) for x in a.val_range.split(':'))
    tr = torch.arange(0, min(a.train_rows, N))
    va = torch.arange(lo, min(hi, N))
    print('[diff] latents %s  存储精度=%s (%.2f GB)  train=%d val=%d'
          % (tuple(L.shape), _Ldtype, L.numel() * L.element_size() / 1e9, len(tr), len(va)),
          flush=True)

    # ★ 潜变量必须归一化到单位尺度！
    #   实测这份 VQ 潜变量 std≈3.98（不是 1）。扩散的噪声日程假设数据是单位尺度：
    #   若 std 远大于 1，则 t=1000 的"纯噪声"其实仍被信号主导（SNR 不为 0），
    #   训练会退化成一个"去小幅噪声"的自动编码器，采样必然失败。
    #   SD 里的 0.18215 就是这个标量的对应物 —— 这里【从数据量出来】并记进 ckpt。
    #
    #   2026-09-23 起默认改为【逐通道】标准化：单标量下逐通道 std 仍差 12.5×，
    #   而消融显示小方差通道同样关键（详见 --chan-norm 的注释）。
    #   采样时必须按同样的参数反变换。
    #   统计量【分块流式】算，避免把整个潜变量上转成 fp32（32x32 时是 7.3 GB）。
    _n = 0
    _s = torch.zeros(C, dtype=torch.float64)
    _ss = torch.zeros(C, dtype=torch.float64)
    for _i in range(0, len(tr), 2048):
        _b = L[tr[_i:_i + 2048]].float()
        _bf = _b.reshape(_b.shape[0], C, -1)
        _s += _bf.sum(dim=(0, 2)).double()
        _ss += _bf.pow(2).sum(dim=(0, 2)).double()
        _n += _bf.shape[0] * _bf.shape[2]
    chan_mean = (_s / _n).float()
    chan_var = (_ss / _n - (_s / _n) ** 2).clamp_min(0)
    chan_std = chan_var.sqrt().float().clamp_min(1e-6)
    # 单标量字段：用通道方差的均值开方（等价于整体 std 的一个稳定估计）
    scale = float(chan_var.mean().sqrt())
    if a.chan_norm:
        print('[diff] 逐通道标准化: 通道 std %.3f~%.3f (差 %.1fx) -> 目标 1.0'
              % (float(chan_std.min()), float(chan_std.max()),
                 float(chan_std.max() / chan_std.min())), flush=True)
    else:
        print('[diff] 单标量归一化（旧行为）: scale=%.6f，通道 std 差 %.1fx'
              % (scale, float(chan_std.max() / chan_std.min())), flush=True)

    def get_x0(idx):
        """按批取潜变量并归一化到单位尺度（fp32）。内存友好。"""
        xb = L[idx].float()
        if a.chan_norm:
            return (xb - chan_mean.view(1, C, 1, 1)) / chan_std.view(1, C, 1, 1)
        return xb / scale

    print('[diff] 归一化参数: chan_std %.4f~%.4f  (scale 字段=%.6f，采样时按 chan_mean/chan_std 反变换)'
          % (float(chan_std.min()), float(chan_std.max()), scale), flush=True)

    # ---- 条件张量 ----
    COND = None
    cond_dim = 0
    if a.cond_pt:
        COND = torch.load(a.cond_pt, map_location='cpu', weights_only=False)
        if not torch.is_tensor(COND):
            COND = COND.get('cond') or COND.get('c')
        COND = COND.float().contiguous()
        if COND.shape[0] != N:
            raise SystemExit('[diff] 条件行数 %d 与 latents %d 不一致（索引会错位，拒绝训练）'
                             % (COND.shape[0], N))
        cond_dim = int(COND.shape[1])
        print('[diff] 条件 %s  形状=%s  两两余弦(前400) mean=%.3f'
              % (os.path.basename(a.cond_pt), tuple(COND.shape),
                 float((COND[:400] @ COND[:400].T)[~torch.eye(400, dtype=bool)].mean())),
              flush=True)
        if a.arch != 'looped':
            raise SystemExit('[diff] 目前只有 --arch looped 支持条件化')
        print('[diff] 条件化开启 cond_dim=%d  训练时置零概率=%.2f（CFG 用）'
              % (cond_dim, a.cond_drop), flush=True)

    if a.arch == 'looped':
        from looped_dit import LoopedDiT, block_executions
        model = LoopedDiT(in_ch=C, latent_size=H, d=a.d, n_unique=a.n_unique,
                          n_heads=a.n_heads,
                          loop_start=a.loop_start, loop_end=a.loop_end,
                          n_loops=a.n_loops, loop_t_emb=a.loop_t_emb,
                          loop_damp=a.loop_damp, cond_dim=cond_dim)
        nblk = block_executions(a.n_unique, a.loop_start, a.loop_end, a.n_loops)
        print('[diff] LoopedDiT d=%d  blocks/forward=%d  n_loops=%d  damp=%s  cond_dim=%d'
              % (a.d, nblk, a.n_loops, a.loop_damp, cond_dim), flush=True)
    else:
        model = build_unet([int(x) for x in a.ch.split(',')], sample_size=H, in_ch=C)
    npar = sum(p.numel() for p in model.parameters())
    print('[diff] %s params=%.2fM  sample_size=%d in_ch=%d  (AR 对照 30.49M)'
          % (a.arch, npar / 1e6, H, C), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    ema = EMA(model)
    g = torch.Generator().manual_seed(a.seed)

    # ---- 推理后端：训练在 CPU，验证可放核显 ----
    from infer_backend import InferBackend
    ibe = InferBackend(model, a.infer_device, verbose=True)
    if a.infer_device != 'cpu':
        # 数值自检：建完副本立刻验一次，不一致就直接退回 CPU，不让它污染训练结论
        try:
            _err, _verdict = None, None
            _xb = torch.randn(min(a.batch, 4), C, H, H)
            _tb = torch.full((_xb.shape[0],), 500, dtype=torch.long)
            with torch.no_grad():
                _o = model(_xb, _tb)
                _ref = _o.sample if hasattr(_o, 'sample') else _o
            _got = ibe.forward(_xb, _tb)
            _err = (_got - _ref).abs().max().item() / (_ref.abs().max().item() + 1e-12)
            if _err >= 1e-2:
                print('[infer] ⚠️ 自检误差 %.3e 过大，退回 CPU 推理' % _err, flush=True)
                ibe = InferBackend(model, 'cpu', verbose=False)
            else:
                print('[infer] 自检通过：相对误差 %.3e（fp32 噪声级）' % _err, flush=True)
        except Exception as e:
            print('[infer] ⚠️ 自检失败（%s: %s），退回 CPU 推理'
                  % (type(e).__name__, str(e)[:80]), flush=True)
            ibe = InferBackend(model, 'cpu', verbose=False)

    def set_lr(step):
        if step < a.warmup:
            lr = a.lr * step / max(1, a.warmup)
        else:
            p = (step - a.warmup) / max(1, a.steps - a.warmup)
            lr = a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * min(1.0, p)))
        for grp in opt.param_groups:
            grp['lr'] = lr
        return lr

    @torch.no_grad()
    def val_loss():
        """固定噪声/时间步的验证 MSE（扩散内部可比，跨范式不可比）。

        ⚠️ 只用前 256 张：全量 1455 张要 182 个 batch × ~0.6s ≈ 109s，
        每 500 步跑一次就是 ~27% 的额外开销。256 张足够看趋势。

        2026-09-23：前向改走 ibe（可选核显）。ibe.sync 保证用的是最新权重；
        噪声/时间步的生成方式与原来完全一致（同一个 gv、同一段代码），
        所以 --infer-device cpu 与 dml 得到的 val 数值应当一致到 fp32 噪声级。
        """
        model.eval()
        ibe.sync(model)          # 把最新训练权重灌进推理副本（cpu 模式下是空操作）
        gv = torch.Generator().manual_seed(12345)
        sel = va[:256]
        tot, cnt = 0.0, 0
        for s in range(0, len(sel), a.batch):
            x0 = get_x0(sel[s:s + a.batch])
            eps = torch.randn(x0.shape, generator=gv)
            t = torch.randint(0, 1000, (x0.shape[0],), generator=gv)
            xt = ac[t].sqrt().view(-1, 1, 1, 1) * x0 + (1 - ac[t]).sqrt().view(-1, 1, 1, 1) * eps
            cc = COND[sel[s:s + a.batch]] if COND is not None else None
            pred = ibe.forward(xt, t, cc)   # 已在 CPU 上；UNet 取 .sample / DiT 直接张量由后端处理
            # ★ 跨预测目标可比：v 目标下模型输出的是 v，先换算回 eps 再算 MSE。
            #   eps = sqrt(1-ac)*xt + sqrt(ac)*v   （与 x0 = sqrt(ac)*xt - sqrt(1-ac)*v 同源）
            #   这样 val 数值在 --pred eps 与 --pred v 之间可以直接比较。
            if PTYPE == 'v_prediction':
                se = (1 - ac[t]).sqrt().view(-1, 1, 1, 1)
                sa = ac[t].sqrt().view(-1, 1, 1, 1)
                pred = se * xt + sa * pred
            tot += float(F.mse_loss(pred, eps)) * x0.shape[0]; cnt += x0.shape[0]
        model.train()
        return tot / max(cnt, 1)

    log = open(os.path.join(a.out, 'train.jsonl'), 'w', encoding='utf-8')
    model.train()
    t0 = time.time()
    losses, ts = [], []
    vl = None
    for step in range(1, a.steps + 1):
        idx = tr[torch.randint(0, len(tr), (a.batch,), generator=g)]
        x0 = get_x0(idx)
        eps = torch.randn(x0.shape, generator=g)
        t = torch.randint(0, 1000, (x0.shape[0],), generator=g)
        xt = ac[t].sqrt().view(-1, 1, 1, 1) * x0 + (1 - ac[t]).sqrt().view(-1, 1, 1, 1) * eps
        # 条件：按 cond_drop 概率整条置零（classifier-free guidance 的无条件分支）。
        # ★ 必须【整条】置零而不是加噪，否则学出来的不是 null 条件。
        if COND is not None:
            cc = COND[idx].clone()
            if a.cond_drop > 0:
                keep = (torch.rand(a.batch, generator=g) >= a.cond_drop)
                cc[~keep] = 0.0
        else:
            cc = None
        t1 = time.perf_counter()
        _o = model(xt, t, cc) if COND is not None else model(xt, t)
        pred = _o.sample if hasattr(_o, 'sample') else _o
        # 目标随 --pred 变：eps 或 v。注意 val_loss 里为了【跨预测目标可比】，
        # 一律把预测换算回 eps 再算 MSE（见下面的 v2eps）。
        if PTYPE == 'v_prediction':
            tgt = sch.get_velocity(x0, eps, t)
        else:
            tgt = eps
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr(step)
        opt.step()
        ema.update(model)
        ts.append(time.perf_counter() - t1); losses.append(float(loss.detach()))

        if step % a.log_every == 0 or step == a.steps:
            if step % a.val_every == 0 or step == a.steps:
                vl = val_loss()
                print('[diff] step %d  train=%.5f  val=%.5f  lr=%.2e  %.2fs'
                      % (step, losses[-1], vl, opt.param_groups[0]['lr'], ts[-1]), flush=True)
            log.write(json.dumps(dict(step=step, loss=losses[-1], val=vl,
                                      lr=opt.param_groups[0]['lr'], step_s=ts[-1],
                                      elapsed_s=time.time() - t0)) + '\n')
            log.flush()
        if step % a.ckpt_every == 0:
            torch.save(dict(model=model.state_dict(), ema=ema.shadow, step=step,
                            ch=a.ch, latents=os.path.basename(a.latents), scale=scale,
                            cond_dim=cond_dim, chan_norm=a.chan_norm,
                            arch=a.arch, d=a.d, n_heads=a.n_heads, n_unique=a.n_unique,
                            loop_start=a.loop_start, loop_end=a.loop_end, n_loops=a.n_loops,
                            pred_type=PTYPE,
                            chan_mean=chan_mean, chan_std=chan_std,
                            sample_size=H, in_ch=C, beta_schedule=a.beta_schedule,
                            beta_start=a.beta_start, beta_end=a.beta_end),
                       os.path.join(a.out, 'ckpt_%d.pt' % step))

    torch.save(dict(model=model.state_dict(), ema=ema.shadow, step=a.steps,
                    ch=a.ch, latents=os.path.basename(a.latents), scale=scale,
                            cond_dim=cond_dim, chan_norm=a.chan_norm,
                            arch=a.arch, d=a.d, n_heads=a.n_heads, n_unique=a.n_unique,
                            loop_start=a.loop_start, loop_end=a.loop_end, n_loops=a.n_loops,
                            pred_type=PTYPE,
                            chan_mean=chan_mean, chan_std=chan_std,
                            sample_size=H, in_ch=C, beta_schedule=a.beta_schedule,
                            beta_start=a.beta_start, beta_end=a.beta_end),
               os.path.join(a.out, 'ckpt_last.pt'))
    log.close()
    ibe.close()
    print('[diff] done  mean_step=%.2fs' % (sum(ts) / max(1, len(ts))), flush=True)
    print(json.dumps(dict(params=npar, steps=a.steps, mean_step_s=sum(ts) / max(1, len(ts)),
                          first=losses[0], last=losses[-1], val=vl)))


if __name__ == '__main__':
    main()
