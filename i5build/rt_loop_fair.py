"""循环 vs 独立层，在神经渲染负载上的干净对照（带 warmup + cosine）。

第一轮（rt_probe.py part B）的结论不能直接用：plain7 拿到 5.04 dB，那是"图基本全错"，
而它和其他臂共用同一套 lr=5e-4、零 warmup —— 7 层 ReLU 叠 L=8 的高频位置编码，
炸掉或死掉都不奇怪。同参数的 loop3 vs plain3（21.65 vs 17.17 dB）可信，因为两条都是浅网。

这一轮补齐三件事：
  1. warmup + cosine，各臂同一套日程，让深层网络有机会训起来；
  2. 加入"门控残差"版循环 —— 豆包那份每轮 cat([x, h]) 把输入层撑到 182 宽（23,296 MAC/轮），
     换成一次投影 + 逐元素残差后每点从 119,552 降到 56,576 MAC（2.11×），而循环语义不变；
  3. 报出真实 MAC（上一轮的 macs_per_point 把循环只数了一遍，低估了 3 倍）。
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\doub')
import neural_rt_demo as rt  # noqa: E402


def true_macs(model, loops=1):
    """每点乘加数。loops>1 表示同一组层被重复执行 loops 次。"""
    tot = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            tot += m.in_features * m.out_features
    return tot * loops


class LoopGatedFine(nn.Module):
    """把豆包的 cat([x, h]) 换成"输入投影一次 + 逐元素残差"。

    原版每轮：h = relu(Linear_182->128([x, h]))，输入宽度 182 ⇒ 23,296 MAC/轮
    这一版  ：h = relu(Linear_128->128(h) + x_proj)，16,384 MAC/轮，参数也更少
    跨轮信息通路保持（每轮都能看到输入 x），语义与 GatedCrossIteration 同源。
    """

    def __init__(self, hidden=128, in_ch=54, loops=3):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.phys = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 4)
        self.hidden = hidden
        self.loops = loops

    def forward(self, p, d):
        x = self.proj(torch.cat([rt.positional_encoding(p, L=8), d], -1))
        h = torch.zeros_like(x)
        for _ in range(self.loops):
            h = F.relu(self.phys(h) + x)
        out = self.head(h)
        return torch.sigmoid(out[..., :3]), F.relu(out[..., 3])


class PlainFine(nn.Module):
    def __init__(self, hidden=128, in_ch=54, n_layer=7):
        super().__init__()
        layers = [nn.Linear(in_ch, hidden), nn.ReLU()]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 4)
        self.n_layer = n_layer

    def forward(self, p, d):
        x = torch.cat([rt.positional_encoding(p, L=8), d], -1)
        out = self.head(self.body(x))
        return torch.sigmoid(out[..., :3]), F.relu(out[..., 3])


def build(kind):
    if kind == 'loop3_concat':
        return rt.RecurrentFineNetwork(loops=3), 3
    if kind == 'loop3_gated':
        return LoopGatedFine(loops=3), 3
    if kind.startswith('plain'):
        n = int(kind[5:])
        return PlainFine(n_layer=n), 1
    raise ValueError(kind)


def run(kind, steps=800, threads=6, seed=42, H=64, W=64, batch=512, lr=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    coarse = rt.CoarseNetwork()
    fine, loops = build(kind)
    params = (sum(p.numel() for p in coarse.parameters())
              + sum(p.numel() for p in fine.parameters()))
    macs = true_macs(coarse, 1) + true_macs(fine, loops)
    allp = list(coarse.parameters()) + list(fine.parameters())
    opt = torch.optim.Adam(allp, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, max(0.0, (s - 100) / max(1, steps - 100)))))))
    c2ws, images = rt.make_dataset(n_views=40, H=H, W=W)
    n_views = len(images)
    t0 = time.perf_counter()
    for step in range(steps):
        vi = np.random.randint(n_views)
        ys = np.random.randint(0, H, batch)
        xs = np.random.randint(0, W, batch)
        target = torch.from_numpy(images[vi, ys, xs])
        o, d = rt.get_rays(c2ws[vi], H, W)
        rc, rf = rt.render_rays(torch.from_numpy(o[ys, xs]),
                                torch.from_numpy(d[ys, xs]), coarse, fine)
        loss = F.mse_loss(rc, target) + F.mse_loss(rf, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(allp, 1.0)
        opt.step()
        sched.step()
    dt = time.perf_counter() - t0
    coarse.eval(); fine.eval()
    ps = []
    with torch.no_grad():
        for th in (2 * np.pi * f for f in (1 / 80, 27 / 80, 53 / 80)):
            c2w = rt.look_at(np.array([3.2 * np.cos(th), 1.2, 3.2 * np.sin(th)]),
                             np.zeros(3))
            gt = rt.ray_trace(c2w, 100, 100)
            o, d = rt.get_rays(c2w, 100, 100)
            chunks = []
            for s in range(0, 10000, 4096):
                _, rgb_f = rt.render_rays(
                    torch.from_numpy(o.reshape(-1, 3)[s:s + 4096]),
                    torch.from_numpy(d.reshape(-1, 3)[s:s + 4096]),
                    coarse, fine, perturb=False)
                chunks.append(rgb_f)
            pred = torch.cat(chunks).numpy().reshape(100, 100, 3)
            ps.append(-10 * np.log10(max(np.mean((gt - pred) ** 2), 1e-12)))
    return params, macs, dt, float(np.mean(ps))


def main():
    print(f"{'变体':<14} {'参数k':>7} {'真实MAC/点':>11} {'训练秒':>8} {'新视角PSNR':>10}")
    for kind in ('loop3_concat', 'loop3_gated', 'plain3', 'plain7', 'plain9'):
        params, macs, dt, psnr = run(kind)
        print(f'{kind:<14} {params/1e3:>7.1f} {macs:>11,} {dt:>8.1f} '
              f'{psnr:>9.2f}dB', flush=True)


if __name__ == '__main__':
    main()
