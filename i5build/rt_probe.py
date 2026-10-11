"""复现豆包那份神经光追探针，并回答两个它没回答的问题。

他们的结论（2 核容器）：推理 97% 花在 fine 网络（PE + 循环 3 次），每 3000 光线 1532 ms。
但 1532 ms 里到底有多少是真的在算、多少是在等调度，那份剖面图看不出来。这里直接量：

  A. 每点 MLP 求值的吞吐（点/秒）随 batch 与线程数的变化。
     一次渲染的点数 = 光线数 × 56（coarse 24 + fine 32），所以点/秒决定了每张图的成本。

  B. 循环深度在"神经渲染"这个负载上到底赚不赚。
     他们的 RecurrentFineNetwork 每个点 119,040 MAC：
         3 × (182×128 + 128×128) = 3 × 39,680
     而一个普通的 7 层 MLP 只要 23,296 + 6×16,384 = 121,600 MAC 就能达到同样的算力开销，
     参数还更多——但神经渲染这里参数根本不是约束（169k 参数 vs 300k 参数都无所谓）。
     所以问题是：同样的墙钟时间，把算力给"循环"还是给"更多独立层"，哪个 PSNR 高？
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\doub')
import neural_rt_demo as rt  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def macs_per_point(model):
    """每点的乘加数（只看 Linear）"""
    tot = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            tot += m.in_features * m.out_features
    return tot


class PlainFine(nn.Module):
    """对照组：不用循环，直接把深度做成独立层。in_ch=54。"""

    def __init__(self, hidden=128, in_ch=54, n_layer=7):
        super().__init__()
        layers = [nn.Linear(in_ch, hidden), nn.ReLU()]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 4)

    def forward(self, p, d):
        x = torch.cat([rt.positional_encoding(p, L=8), d], -1)
        out = self.head(self.body(x))
        return torch.sigmoid(out[..., :3]), F.relu(out[..., 3])


def throughput(model, n_points, threads, reps=3):
    torch.set_num_threads(threads)
    model.eval()
    p = torch.randn(1, n_points, 3)
    d = F.normalize(torch.randn(1, n_points, 3), dim=-1)
    with torch.no_grad():
        model(p, d)                                    # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            model(p, d)
        dt = (time.perf_counter() - t0) / reps
    m = macs_per_point(model)
    return dt, n_points / dt, 2 * m * n_points / dt / 1e9      # 点/秒, GFLOPS


def train_variant(kind, steps, threads, seed=42, H=64, W=64, batch=512):
    """在同一个红球场景上短训，比较 PSNR 与墙钟。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    coarse = rt.CoarseNetwork()
    fine = (rt.RecurrentFineNetwork(loops=3) if kind == 'loop3'
            else PlainFine(n_layer=7 if kind == 'plain7' else 3))
    params = (sum(p.numel() for p in coarse.parameters())
              + sum(p.numel() for p in fine.parameters()))
    macs = macs_per_point(coarse) + macs_per_point(fine)
    opt = torch.optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=5e-4)
    c2ws, images = rt.make_dataset(n_views=40, H=H, W=W)
    n_views = len(images)
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if step == int(steps * 0.66):
            for g in opt.param_groups:
                g['lr'] = 1e-4
        vi = np.random.randint(n_views)
        ys = np.random.randint(0, H, batch)
        xs = np.random.randint(0, W, batch)
        target = torch.from_numpy(images[vi, ys, xs])
        o, d = rt.get_rays(c2ws[vi], H, W)
        origin = torch.from_numpy(o[ys, xs])
        dirs = torch.from_numpy(d[ys, xs])
        rc, rf = rt.render_rays(origin, dirs, coarse, fine)
        loss = F.mse_loss(rc, target) + F.mse_loss(rf, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(coarse.parameters()) + list(fine.parameters()), 1.0)
        opt.step()
    dt = time.perf_counter() - t0
    # 新视角 PSNR
    coarse.eval(); fine.eval()
    thetas = [2 * np.pi * f for f in (1 / 80, 27 / 80, 53 / 80)]
    ps = []
    with torch.no_grad():
        for th in thetas:
            cam = np.array([3.2 * np.cos(th), 1.2, 3.2 * np.sin(th)])
            c2w = rt.look_at(cam, np.zeros(3))
            gt = rt.ray_trace(c2w, 100, 100)
            o, d = rt.get_rays(c2w, 100, 100)
            chunks = []
            for s in range(0, 100 * 100, 4096):
                _, rgb_f = rt.render_rays(torch.from_numpy(o.reshape(-1, 3)[s:s + 4096]),
                                          torch.from_numpy(d.reshape(-1, 3)[s:s + 4096]),
                                          coarse, fine, perturb=False)
                chunks.append(rgb_f)
            pred = torch.cat(chunks).numpy().reshape(100, 100, 3)
            ps.append(-10 * np.log10(np.mean((gt - pred) ** 2)))
    return dict(kind=kind, params=params, macs_per_point=macs, seconds=dt,
                psnr=float(np.mean(ps)))


def main():
    print('=' * 78)
    print('A. 每点吞吐：batch × 线程（他们推理用的点数是 光线×56）')
    print('=' * 78)
    fine = rt.RecurrentFineNetwork(loops=3)
    print(f'   RecurrentFineNetwork(loops=3)  参数 '
          f'{sum(p.numel() for p in fine.parameters())/1e3:.1f}k  '
          f'每点 {macs_per_point(fine):,} MAC')
    print(f"{'rays':>7} {'points':>9} {'threads':>8} {'ms':>9} {'points/s':>12} "
          f"{'GFLOPS':>8} {'512² 预计':>11}")
    for rays in (512, 3000, 20000, 60000):
        pts = rays * 56
        for th in (2, 6):
            dt, pps, gf = throughput(fine, pts, th)
            # 512² = 262144 光线 × 56 点
            secs = 262144 * 56 / pps
            print(f'{rays:>7} {pts:>9} {th:>8} {dt*1e3:>9.1f} {pps:>12.0f} '
                  f'{gf:>8.2f} {secs:>10.1f}s', flush=True)

    print()
    print('=' * 78)
    print('B. 循环 vs 独立层：同一场景、同步数、同线程')
    print('=' * 78)
    print(f"{'变体':<12} {'参数k':>8} {'每点MAC':>10} {'训练秒':>9} {'新视角PSNR':>11}")
    for kind in ('loop3', 'plain3', 'plain7'):
        r = train_variant(kind, steps=800, threads=6)
        print(f"{r['kind']:<12} {r['params']/1e3:>8.1f} {r['macs_per_point']:>10,} "
              f"{r['seconds']:>9.1f} {r['psnr']:>10.2f}dB", flush=True)


if __name__ == '__main__':
    main()
