# -*- coding: utf-8 -*-
"""ar_spike_analyze.py — 核显训练日志的 loss 尖峰诊断。

输入是 `gt1030/ar_v2_train.py` 的日志（每步 loss/gnorm/lr，含 OVERFLOW/SPIKE 行）。
输出：
  1) 每步 loss 的滚动 EMA(20) 与相对偏差；
  2) 尖峰计数：loss > EMA + 3×MAD（MAD = 偏差中位数绝对偏差）；
  3) 与 gnorm、lr、OVERFLOW 事件的对应关系；
  4) 冷却期检查：尖峰后 20 步的 EMA 是否回到原水平（真尖峰 vs 状态漂移）。

用法：
  python ar_spike_analyze.py --log a.log --log b.log
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

RE_STEP = re.compile(r"step (\d+)/(\d+) loss=([-\d.eE+]+) gnorm=([-\d.eE+inf]+) lr=([-\d.eE+]+)")
RE_BAD = re.compile(r"step (\d+)/(\d+) (OVERFLOW|SPIKE|SKIP|GUARD)")


def parse(path):
    loss, gn, lr, bad = [], [], [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = RE_STEP.search(ln)
            if m:
                loss.append(float(m.group(3)))
                g = m.group(4)
                gn.append(float(g) if g not in ("inf", "-inf") else np.inf)
                lr.append(float(m.group(5)))
                continue
            mb = RE_BAD.search(ln)
            if mb:
                bad[int(mb.group(1))] = mb.group(3)
    return dict(loss=np.array(loss), gnorm=np.array(gn), lr=np.array(lr), bad=bad)


def analyze(path, ema_n=20, k=3.0):
    d = parse(path)
    L = d["loss"]
    if len(L) < ema_n + 5:
        return dict(log=path, n=len(L), error="步数太少")
    ema = np.full(len(L), np.nan)
    dev = np.zeros(len(L))
    for i in range(len(L)):
        lo = max(0, i - ema_n)
        ref = L[lo:i]
        if len(ref) < ema_n:
            continue
        ema[i] = ref.mean()
        dev[i] = L[i] - ema[i]
    valid = np.where(np.isfinite(ema))[0]
    mad = np.median(np.abs(dev[valid] - np.median(dev[valid])))
    thr = np.median(dev[valid]) + k * max(mad, 1e-6)
    spikes = [int(i) for i in valid if dev[i] > thr]
    # 冷却：尖峰后 20 步 EMA 是否回落（用 ema 变化衡量）
    cooled = 0
    for i in spikes:
        j = min(i + 20, len(L) - 1)
        if np.isfinite(ema[j]) and np.isfinite(ema[i]) and ema[j] <= ema[i] + 0.05:
            cooled += 1
    gspike = [i for i in spikes if i < len(d["gnorm"]) and not np.isfinite(d["gnorm"][i])]
    res = dict(log=path.split("\\")[-1], n=int(len(L)),
               loss_mean=round(float(np.mean(L)), 4),
               loss_min=round(float(np.min(L)), 4), loss_max=round(float(np.max(L)), 4),
               loss_last20=round(float(np.mean(L[-20:])), 4),
               step_std=round(float(np.std(L)), 4),
               mad=round(float(mad), 4), thr=round(float(thr), 4),
               n_spikes=len(spikes), spike_rate=round(len(spikes) / len(L), 4),
               spike_steps=spikes[:20],
               spikes_with_inf_gnorm=len(gspike),
               cooled_after_20=round(cooled / max(1, len(spikes)), 3),
               bad_events=dict(list(sorted(d["bad"].items()))[:10]),
               bad_count=len(d["bad"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = []
    for p in a.log:
        r = analyze(p, k=a.k)
        out.append(r)
        if "error" in r:
            print(f"  {r['log']}: {r['error']}")
            continue
        print(f"  {r['log']}: n={r['n']} loss {r['loss_min']}–{r['loss_max']} "
              f"(mean {r['loss_mean']}, last20 {r['loss_last20']}, std {r['step_std']})")
        print(f"    尖峰 {r['n_spikes']} 次 (rate {r['spike_rate']:.3f}) | 其中 gnorm=inf {r['spikes_with_inf_gnorm']} 次 "
              f"| 20 步内回落 {r['cooled_after_20']:.0%} | 异常事件 {r['bad_count']} 次")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(out if len(out) > 1 else out[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
