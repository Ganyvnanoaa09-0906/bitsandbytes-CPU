"""A/B harness that can actually resolve differences on THIS machine.

Why it exists: today's shape experiments were worthless because they used
"run A, then run B, compare single values" on a machine whose GEMM throughput
has a ~10% coefficient of variation even on AC with the Ultimate Performance
plan (measured: min 208.5 / median 231.1 / max 293.8 on one shape). Any effect
below ~20% is invisible to that method -- and every effect I chased today was
below it (qkv fusing +6%, K-trend +/-20%).

The rules this encodes, each learned the hard way:
  1. INTERLEAVE the variants (A/B/A/B...). Sequential blocks let a slow clock
     period land entirely on one variant.
  2. Report DISTRIBUTIONS, never a single number.
  3. Keep a REFERENCE workload in the same run. Reporting the variant/reference
     ratio removes common-mode clock drift -- the same trick that fixed the
     iGPU measurements (CV 32.8% -> 2.9%).
  4. Record the POWER SOURCE. Battery vs AC changes the whole power budget;
     this flipped between two measurements today.
  5. Judge with a PAIRED statistic, and say so when the difference is inside
     the noise instead of reporting a winner.

Usage: define variants as (name, callable) and a reference callable.
"""
import statistics as st
import sys
import time

import torch

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)


def power_source():
    try:
        import subprocess
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Battery).BatteryStatus'],
                           capture_output=True, text=True, timeout=15)
        return {'1': '电池(放电) !!', '2': '交流电', '6': '充电中'}.get(
            r.stdout.strip(), r.stdout.strip() or '?')
    except Exception:
        return '?'


def timeit(fn, reps=3):
    fn()                                  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def compare(variants, reference, rounds=9, reps=3):
    """variants: list of (name, fn). reference: fn used for normalisation.
    Returns per-variant ratio distributions and a paired verdict."""
    print('电源: %s | 线程: %d | 交替 %d 轮\n'
          % (power_source(), torch.get_num_threads(), rounds))
    acc = {name: [] for name, _ in variants}
    refs = []
    for r in range(rounds):
        t_ref = timeit(reference, reps)
        refs.append(t_ref)
        line = '  轮 %2d  参照 %7.2f ms  ' % (r + 1, t_ref * 1000)
        for name, fn in variants:
            t = timeit(fn, reps)
            acc[name].append(t / t_ref)        # ratio to the reference
            line += '| %s %.4f  ' % (name[:12], t / t_ref)
        print(line)

    print('\n%-18s %9s %9s %9s %8s' % ('变体', '中位比值', 'min', 'max', 'CV'))
    print('-' * 58)
    for name, _ in variants:
        v = acc[name]
        print('%-18s %9.4f %9.4f %9.4f %7.1f%%'
              % (name, st.median(v), min(v), max(v),
                 st.pstdev(v) / st.mean(v) * 100))

    # paired verdict: compare each variant against the FIRST one, per round
    if len(variants) >= 2:
        base_name = variants[0][0]
        b = acc[base_name]
        print('\n配对比较（每轮内相除，看是否稳定地偏离 1.0）：')
        for name, _ in variants[1:]:
            v = acc[name]
            ratios = [x / y for x, y in zip(v, b)]
            med = st.median(ratios)
            spread = st.pstdev(ratios) / st.mean(ratios) * 100
            verdict = ('差异可信 ✓' if abs(med - 1) > 2 * st.pstdev(ratios)
                       and spread < 15 else '在噪声内 ✗ 不可判定')
            print('  %-16s 比值中位 %.4f  (离散 %.1f%%)  ⇒ %s'
                  % (name, med, spread, verdict))
        print('\n  判读规则：只有当"比值中位偏离 1.0"超过其自身 2 倍标准差时，')
        print('            才认为两个变体真的有差别；否则一律记为不可判定。')
