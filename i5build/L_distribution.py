"""What L does the training actually sample?

The training uses --loop_random with --loop_L 1 --loop_L_max 3, so each step
draws L from sample_L(). If the mean is ~2, then pinning L=1 would cut the
per-step FLOPs by ~29% (measured: L=1 3.351 s, L=2 4.738 s -> 1.41x). If the
mean is ~1, there is nothing to win and the loop is already nearly free.

This asks sample_L() directly instead of instrumenting a training run: the
distribution is a property of the sampler, so 20000 draws give the exact answer
in seconds, with no val overhead, no checkpoints and no clock noise.

Class is located by introspection so a rename does not break this probe.
"""
import inspect
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

import loop_transformer as LT  # noqa: E402

# locate whichever class exposes sample_L
cls = None
for name, obj in vars(LT).items():
    if inspect.isclass(obj) and hasattr(obj, 'sample_L'):
        cls = obj
        break
if cls is None:
    print('没找到带 sample_L 的类；模块里的类：',
          [n for n, o in vars(LT).items() if inspect.isclass(o)])
    sys.exit(1)
print('类:', cls.__name__)
print('sample_L 源码:')
for ln in inspect.getsource(cls.sample_L).splitlines():
    print('   ', ln)

# build it the way train_ar_v2.py does for the run we care about:
#   --loop_start 4 --loop_end 7 --loop_L 1 --loop_L_max 3 --loop_random
sig = inspect.signature(cls.__init__)
kw = {}
for p in sig.parameters.values():
    if p.name == 'self':
        continue
    if p.name in ('loop_start',):
        kw[p.name] = 4
    elif p.name == 'loop_end':
        kw[p.name] = 7
    elif p.name == 'loop_L':
        kw[p.name] = 1
    elif p.name == 'loop_L_max':
        kw[p.name] = 3
    elif p.name == 'loop_random':
        kw[p.name] = True
    elif p.default is not inspect.Parameter.empty:
        pass          # leave other defaults alone
print('\n构造参数:', {k: v for k, v in kw.items()})
try:
    obj = cls(**kw)
except Exception as e:
    print('构造失败: %s' % str(e)[:120])
    print('参数签名:', sig)
    sys.exit(1)

N = 20000
hits = {}
for _ in range(N):
    L = obj.sample_L()
    hits[L] = hits.get(L, 0) + 1

print('\n=== sample_L() 的分布（%d 次抽样）===' % N)
tot = 0.0
for L in sorted(hits):
    p = hits[L] / N
    tot += L * p
    npass = 10 - 4 + 4 * L          # n_layer=10, loop 覆盖 [4..7] 共 4 层
    print('  L=%d  %6.2f%%   ⇒ 该步 %2d 个 block pass' % (L, p * 100, npass))
print('\n  平均 L = %.3f' % tot)
mean_pass = 10 - 4 + 4 * tot
print('  平均 pass 数 = %.2f  (L=1 时是 10)' % mean_pass)
print('  ⇒ 平均 FLOPs 相对 L=1 = %.2fx' % (mean_pass / 10.0))
if mean_pass / 10.0 > 1.05:
    print('  ⇒ 固定 L=1 可省约 %.0f%% 的每步 FLOPs ✓✓（远超 20%% 的分辨力下限）'
          % ((1 - 10.0 / mean_pass) * 100))
else:
    print('  ⇒ 循环几乎不额外花钱，固定 L=1 无收益 ✗（§7.12 的 97%% 贴顶成立）')
