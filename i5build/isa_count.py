"""读 ISA 反汇编，数指令 —— 不再猜"2/3 的时间去哪了"。

对象：_vgreg/fp32 那次 -save-temps 落下的 .isa（156 KB，含 gemm32 的反汇编）。
数四类：
  · FMA（v_fma_f32 / v_mac_f32）          ← 有效功
  · LDS 读（ds_read*）                     ← 访存
  · 地址/整数运算（v_add/v_lshl/v_mul/v_and/...）  ← 纯开销
  · 同步与等待（s_waitcnt/s_barrier/s_cbranch）    ← 延迟暴露的位置
另外找 scratch（spill）痕迹。
"""
import collections
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

CAND = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_vgreg', '*', '*.isa'))
CAND += glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '_clean', '*', '*.isa'))
if not CAND:
    print('没有 .isa 文件，先跑一次带 -save-temps 的构建')
    raise SystemExit(1)

PATH = max(CAND, key=os.path.getsize)
print(f'读 {PATH}  ({os.path.getsize(PATH)/1024:.0f} KB)\n')
txt = open(PATH, encoding='utf-8', errors='replace').read()

# 按 kernel code object 切
blocks = {}
for chunk in txt.split('AMD Kernel Code for &'):
    name = chunk.split(':')[0].strip()
    if name.startswith('__OpenCL_'):
        blocks[name] = chunk
print('kernel code object:', list(blocks))

INS = re.compile(r'^\s+([a-z][a-z0-9_]*)\s.*//\s*[0-9a-f]+:', re.M)

VALU = ('v_add', 'v_sub', 'v_mul', 'v_lshl', 'v_lshr', 'v_and', 'v_or', 'v_xor',
        'v_mov', 'v_cmp', 'v_cndmask', 'v_min', 'v_max', 'v_cvt', 'v_lerp')
SALU = ('s_add', 's_sub', 's_mul', 's_lshl', 's_lshr', 's_and', 's_or', 's_xor',
        's_mov', 's_cmp', 's_cselect', 's_pack', 's_bfe', 's_abs')

for name, body in blocks.items():
    dis = body.split('Disassembly:')
    if len(dis) < 2:
        print(f'\n{name}: 没有 Disassembly 段')
        continue
    ops = INS.findall(dis[1])
    if not ops:
        print(f'\n{name}: Disassembly 段是空的（{len(dis[1])} 字符）')
        continue
    hist = collections.Counter(ops)

    def bucket(pred):
        return sum(v for k, v in hist.items() if pred(k))

    fma = bucket(lambda k: k.startswith(('v_fma', 'v_mac', 'v_pk_fma')))
    lds = bucket(lambda k: k.startswith('ds_'))
    glb = bucket(lambda k: k.startswith(('buffer_load', 'flat_load', 'global_load')))
    valu = bucket(lambda k: k.startswith(VALU))
    salu = bucket(lambda k: k.startswith(SALU))
    sync = bucket(lambda k: k.startswith(('s_waitcnt', 's_barrier', 's_cbranch',
                                          's_branch', 's_setpc')))
    other = len(ops) - fma - lds - glb - valu - salu - sync
    print(f'\n=== {name} ===  反汇编总指令数 {len(ops)}')
    print(f'  FMA          {fma:>7}  ({fma/len(ops)*100:>5.1f}%)')
    print(f'  LDS 读       {lds:>7}  ({lds/len(ops)*100:>5.1f}%)')
    print(f'  全局读写     {glb:>7}  ({glb/len(ops)*100:>5.1f}%)')
    print(f'  向量整数/地址{valu:>7}  ({valu/len(ops)*100:>5.1f}%)')
    print(f'  标量         {salu:>7}  ({salu/len(ops)*100:>5.1f}%)')
    print(f'  同步/等待/分支{sync:>6}  ({sync/len(ops)*100:>5.1f}%)')
    print(f'  其它         {other:>7}  ({other/len(ops)*100:>5.1f}%)')
    print(f"  ⚠️ scratch/spill 迹象: v_scratch={hist.get('v_scratch', 0)} "
          f"buffer_store={sum(v for k, v in hist.items() if 'scratch' in k)}")
    print('  出现最多的 12 条：')
    for k, v in hist.most_common(12):
        print(f'     {k:<24} {v:>6}')
