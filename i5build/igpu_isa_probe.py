"""能不能拿到核显内核的 GCN 汇编？（决定"读汇编"这条路走不走得通）

三条可能的路：
  1. -save-temps  —— AMD OpenCL 的编译选项，直接把 .s / .ll / .bc 落到磁盘
  2. CL_PROGRAM_BINARIES —— 拿 ELF，但需要 llvm-objdump 才能反汇编（本机没装）
  3. Radeon GPU Analyzer (RGA) —— AMD 官方离线编译器，能给寄存器用量/占用率（未安装）
"""
import glob
import os
import sys

import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

SRC = r'''
#define TS 64
#define TK 8
__kernel void probe(__global const float* A, __global float* C) {
    __local float As[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    As[tx % TK][ty * 8 + tx % 8] = A[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < TK; k++) acc = fma(As[k][ty * 8 + tx % 8], 1.0001f, acc);
    C[get_global_id(0)] = acc;
}
'''

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_isa_dump')
os.makedirs(work, exist_ok=True)
os.chdir(work)
for f in glob.glob('*'):
    os.remove(f)

ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
p = cl.Program(ctx, SRC)
opts = ['-save-temps', '-cl-fast-relaxed-math']
try:
    p.build(options=opts)
    print('build ok with', opts)
except Exception as exc:                                    # noqa: BLE001
    print('build 失败：', str(exc)[:400])
    raise SystemExit(1)

files = sorted(glob.glob('*'))
print(f'\n-save-temps 落盘 {len(files)} 个文件：')
for f in files:
    print(f'   {f:<50} {os.path.getsize(f):>9,} B')

for pat in ('*.s', '*.isa', '*.ll'):
    for f in glob.glob(pat):
        print(f'\n===== {f} 前 30 行 =====')
        with open(f, encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                if i >= 30:
                    break
                print('  ' + line.rstrip())

try:
    bins = p.binaries
    print(f'\nCL_PROGRAM_BINARIES: {len(bins)} 个，大小 {[len(b) for b in bins]}，'
          f'头 4 字节 {bins[0][:4]}')
except Exception as exc:                                    # noqa: BLE001
    print('\n取 binaries 失败：', str(exc)[:200])
