"""Add the exported 4-bit entry point and drop the now-redundant header.

Why the entry point goes next to the 8-bit one rather than in a header: every
helper it needs (OptParams, opt_update_element, opt_update_p, opt_load/opt_store,
kOptBlockSize, the 4-bit kernel itself) lives in cpu_ops.cpp's anonymous
namespace, and the exported entry point must sit OUTSIDE that namespace -- same
arrangement as optimizer_update_8bit_blockwise_cpu.
"""
import ast
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\cpu_ops.cpp'
s = open(P, encoding='utf-8').read()

if 'optimizer_update_4bit_blockwise_cpu' in s:
    print('  已经加过入口')
else:
    # anchor: the closing brace of the 8-bit entry point, then its trailing
    # blank line and the next comment banner.
    m = re.search(r'(void optimizer_update_8bit_blockwise_cpu\(.*?\n\}\n)', s, re.S)
    if not m:
        print('  ✗ 找不到 8-bit 入口函数')
        sys.exit(1)
    END = m.group(1)

    ENTRY = r'''

// =====================================================================
// 4-bit blockwise optimizer -- exported entry point
// ---------------------------------------------------------------------
// 与 8-bit 版同构；差别只在状态的存取宽度。调用方（Python 侧）负责：
//   · state1/state2 各 ceil(n/2) 字节（一字节两个 4-bit 码）
//   · absmax1/absmax2 每 kOptBlockSize 个元素一个 fp32
//   · qmap1/qmap2 用 opt4_fill_qmap 填好的 256 项表（只有前 16 项会被读）
// ⚠️ ademamix 的第三个状态接在 state1 之后，偏移 (n+1)/2 字节 —— 与 8-bit 版
//   （接在 n 字节之后）不同，因为打包后字节数减半。
// =====================================================================
void optimizer_update_4bit_blockwise_cpu(
    int optimizer_id, void* g, void* p, unsigned char* state1, unsigned char* state2, float beta1,
    float beta2, float beta3, float alpha, float eps, int step, float lr, const float* qmap1,
    const float* qmap2, float* absmax1, float* absmax2, float weight_decay, float gnorm_scale,
    bool skip_zeros, long long n, int dtype
) {
    if (n <= 0 || g == nullptr || p == nullptr || state1 == nullptr || qmap1 == nullptr ||
        absmax1 == nullptr)
        return;

    OptParams P;
    P.optimizer_id = optimizer_id;
    P.beta1 = beta1;
    P.beta2 = beta2;
    P.beta3 = beta3;
    P.alpha = alpha;
    P.eps = eps;
    P.lr = lr;
    P.weight_decay = weight_decay;
    P.gnorm_scale = gnorm_scale;
    P.step = step;
    P.skip_zeros = skip_zeros;
    P.correction1 = 1.0f - (float)std::pow((double)beta1, (double)step);
    P.correction2 = std::sqrt(1.0f - (float)std::pow((double)beta2, (double)step));
    P.step_size = -lr * P.correction2 / P.correction1;

    switch (dtype) {
    case 0:
        optimizer_4bit_blockwise_scalar<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 1:
        optimizer_4bit_blockwise_scalar<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 2:
        optimizer_4bit_blockwise_scalar<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    default:
        break;
    }
}
'''
    s = s.replace(END, END + ENTRY, 1)
    open(P, 'w', encoding='utf-8', newline='').write(s)
    print('  + 入口函数已加（跟在 8-bit 入口之后）')

# drop the standalone header -- its content now lives in cpu_ops.cpp
H = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\optimizer_4bit_blockwise.h'
if os.path.exists(H):
    os.remove(H)
    print('  - 删掉独立头文件（内容已并入 cpu_ops.cpp，那里才看得到匿名命名空间的 helper）')

# sanity: entry point must be outside the anonymous namespace. Count braces
# between the last '} // namespace' before it and the function itself.
i = s.find('void optimizer_update_4bit_blockwise_cpu')
j = s.rfind('} // namespace', 0, i)
k = s.rfind('\nnamespace {', 0, i)
print('  位置检查: 入口在最后一个 } // namespace 之后? %s'
      % ('是 ✓' if j < i else '否 ✗'))
print('  (最近一次 namespace { 在第 %d 字符，} // namespace 在第 %d 字符，入口在第 %d 字符)'
      % (k, j, i))
