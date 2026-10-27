# -*- coding: utf-8 -*-
"""利用因果结构跳掉上三角：attention 的 S/dP/O/dQ 都只需要下三角部分。

依据：`softmax_causal` 只读 `k < row+1`、并把 `k >= row+1` 写成 0；`softmax_bwd_causal` 同理。
所以：
  · S、dP 的 tile：`tj > ti` 的完全在严格上三角 → 直接 return（省 ~50%）
  · O = P·Vb、dQ = dS·Kn：A 的第 r 行块只需要 k ≤ r 的列块（P/dS 上三角为 0）
    → k 循环只跑到 `mb + MS` 而不是 T（平均再省一半）
数学上被跳过的项恒等于 0，所以结果不变。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# 1) k=64 kernel（S/dP）：跳上三角 tile
s = s.replace("""    int mb = ti * 128, nb = tj * 64;
    for (int t = tid; t < 128 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);""",
"""    int mb = ti * 128, nb = tj * 64;
    // 因果：整块在严格上三角就不用算（softmax 只读下三角，上三角它自己写 0）
    if (nb > mb + 127) return;
    for (int t = tid; t < 128 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);""")

# 2) nn batched kernel（O/dQ）：k 循环截到 mb+MS
s = s.replace("""    for (int k0 = 0; k0 < T; k0 += 64) {
        for (int t = tid; t < 128 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Ap + (long)(mb + r) * T + k0 + c4);""",
"""    int kend = (mb + 128 < T) ? (mb + 128) : T;   // 因果：P/dS 上三角为 0，k 不必跑满
    for (int k0 = 0; k0 < kend; k0 += 64) {
        for (int t = tid; t < 128 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Ap + (long)(mb + r) * T + k0 + c4);""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched causal skip (S/dP tiles, O/dQ k-range)')
