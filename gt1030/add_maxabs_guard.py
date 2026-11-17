# -*- coding: utf-8 -*-
"""add_maxabs_guard.py — 给核显训练加"梯度最大值"守卫（在 fp16 饱和之前就拦住）。

动机（本次调查结论）：
  - 权重梯度以 **fp16** 存储，且被 **×CE_SCALE(=1024)** 放大；
  - 于是单个梯度元素一旦真值 ≥ 64，存储值就撞上 fp16 上限 65504 → inf → 下一步
    AdamW 把 inf 带进 fp32 master → 永久 NaN（历史 run 在 13334 步就是这样死的）；
  - 现有守卫比较的是 `grad_norm()`（已 ÷1024），所以阈值 100 实际等于真范数 102400，
    等于"救护车在着火之后才出车"。

本补丁新增 kernel `maxabs_half` 与 `Model.grad_maxabs()`，并在训练器里：
  当 `AR_GUARD_MAXABS`（如 16384）被超过时跳过该步 —— 此时还剩 4× 余量，weights 不会被污染。
"""
import io
import os

REPO = r"D:\work\bitsandbytes-CPU"
MODEL = os.path.join(REPO, "gt1030", "ar_v2_multi.py")
TRAIN = os.path.join(REPO, "gt1030", "ar_v2_train.py")

m = io.open(MODEL, encoding="utf-8").read()
if "maxabs_half" in m:
    print("[skip] kernel 已存在")
else:
    # 1) kernel 源码：紧跟在 sumsq_half 之后
    anchor = """__kernel void sumsq_half(__global const half* g, __global float* out, int n) {
    int i = get_global_id(0), lsz = get_global_size(0);
    float acc = 0.f;
    for (int k = i; k < n; k += lsz) { float v = (float)g[k]; acc += v * v; }
    out[i] = acc;
}"""
    add = anchor + """

// 梯度绝对值最大值（每 work-item 归约一段，输出 256 个局部最大，host 再取最大）
__kernel void maxabs_half(__global const half* g, __global float* out, int n) {
    int i = get_global_id(0), lsz = get_global_size(0);
    float mx = 0.f;
    for (int k = i; k < n; k += lsz) {
        float v = fabs((float)g[k]);
        if (v != v) { out[i] = INFINITY; return; }   // NaN 直接上报
        if (v > mx) mx = v;
    }
    out[i] = mx;
}"""
    assert anchor in m, "sumsq_half 锚点没命中"
    m = m.replace(anchor, add, 1)
    # 2) 注册名字
    assert "'sumsq_half'," in m
    m = m.replace("'sumsq_half',", "'sumsq_half', 'maxabs_half',", 1)
    # 3) Dev.maxabs_half 方法（放在 sumsq 后面）
    anchor2 = """    def sumsq(self, buf, n):"""
    meth = """    def maxabs_half(self, buf, n):
        \"\"\"fp16 梯度缓冲的 |g| 最大值（256 路归约，host 取最大）。\"\"\"
        if not hasattr(self, '_mx_pool'):
            self._mx_pool = [self._alloc(256 * 4) for _ in range(4)]
        while len(self._mx_pool) < 2:
            self._mx_pool.append(self._alloc(256 * 4))
        self.K['maxabs_half'](self.q, (256,), None, buf, self._mx_pool[0], np.int32(n))
        o = np.empty(256, np.float32)
        cl.enqueue_copy(self.q, o, self._mx_pool[0])
        self.q.finish()
        return float(np.max(o))

    def sumsq(self, buf, n):"""
    assert anchor2 in m
    m = m.replace(anchor2, meth, 1)
    # 4) Model.grad_maxabs
    anchor3 = """    def clip_grads(self, f):"""
    add3 = """    def grad_maxabs(self):
        \"\"\"所有 fp16 权重梯度缓冲里的 |g| 最大值（用来在饱和前拦截）。\"\"\"
        mx = 0.0
        for blk in self.blocks:
            d = blk.dev
            d.activate()
            for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                p = blk.W[k]
                v = d.maxabs_half(p.grad, p.n)
                if v > mx:
                    mx = v
        h = self.head
        h.dev.activate()
        v = h.dev.maxabs_half(h.W.grad, h.W.n)
        return max(mx, v)

    def clip_grads(self, f):"""
    assert anchor3 in m
    m = m.replace(anchor3, add3, 1)
    io.open(MODEL, "w", encoding="utf-8", newline="\n").write(m)
    print("[ok] ar_v2_multi：新增 maxabs_half kernel + Dev.maxabs_half + Model.grad_maxabs")

t = io.open(TRAIN, encoding="utf-8").read()
if "AR_GUARD_MAXABS" in t:
    print("[skip] trainer 已改")
else:
    OLD = """        model.backward(tokens)
        if args.spike:"""
    NEW = """        model.backward(tokens)
        _gmax = os.environ.get('AR_GUARD_MAXABS')
        if _gmax:
            _mx = model.grad_maxabs()
            if not np.isfinite(_mx) or _mx > float(_gmax):
                n_spike += 1
                msgx = '[igpu] step %d/%d GUARD maxabs=%.4g (>%s) → 跳过该步' % (
                    st, args.steps, _mx, _gmax)
                print(msgx, flush=True)
                if logf:
                    logf.write(msgx + "\\n"); logf.flush()
                continue
        if args.spike:"""
    assert OLD in t, "trainer backward 锚点没命中"
    t = t.replace(OLD, NEW, 1)
    io.open(TRAIN, "w", encoding="utf-8", newline="\n").write(t)
    print("[ok] ar_v2_train：AR_GUARD_MAXABS 生效")
