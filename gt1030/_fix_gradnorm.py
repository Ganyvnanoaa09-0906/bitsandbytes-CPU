# -*- coding: utf-8 -*-
"""修 grad_norm：把 40 个权重梯度（59MB）下载到 host 再平方和，白白花 200ms。
改成设备端 sumsq 归约（每个参数只回传 256 个部分和 = 1KB）→ 预计 200ms → ~2ms。

顺带把 clip 的 scale_half 循环保持原样（它本来就在 device 上）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# 1) kernel
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
"""// 梯度平方和：每个 work-item 归约一段，输出 256 个部分和（host 侧再求和）
__kernel void sumsq_half(__global const half* g, __global float* out, int n) {
    int i = get_global_id(0), lsz = get_global_size(0);
    float acc = 0.f;
    for (int k = i; k < n; k += lsz) { float v = (float)g[k]; acc += v * v; }
    out[i] = acc;
}
__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'sgemm_nt64', 'sgemm_nn_bt')""",
              """          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half')""")

# 2) Dev 里加一个可复用的部分和 buffer + 便捷方法
s = s.replace("""        self._tick_buf = self._alloc(2048)""",
"""        self._tick_buf = self._alloc(2048)
        self._part = self._alloc(256 * 4)      # sumsq 的部分和""")

s = s.replace("""    def finish(self):
        self.q.finish()""",
"""    def finish(self):
        self.q.finish()

    def sumsq(self, buf, n):
        \"\"\"设备端求 half 数组的平方和（回传 256 个部分和，再在 host 求和）\"\"\"
        self.K['sumsq_half'](self.q, (256,), None, buf, self._part, np.int32(n))
        p = np.empty(256, np.float32)
        cl.enqueue_copy(self.q, p, self._part)
        self.q.finish()
        return float(p.sum())""")

# 3) Block.grad_norm / Block.opt_step 里的 grad_norm 走 sumsq
s = s.replace("""    def grad_norm(self):
        d = self.dev
        d.activate()
        tot = 0.0
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            p = self.W[k]
            g = d.down(p.grad, (p.n,), np.float16).astype(np.float32)
            tot += float((g * g).sum())
        for k, buf in self.dw.items():
            g = d.down(buf, (len(self.host[k]),), np.float32)
            tot += float((g * g).sum())
        return tot""",
"""    def grad_norm(self):
        d = self.dev
        d.activate()
        tot = 0.0
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            p = self.W[k]
            tot += d.sumsq(p.grad, p.n)
        for k, buf in self.dw.items():
            tot += d.sumsq_f32(buf, len(self.host[k]))
        return tot""")

# Dev: float32 版本的 sumsq
s = s.replace("""    def sumsq(self, buf, n):""",
"""    def sumsq_f32(self, buf, n):
        g = np.empty(n, np.float32)
        cl.enqueue_copy(self.q, g, buf)
        self.q.finish()
        return float((g * g).sum())

    def sumsq(self, buf, n):""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched grad_norm -> device sumsq')
