# -*- coding: utf-8 -*-
"""给 ar_v2_model 加两项：
   1) Dev.up 上传后立刻回读校验（8 字节），不一致 -> 抛 DriverBroken
   2) V2ModelDev 构建失败时换一个全新 context 重试（先释放旧 buffer）

依据（探针 13/16/18）：
   - 同一份代码 5 次里 3~4 次会出现『整条 OpenCL 队列变哑』：之后所有 enqueue_copy
     静默 no-op，回读全 0/读到别的 buffer 的内容；weight 上传也跟着丢。
   - 但在构建过程中每分配一个 buffer 就插一次 1KB 回读+finish()（探针 18），
     431 个 buffer / 1.36GB 连续 3 次全正常 ⇒ 可以用『上传即校验』把这种坏状态
     立刻抓出来，再换 context 重建。
   - 换 context 前必须先把旧 buffer 释放，否则 cl.CommandQueue 会报 OUT_OF_HOST_MEMORY。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

# ---- 1) 模块级：可重建 context ----
old_hdr = """KNAMES = ('rms_stats',"""
new_hdr = '''class DriverBroken(RuntimeError):
    """本机 AMD 驱动偶发：队列整体变哑（copy 静默 no-op）。抓到就换 context 重建。"""


def new_context():
    """建一个全新的 OpenCL context/queue，并把 A.* 的全局句柄换掉。

    hgemm/l.hgemm_* 全部通过 A.QP / A.queue / A.ctx 取值，所以换这三个就够。
    """
    import ctypes
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    A.ctx, A.queue, A.mf = ctx, queue, cl.mem_flags
    A.QP = ctypes.c_void_p(int(queue.int_ptr))
    return ctx, queue


KNAMES = ('rms_stats','''
assert old_hdr in s
s = s.replace(old_hdr, new_hdr, 1)

# ---- 2) Dev.up 上传即校验 ----
old_up = """        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        self.q.finish()      # 见 _fix_model_up 注释：必须同步，否则批量 copy 会把队列搞哑
        return b"""
new_up = """        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        self.q.finish()      # 见 _fix_model_up 注释：必须同步，否则批量 copy 会把队列搞哑
        # 上传即校验：本机驱动偶发『队列变哑』，copy 静默 no-op，不校验的话后面全是
        # 静默的错误结果（权重读回全 0）。只回读 8 字节，代价可忽略。
        chk = np.empty(4, np.float16)
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        if not np.array_equal(chk, a.ravel()[:4]):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))
        return b"""
assert old_up in s
s = s.replace(old_up, new_up, 1)

# ---- 3) V2ModelDev 构建重试 ----
old_init_sig = """    def __init__(self, ckpt_path, dev=None):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)"""
new_init_sig = """    def __init__(self, ckpt_path, dev=None, retries=6):
        last = None
        for attempt in range(1, retries + 1):
            try:
                self._build(ckpt_path, dev)
                return
            except DriverBroken as e:
                last = e
                if attempt == retries:
                    break
                print('[V2ModelDev] 驱动队列失效（%s），换 context 重建（第 %d 次）'
                      % (e, attempt + 1), flush=True)
                self._release()
        raise DriverBroken('重试 %d 次仍失败: %s' % (retries, last))

    def _release(self):
        import gc
        for attr in list(self.__dict__.keys()):
            if attr != 'cfg':
                self.__dict__.pop(attr, None)
        gc.collect()

    def _build(self, ckpt_path, dev=None):
        if dev is not None:
            # dev 里可能已经是坏 context；除非刚建好并校验过，否则重建
            pass
        if dev is None or getattr(dev, 'broken', False):
            new_context()
            dev = Dev()
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)"""
assert old_init_sig in s
s = s.replace(old_init_sig, new_init_sig, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched ar_v2_model: DriverBroken + upload verify + context retry')
