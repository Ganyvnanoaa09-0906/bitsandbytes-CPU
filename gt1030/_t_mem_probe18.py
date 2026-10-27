# -*- coding: utf-8 -*-
"""探针 18：在整机构建过程中，每分配一个 buffer 就做一次 1KB 健康自检，
            找出『队列变哑』到底发生在哪一个分配/拷贝上。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'


class Dead(Exception):
    pass


def main():
    dev = MM.Dev()
    q, ctx, mf = dev.q, dev.ctx, dev.mf
    probe_buf = cl.Buffer(ctx, mf.READ_WRITE, 2048)
    state = {'n': 0, 'bytes': 0, 'log': []}

    def probe(where):
        a = np.full(1024, 5.5, np.float16)
        try:
            cl.enqueue_copy(q, probe_buf, a, is_blocking=True)
            out = np.empty(1024, np.float16)
            cl.enqueue_copy(q, out, probe_buf)
            q.finish()
        except Exception as e:
            print('EXC at %s: %r' % (where, e))
            return False
        ok = float(out[0]) == 5.5
        if not ok:
            print('!!! 队列变哑：%s 之后（第 %d 个 buffer，累计 %.1f MB）读回 %.3f'
                  % (where, state['n'], state['bytes'] / 1e6, float(out[0])))
        return ok

    orig_buf = MM.Dev.buf
    orig_up = MM.Dev.up

    def buf(self, nb):
        b = orig_buf(self, nb)
        state['n'] += 1
        state['bytes'] += nb
        if not probe('buf(%d bytes) #%d' % (nb, state['n'])):
            raise Dead
        state['log'].append(('raw', nb))
        return b

    def up(self, a):
        a = np.ascontiguousarray(a)
        b = orig_up(self, a)
        state['n'] += 1
        state['bytes'] += a.nbytes
        if not probe('up%s #%d' % (a.shape, state['n'])):
            raise Dead
        state['log'].append(('up', a.nbytes, a.shape))
        return b

    MM.Dev.buf = buf
    MM.Dev.up = up
    try:
        model = MM.V2ModelDev(CKPT, dev=dev)
        print('构建完成：%d 个 buffer，%.1f MB，队列正常' % (state['n'], state['bytes'] / 1e6))
    except Dead:
        tail = state['log'][-6:]
        print('最后 6 个分配：')
        for t in tail:
            print('   ', t)


main()
