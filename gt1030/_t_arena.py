# -*- coding: utf-8 -*-
"""鎺㈤拡 22锛氬崟鍧?1.4GB arena + 鍋忕Щ璇诲啓 鏄惁绋冲畾锛堜负 arena 鏂规鍋氬彲琛屾€у垽鏂級銆?""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
ARENA = 1400 << 20
arena = cl.Buffer(ctx, mf.READ_WRITE, ARENA)
print('arena %.2f GB allocated' % (ARENA / 1e9))
rng = np.random.default_rng(0)
bad = 0
# 妯℃嫙 400 涓瓙鍖洪棿锛堜笉鍚岀殑鍋忕Щ/澶у皬锛夛紝閫愪釜鍐欎笉鍚屽€煎啀鍥炶
slots = []
off = 0
for i in range(400):
    n = int(rng.choice([512, 1024, 2048, 8192, 65536, 1 << 20, 4 << 20]))
    if off + n * 2 > ARENA:
        break
    slots.append((off, n))
    off += n * 2
print('%d 涓瓙鍖洪棿, 鍏?%.1f MB' % (len(slots), off / 1e6))
refs = []
for i, (o, n) in enumerate(slots):
    v = (rng.standard_normal(n) * 0.03).astype(np.float16)
    cl.enqueue_copy(q, arena, v, dst_offset=o)
    q.finish()
    refs.append((o, n, v))
bad = 0
for i, (o, n, v) in enumerate(refs):
    out = np.empty(n, np.float16)
    cl.enqueue_copy(q, out, arena, src_dst_offset=o)
    q.finish()
    if not np.array_equal(out, v):
        bad += 1
        if bad <= 3:
            print('  绗?%d 涓尯闂?(off=%d, n=%d) 涓嶄竴鑷? out[:3]=%s v[:3]=%s'
                  % (i, o, n, out[:3], v[:3]))
print('鍥炶涓嶄竴鑷? %d/%d' % (bad, len(refs)))

