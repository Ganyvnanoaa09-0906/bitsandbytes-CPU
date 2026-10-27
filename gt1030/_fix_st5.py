# -*- coding: utf-8 -*-
"""selftest: 用进程级重启替代进程内重试（驱动在失败后会 OUT_OF_HOST_MEMORY，换 context 也救不回）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s = io.open(p, encoding='utf-8').read()

old = """    model = None
    for _att in range(6):
        try:
            model = MM.Model(args.ckpt)
            break
        except RuntimeError as e:
            print('[build] 第 %d 次构建失败（%r），换 context 重来' % (_att + 1, e), flush=True)
            import gc
            gc.collect()
    if model is None:
        raise RuntimeError('构建 6 次都失败')"""
new = """    model = MM.Model(args.ckpt)"""
assert old in s
s = s.replace(old, new)

old2 = """if __name__ == '__main__':
    main()"""
new2 = """if __name__ == '__main__':
    import os
    _n = int(os.environ.get('AR_REEXEC', '0'))
    try:
        main()
    except RuntimeError as e:
        msg = str(e)
        if ('upload verify' not in msg and 'OUT_OF_HOST_MEMORY' not in msg) or _n >= 8:
            raise
        print('[reexec] 构建撞上驱动问题（%s），重启进程重试（第 %d 次）' % (msg[:60], _n + 1),
              flush=True)
        os.environ['AR_REEXEC'] = str(_n + 1)
        os.execv(sys.executable, [sys.executable] + sys.argv)"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched reexec')
