# -*- coding: utf-8 -*-
"""selftest 加『context 重建也救不回就整个进程重启』的保护。

依据：本机驱动偶发整个进程的 OpenCL 队列从第一次上传就是哑的（upload verify 直接失败），
这种状态换 context 也救不回来（同一进程内 6 次重建都失败），只能重启进程。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model_selftest.py'
s = io.open(p, encoding='utf-8').read()

old = """if __name__ == '__main__':
    main()"""
new = """if __name__ == '__main__':
    import os
    _try = int(os.environ.get('AR_REEXEC', '0'))
    try:
        main()
    except MM.DriverBroken as e:
        if _try >= 5:
            print('!! 连续 %d 次进程级重试仍撞上驱动队列失效，放弃: %s' % (_try, e))
            raise
        print('[reexec] 驱动队列整体失效（%s），重启本进程重试（第 %d 次）'
              % (e, _try + 1), flush=True)
        os.environ['AR_REEXEC'] = str(_try + 1)
        os.execv(sys.executable, [sys.executable] + sys.argv)"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched selftest with reexec guard')
