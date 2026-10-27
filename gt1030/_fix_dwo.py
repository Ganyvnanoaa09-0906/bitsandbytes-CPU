# -*- coding: utf-8 -*-
import io
p = r'D:\work\bitsandbytes-CPU\gt1030\_t_dwo.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("dout = torch.autograd.grad(loss, xb)[0].view(M, MD)",
              "dout = torch.autograd.grad(loss, xb, retain_graph=True)[0].view(M, MD)")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
