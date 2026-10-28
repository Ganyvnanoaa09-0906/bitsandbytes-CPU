import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, time
v15=r"D:\work\textmodel\animatediff-motion-adapter-v1-5\diffusion_pytorch_model.safetensors"
frcnn=r"D:\work\textmodel\fsrcnn_x4_v2.pth"
start=time.time(); MAX=5*3600
while time.time()-start < MAX:
    done_v15 = os.path.exists(v15) and os.path.getsize(v15) >= 1500000000  # 1.5GB 够用(v1-5 1.56 可加载)
    done_frcnn = os.path.exists(frcnn) and os.path.getsize(frcnn) > 10000
    if done_v15 and done_frcnn:
        print("两任务完成, 关机", flush=True); break
    time.sleep(30)
else:
    print("超时关(防卡)", flush=True)
os.system(r"shutdown /s /t 0")
print("已触发关机", flush=True)
