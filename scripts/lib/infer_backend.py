# -*- coding: utf-8 -*-
"""infer_backend.py — 把推理（验证/采样）放到核显上跑，训练留在 CPU。

背景（2026-09-23 实机实测）：
  自家 UNet 25.98M 在 DirectML 上：
      前向 batch=1 快 1.55x，batch=8 快 2.41x，端到端 DDIM 20 NFE 快 1.72x
      数值误差 2.5e-06（fp32 噪声级）
  但【训练】不能用核显：完整训练步 DML 比 CPU 慢 1.45x，
  因为 aten::native_group_norm_backward 回退 CPU。
  => 所以正确用法是：CPU 训练 + 核显验证/采样。

⚠️⚠️ 一个必须记住的坑 ⚠️⚠️
  torch 的 Module.to() 是【原地】修改。直接对训练模型调用 model.to(dml)
  会把它换成 DML 模型，之后再用 CPU 张量喂它就会撞：
      RuntimeError: tensor.device().type() == at::DeviceType::PrivateUse1
      INTERNAL ASSERT FAILED at ".../DMLTensor.cpp":31
      unbox expects Dml at::Tensor as inputs
  这不是 DML 坏了（历史文档曾误记为"DMLTensor.cpp 插件内部 bug"），
  是设备不匹配的断言。本模块一律用 deepcopy 规避。

设备内存约束（实测）：单块分配上限 2048 MB，累计上限 ≈7168 MB，
且流式加载真实成本 ≈ 2.1x 权重字节。所以本模块只适合 ≤ ~1 GB 的模型。

可移植性：全部走 torch 的设备抽象，没有写死任何 4500U 专属值；
换到别的机器上，torch_directml 在就用、不在就自动退回 CPU。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import copy

import torch


def dml_available():
    """torch_directml 是否可用。任何异常都当作不可用。"""
    try:
        import torch_directml
        if not torch_directml.is_available():
            return False
        # 真正建一次设备对象，光 is_available 不够
        torch_directml.device()
        return True
    except Exception:
        return False


def resolve_device(spec):
    """把用户给的设备字符串解析成 torch.device。

    spec: 'cpu' | 'dml' | 'auto'
      auto -> 有 torch_directml 就用核显，否则 CPU
    返回 (device, 说明字符串)
    """
    spec = (spec or 'cpu').lower()
    if spec == 'cpu':
        return torch.device('cpu'), 'cpu（显式指定）'
    if spec == 'auto':
        if dml_available():
            import torch_directml
            return torch_directml.device(), 'dml（auto 探测到核显）'
        return torch.device('cpu'), 'cpu（auto 未探测到 torch_directml）'
    if spec == 'dml':
        if not dml_available():
            raise RuntimeError("指定了 --infer-device dml，但 torch_directml 不可用")
        import torch_directml
        return torch_directml.device(), 'dml（显式指定）'
    raise ValueError("未知设备 %r，应为 cpu/dml/auto" % spec)


class InferBackend:
    """持有一份模型的推理副本，与 CPU 训练副本按需同步。

    用法：
        be = InferBackend(model, 'dml')          # 建副本
        ...
        be.sync(model)                            # 每次验证前同步权重
        out = be.forward(xt, t)                   # 在推理设备上跑，返回 CPU 张量
    """

    def __init__(self, cpu_model, device_spec='cpu', verbose=True):
        self.device, self.desc = resolve_device(device_spec)
        self.on_gpu = self.device.type != 'cpu'
        self._n_sync = 0
        # 模型是否支持文本条件。优先看 cond_dim 属性（LoopedDiT 显式声明），
        # 退化到签名探测，这样 UNet2DModel 等其它模型也能照常工作。
        self.has_cond = getattr(cpu_model, 'cond_dim', 0) > 0
        if not self.has_cond:
            try:
                import inspect
                self.has_cond = 'c' in inspect.signature(cpu_model.forward).parameters
            except Exception:
                pass
        if self.on_gpu:
            # deepcopy：绝不能原地动训练模型（见文件头警告）
            self._model = copy.deepcopy(cpu_model).to(self.device)
            self._model.eval()
            if verbose:
                np_ = sum(p.numel() for p in self._model.parameters())
                print('[infer] 推理设备=%s  模型副本 %.2fM 参数（deepcopy，未污染训练模型）'
                      % (self.desc, np_ / 1e6), flush=True)
        else:
            # 同设备：直接用训练模型本身，不复制、不同步、永远最新
            self._model = cpu_model
            if verbose:
                print('[infer] 推理设备=%s（与训练同设备，无额外副本）' % self.desc, flush=True)

    # -- 权重同步 ---------------------------------------------------------
    @torch.no_grad()
    def sync(self, cpu_model):
        """把 CPU 训练模型的最新权重灌进推理副本。"""
        if not self.on_gpu:
            return
        sd = cpu_model.state_dict()
        tgt = self._model.state_dict()
        for k, v in sd.items():
            if k in tgt:
                tgt[k].copy_(v.to(tgt[k].dtype))
        self._n_sync += 1

    # -- 前向 -------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x, t, c=None):
        """跑一次前向，返回【CPU 上的】输出张量。

        兼容两种返回：UNet2DModel 返回对象（取 .sample），LoopedDiT 返回张量。
        c 为文本条件（仅 cond_dim>0 的模型有意义）；None 表示无条件。
        """
        if not self.on_gpu:
            return self._model_forward(x, t, c)
        xd = x.to(self.device)
        td = t.to(self.device)
        cd = c.to(self.device) if c is not None else None
        o = self._model(xd, td, cd) if self.has_cond else self._model(xd, td)
        pred = o.sample if hasattr(o, 'sample') else o
        return pred.cpu()

    def _model_forward(self, x, t, c=None):
        o = self._model(x, t, c) if self.has_cond else self._model(x, t)
        return o.sample if hasattr(o, 'sample') else o

    # -- 设备态前向（采样循环用，避免每步来回搬运）-------------------------
    @torch.no_grad()
    def forward_dev(self, x_on_dev, t_on_dev, c_on_dev=None):
        """输入/输出都留在推理设备上的前向。调用方自己负责搬运。"""
        o = self._model(x_on_dev, t_on_dev, c_on_dev) if self.has_cond \
            else self._model(x_on_dev, t_on_dev)
        return o.sample if hasattr(o, 'sample') else o

    def to_dev(self, t):
        """把一个张量搬到推理设备（cpu 模式下是 no-op）。"""
        return t.to(self.device) if self.on_gpu else t

    # -- 采样循环 ---------------------------------------------------------
    @torch.no_grad()
    def sample(self, scheduler, shape, seed=0, steps=20):
        """跑一个完整 DDIM 采样循环，返回 CPU 张量。

        这里把整条循环都放在推理设备上（不只是模型），
        实测端到端 20 NFE 比 CPU 快 1.72x。
        """
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(shape, generator=g)
        sched = type(scheduler).from_config(scheduler.config)
        sched.set_timesteps(steps)
        if self.on_gpu:
            x = x.to(self.device)
        for ts in sched.timesteps:
            tt = torch.full((shape[0],), int(ts), device=x.device, dtype=torch.long)
            if self.on_gpu:
                o = self._model(x, tt)
                eps = o.sample if hasattr(o, 'sample') else o
            else:
                o = self._model(x, tt)
                eps = o.sample if hasattr(o, 'sample') else o
            x = sched.step(eps, ts, x).prev_sample
        return x.cpu()

    def close(self):
        # 同设备时 _model 就是训练模型本身，不能置 None（那会连带丢掉训练模型）
        if self.on_gpu:
            self._model = None


def selftest(model, batch_shape, device_spec='auto'):
    """数值自检：推理后端 vs CPU，前向输出应一致到 fp32 噪声级。

    返回 (最大相对误差, 说明)。相对误差 > 1e-2 视为不可用。
    """
    x = torch.randn(*batch_shape)
    t = torch.full((batch_shape[0],), 500, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        o = model(x, t)
        ref = o.sample if hasattr(o, 'sample') else o

    be = InferBackend(model, device_spec, verbose=False)
    got = be.forward(x, t)
    be.close()

    scale = ref.abs().max().item() + 1e-12
    err = (got - ref).abs().max().item() / scale
    ok = err < 1e-2
    return err, ("OK" if ok else "不可用（误差过大）")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
    print("设备探测:", resolve_device('auto')[1])
    print()
    from diffusers import UNet2DModel
    m = UNet2DModel(sample_size=16, in_channels=64, out_channels=64,
                    layers_per_block=2, block_out_channels=(128, 256, 256),
                    down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
                    up_block_types=('AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
                    norm_num_groups=32)
    err, verdict = selftest(m, (4, 64, 16, 16), 'auto')
    print("自检 UNet:  相对误差 %.3e  %s" % (err, verdict))

    from looped_dit import LoopedDiT
    m2 = LoopedDiT(d=384, in_ch=64, latent_size=16, n_unique=8,
                   loop_start=2, loop_end=3, n_loops=1)
    err2, v2 = selftest(m2, (4, 64, 16, 16), 'auto')
    print("自检 LoopedDiT: 相对误差 %.3e  %s" % (err2, v2))
