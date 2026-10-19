"""Fix large-model memory: free the CPU copy of frozen base weights once they are on the iGPU.

Problem: a 1.3B model in fp32 holds 5.2 GB on the CPU *and* 5.2 GB in iGPU buffers (UMA = the same
physical RAM), so the box runs out of memory after a few dozen steps (the 500-step run died at step
100+ with "Unable to allocate 1.34 MiB").

Fix: in LoRA mode the base weights are frozen -- after they are uploaded to the device, the CPU tensor
is no longer needed. The wrappers therefore drop their reference to the base weight and keep only the
device buffer + shape. Full fine-tuning keeps the CPU weights (it needs them for the optimizer).

Requires decoupling the autograd Function from the CPU weight tensor: it now takes explicit
out_features/in_features and accepts a None weight (LoRA mode returns no weight gradient).
"""
from __future__ import annotations

import io

P = r'D:\work\bitsandbytes-CPU\gt1030\igpu_train.py'
t = io.open(P, encoding='utf-8').read()

# ---- Function: explicit dims, weight may be None ----
old_fwd = """    @staticmethod
    def forward(ctx, x, w, w_dev, want_dw, layout):
        orig = x.shape
        k = orig[-1]
        n = w.shape[1] if layout else w.shape[0]
"""
new_fwd = """    @staticmethod
    def forward(ctx, x, w, w_dev, want_dw, layout, n_dim, k_dim):
        orig = x.shape
        k = orig[-1]
        n = n_dim
"""
assert t.count(old_fwd) == 1
t = t.replace(old_fwd, new_fwd, 1)
t = t.replace("        return torch.from_numpy(dx), dW, None, None, None\n",
              "        return torch.from_numpy(dx), dW, None, None, None, None, None\n", 1)

# ---- Linear wrappers: pass dims; free host weights after upload ----
t = t.replace("""        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.w_dev = None

    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 0)
        if self.bias is not None:
            y = y + self.bias
        return y
""", """        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.w_dev = None

    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 0,
                         self.out_features, self.in_features)
        if self.bias is not None:
            y = y + self.bias
        return y
""")

t = t.replace("""    def refresh(self):
        w = np.ascontiguousarray(self.base_weight.float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.base_weight, self.w_dev, False, 0)
""", """    def refresh(self):
        w = np.ascontiguousarray(self.base_weight.float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)
        if not self.base_weight.requires_grad:
            self.base_weight = None      # frozen base: keep only the device copy (halves host RAM)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, None, self.w_dev, False, 0, self.out_features, self.in_features)
""")

# ---- Conv1D wrapper ----
t = t.replace("""    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 1)
""", """    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 1,
                         self.out_features, self.in_features)
""")

# ---- proactive weight upload + host release after patching (LoRA only) ----
old_patch = """    n = patch_model(model, args.mode, args.lora_r, skip_lm)
"""
new_patch = """    n = patch_model(model, args.mode, args.lora_r, skip_lm)
    if args.mode == 'lora':
        # upload every frozen base weight once and drop the CPU copies (saves ~model-size host RAM)
        for m in model.modules():
            if isinstance(m, IgpuLoRALinear) and m.w_dev is None:
                m.refresh()
        import gc as _gc
        _gc.collect()
        print('  froze base on device; freed CPU copies of base weights')
"""
assert t.count(old_patch) == 1
t = t.replace(old_patch, new_patch, 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('large-model memory fix applied (LoRA releases CPU base weights)')
