"""disk_balancer 零拷贝改动 —— 端到端正确性验证。

判定标准（全部必须成立）：
  V1. 盘上文件字节与源张量的原始字节 **逐字节一致**（fp32/fp16/bf16）。
  V2. 非连续张量（transpose 后）也能正确写盘。
  V3. get_cold() 读回的张量与原张量数值完全相等（dtype / shape 都对）。
  V4. 写完后 _write_queue 里没有残留 memoryview（release 必须被调用），
      且源 storage 真的被回收（weakref 判定）。
  V5. update_step() 在内存紧张时一次能迁移多个（视频模型场景）。
  V6. 负控制：故意把源张量写成不同数值，验证比对方法能抓到差异（防止"永远 PASS"）。
"""
import os
import gc
import sys
import json
import shutil
import weakref
import tempfile

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import disk_balancer as db

# Paths are resolved relative to this file, and the scratch tree goes under the
# system temp directory. They used to be hard-coded to D:\work\..., which meant
# this test could only ever run on one machine.
ROOT = os.path.join(tempfile.gettempdir(), "_verify_zerocopy")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


def fresh_bal(sub, quota=2048):
    d = os.path.join(ROOT, sub)
    os.makedirs(d, exist_ok=True)
    mount = os.path.splitdrive(os.path.abspath(d))[0].upper()
    free = shutil.disk_usage(d).free // (1 << 20)
    cfg = db.DiskBalancerConfig(mode="manual", sizes={mount: min(quota, int(free * 0.5))},
                                paths=[d], min_param_size=1)
    b = db.DiskLoadBalancer(cfg)
    b.attach_model(nn.Linear(2, 2))
    b.start()
    return b, d


def raw_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


# ----------------------------------------------------------------------
# V1/V2：各 dtype + 非连续
# ----------------------------------------------------------------------
def v1_v2():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for contig in (True, False):
            tag = f"{str(dtype).replace('torch.', '')}{'_contig' if contig else '_noncontig'}"
            bal, d = fresh_bal(f"v12_{tag}")

            # 构造：contig=True 用连续张量；False 用转置后的非连续张量。
            # 注意：这里必须把构造用的中间变量清掉，否则它和参数共享同一块 storage，
            # weakref 判定"源 storage 是否回收"会永远是 False（第一版就踩了这个）。
            if contig:
                t = nn.Parameter(torch.randn(64, 128).to(dtype))
            else:
                t = nn.Parameter(torch.randn(128, 64).to(dtype).t())
            if not contig:
                assert not t.is_contiguous(), f"{tag} 期望非连续，实际连续"

            shape = t.shape
            numel = t.numel()
            ref = raw_bytes(t)          # raw_bytes 内部 contiguous()，结果是扁平字节
            src = weakref.ref(t.untyped_storage())
            key = f"p_{tag}"
            path = bal.put_cold(key, t, tier="ssd")
            bal.wait_writes()

            # put_cold 的职责就是把源张量的 data 置空。必须在迁移【之前】把 numel/shape
            # 存下来 —— 第一版这里直接写 t.numel()，迁移后它是 0，于是 "back.numel() ==
            # t.numel()" 变成 "8192 == 0" 恒假，白白报了 12 个 FAIL。
            check(f"V0 put_cold 后源张量已置空 [{tag}]", t.numel() == 0 and numel > 0,
                  f"迁移前 {numel} -> 迁移后 {t.numel()}")

            with open(path, "rb") as f:
                got = f.read()
            check(f"V1 盘上字节 == 源字节 [{tag}]", got == ref,
                  f"{len(got)}B vs {len(ref)}B")

            # V3：get_cold 返回的是【扁平】张量（只按 numel 还原，不带 shape）。
            #     要比数值必须自己 reshape，或用 get_cold_shape()。
            back_flat = bal.get_cold(key)
            ok_flat = (back_flat is not None and back_flat.dtype == dtype
                       and back_flat.numel() == numel
                       and raw_bytes(back_flat) == ref)
            check(f"V3a get_cold 字节一致(扁平) [{tag}]", ok_flat,
                  f"dtype={None if back_flat is None else back_flat.dtype} "
                  f"numel={None if back_flat is None else back_flat.numel()} 期望 {numel}")

            back = bal.get_cold_shape(key, shape)
            ok_shaped = (back is not None and back.shape == shape
                         and raw_bytes(back) == ref)
            check(f"V3b get_cold_shape 数值一致 [{tag}]", ok_shaped,
                  f"shape={None if back is None else tuple(back.shape)} 期望 {tuple(shape)}")

            del t
            gc.collect()
            check(f"V4 源 storage 已回收 [{tag}]", src() is None)
            bal.cleanup()


# ----------------------------------------------------------------------
# V4b：队列无残留 memoryview + 全流程后无泄漏
# ----------------------------------------------------------------------
def v4_queue_clean():
    bal, d = fresh_bal("v4")
    params = [nn.Parameter(torch.randn(512, 512)) for _ in range(16)]
    qt = bal._write_queue
    for i, p in enumerate(params):
        bal.put_cold(f"p{i}", p, tier="ssd")
    bal.wait_writes()
    # 队列应已排空
    check("V4 队列已排空", qt.qsize() == 0 and qt.unfinished_tasks == 0,
          f"qsize={qt.qsize()} unfinished={qt.unfinished_tasks}")
    # 全部文件都在且非空
    sizes = [os.path.getsize(e[0]) for e in bal._cold_index.values()]
    check("V4 全部文件已落盘且非空", len(sizes) == 16 and all(s == 512 * 512 * 4 for s in sizes),
          f"{len(sizes)} 个文件，尺寸 {sorted(set(sizes))}")
    bal.cleanup()


# ----------------------------------------------------------------------
# V5：update_step 一次迁移多个
# ----------------------------------------------------------------------
def v5_multi_offload():
    bal, d = fresh_bal("v5", quota=4096)
    # 造一个"几百个冻结张量"的模型（视频模型形态）
    layers = []
    for _ in range(40):
        lin = nn.Linear(1024, 1024, bias=False)
        for p in lin.parameters():
            p.requires_grad_(False)
        layers.append(lin)
    model = nn.Sequential(*layers)
    bal.attach_model(model)

    cold = sum(1 for n, p in model.named_parameters() if bal._is_cold_param(n, p))

    # 强制"内存紧张"：直接替换 _mem_pressure 的判定来源不可靠（psutil 是 C 扩展），
    # 所以改用最小侵入的方式——把阈值设成当前占用率之下，使 _mem_pressure() 为真。
    mem = __import__("psutil").virtual_memory()
    used_ratio = mem.percent / 100.0
    bal._cfg.memory_threshold = max(0.01, used_ratio - 0.05)
    tight = bal._mem_pressure()
    check("V5 前置条件：内存压力判定为真", tight,
          f"used={used_ratio:.3f} thr={bal._cfg.memory_threshold:.3f}")

    n = bal.update_step()
    check("V5 update_step 一次迁移多个（旧实现恒为 1）", n > 1, f"迁移了 {n} 个 / 共 {cold} 个冷参数")
    check("V5 不限量时能把全部冷参数迁完", n == cold, f"{n} vs {cold}")
    bal.wait_writes()
    bal.cleanup()

    # 限流仍然生效
    bal2, d2 = fresh_bal("v5b", quota=4096)
    bal2._cfg.offload_per_step = 5
    bal2._cfg.memory_threshold = max(0.01, used_ratio - 0.05)
    layers2 = []
    for _ in range(40):
        lin = nn.Linear(1024, 1024, bias=False)
        for p in lin.parameters():
            p.requires_grad_(False)
        layers2.append(lin)
    bal2.attach_model(nn.Sequential(*layers2))
    n2 = bal2.update_step()
    check("V5 offload_per_step=5 限流生效", n2 == 5, f"迁移了 {n2} 个")
    bal2.wait_writes()
    bal2.cleanup()


# ----------------------------------------------------------------------
# V6：负控制 —— 比对方法必须能抓到差异
# ----------------------------------------------------------------------
def v6_negative_control():
    bal, d = fresh_bal("v6")
    t = nn.Parameter(torch.randn(256, 256))
    ref = raw_bytes(t)
    path = bal.put_cold("neg", t, tier="ssd")
    bal.wait_writes()

    with open(path, "rb") as f:
        got = f.read()
    check("V6a 正样本比对为真", got == ref)

    # 故意改一个字节，比对必须为假
    tampered = bytearray(got)
    tampered[12345] ^= 0xFF
    check("V6b 篡改 1 字节后比对为假（负控制）", bytes(tampered) != ref)

    # 故意用不同数值的张量当参照，比对必须为假
    other = raw_bytes(torch.randn(256, 256))
    check("V6c 用另一个张量当参照比对为假（负控制）", other != ref)
    bal.cleanup()


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)
    print("=== V1/V2/V3/V4 各 dtype + 非连续 ===", flush=True)
    v1_v2()
    print("=== V4b 队列与落盘 ===", flush=True)
    v4_queue_clean()
    print("=== V5 多参数迁移 ===", flush=True)
    v5_multi_offload()
    print("=== V6 负控制 ===", flush=True)
    v6_negative_control()

    npass = sum(1 for r in RESULTS if r["pass"])
    print(f"\n{'='*60}\n总计 {npass}/{len(RESULTS)} PASS\n{'='*60}")
    for r in RESULTS:
        if not r["pass"]:
            print(f"  FAIL: {r['check']}  {r['detail']}")

    out_dir = os.path.join(HERE, "cloud_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verify_zerocopy.json"), "w", encoding="utf-8") as f:
        json.dump({"results": RESULTS, "passed": npass, "total": len(RESULTS)},
                  f, indent=2, ensure_ascii=False)

    shutil.rmtree(ROOT, ignore_errors=True)
    sys.exit(0 if npass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
