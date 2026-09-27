"""直接测「队列里到底扣着多少源内存」—— 不再靠 RSS 间接推断。

为什么换口径：同一问题我得到过三组互相矛盾的 RSS 观测（集成测试"只降1.2MB"、
峰值口径"0.02MB 峰值"、拷贝口径"峰值 1008MB 后 -4550MB"）。RSS 受 torch 分配器
缓存、Windows 工作集回收策略影响，不适合当这个问题的尺子。

直接量：
  Q_max      队列峰值深度（排队中的 payload 个数）
  Q_bytes    排队 payload 的字节总量（= 被扣住的源内存）
  inflight   同时在飞的视图数
  源storage  用 weakref 逐个判定"是否仍存活"
"""
import os
import sys
import gc
import json
import time
import queue
import shutil
import weakref
import threading

import torch
import psutil

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import disk_balancer as db
import video_models as vm

ROOT = r"D:\work\_qprobe"
os.makedirs(ROOT, exist_ok=True)


class TrackingWriter(threading.Thread):
    def __init__(self, q, delay=0.0):
        super().__init__(daemon=True)
        self.q, self.stop, self.delay = q, False, delay
        self.done = 0
        self.max_q = 0
        self.max_bytes = 0
        self.lock = threading.Lock()

    def run(self):
        while not self.stop:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                continue
            _, payload, path = item
            try:
                if self.delay:
                    time.sleep(self.delay)
                with open(path, "wb") as f:
                    f.write(payload)
            finally:
                if isinstance(payload, memoryview):
                    payload.release()
                with self.lock:
                    self.done += 1
                self.q.task_done()


def nbytes_of(payload):
    return payload.nbytes if isinstance(payload, memoryview) else len(payload)


def run(zero_copy, bounded, delay=0.0):
    cache = os.path.join(ROOT, f"{int(zero_copy)}_{bounded}_{delay}")
    os.makedirs(cache, exist_ok=True)

    model = vm.load_animatediff()
    for p in model.parameters():
        p.requires_grad_(False)

    mount = os.path.splitdrive(os.path.abspath(cache))[0].upper()
    free = shutil.disk_usage(cache).free // (1 << 20)
    cfg = db.DiskBalancerConfig(mode="manual", sizes={mount: min(8192, int(free * 0.6))},
                                paths=[cache], min_param_size=100000)
    bal = db.DiskLoadBalancer(cfg)
    bal.attach_model(model)
    bal.start()

    bal._write_queue = queue.Queue(maxsize=bounded) if bounded else queue.Queue()
    w = TrackingWriter(bal._write_queue, delay=delay)
    w.start()

    cold = [(n, p) for n, p in model.named_parameters() if bal._is_cold_param(n, p)]
    cold_mb = sum(p.numel() * p.element_size() for _, p in cold) / 1024 / 1024

    # 逐个记录源 storage 的 weakref，判定它什么时候真的没了
    src_refs = {n: weakref.ref(p.untyped_storage()) for n, p in cold}
    queued_bytes = 0
    max_queued = 0
    max_qdepth = 0
    alive_peak = 0

    gc.collect()
    t0 = time.perf_counter()
    for n, p in cold:
        path = os.path.join(cache, f"{n.replace('.', '_')}.bin")
        with bal._lock:
            bal._cold_index[n] = (path, p.numel(), p.dtype)
        if zero_copy:
            payload = memoryview(p.detach().cpu().contiguous().numpy())
        else:
            payload = p.detach().cpu().contiguous().numpy().tobytes()
        b = nbytes_of(payload)
        bal._write_queue.put((n, payload, path))
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        bal._migrated_count += 1

        qd = bal._write_queue.qsize()
        max_qdepth = max(max_qdepth, qd)
        alive = sum(1 for r in src_refs.values() if r() is not None)
        alive_peak = max(alive_peak, alive)

    t_calls = time.perf_counter() - t0
    bal._write_queue.join()
    t_total = time.perf_counter() - t0

    # 队列排空后，源 storage 应当全部消失
    gc.collect()
    alive_after = sum(1 for r in src_refs.values() if r() is not None)

    res = {
        "zero_copy": zero_copy, "queue": bounded or "unbounded", "writer_delay_s": delay,
        "cold_params": len(cold), "cold_MB": cold_mb,
        "max_queue_depth": max_qdepth,
        "src_storages_alive_peak": alive_peak,
        "src_storages_alive_after_join": alive_after,
        "call_site_s": t_calls, "total_s": t_total,
        "call_site_per_param_ms": t_calls / max(len(cold), 1) * 1000,
    }
    for n in list(bal._cold_index.keys()):
        pth = bal._cold_index[n][0]
        bal._cold_index.pop(n, None)
        if os.path.exists(pth):
            os.remove(pth)
    bal.cleanup()
    w.stop = True
    bal._write_queue.put(None)
    w.join(timeout=5)
    del model
    gc.collect()
    return res


def main():
    rep = {"results": []}
    # delay=0.02s 模拟慢盘，把队列真实撑起来（否则 D: 太快，队列根本排不长）
    for zc, bnd, dly in [(False, None, 0.0), (True, None, 0.0),
                         (False, None, 0.02), (True, None, 0.02),
                         (True, 2, 0.02)]:
        print(f"  zero_copy={zc} queue={bnd} delay={dly} ...", flush=True)
        rep["results"].append(run(zc, bnd, dly))
        gc.collect()
    with open(r"D:\work\cloud_results\queue_probe.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    for r in rep["results"]:
        print(f"  zc={r['zero_copy']} q={r['queue']} delay={r['writer_delay_s']}: "
              f"max_qdepth={r['max_queue_depth']} "
              f"src_alive_peak={r['src_storages_alive_peak']}/{r['cold_params']} "
              f"alive_after_join={r['src_storages_alive_after_join']} "
              f"call={r['call_site_per_param_ms']:.3f}ms/param total={r['total_s']:.2f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
