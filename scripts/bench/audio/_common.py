# -*- coding: utf-8 -*-
"""_common.py — 音频 bench 的共享契约：参数校验、JSON 读写、干净失败。

约定（与 README 一致）：
  * 缺依赖/缺权重/坏输入/坏输出路径 → stderr 提示 + exit 2，绝不 traceback
  * --threads 必须是正整数（0/负数在不同后端行为不一致：torch 抛异常、
    faster-whisper 当 auto —— 统一拒绝，避免同一参数两种语义）
"""

from __future__ import annotations

import json
import os
import sys


def fail(msg: str, code: int = 2):
    print("[skip] %s" % msg, file=sys.stderr)
    raise SystemExit(code)


def check_threads(n) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        fail("--threads 必须是正整数（收到 %r）" % (n,))
    if n < 1:
        fail("--threads 必须是正整数（收到 %r）" % (n,))
    return n


def load_json(path: str, what: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        fail("无法读取 %s %s: %s" % (what, path, e))


def write_json(path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except OSError as e:
        fail("无法写 JSON %s: %s" % (path, e))
