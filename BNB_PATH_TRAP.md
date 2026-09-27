# bitsandbytes 本地 fork：路径遮蔽陷阱与修复

## 症状

`diffusers` 加载任意 LoRA 时报：

```
Loading lcm was unsuccessful with the following error:
AttributeError: module 'bitsandbytes' has no attribute 'nn'
```

看起来像 bitsandbytes 装坏了或缺少 `nn` 子模块。**实际原因与 bitsandbytes 无关。**

## 真因

本工作区的布局是「内层才是包」：

```
D:\work\bitsandbytes-CPU\                     ← git 根，没有 __init__.py
├── bitsandbytes\                             ← 构建根（pyproject.toml / setup.py / csrc\）
│   └── bitsandbytes\__init__.py              ← 真正的包在这里
```

而**约 190 个脚本**里有 `sys.path.insert(0, r"D:\work\bitsandbytes-CPU")`（外层）。

外层 `bitsandbytes\` 没有 `__init__.py` ⇒ Python 按 **PEP 420 命名空间包**处理它，
`import bitsandbytes` 会"成功"但拿到一个空对象。

两种路径的实际差异（子进程实测）：

| sys.path 指向 | find_spec | `__file__` | `has nn` | `__version__` | peft 判定 |
|---|---|---|---|---|---|
| **外层**（脚本常用） | namespace | `None` | **False** | **None** | 抛 AttributeError |
| **内层**（真正的包） | module | 真路径 | **True** | 0.50.2.dev0 | `is_bnb_available()=True` |

于是 `import bitsandbytes.nn` 直接 `ModuleNotFoundError`。

peft 的探测链是**两级**的，这是报错信息具有误导性的原因：

```python
# peft/import_utils.py
def is_bnb_available():        return importlib.util.find_spec("bitsandbytes") is not None
def is_bnb_4bit_available():
    if not is_bnb_available(): return False
    import bitsandbytes as bnb
    return hasattr(bnb.nn, "Linear4bit")     # ← 先取 .nn，再对 .nn 做 hasattr
```

它假设「bnb 缺失 ⇒ 短路返回 False」；但命名空间包让 `find_spec` 非 None、
`bnb` 又是个空对象 ⇒ `bnb.nn` 抛 `AttributeError`。
**只要外层路径在 `sys.path` 上，diffusers 的所有 LoRA 加载都会失败。**

## 修复

让 `import bitsandbytes` 不依赖任何路径操作就解析到真包。二选一：

### 方案 A：`.pth` 文件（本机已采用）

```
<site-packages>\_bitsandbytes_cpu_fork.pth
内容（一行）：D:\work\bitsandbytes-CPU\bitsandbytes
```

验证（从任意目录、不做任何 `sys.path` 操作）：

```
find_spec: ModuleSpec(loader=SourceFileLoader,
                      origin='...\bitsandbytes\bitsandbytes\__init__.py')
version  : 0.50.2.dev0     has nn : True
from bitsandbytes.nn import Linear8bitLt, Linear4bit   -> OK
peft is_bnb_available / is_bnb_4bit_available          -> True / True
```

**并且普通模块优先于命名空间部分（与 `sys.path` 顺序无关）**，
所以即使脚本仍然插着外层目录，`import bitsandbytes` 也会解析到真包 ——
遮蔽被结构性消除，**不需要改那 190 个脚本**。

### 方案 B：正规安装（推荐用于分发，本机暂未采用）

```
cd D:\work\bitsandbytes-CPU\bitsandbytes
python -m pip install -e .
```

内层 `pyproject.toml` 用 flat-layout 发现：`include = ["bitsandbytes*"]`，
自动找到内层包目录；editable 安装会写入正确的 `.pth`。

**前置条件**：构建后端 `scikit_build_core` 当前**未安装**，
本机 DLL（`bitsandbytes\libbitsandbytes_cpu.dll`，368 KB）已手工构建好，
所以选方案 A 避免触发重新构建。

## 边界

- `.pth` 是**本机环境配置**，不在版本控制内；换机器 / 重装 Python 需要重做。
- 方案 A 不是「安装完成」—— 它只解决 import 解析。
  若要让包内元数据（`pip show bitsandbytes`）也正确，需要方案 B。
- **PyPI 打包**（用户清单第 9 项）仍需解决「手工构建的 DLL 如何进 wheel」的问题，
  与本文档是两件事。

## 相关文件

- `D:\work\diag_bnb_path.py` —— 外层 vs 内层的逐项对照（6/6 PASS）
- `D:\work\diag_bnb_kernels.py` —— 命名空间路径下内核注册缺失的证据
- `D:\work\diag_namespace.py` —— 受控实验：命名空间包是被哪一步带进 sys.path 的
- 报告 `CPU_FORGE_TECHNICAL_REPORT.md` §10.160
