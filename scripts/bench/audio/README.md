# CPU 音频能力基准（可复现）

这组脚本回答一个问题：**在这台没有独显的笔记本上，音频推理和训练到底能跑到什么程度？**
每个脚本都输出同一套 JSON（schema v1），带环境指纹，数字可直接进发布记录。

> 权重一律**不入库**。所有模型路径通过参数传入；缺依赖/缺权重时脚本以 **exit 2 + 明确提示** 退出，
> 不会静默跑出错误数字。

出处：`AUDIO_ON_CPU_2026-11-13.md`、`SINGING_ON_CPU_2026-11-14.md`（这两份文档里的数字是基准值的来源）。

## 能力矩阵（参考值，空闲机器实测）

| 能力 | 脚本 | 参考结果 | 依赖 |
|---|---|---|---|
| ASR（语音识别） | `asr_bench.py` | faster-whisper small **int8**：**5.8× 实时**、WER 0（含参考文本时算 WER） | `faster-whisper` |
| TTS（Piper） | `tts_piper_bench.py` | **15–18× 实时**（长短句） | `piper-tts` |
| TTS（Kokoro-82M） | `tts_kokoro_bench.py` | **1.9–2.1× 实时** | `kokoro`、`torch` |
| 训练成本（三档规模） | `train_bench.py` | tiny_cnn **0.03 s/步**、VITS 量级 **0.57 s/步**、Whisper-small 量级 **13.3 s/步** | `torch` |
| 真训一个 KWS 小模型 | `train_kws_demo.py` | 23,556 参数、60 epoch **20.4 s**、测试 acc **0.729** | `piper-tts`、`torch`、`torchaudio` |

> **注意**：参考值是在**空闲机器**上测的。同机有训练任务在跑时，RTF 会明显变差（本仓库的 smoke 就是在
> 训练占 5 核的情况下跑的，所以看到 0.9× 而不是 15× 是正常的）。README 只承诺"方法论可复现"，
> 不承诺"换个负载还得到同一个数"。

## 快速验证（smoke，各 ≤ 1 分钟）

```bash
# 1) 训练成本（不需要任何权重，最安全的自检）
python scripts/bench/audio/train_bench.py --smoke --threads 6 --out out/train.json

# 2) ASR（需要 faster-whisper 模型目录 + 若干 wav）
python scripts/bench/audio/asr_bench.py --smoke \
    --model <faster-whisper-dir> --audio-dir <wav-dir> --out out/asr.json

# 3) Piper（需要 *.onnx 语音包）
python scripts/bench/audio/tts_piper_bench.py --smoke \
    --voice <xxx.onnx> --out out/piper.json

# 4) Kokoro（需要 Kokoro-82M 目录：config.json + *.pth + voices/*.pt）
python scripts/bench/audio/tts_kokoro_bench.py --smoke \
    --kokoro-dir <kokoro-dir> --voice af_heart.pt --out out/kokoro.json

# 5) 真训 KWS（需要 Piper 语音包；smoke = 每类 4 条 + 1 epoch）
python scripts/bench/audio/train_kws_demo.py --smoke \
    --voice <xxx.onnx> --out out/kws.json
```

本机 smoke 实测（**在 5 核被训练占用时**）：`train_bench` ≈ 40 s，`asr_bench` ≈ 16 s，
`tts_piper_bench` ≈ 14 s，`tts_kokoro_bench` ≈ 49 s（含加载），`train_kws_demo` ≈ 27 s。

## 输出格式（schema v1）

```json
{
  "schema_version": 1,
  "kind": "asr_bench | tts_piper_bench | tts_kokoro_bench | train_bench | train_kws_demo",
  "created": "2026-11-16T22:00:00+0800",
  "env": {"python": "3.11.3", "platform": "...", "machine": "AMD64",
          "threads": 6, "cpu_count": 12, "rss_gb": 0.4, "torch": "2.4.1"},
  "args": {"...": "本次调用的关键参数"},
  "results": ["每个样本/每档规模一行"],
  "summary": {"...": "汇总指标"}
}
```

字段约定：时间一律 ISO8601（带时区）；`rtf` = 音频时长 / 墙钟耗时（>1 快于实时）；
显存/内存统一 GB；所有数值四舍五入到固定小数位，便于 diff。

## 各脚本依赖

| 脚本 | 依赖 |
|---|---|
| `asr_bench.py` | `faster-whisper`、`soundfile`（可选 `psutil`） |
| `tts_piper_bench.py` | `piper-tts`（可选 `psutil`、`soundfile` 写 wav） |
| `tts_kokoro_bench.py` | `kokoro`、`torch`、`numpy`（写 wav 需 `soundfile`） |
| `train_bench.py` | `torch`（可选 `psutil`） |
| `train_kws_demo.py` | `piper-tts`、`torch`、`torchaudio`、`numpy` |

## 不推荐路径（负结果，供后来者省时间）

这些是本项目实测过、**不要重复踩**的路线（数据见 `AUDIO_ON_CPU_2026-11-13.md`）：

- **DirectML**：本机 PyTorch 无可用 DirectML 后端（版本约束 + `aten::lerp` 不支持 + fp16 NaN）。
- **OpenCL/CLBlast 跑音频训练**：GEMM 在 ≥1024³ 时快 1.8–2.2×，但**卷积只有 0.07–0.5×**，
  FFN 最好 1.15×，端到端 AST 式 Transformer 仅 1.12×——而同任务 torch CPU 快 7.4×。
- **CPU+iGPU 并发**：异步成立，但重叠只有 1.04–1.21×（流水线式 1.16–1.34×），不值得为此加复杂度。

## 约定

- 只读脚本，不写仓库内文件（`--out` 需显式给出，默认只打印）。
- 不下载权重；`kokoro` 用 `KModel(repo_id=None, model=..., config=...)` 完全离线加载。
- 所有脚本 `--smoke` 都保证 ≤1 分钟，可在 CI/人工验收里跑。
