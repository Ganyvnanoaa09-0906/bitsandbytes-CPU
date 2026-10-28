# 动态崩坏边界表（速查）

> 管线：**冻结配置** 512×512 · DPM-Solver++(2M) · 5 步 · SD1.5 VAE · 独立分块 + 硬切 · FSRCNN 后处理
> 来源：`motion_boundary.py` + `face_consistency.py` + `scripts/misc/streak_metrics.py`，2026-09-11 下午实测
> 详细分析：这是视频生成项目的独立实验记录，不属于 `bitsandbytes` 库 API；数据由上述脚本产生。

## 一、五档实测数据

| 档位 | 内容 | mad | ncc | lap | **det%** | **face_conf** | 尺度漂移 | radial_align | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| L1_micro | 微表情 | 10.11 | 0.990 | 73.9 | **100** | **10.80** | +4.0% | 0.726 | 750s |
| L2_headturn | 转头 | 10.80 | 0.989 | 130.8 | **100** | 10.21 | +1.3% | 0.692 | 724s |
| L3_wave | 挥手 | 12.79 | **0.973** | 302.0 | **100** | 10.07 | −3.3% | **0.510** | 754s |
| **L4_runjump** | 跑跳 | 13.86 | **0.964** | 252.2 | **80** | **7.68** | +11.9% | **0.580** | 768s |
| **L5_fight** | 打斗 | 13.25 | **0.959** | 208.1 | **40** | **7.56** | **−20.3%** | 0.628 | 808s |

**关键判据**
- `det%` = `keypointrcnn` 能检出"人"的帧比例 —— **最灵敏的崩坏信号**
- `ncc` = 相邻帧归一化互相关 —— 内容在"移动"还是被"替换"
- `radial_align` < 0.637（理论值 2/π）= 出现放射状速度线 = **模型在用特效掩盖崩坏**
- ⚠️ **`lap`（Laplacian）在动态档完全失真**：速度线是高频内容，会把 lap 抬高（L3 的 302 > L1 的 73.9，但 L3 明显更差）

## 二、边界与崩坏形态

| 档位 | 崩坏程度 | 具体表现 |
|---|---|---|
| L1 微表情 | **无** | 五官端正、表情自然、发丝柔和 |
| L2 转头 | **无** | 侧脸结构完整 |
| L3 挥手 | **轻度** | 脸部半抽象（单眼模糊发光、面部惨白、头发成糊块）；开始出现放射速度线 |
| L4 跑跳 | **中度** | 手臂与手不可辨、躯干含糊；模型用放射光束与速度线掩盖 |
| L5 打斗 | **重度** | 脸完全抽象（仅剩黑点当眼）、身体成色块、角色尺度缩水 20%；**出现像素化垃圾/水印状伪影** |

**✅ 边界位置：L3（挥手）可用 → L4（跑跳）开始退化 → L5（打斗）崩坏。**

## 三、处理方案（零成本，仅改提示词）

| 动作类型 | 策略 | 实测效果 |
|---|---|---|
| 微表情 / 静态 / 口播 | **直接生成** | det% 100%、face_conf 10.8 |
| 转头 / 侧脸 | **直接生成** | det% 100%、face_conf 10.2 |
| 挥手 / 手势 | **直接生成**（目视抽检） | det% 100%、face_conf 10.1 |
| **跑跳 / 位移** | **加** `motion blur, bokeh, blurry background, depth of field` | det% **80 → 100**；lap 252 → **97**；目视双臂可见、腿呈跨步 |
| **格斗 / 激烈冲突** | **改"脸部特写 + speed lines + bokeh"**，不用全景 | 全景 det% 仅 40%；特写五官完整（目视确认） |

### 缓解方案实测对照

| 组 | 改动 | det% | lap | 目视 |
|---|---|---|---|---|
| L4 基准 | — | 80 | 252.2 | 手臂不可辨 |
| **MB** | + motion blur + bokeh + DOF | **100** | **96.5** | **双臂可见、腿跨步、比例正常** |
| L5 基准 | — | 40 | 208.1 | 脸全抽象、身体色块 |
| **MA** | → 特写 + bokeh + DOF | 20\* | 40.8 | **眼睛清晰完整、五官端正** |

\* MA 的 det%=20 是**测量假象**：`keypointrcnn` 在 COCO 全身/半身人上训练，**遇到"只有脸"的极端特写无法检出"人"**；该图经目视确认五官完整。**此指标对极端特写失效。**

## 四、生产建议

**激烈段落的表现手法**（规避模型短板，而非硬碰）：
- 用 **「特写 + 速度线」承载情绪**
- 用 **「虚化 + 运动模糊的中景」承载动作**
- **避免全景格斗镜头**（det% 仅 40%）

**验收判据按场景选择**：

| 场景 | 主判据 | **不可用** |
|---|---|---|
| 静态 / 软帧后处理 | `Laplacian` | — |
| **动态场景** | **`det%`** | **`Laplacian`（完全失真）** |
| 白化 / 过曝 | `white%`>245、`p99` | `MAD`、`hist_drift` |
| 绿斑 | `green_px`（须先 `astype(float32)`） | — |

## 五、复现命令

```powershell
# 崩坏扫描（5 档，约 64 分钟）
$env:ADIFF_VAE = "D:\work\textmodel\sd15_full\vae"
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u motion_boundary.py --out "D:\work\output\motion"

# 纯分析（不生成）
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u scripts/video/motion_report.py --root "D:\work\output\motion"

# 面部一致性
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u face_consistency.py `
    --root "D:\work\output\motion" --out "D:\work\output\motion\face.json"

# 放射条纹量化
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u scripts/misc/streak_metrics.py `
    --root "D:\work\output\motion" --more "D:\work\output\motion_mit"
```
