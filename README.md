# 听水·知微

**16 通道均匀线阵水声异常检测、近场测向、事件分类与连续回放系统**

本仓库当前以团队《中期初步方案》为唯一主线：同步 16 通道输入，100 kHz 采样，1 秒分析窗、250 ms 更新；智能链路为 **OS-SCL 异常检测 → 学习式宽带空间聚焦与测向 → SSL＋TinySwin 事件分类**。CFAR、等权 CBF、MUSIC 和 log-Mel 逻辑回归仅作为可运行对照基线。

> 当前完成的是代码实现、专项测试和固定仿真演示。尚未完成三阶段 25 轮全量训练、真实水声实录验收、无人船硬件联调，不能宣称方案指标已经达标。

## 快速运行

环境：Windows、Python 3.12。依赖见 `requirements.txt`。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

在仓库根目录运行新方案基线演示：

```bash
.venv/Scripts/python run_pipeline.py --demo --method baseline
```

该命令真实生成 16 通道仿真波形，依次执行 CFAR、近场等权 CBF 和 ch7 log-Mel 逻辑回归，并输出：

- `outputs/ula16_demo/baseline_results.json`：逐会话、逐告警原始结果；
- `outputs/ula16_demo/baseline_overview.png`：参考通道波形和 CBF 方位谱，明确标注“仿真数据”。

当前固定演示结果如下，数值仅用于说明代码真实运行，不是性能验收：

| 会话 | 设定类别 | 设定方位 | CBF 方位 | 基线预测 | 未校准模型得分 |
|---|---|---:|---:|---|---:|
| sim_A | A | −25° | −25° | A | 0.984 |
| sim_B | B | 5° | 6° | B | 0.699 |
| sim_C | C | 30° | 30° | **B（误判）** | 0.797 |

保留这次误判是为了如实暴露基线局限，不能用挑选后的结果代替测试集评估。

## 智能模式

智能模式必须同时提供三份由训练入口生成的权重：

```bash
.venv/Scripts/python run_pipeline.py --demo --method intelligent \
  --detector-model models/ula16/os_scl.pt \
  --focus-model models/ula16/learned_focus.pt \
  --classifier-model models/ula16/tiny_swin.pt
```

缺少任何权重、归一化统计或门限都会明确报错，**不会用随机模型或基线静默替代智能结果**。当前仓库尚未生成这三份正式权重，因此智能端到端路径已编码和做接口测试，但尚未用正式权重跑仿真回放。

### 串行训练

```bash
.venv/Scripts/python train_all.py \
  --detection-data data/detection_train.npz \
  --focus-data data/focus_train.npz \
  --classifier-data data/classifier_train.npz \
  --output-dir models/ula16 \
  --epochs 25
```

训练固定 CPU 单线程、批大小 8，每个阶段最多 25 轮，顺序为检测 → 测向 → 分类；分类阶段先自监督预训练，再监督微调并用验证集选择未知拒识阈值。每完成一个阶段就更新 `training_manifest.json`，记录数据与权重哈希、实际训练轮数和验证选模信息。

NPZ 输入契约：

| 数据文件 | 必需字段 |
|---|---|
| detection | `train_x/train_y/val_x/val_y`、`feature_mean/feature_std`、`tau_on/tau_off` |
| focus | 训练/验证协方差、频带谱、有效掩码和角度，`grid_angles`、`angle_indices_json`、`background_power`、`peak_ratio_threshold` |
| classifier | `train_x/train_y/val_x/val_y/unknown_x`、`feature_mean/feature_std` |

分类特征形状为 `[N,1,128,193]`。测向协方差形状为 `[N,8,16,16]`，频带谱为 `[N,8,P]`。测试集不得出现在训练、阈值标定或模型选择中。

## 连续回放与评估

`src/pipeline.py` 以 `session_id / seq / event_id` 串联每 250 ms 的推理结果：

- 最近 3 窗至少 2 窗超过开启阈值才触发；
- 连续 3 窗低于关闭阈值才结束；
- 两秒回溯缓存只写入非重叠的新样本；
- 采集异常、序号断点或新会话会清空门控、回溯和分类暂存；
- 事件期间对检测分最高的 3 个窗做分类概率平均，关闭时冻结最终结果；
- 无有效频点或峰值比不足时返回“方位不确定”，不强制输出角度；
- 低置信度返回“未知”，不强制分到 A/B/C。

`src/evaluate.py` 分开统计仿真与实录，输出事件召回、漏报、额外误报、背景误报/分钟、角度 MAE/P90、角度覆盖率与拒判数、宏 F1、未知误接受和触发延迟。

## PyQt5 界面

```bash
.venv/Scripts/python -m src.ui.pyqt_app
```

或让基线演示结束后打开界面：

```bash
.venv/Scripts/python run_pipeline.py --demo --method baseline --ui
```

界面显示 ch7 波形、事件状态、方位射线、类别与失效原因。Windows 下 Qt5 可能因中文绝对插件路径阻塞，本项目在仓库根目录运行时自动使用 ASCII 相对插件路径；不要从其他工作目录启动。

## 测试

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

当前本地结果为 **51 项全部通过**。测试覆盖配置与 193 帧 STFT、PCM16 会话存取、分数延时无回绕、18/6/6 母素材分组、近场 CBF/MUSIC、峰值比拒判、OS-SCL 一轮真实训练、学习式频带加权一轮真实训练、TinySwin 前向与 NT-Xent 反传、未知拒识、事件门控、非重叠回溯、评估、主入口和 PyQt5 offscreen 界面。

## 代码地图

| 模块 | 文件 | 职责 |
|---|---|---|
| 配置 | `src/config.py`、`configs/*_ula16.json` | 水中/空气独立参数、阵列与特征契约 |
| 仿真 | `sim/abc_sources.py`、`sim/backgrounds.py`、`sim/ula_sim.py` | A/B/C、8 类背景、球面波直达与一阶反射 |
| 数据 | `src/data/sessions.py`、`src/data/splits.py` | 多通道 WAV/JSON、哈希、按母素材防泄漏划分 |
| 特征 | `src/features/logmel.py` | 16 通道 STFT、128×193 log-Mel、冻结训练归一化 |
| 检测 | `src/detect/os_scl.py`、`event_gate.py` | OS-SCL、原型分数、双阈值事件状态机 |
| 测向 | `src/doa/nearfield.py`、`learned_focus.py`、`focus.py` | CBF/MUSIC、频带权重、方向聚焦与拒判 |
| 分类 | `src/classify/ssl.py`、`tiny_swin.py` | 掩码重建＋NT-Xent、TinySwin、top-3 事件平均 |
| 主控 | `src/pipeline.py`、`run_pipeline.py` | 连续推理、显式模型加载、唯一 CLI |
| 评估 | `src/evaluate.py` | 事件匹配和仿真/实录独立指标 |
| 界面 | `src/ui/pyqt_app.py` | 单屏事件展示，不参与算法计算 |

更详细的接口见 `docs/algorithm-framework.md`，算法说明见 `docs/ula16-algorithm.md`，真实进度与未完成项见 `docs/handoff.md`。

## 当前边界

- 没有正式训练权重，没有 25 轮全量训练结果；
- 固定基线演示存在 C 类误判，不能据此声称分类达标；
- 没有真实水听器同步采集、实录数据集或水域测试；
- 没有船体、电机、水密结构或动力控制闭环；
- 仿真得分未经校准，不等于现实中的概率或置信度。
