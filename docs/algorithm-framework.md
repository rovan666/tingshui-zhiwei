# 算法框架（algorithm-framework.md）

> 用途：写代码的总图纸。每个模块先按这里的接口约定实现，互不干扰、可独立替换。
> 修改接口必须先改这份文档，再改代码。

## 1. 总体数据流

```
仿真/真实采集
   │  x: (4, N) 时域波形, fs=96000 Hz
   ▼
[detect] CFAR 检测 ──无异常──▶ 继续监听
   │ 有异常 → AlarmEvent{t_start, t_end, snr_est}
   ▼
[doa] 方位估计（CBF / MUSIC）
   │  → bearing_deg, spectrum(方位谱), confidence
   ▼
[classify] 目标分类（log-Mel + MLP，CNN 为后续升级）
   │  → label(ship/human/fish), prob
   ▼
[viz] 可视化 + 告警记录
   │  → 波形/时频谱/方位雷达/告警列表
   ▼
（远期）[control] 船体决策：调整航向对准声源
```

## 2. 全局约定（所有模块必须遵守）

| 约定 | 值 |
|---|---|
| 采样率 `fs` | 96000 Hz |
| 阵元数 | 4（均匀圆阵，45/135/225/315°） |
| 圆阵半径 `r` | 0.0375 m（10 kHz 下 λ/4，小孔径） |
| 水中声速 `c` | 1500 m/s |
| 时域波形 | `x: np.ndarray (4, N)`，通道顺序与 `sim/uca_sim.ELEMENT_ANGLES_DEG` 一致 |
| 频域波形 | `X: np.ndarray (4, F)`，`np.fft.rfft(x, axis=1)` |
| 分帧 | CFAR 与 log-Mel 均为帧长 1024 点、步长 512 点；DOA 窄带快拍按各算法脚本配置 |
| 频段 | DOA 基础仿真重点关注 8–12 kHz；CFAR 100 Hz–24 kHz；分类 log-Mel 50 Hz–24 kHz |
| 方位角 | 度，0°=x 轴正方向，逆时针，范围 [0, 360) |

## 3. 模块清单与接口

### sim —— 仿真数据（✅ 已完成）
- 文件：`sim/uca_sim.py`
- 接口：`simulate_uca(phi_deg, f0, fs, duration, snr_db, r, c, seed) -> (t, x, X, freqs)`
- 待扩展：`simulate_moving_source()`（声源移动，出方位-时间记录图）、多类信号生成器（供分类用）

### detect —— CFAR 异常检测（✅ 已完成，仿真验证）
- 文件：`src/detect/cfar.py`
- 接口：`cfar_detect(x, fs, frame_len=1024, hop=512, pfa=1e-2) -> (alarms, debug)`
- 原理：频率向谱线 CFAR + 时间向宽带能量 CFAR 双通道；低频边缘使用单侧参考窗，门限随背景自适应
- 验收图：检测统计量+门限图；检测率/虚警率随 SNR 变化曲线（当前每个 SNR 蒙特卡洛 40 次）
- `AlarmEvent = dict(t_start, t_end, snr_est)`（字段写入数据接口规范）

### doa —— 方位估计（✅ 已完成）
- 文件：`src/doa/cbf.py`、`src/doa/music.py`
- 接口：
  - `cbf_spectrum(X, freqs, r, c, f_band, n_scan) -> (angles_deg, P_db, est_deg)`
  - `music_spectrum(x, fs, f0, r, c, n_sources=1, ...) -> (angles_deg, P_db, est_deg)`
- 待扩展：多信源 MUSIC（n_sources>1）；多途/相干源 → 空间平滑（列为"已知问题与后续工作"）

### classify —— 智能分类（✅ MLP 版已完成，仿真验证）
- 文件：`src/classify/features.py`（log-Mel 提取）、`src/classify/mlp.py`（训练与推理）
- 接口：
  - `logmel(x_mono, fs, n_mels=40) -> S: (n_mels, T)`
  - `classify(x_mono, clf, fs) -> (label, prob)`，label ∈ {ship, human, fish, noise}
- 输入约定：当前 MLP 使用 1 s 定长波形，流水线负责截取以告警为中心的 1 s 分析窗
- 数据：当前仅三类仿真信号 + 仿真噪声；公开数据集 ShipsEar/DeepShip 尚未接入
- 模型：log-Mel(40) + sklearn MLP(128)；CNN 留作后续升级，不再作为当前完成项表述
- 验收图：混淆矩阵 + 四类 log-Mel 特征图；测试准确率 96.2%（仿真测试集）

### viz —— 可视化（⬜ 待做）
- 文件：`src/viz/app.py`（Streamlit）
- 四个区：实时波形 / 时频谱 / 方位极坐标图 / 告警表（时间、方位、类型、置信度）
- 输入：完整流水线跑完一段仿真数据的结果
- 验收：录屏 1–2 分钟（OBS）

### pipeline —— 主控串联（✅ 已完成，仿真验证）
- 文件：`run_pipeline.py`
- 逻辑：参考通道 CFAR 检测 → 告警定长分析窗 → 四通道 CBF → MLP 分类 → 输出告警
- 终端输出格式：`【告警】1.01s–3.00s | 方位 61.4° | 类型 ship | 置信度 1.00`
- 回归：`python -m unittest discover -s tests -v` 当前 5 项全部通过

## 4. 答辩图清单（每张图对应一个脚本，完成即打勾）

- [x] 四通道波形图（`run_demo_d1.py`）
- [x] CBF 方位谱（`run_demo_d1.py`）
- [x] CBF vs MUSIC 对比，0/-10 dB（`run_demo_music.py`）
- [x] CFAR 检测统计量+门限曲线（`run_demo_cfar.py`）
- [x] 检测率/虚警率-SNR 曲线（`run_demo_cfar.py`）
- [x] 分类混淆矩阵（`run_demo_classify.py`）
- [ ] 方位-时间记录图 BTR（sim 移动声源 + doa）
- [ ] 可视化平台截图/录屏（viz）

## 5. 开发顺序

detect、classify(MLP)、pipeline 已完成；下一步为数据接口规范与 viz，再做 BTR 扩展。
时间不够时从后往前砍；detect + doa + 对比图是保底。
