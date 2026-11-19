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
[classify] 目标分类（log-Mel + CNN）
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
| 分帧 | 帧长 256 点（2.67 ms），步长 128 点 |
| 频段 | 重点关注 8–12 kHz（仿真频段），分类特征用 log-Mel（0–24 kHz） |
| 方位角 | 度，0°=x 轴正方向，逆时针，范围 [0, 360) |

## 3. 模块清单与接口

### sim —— 仿真数据（✅ 已完成）
- 文件：`sim/uca_sim.py`
- 接口：`simulate_uca(phi_deg, f0, fs, duration, snr_db, r, c, seed) -> (t, x, X, freqs)`
- 待扩展：`simulate_moving_source()`（声源移动，出方位-时间记录图）、多类信号生成器（供分类用）

### detect —— CFAR 异常检测（⬜ 待做）
- 文件：`src/detect/cfar.py`
- 接口：`cfar_detect(x, fs, frame_len=256, hop=128, pfa=1e-3) -> list[AlarmEvent]`
- 原理：滑动参考窗估计背景噪声功率，门限=噪声×系数，系数由虚警率 pfa 决定；门限随环境自适应
- 验收图：能量曲线+自适应门限叠加图；检测率/虚警率随 SNR 变化曲线（蒙特卡洛 100 次）
- `AlarmEvent = dict(t_start, t_end, snr_est)`（字段写入数据接口规范）

### doa —— 方位估计（✅ 已完成）
- 文件：`src/doa/cbf.py`、`src/doa/music.py`
- 接口：
  - `cbf_spectrum(X, freqs, r, c, f_band, n_scan) -> (angles_deg, P_db, est_deg)`
  - `music_spectrum(x, fs, f0, r, c, n_sources=1, ...) -> (angles_deg, P_db, est_deg)`
- 待扩展：多信源 MUSIC（n_sources>1）；多途/相干源 → 空间平滑（列为"已知问题与后续工作"）

### classify —— 智能分类（⬜ 待做）
- 文件：`src/classify/features.py`（log-Mel 提取）、`src/classify/cnn.py`（模型）、`src/classify/train.py`
- 接口：
  - `logmel(x_mono, fs, n_mels=64) -> S: (n_mels, T)`
  - `classify(x, fs, model) -> (label, prob)`，label ∈ {ship, human, fish, noise}
- 数据：三类仿真信号 + 公开数据集（ShipsEar/DeepShip，仅船舶类真实数据）
- 模型：4 层小 CNN（64×T 时频图 → conv→pool×3 → fc），CPU 可训练
- 验收图：混淆矩阵 + 训练/验证准确率曲线

### viz —— 可视化（⬜ 待做）
- 文件：`src/viz/app.py`（Streamlit）
- 四个区：实时波形 / 时频谱 / 方位极坐标图 / 告警表（时间、方位、类型、置信度）
- 输入：完整流水线跑完一段仿真数据的结果
- 验收：录屏 1–2 分钟（OBS）

### pipeline —— 主控串联（⬜ 待做）
- 文件：`run_pipeline.py`
- 逻辑：`for 每一帧: detect → 有告警则 doa + classify → 输出 AlarmEvent 并可视化`
- 终端输出格式：`【告警】14:32:05 | 方位 35.0° | 类型 ship | 置信度 0.93`

## 4. 答辩图清单（每张图对应一个脚本，完成即打勾）

- [x] 四通道波形图（`run_demo_d1.py`）
- [x] CBF 方位谱（`run_demo_d1.py`）
- [x] CBF vs MUSIC 对比，0/-10 dB（`run_demo_music.py`）
- [ ] CFAR 能量+门限曲线（detect）
- [ ] 检测率/虚警率-SNR 曲线（detect）
- [ ] 分类混淆矩阵（classify）
- [ ] 方位-时间记录图 BTR（sim 移动声源 + doa）
- [ ] 可视化平台截图/录屏（viz）

## 5. 开发顺序

detect(CFAR) → classify(CNN) → pipeline 串联 → viz → BTR 扩展。
时间不够时从后往前砍；detect + doa + 对比图是保底。
