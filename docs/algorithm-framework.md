# 算法框架（algorithm-framework.md）

> 用途：写代码的总图纸。每个模块先按这里的接口约定实现，互不干扰、可独立替换。
> 修改接口必须先改这份文档，再改代码。

## 1. 总体数据流

```
仿真/真实采集
   │  x: (4, N) 时域波形, fs=96000 Hz
   ▼
[detect] CFAR 检测 ──无异常──▶ 继续监听
   │ 有异常 → AlarmEvent{t_start, t_end, detection_stat_db}
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
- 原理：矩形窗分帧 FFT + 频率向谱线 CFAR + 时间向宽带能量 CFAR 双通道，各占 `pfa/2`；低频边缘使用单侧参考窗，门限随背景自适应。矩形窗用于避免 Hann 窗造成的相邻频点相关，匹配当前白噪声下参考单元近似独立的假设
- 验收图：归一化检测裕量+0 dB 门限；分别报告检测率、场景级虚警概率、帧级虚警率与虚警次数/小时（当前每个 SNR 蒙特卡洛 40 次）
- 校准状态：100 组纯噪声对照中，Hann 窗与矩形窗的频谱通道帧级虚警率分别为 0.02116 和 0.00507；改用矩形窗后，六档 SNR 实验的帧级虚警率为 0.00374–0.00517，均低于设计预算 0.01。该结论仅适用于当前仿真白噪声与样本规模，不代表真实水域中的恒虚警保证
- `AlarmEvent = dict(t_start, t_end, detection_stat_db)`（字段写入数据接口规范）

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

### viz —— 可视化（✅ 交互页面已完成，录屏待做）
- 文件：`src/viz/export_demo.py`（真实结果导出）、`app/`（React + TypeScript 工作台）
- 五个区：四通道波形包络 / CFAR 检测裕量 / CBF 方位极坐标谱 / log-Mel 与四类模型得分 / 告警表
- 输入：`python -m src.viz.export_demo` 从完整流水线生成的 `app/public/demo-data.json`，不在前端伪造结果
- 交互：播放、暂停、时间拖动、1/2/4 倍速、通道开关、告警联动与数据下载
- 验收：3 项 Playwright 测试通过，控制台无错误、无 404；桌面与移动端截图位于 `figures/day4/`；1–2 分钟 OBS 录屏待做

### pipeline —— 主控串联（✅ 已完成，仿真验证）
- 文件：`run_pipeline.py`
- 逻辑：参考通道 CFAR 检测 → 告警定长分析窗 → 四通道 CBF → MLP 分类 → 输出告警
- 终端输出示例：`【告警】1.00s–3.00s | 方位 60.9° | 类型 ship | 模型得分 1.00`（该分数为未校准 softmax 输出）
- 回归：`python -m unittest discover -s tests -v` 当前 8 项全部通过

## 4. 答辩图清单（每张图对应一个脚本，完成即打勾）

- [x] 四通道波形图（`run_demo_d1.py`）
- [x] CBF 方位谱（`run_demo_d1.py`）
- [x] CBF vs MUSIC 对比，0/-10 dB（`run_demo_music.py`）
- [x] CFAR 检测统计量+门限曲线（`run_demo_cfar.py`）
- [x] 检测率/虚警率-SNR 曲线（`run_demo_cfar.py`）
- [x] 分类混淆矩阵（`run_demo_classify.py`）
- [ ] 方位-时间记录图 BTR（sim 移动声源 + doa）
- [x] 可视化平台桌面/移动端截图（`figures/day4/`）
- [ ] 可视化平台 1–2 分钟操作录屏（OBS）

## 5. 开发顺序

detect、classify(MLP)、pipeline 与 viz 页面已完成；下一步为数据接口规范和演示录屏，再做 BTR 扩展。
时间不够时从后往前砍；detect + doa + 对比图是保底。
