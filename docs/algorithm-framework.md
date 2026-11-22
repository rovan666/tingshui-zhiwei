# 16 通道线阵算法框架

> 本文件是新方案的接口总图。修改数据形状、事件状态或模型契约时，先更新这里，再改代码。

## 1. 唯一主数据流

```text
16 通道同步波形 [16,N]，fs=100000 Hz
                 │
                 ▼
      1 秒窗，250 ms 更新，ch7 为参考通道
                 │
       ┌─────────┴──────────┐
       ▼                     ▼
OS-SCL 原型异常分数       16 通道 STFT [16,1025,193]
       │                     │
       ▼                     ▼
2/3 开启＋连续三低关闭     1–6 kHz 分 8 带，每带至多 2 个频点
       │                     │
       │                     ▼
       │               等权 CBF / 学习式频带加权
       │                     │
       │                     ▼
       │               峰值比拒判＋近场候选点
       │                     │
       │                     ▼
       │               方向引导 ISTFT 聚焦波形
       │                     │
       └─────────┐     ┌─────┘
                 ▼     ▼
             top-3 检测分窗口
                 │
                 ▼
        SSL＋TinySwin → A/B/C/未知
                 │
                 ▼
  session_id/seq/event_id 日志＋PyQt5＋分模式评估
```

CFAR → 近场等权 CBF → ch7 log-Mel 逻辑回归是独立基线，不冒充上述智能链路。

## 2. 全局参数

| 约定 | 值 |
|---|---|
| 通道 | 固定 16，参考通道 `ch7`（从 0 编号） |
| 阵元坐标 | `x_m=(m-7.5)×0.05 m`，`y_m=0` |
| 采样率 | 100000 Hz |
| 声速 | 水中 1500 m/s；空气 343 m/s，配置不得混用 |
| 示例区域 | `L=4 m`、`W=3 m`，声源 `y>0` |
| 角度 | `atan2(x,y)`，正右为正，工作角域 −60°～60° |
| 窗与更新 | 100000 点、25000 点 |
| STFT | Hann、FFT 2048、hop 512、`center=False` |
| 帧数 | 100000 点右补零 352 点，得到 193 帧 |
| log-Mel | 128 带、1–6 kHz、形状 `[1,128,193]` |
| 测向频率 | 8 带，每带至多 2 个高于背景门限的频点 |
| 测向上限 | `min(6000,c/(2d),fs/2)`；水中 6 kHz，空气 3430 Hz |
| 训练 | AdamW，初始 lr 1e-3、weight decay 1e-4、batch 8、单阶段最多 25 轮、CPU 单线程 |

## 3. 会话与标签

`Session` 字段：

- `session_id`、`fs`、`audio[16,N]`；
- `events: list[EventLabel]`；
- `mode ∈ {simulation, recording}`；
- `config_id`、`metadata`。

`EventLabel` 包含 `source_id/class_id/start_sample/end_sample/x_m/y_m/theta_deg/snr_db/seed/mode/config_id`。保存为 16 通道 PCM16 WAV＋JSON；JSON 保存 PCM 缩放和波形 SHA-256，读取时校验。

A/B/C 每类 30 份母素材按 18/6/6 划分。必须先划分母素材、背景和会话，再生成位置、反射、混音和重叠窗；同一母素材的任何变体不得跨训练、验证和测试集合。

## 4. 模块接口

### 配置与特征

- `default_water_config()`、`load_config()`、`doa_fmax(config)`；
- `stft16(x,fs,feature_cfg) -> X[16,F,T],freqs,times`；
- `LogMelExtractor.fit(train_waves)` 只在训练集估计均值和标准差；
- `transform(wave)` 推理时统计量冻结。

### 仿真

- `generate_source(class_id,rng,duration,fs)`：A 脉冲串、B 调频、C 调幅谐波，限制在 1–6 kHz；
- `generate_background(kind,...)`：white/pink/brown/tonal/modulated/impulsive/narrowband/mixed；
- `simulate_session(...)`：球面波直达和三边一阶像源反射；
- 分数延时使用线性插值，不允许 FFT 循环移位造成事件前回绕；
- 所有通道共用同一 SNR 增益，不在各通道分别缩放到相同功率。

### 检测

- `OSSCLModel`：CNN＋FPH＋ArcFace＋投影头；
- 总损失：监督对比＋Noisy-ArcMix；
- `fit_prototypes()` 使用训练嵌入，`prototype_score()` 为最近原型余弦距离；
- `EventGate`：最近 3 窗至少 2 窗高于 `tau_on` 开启，连续 3 窗低于 `tau_off` 关闭，中间保持；
- 两秒缓存只追加当前窗相对上一窗新增的 250 ms；采集异常和序号断点清空缓存与事件暂存；
- `calibrate_gate()` 只用验证背景与验证事件选择阈值，测试集不参与。

### 测向与聚焦

- `build_grid()`：矩形合法近场候选点；
- `build_steering()`：相对 ch7 的球面波相位与幅度，导向向量单位范数；
- `spatial_spectra(method="cbf"|"music")`：协方差迹归一化、8 带空间谱和有效掩码；
- `BandWeightNet`：三层协方差输入（实部、虚部绝对值、虚部符号），共享两层 3×3 CNN，掩码 softmax；
- 软角度标签在相邻网格线性插值，损失温度 0.1，验证角度 MAE 选模；
- `estimate_bearing()` 折叠为角度射线；无有效频点或峰值比不足时 `valid=False`；
- `focus_event()` 使用当前窗、未平滑候选点做方向聚焦，输出与输入等长；
- 显示端可用最近三个有效角度中位数，但不能用平滑后的角度反推另一时刻的聚焦结果。

### 分类

- `MaskedSwinSSL`：增益/噪声双视图、30% 谱块遮盖、掩码重建＋NT-Xent；
- `TinySwinClassifier`：`torchvision.swin_t`，单通道输入，内部适配 224×224，输出 A/B/C 三分类；
- `EventClassifier`：按检测分保留 top-3 窗口，事件关闭时平均概率并冻结；
- `select_unknown_threshold()`：在 0.50–0.95 间先满足未知误接受约束，再优化宏 F1；
- `LogMelLogisticBaseline`：八时间块 log-Mel＋标准化逻辑回归，仅作同数据对照。

### 主流水线

`InferenceEngine.process_window(window,session_id,seq,acquisition_error=False)` 输出至少包含：

```text
session_id, seq, event_id, time_s, status, trigger, close,
detection_score, angle_deg, point_index, peak_ratio,
label, probability, reason, config_hash, model_versions, elapsed_ms
```

采集异常必须短路测向和分类。事件进行中分类为暂态，关闭时输出最终结果；不能将采集异常、方位拒判或低置信度伪装成正常角度或 A/B/C。

`replay_session()` 使用 1 秒窗和 250 ms 更新，会话间清空门控和分类状态，但 `event_id` 保持全局递增。

## 5. 训练与权重契约

`train_all.py` 串行执行检测、测向、分类。除训练数组外，NPZ 还必须携带推理所需统计：

- 检测：`feature_mean/feature_std/tau_on/tau_off`；
- 测向：`background_power/peak_ratio_threshold`；
- 分类：`feature_mean/feature_std`，未知验证样本 `unknown_x` 用于阈值标定。

权重只保存张量和基本类型，主入口使用 `torch.load(...,weights_only=True)`。缺文件或缺字段直接失败，不随机初始化替代。

## 6. 评估口径

- 每个真值事件只匹配时间上首个预测；同一真值的后续触发计额外误报；
- 背景会话单独统计误报/分钟；
- 角度 MAE/P90 只在有效角度上计算，同时必须报告覆盖率、拒判率和拒判数；
- 已知类宏 F1 将“未知”预测作为未正确识别；
- 未知事件被分成 A/B/C 计入未知误接受；
- 仿真与实录分组成 `by_mode`，不得混合平均；
- 固定演示只说明链路运行，不能代替独立测试集验收。
