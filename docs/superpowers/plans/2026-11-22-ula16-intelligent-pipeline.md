# 16 通道线阵水声智能处理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库主实现改为《中期初步方案》规定的 16 通道线阵、OS-SCL 检测、学习式宽带空间聚焦、SSL＋TinySwin 分类与连续事件验收系统。

**Architecture:** 统一配置和会话数据契约贯穿仿真、训练、推理与评估。传统 CFAR、等权 CBF、MUSIC 和逻辑回归是可运行的对照基线；三个学习模块分别训练、独立替换，并由连续回放统一串联。PyQt5 只消费推理事件，不参与算法计算。

**Tech Stack:** Python 3.12、NumPy、SciPy、scikit-learn、PyTorch、torchvision、PyQt5、unittest。

**Spec:** `D:\学习用的\大创\references\中期初步方案.pdf`

## Global Constraints

- 主输入为同步波形 `[16,N]`，名义 `fs=100000 Hz`、`d=0.05 m`，阵元 `x_m=(m-7.5)d`，参考通道 `ch7`（从 0 编号）。
- 示例矩形 `L=4 m`、`W=3 m`，声源 `y>0`，角度 `atan2(x,y)`，演示角域 `[-60,60]°`。
- 1 秒窗、250 ms 更新；STFT 为 Hann、FFT 2048、hop 512、`center=False`，100000 点右补零 352 点得到 193 帧。
- log-Mel 为 128 维、1–6 kHz；测向上限为 `min(6000,c/(2d),fs/2)`，水中和空气配置不得混用。
- A/B/C 每类 30 份母素材，按 18/6/6 划分；先划分素材、背景和会话，再混音切窗，禁止同源变体跨集合。
- 训练起点为 AdamW、学习率 `1e-3`、权重衰减 `1e-4`、批大小 8、单阶段最多 25 轮、CPU 单线程。
- 系统必须显式表示“采集异常”“方位不确定”“未知”，不能把失败强制转成角度或类别。
- 所有指标区分仿真与实录；测试集不得参与训练、阈值选择或模型选择。
- 旧四元圆阵不再是主交付；现有未提交文档与报告文件不得删除或覆盖。

## Review Focus

- 分数时延不得使用 FFT 循环移位造成首达前回绕；由仿真测试检查事件前为零。
- 背景或无效频点不得产生伪角度；由空掩码和峰值比测试检查拒判。
- 同源素材的多位置、反射和重叠窗不得跨集合；由分组划分测试检查。
- 采样断点必须清空事件状态和回溯缓存；由状态机测试检查。
- 拒判后的角度误差必须同时带覆盖率和拒判数；由事件评估测试检查。

---

### Task 1: 统一配置、特征和会话契约

**Files:**
- Create: `src/config.py`, `src/features/logmel.py`, `src/data/sessions.py`, `configs/water_ula16.json`, `configs/air_ula16.json`
- Test: `tests/test_ula16_config_features.py`

**Interfaces:**
- Produces: `ArrayConfig`, `FieldConfig`, `FeatureConfig`, `default_water_config()`, `doa_fmax(config)`；`stft16(x,fs,cfg)`；`LogMelExtractor.fit(x)`、`transform(x)`；`Session`、`EventLabel`、`save_session()`、`load_session()`。

- [ ] 写配置值、193 帧、Mel 形状、空气中 3.43 kHz 上限和 WAV/JSON 往返测试。
- [ ] 运行测试并确认因模块不存在而失败。
- [ ] 实现配置校验、精确右补零、训练集归一化冻结、PCM16 多通道 WAV 与 JSON 标注存取。
- [ ] 运行测试并确认通过。

### Task 2: A/B/C、八类背景与近场传播仿真

**Files:**
- Create: `sim/abc_sources.py`, `sim/backgrounds.py`, `sim/ula_sim.py`, `src/data/splits.py`
- Test: `tests/test_ula16_sim_data.py`

**Interfaces:**
- Consumes: Task 1 的配置和会话类型。
- Produces: `generate_source(class_id,rng,duration,fs)`、`generate_background(kind,rng,duration,fs)`、`simulate_session(source,xy,background,cfg,snr_db,reflections)`、`split_master_ids(ids,seed)`。

- [ ] 测试 A/B/C 时长和频段、参考通道 SNR、通道幅度比、分数时延无回绕、三处角度符号、18/6/6 分组及一阶反射路径清单。
- [ ] 运行测试并确认失败。
- [ ] 实现源波形 RMS 归一化、共时钟 16 通道传播、共用 SNR 增益、矩形三边一阶像源和按母素材划分。
- [ ] 运行测试并确认通过。

### Task 3: 近场导向、CBF/MUSIC、聚焦和方位拒判

**Files:**
- Create: `src/doa/nearfield.py`, `src/doa/focus.py`
- Test: `tests/test_ula16_doa.py`

**Interfaces:**
- Consumes: Task 1 STFT、Task 2 仿真。
- Produces: `CandidateGrid`、`SteeringTable`、`build_grid()`、`build_steering()`、`select_band_frequencies()`、`spatial_spectra()`、`estimate_bearing()`、`focus_event()`、`BearingSmoother`。

- [ ] 测试单位范数导向、矩形合法点、每带至多 2 频点、空频点拒判、正左正右测向、峰值比和三次中位数平滑。
- [ ] 运行测试并确认失败。
- [ ] 实现球面波相对相位/幅度、协方差迹归一化、8 带等权 CBF、单源 MUSIC、角度射线折叠和同窗未平滑候选聚焦。
- [ ] 运行测试并确认通过。

### Task 4: OS-SCL 检测、Noisy-ArcMix 与事件门控

**Files:**
- Create: `src/detect/os_scl.py`, `src/detect/event_gate.py`, `src/detect/train_os_scl.py`
- Test: `tests/test_ula16_detection.py`

**Interfaces:**
- Consumes: Task 1 log-Mel、Task 2 八类背景。
- Produces: `OSSCLModel`、`FPH`、`ArcFaceHead`、`supervised_contrastive_loss()`、`noisy_arcmix_loss()`、`fit_prototypes()`、`prototype_score()`、`EventGate`、`calibrate_gate()`。

- [ ] 测试投影头梯度、MixUp 加权损失、原型分数、2/3 开启、连续 3 窗关闭、断点清空和 2 秒回溯。
- [ ] 运行测试并确认失败；无 PyTorch 时先安装项目依赖再执行。
- [ ] 实现 CNN＋FPH＋ArcFace＋投影头、监督对比与 Noisy-ArcMix 总损失、验证背景准确率选权重和连续会话阈值标定。
- [ ] 运行测试并确认通过。

### Task 5: 学习式宽带空间聚焦

**Files:**
- Create: `src/doa/learned_focus.py`, `src/doa/train_focus.py`
- Test: `tests/test_ula16_learned_focus.py`

**Interfaces:**
- Consumes: Task 3 协方差、有效掩码和频带谱。
- Produces: `BandWeightNet`、`covariance_bands()`、`soft_angle_targets()`、`focus_cross_entropy()`、`train_focus()`。

- [ ] 测试三层协方差输入、无效带不参与 softmax、权重和为 1、相邻角度线性软标签及反传梯度。
- [ ] 运行测试并确认失败。
- [ ] 实现共享两层 3×3 卷积（16、32）、全局池化、掩码 softmax、温度 0.1 和验证选模。
- [ ] 运行测试并确认通过。

### Task 6: SSL＋TinySwin、方向聚焦输入与未知拒识

**Files:**
- Create: `src/classify/tiny_swin.py`, `src/classify/ssl.py`, `src/classify/train_classifier.py`, `src/classify/baseline.py`
- Test: `tests/test_ula16_classifier.py`

**Interfaces:**
- Consumes: Task 1 特征、Task 3 预测候选聚焦。
- Produces: `TinySwinClassifier`、`MaskedSwinSSL`、`nt_xent_loss()`、`mask_spectrogram()`、`select_unknown_threshold()`、`EventClassifier`、`LogMelLogisticBaseline`。

- [ ] 测试 30% 谱块遮盖、NT-Xent 对称性、模型输出三分类概率、阈值 0.50–0.95 选择、top-3 检测分窗口平均和事件结束冻结。
- [ ] 运行测试并确认失败。
- [ ] 实现增益/噪声/局部掩码双视图、掩码重建＋NT-Xent、TinySwin 微调、三折分组交叉预测数据生成接口和逻辑回归基线。
- [ ] 运行测试并确认通过。

### Task 7: 主流水线、统一评估与 PyQt5

**Files:**
- Create: `src/pipeline.py`, `src/evaluate.py`, `src/ui/pyqt_app.py`, `train_all.py`
- Modify: `run_pipeline.py`, `requirements.txt`
- Test: `tests/test_ula16_pipeline_eval.py`, `tests/test_ula16_ui.py`

**Interfaces:**
- Consumes: 前六任务的模型和数据契约。
- Produces: `InferenceEngine.process_window()`、`replay_session()`、`match_events()`、`evaluate_sessions()`、`MainWindow`。

- [ ] 测试 `session_id/seq/event_id` 串联、采集异常短路、首次结果日志、首个时间匹配、重复触发计误报、角度 MAE/P90 同时输出拒判率。
- [ ] 运行测试并确认失败。
- [ ] 实现 2 秒缓存、每 250 ms 推理、事件日志、仿真/实录独立统计、模型版本与配置哈希；PyQt5 一屏显示波形、状态、方位射线、A/B/C/未知。
- [ ] 提供顺序训练入口和可显式选择基线的端到端 CLI；无模型时明确报错，不静默伪造智能结果。
- [ ] 运行测试并确认通过。

### Task 8: 可复现演示、文档和全量验证

**Files:**
- Create: `run_demo_ula16.py`, `docs/ula16-algorithm.md`
- Modify: `README.md`, `docs/algorithm-framework.md`, `docs/handoff.md`
- Test: 全量 `tests/`

- [ ] 写新主流程的使用、训练、连续回放和失效状态文档，明确训练权重与实录性能是否真实存在。
- [ ] 运行 `python -m unittest discover -s tests -v`、`python run_pipeline.py --demo --method baseline` 和 PyQt5 offscreen  smoke test。
- [ ] 检查所有输出数字来自本次真实运行，记录未能完成的全量训练或实录验收，不宣称达到方案指标。
- [ ] 更新 handoff，写清已验证命令、结果、剩余瓶颈和下一轮入口。
