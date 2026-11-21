# 听水·知微 · 水声监测工作台

用于中期答辩的本地交互演示页。展示同一段 8 秒四通道仿真场景的真实流水线输出：波形包络、CFAR 检测裕量、CBF 方位谱、log-Mel 特征、四类模型得分和告警记录。

## 运行

先在仓库根目录重新导出数据：

```bash
python -m src.viz.export_demo
```

再启动前端：

```bash
cd app
npm install
npm run dev
```

## 验证

```bash
npm run lint
npm run build
npm run test:ui
```

首次在新机器运行浏览器测试时，先执行 `npx playwright install chromium`。

## 数据口径

- `public/demo-data.json` 由现有 Python 流水线生成，不在前端伪造算法结果。
- 波形显示真实采样的时间桶最小/最大包络，以减小传输体积。
- 分类百分比为未校准的 softmax 模型得分，不是现实置信度。
- 当前仅用于仿真验证，尚未接入真实水听器或公开水声数据集。
