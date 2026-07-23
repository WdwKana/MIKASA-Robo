# 实验结构 & 图表规划 (Structure & Figure/Table Plan)

## 推荐 Experiments 结构

```
§4 Experiments
  §4.1 Benchmark, Baselines, and Protocol  (~0.5 col)
       - MIKASA-Robo benchmark (12 tasks, 3 families)
       - 基线描述（perception-matched）
       - 评估协议：7M steps, 3 seeds, success_once (last-3 eval mean)
       - CRES/CAPS 规则（所有方法统一）
       - 与 ELMUR/offline 的定位说明

  §4.2 Main Results  (~1.0 col)
       - Table 1: 主表 (已有)
       - Figure 2: 训练曲线 (已有)
       - 正文分析：整体优势、任务族差异、两个失败点

  §4.3 Does Memory-Relative Surprise Matter?  (~0.4 col)
       - Table 2: Write-rule ablation (RC5, 3 variants × 3 seeds)
       - 因果论证：相同 reader/buffer，仅 write rule 不同

  §4.4 What Role Does Each Memory Pathway Play?  (~0.5 col)
       - 合并原 §4.4 和 §4.5 的内容
       - Figure 3: 记忆存活探针 (已有, 需 restyle)
       - 路由消融结论（NoLSTM 数据融入段落）
       - 叙事：buffer 存线索（平坦 ≥0.90），LSTM 跟运动（Intercept -11pp）
```

## 各小节详细内容规划

### §4.1 Benchmark, Baselines, and Protocol
**必含要素**：
1. MIKASA-Robo 简介 + 引用（不重复 §2）
2. 12 任务列表（3 族：Remember × {Color, Shape, ShapeAndColor}，ShellGame × {Touch, Push}，Intercept × {GrabFast, Fast}）
3. 基线：MLP (no memory), GRU, LSTM, FFM [cite], SHM [cite]
4. 感知匹配说明：all share frozen DINOv2-S/14, same CRES/CAPS by task
5. 训练：PPO, 7M steps, 256 envs, seeds {33,42,99}
6. 评估：success_once, mean of last-3 evaluations
7. CRES/CAPS 分配规则（一句话 + cross-ref Appendix）
8. ELMUR 定位：separate track (offline/end-to-end), cite their numbers

### §4.2 Main Results
**Table 1** (已有 main_table_12.tex)
- 格式：已有 booktabs, bold best, ours last column
- **修改建议**：加 task-family 分组线（Remember / ShellGame / Intercept）
- 加 per-family Average 行

**Figure 2** (已有 main_success_12.pdf)
- 格式：4×3 panel, 已可用
- Caption 需写（当前正文 caption 基本完整但可优化）

**正文分析** (~100 words)：
1. 整体：STRM 51.4% vs GRU 30.4% (+21pp), 10/12 best
2. 强势任务：ShellGameTouch 98.1%（near-ceiling）, RS5 52.7% vs next 25.9%
3. RC9 失败分析：9-way 颜色辨别超出 CRES 能力？GRU 的隐式颜色编码反而更有效
4. InterceptFast：LSTM 82.8% vs STRM 80.9%，差距在 std 内 (±14.5)
5. ShellGamePush 高方差 (31.9% std)：原因分析

### §4.3 Write-Rule Ablation
**Table 2** (需创建):
```
| Write Rule | RC5 Success (%) |
|---|---|
| Surprise (ours) | 55.3 ± 19.7 |
| Random | 21.9 ± 1.2 |
| FIFO | 23.3 ± 4.9* |
```
*FIFO 如果 seed99 不可靠，标注或仅用 2 seeds

**正文**：
- surprise 相对 random/FIFO 提升 2.5×
- 说明实验设置：相同 buffer capacity/reader/LSTM/CRES/CAPS, 仅替换 write rule
- 含义：信息的选择性写入（而非随机或时序）是关键

### §4.4 Memory Pathway Analysis (合并 probes + routing)
**Figure 3**: 记忆存活探针
- 需要 restyle 为 AAAI 格式（当前是原始 matplotlib）
- 建议：2-panel（左：RC5 各 substrate decodability over time；右：NoLSTM ablation bar chart）

**段落 1**：探针结果
- buffer 线索可解码性 ≥ 0.90 全 episode
- LSTM c state τ ≈ 8 步
- FFM / GRU 也快速衰减
- 结论：episodic buffer 是唯一能长期保持的基质

**段落 2**：路由消融
- NoLSTM on Intercept: -11.8pp / -13.2pp
- NoLSTM on Remember: 持平或更好
- 结论：两个路径互补——buffer 记线索，LSTM 跟运动

## 图表总览

| 编号 | 类型 | 内容 | 位置 | 状态 |
|---|---|---|---|---|
| Fig 1 | figure* | Architecture diagram | Method | ❌ caption 需写 |
| Fig 2 | figure* | 12-task training curves | §4.2 | ✅ 已有 |
| Fig 3 | figure | Memory survival + routing ablation | §4.4 | ⚠️ 需 restyle |
| Tab 1 | table* | Main results (12 tasks × 6 methods) | §4.2 | ✅ 已有 |
| Tab 2 | table | Write-rule ablation | §4.3 | ❌ 需创建 |

**附录图表**：
| 编号 | 类型 | 内容 | 状态 |
|---|---|---|---|
| Tab A1 | table | Hyperparameters | ✅ 已有 |
| Tab A2 | table* | 16-task full results | ✅ 可从 table_eval_last3.md 生成 |
| Tab A3 | table | NoLSTM 12-task ablation | ✅ 可从数据生成 |
| Fig A1 | figure | LSTM checkpoint probes | ✅ 已有 |
| Fig A2 | figure | Intercept tracking probe | ✅ 已有 |
