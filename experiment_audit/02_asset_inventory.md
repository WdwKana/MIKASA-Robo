# 实验资产清单 (Experiment Asset Inventory)

## A. 主表结果 (Main Results)

| 资产 | 位置 | 状态 | 说明 |
|---|---|---|---|
| 12-task 主表 (markdown) | `final_results/paper_figures/main_table_12.md` | ✅ 完整 | 6方法×12任务, success_once, last-3 eval mean |
| 12-task 主表 (LaTeX) | `final_results/paper_figures/main_table_12.tex` | ✅ 完整 | 同上 |
| 12-task 训练曲线 (PDF) | `final_results/paper_figures/main_success_12.pdf` | ✅ 完整 | 4×3 panel, mean±std |
| 16-task 完整表 (3指标) | `final_results/paper_figures/table_eval_last3.md` | ✅ 完整 | success_once, success_at_end, return |
| 16-task max-over-evals | `final_results/paper_figures/table_max_over_evals.md` | ✅ 完整 | peak metric 参考 |
| 生成脚本 | `analysis/paper_main.py` | ✅ 完整 | 可再生成 |
| 16-task 曲线 (3指标) | `final_results/paper_figures/curves_*.pdf` | ✅ 完整 | `paper_figures.py` 生成 |

## B. Write-Rule 消融 (Write-Rule Ablation)

| 变体 | 任务 | 种子 | 状态 | 数值 (success_once, last-3 mean) |
|---|---|---|---|---|
| w-random (RC5) | RememberColor5 | 33/42/99 | ✅ 3/3 完成 | .207/.229/.222 → mean 21.9% |
| w-fifo (RC5) | RememberColor5 | 33/42/99 | ⚠️ seed99 仅3 evals | .198/.267/.109 → mean 19.1% (seed99 不可靠) |
| w-random (RS5) | RememberShape5 | 33 only | ⚠️ 仅1种子, 12 evals | .250 |
| w-fifo (RS5) | RememberShape5 | — | ❌ 未运行 | — |
| STRM 对照 (RC5) | RememberColor5 | 33/42/99 | ✅ | 55.3±19.7% |

**问题**：
1. RC5 w-fifo seed99 只有 3 次 eval（可能还在跑或被 kill），数值不可靠
2. RS5 仅 wrandom seed33 有数据（12 evals = 还没跑完）
3. 缺少 w-stride 变体（论文 TODO 提到 stride）
4. **建议**：仅报告 RC5 消融（wrandom 3种子完整），RS5 视为补充或放弃

## C. 组件消融 (Component Ablations)

### C1. NoLSTM 消融（去除循环路径）

| 任务 | 种子 | Evals | 均值 success_once | STRM 完整版 | 差距 |
|---|---|---|---|---|---|
| RememberColor3 | 3 | 19 each | 72.7% | 50.1% | **NoLSTM +22.6pp！** |
| RememberColor5 | 3 | 19 each | 65.6% | 55.3% | **NoLSTM +10.3pp！** |
| RememberColor9 | 3 | 19 each | 18.3% | 18.1% | 持平 (+0.2) |
| RememberShape3 | 3 | 19 each | 44.4% | 46.2% | 略低 (-1.8) |
| RememberShape5 | 3 | 19 each | 45.6% | 52.7% | 低 (-7.1pp) |
| RememberShape9 | 3 | 19 each | 25.0% | 26.8% | 低 (-1.8) |
| ShapeAndColor3x2 | 3 | 19 each | 35.2% | 39.9% | 低 (-4.7pp) |
| ShapeAndColor3x3 | 3 | 19 each | 20.8% | 19.6% | **NoLSTM +1.2pp** |
| InterceptFast | 3 | 19 each | 69.1% | 80.9% | **低 -11.8pp** |
| InterceptGrabFast | 3 | 19 each | 76.0% | 89.2% | **低 -13.2pp** |
| **10-task Average** | — | — | **47.3%** | **47.9%** | **-0.6pp (几乎相同!)** |

**关键发现**：
- **10 任务平均几乎相同** (47.3% vs 47.9%)——LSTM 的正负效应近乎抵消
- LSTM 对 Intercept 任务至关重要（动态追踪），去除后下降 11-13pp
- LSTM 对 Color 任务有负面影响（NoLSTM 在 RC3 上 +22.6pp, RC5 上 +10.3pp！）
- **叙事方向**：两条路径的价值在于*任务适应性*——buffer 专精线索存储，LSTM 专精运动追踪。在纯记忆任务上 LSTM 是噪声源；在动态任务上是性能保障
- 缺 ShellGame 2 任务的 NoLSTM 数据（ShellGameTouch/Push 未运行 NoLSTM）

### C2. CRES 消融

| 变体 | 任务 | 种子 | 均值 | 说明 |
|---|---|---|---|---|
| srbtr (no CRES, no CAPS) | RC5 | 3 (14 evals) | 31.9% | 旧版，14 evals |
| srbtrcaps (CAPS only) | RC5 | 3 (19 evals) | 24.2% | **比 no-CAPS 还低？** |
| srbtrcres (CRES only) | RC5 | 3 (14 evals) | 56.8% | 接近完整版 |
| srbtr-crescaps (full) | RC5 | 3 (19 evals) | 55.3% | 主表 |

**关键发现**：
- CRES 是 RC5 性能的主要驱动力（56.8% vs 31.9%）
- CAPS 在 RC5 上看不到明显帮助（srbtrcaps 24.2% 甚至低于 bare 31.9%，但 eval 数不同）
- 不同 eval 数（14 vs 19）使得直接比较不完全公平

### C3. MV / CCAT 变体

| 变体 | 任务 | 均值 | 说明 |
|---|---|---|---|
| srbtrmv (MV-SPLIT) | RC5 | 66.8% | 高于 CRES，但方差极大 (38-82%) |
| srbtrccat (concat color) | RC5 | 44.7% | 低于 CRES |

### C4. CAPS λ 扫描

| λ_t | RC5 (seed33, 14 evals) | RS5 (seed33, 各不同) |
|---|---|---|
| 0.0 | 43.1% | — |
| 0.05 | 22.9% | — |
| 0.15 | 28.1% | — |
| 0.5 | 35.8% | — |

**注意**：这些只有 seed33，不足以做表。

## D. 探针分析 (Probes)

| 资产 | 位置 | 状态 | 内容 |
|---|---|---|---|
| exp1 曲线签名 | `analysis/rnn_diagnosis/out/exp1_curves.png` | ✅ | 训练曲线特征分析 |
| exp1 摘要 | `analysis/rnn_diagnosis/out/exp1_summary.csv` | ✅ | takeoff/half/final/plateau 统计 |
| LSTM/GRU/FFM/SRBTR 探针 | `analysis/rnn_diagnosis/out/probe_*_rc5_*` | ✅ | 线索可解码性探针曲线 |
| LSTM 检查点序列探针 | `analysis/rnn_diagnosis/out/exp3_lstm_rc5_s99` | ✅ | 19个 checkpoint 的探针演变 |
| 记忆存活率图 | `analysis/rnn_diagnosis/out/memory_survival_rc5.png` | ✅ | buffer ≥0.90 vs LSTM τ≈8 |
| Intercept 追踪探针 | `analysis/rnn_diagnosis/out/track_*` | ✅ | SRBTR/GRU 在 Intercept 上的追踪 |
| 探针控制 | `analysis/rnn_diagnosis/out/controls_*.json` | ✅ | 基线控制条件 |

## E. 训练曲线 / 可视化

| 资产 | 位置 | 状态 |
|---|---|---|
| 逐任务训练曲线 (旧版) | `final_results/*_success_once_*.png` | ✅ 6任务 |
| 动作结构 t-SNE | `final_results/latent_action_structure/` | ✅ InterceptFast |
| CAPS sweep 图 | `final_results/*lambda_act_sweep*` | ✅ InterceptFast |

## F. 缺失实验 (Missing)

| 实验 | 优先级 | 预估成本 | 说明 |
|---|---|---|---|
| CRES 消融 (aaai_final config) | **P0** | 3 GPU-hours | RC5 srbtr-plain-crescaps = full without CRES |
| Write-rule RS5 完整 | P1 | 6 GPU-hours | w-random/w-fifo × 3 seeds |
| Write-rule RC5 fifo seed99 重跑 | P1 | 1 GPU-hour | 当前仅3 evals |
| ShellGame NoLSTM | P2 | 2 GPU-hours | 2 tasks × 3 seeds |
| CAPS λ 多种子 | P3 | 12 GPU-hours | 4 λ × 3 seeds × RC5 |
| Routing overlap 统计 | P2 | 0 cost (推理) | 给现有检查点加仪器 |
