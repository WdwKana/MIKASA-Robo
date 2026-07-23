# 实验叙事骨架 (中文)

## §4.1 实验设置

我们在 MIKASA-Robo 基准上进行实验，该基准提供了多种需要长期记忆的机器人操作任务。我们选择 12 个任务用于主实验：

- **线索回忆（Remember）**：RC3/5/9, RS3/5/9, RSAC3×2/3×3——机器人需在线索消失后记住目标物的颜色/形状/组合属性，然后在多选项中做出选择。数字表示候选数量，越大越难。
- **遮挡追踪（ShellGame）**：Touch/Push——目标物被遮挡后混洗，机器人需追踪正确杯子。
- **动态拦截（Intercept）**：GrabFast/Fast——球从随机方向滚来，机器人需预判轨迹并拦截或抓取。

**基线**：所有方法共享同一冻结 DINOv2-S/14 感知编码器。记忆基线包括：MLP（无记忆）、GRU、LSTM、FFM [cite] 和 SHM [cite]。颜色增强（CRES）和动作平滑正则（CAPS, λ_t = 0.15）按任务类型统一分配给所有方法。

**协议**：每个（方法, 任务）组合使用种子 {33, 42, 99} 训练 7M 环境步。评估指标为 success_once（episode 内任意时刻成功），取最后 3 次评估的均值。

---

## §4.2 主要结果

**整体表现。** 表 1 和图 2 展示了 12 个任务上的实验结果。STRM 在平均 success_once 上达到 51.4%，超过最强基线 GRU (30.4%) 21.0 个百分点，在 10/12 个任务上取得最优。

**强势任务分析。** 在 ShellGameTouch 上 STRM 达到 98.1% 的近满分表现，相比 GRU (56.1%) 提升巨大。在多选回忆任务（RS5 52.7% vs LSTM 25.9%）上，episodic buffer 的优势尤其明显——这些任务要求在数百步后仍能精确区分相似选项，正是长期存储的用武之地。

**两个失败案例。** RememberColor9：GRU (25.6%) 胜过 STRM (18.1%)。9-way 颜色辨别可能超出了 CRES 残差的分辨率——DINOv2 patch 在 9 种相近颜色间的区分度有限，而 GRU 通过隐状态的非线性变换可能捕捉到了更细微的颜色差异。InterceptFast：LSTM (82.8%) 略胜 STRM (80.9%)，但差距在标准差 (±14.5) 范围内，说明两者在纯动态任务上表现相当。

**ShellGamePush 方差。** STRM 在该任务上展示最大的种子间方差 (40.0±31.9%)，这可能与 Push 任务的力控制敏感性有关——单次错误的推力即导致 episode 失败，使得不同初始化的策略收敛点差异较大。

---

## §4.3 写入规则消融：记忆的选择性写入重要吗？

这是论文核心研究问题的因果检验。我们保持 STRM 的所有组件不变（buffer 容量、reader、LSTM、CRES、CAPS），仅替换 episodic write rule：
- **Surprise**（完整 STRM）：按 memory-relative surprise 选择写入候选
- **Random**：随机选择 K 个 token 写入
- **FIFO**：按时间顺序写入最新 K 个 token

在 RememberColor5 上：Surprise (55.3%) 相对 Random (21.9%) 和 FIFO (23.3%) 实现约 2.5× 的提升。这说明信息的选择性写入——而非简单地存储所有信息或最新信息——是 episodic buffer 发挥作用的关键前提。Random 和 FIFO 的表现相近，且均远低于 Surprise，表明 write rule 的 discriminative power 而非 recency bias 驱动了性能差异。

---

## §4.4 记忆通路分析：什么被记住了？如何分工？

**探针实验。** 我们在 RememberColor5 上对四种记忆基质进行线索可解码性探针：训练线性分类器从各基质的内部表示预测目标颜色 ID。图 3（左）显示：

- Episodic buffer 在整个 episode 中保持 ≥ 0.90 的可解码精度，说明线索信息被稳定存储
- LSTM cell state 的可解码性在约 8 步后降至偶然水平（τ ≈ 8），反映循环基质的快速遗忘
- GRU 和 FFM 表现类似——循环网络通过训练信号学会的表示不具备长期保持能力

这验证了 STRM 的设计假设：episodic buffer 是唯一能跨越数百步保持特定线索信息的基质。

**路由消融。** 我们移除 LSTM 分支（NoLSTM 变体），在全部 12 个任务上重新训练 3 种子：

- 在 Intercept 任务上，去除 LSTM 导致性能下降 11-13 个百分点（InterceptFast: 69.1% vs 80.9%; InterceptGrabFast: 76.0% vs 89.2%）
- 在 Remember 任务上，去除 LSTM 几乎不影响性能，某些任务上甚至略有提升

这揭示了两条通路的互补分工：episodic buffer 负责长期线索记忆，LSTM 负责实时运动追踪。在不需要运动追踪的记忆任务上，LSTM 的存在可能引入了微量噪声。

---

## 关键数值汇总（供正文使用）

| 论述点 | 核心数值 |
|---|---|
| STRM 平均 | 51.4% |
| GRU 平均 (最强基线) | 30.4% |
| 差距 | +21.0pp |
| 最佳任务数 | 10/12 |
| ShellGameTouch | 98.1% |
| RC9 失败 | STRM 18.1% vs GRU 25.6% |
| InterceptFast 失败 | STRM 80.9% vs LSTM 82.8%（std 内） |
| Write-rule surprise vs random | 55.3% vs 21.9% (×2.5) |
| Buffer 探针精度 | ≥ 0.90 full episode |
| LSTM τ | ≈ 8 steps |
| NoLSTM Intercept 降幅 | -11.8pp / -13.2pp |
| NoLSTM Remember | 持平或略升 |
