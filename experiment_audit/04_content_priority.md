# 内容优先级分类 (Content Priority Classification)

## A 级：必须放主文（Main Paper Must-Have）

| 内容 | 预估空间 | 资产状态 | 说明 |
|---|---|---|---|
| **主表**：6方法 × 12任务 + Average | Table* (~0.5 page) | ✅ 已有 | success_once, last-3 eval mean |
| **训练曲线**：4×3 panel 图 | Figure* (~0.5 page) | ✅ 已有 | main_success_12.pdf |
| **主结果分析**：整体优势 + 两个失败点讨论 | 1 paragraph (~100 words) | ❌ 需写 | RC9/InterceptFast 分析 |
| **Write-rule 消融**：surprise vs random/FIFO | Small table (~0.15 page) | ⚠️ RC5 基本可用 | 核心论断的因果检验 |
| **记忆存活探针图**：buffer vs LSTM 保持力 | Figure (~0.3 page) | ✅ 已有 (需 restyle) | 核心 insight 的直观证据 |
| **路由消融**：NoLSTM 结果 | 主文段落 or 小表 | ✅ 数据齐全 | 互补分工的证据 |

## B 级：尽量放主文，空间不够移附录

| 内容 | 预估空间 | 资产状态 | 说明 |
|---|---|---|---|
| ShellGamePush 高方差讨论 | 2-3 sentences | ✅ 可从数据导出 | std=31.9%, 最大方差 |
| 记忆探针 per-task 分析 | 段落 | ✅ 已有 | buffer 和 LSTM 在不同任务类型的角色 |
| CRES 消融（附方法说明） | 小表 | ⚠️ 旧版数据 | 如果补了 P0 实验可升 A |
| 与 ELMUR 的比较定位 | 2-3 sentences | ✅ 引用数据 | RC5 0.19, RC9 0.23 → STRM 大幅超越 |

## C 级：附录（Appendix）

| 内容 | 预估空间 | 资产状态 | 说明 |
|---|---|---|---|
| 16-task 完整表（含 Intercept Slow/Med/Grab） | Table (~0.4 page) | ✅ 已有 | table_eval_last3.md |
| success_at_end 补充表 | Table (~0.3 page) | ✅ 已有 | 同一 table |
| Return 补充表 | Table (~0.3 page) | ✅ 已有 | 同一 table |
| NoLSTM 完整 12-task 表 | Table (~0.3 page) | ✅ 可生成 | 已有所有数据 |
| CAPS 说明 | 段落 | ✅ 可写 | 已有 memory 记录 |
| Intercept 追踪探针 | Figure | ✅ 已有 | track_srbtr_ifast |
| LSTM 检查点探针序列 | Figure | ✅ 已有 | exp3 summary |
| 动作 t-SNE | Figure | ✅ 已有 | latent_action_structure |

## D 级：砍掉（Drop / Future Work）

| 内容 | 原因 |
|---|---|
| MV-SPLIT / CCAT 变体比较 | 不在主方法中，是开发过程尝试 |
| CAPS λ 扫描 | 单种子，不足以做结论 |
| 旧版 _summary.csv / _summary_per_task.csv | 过时数据，用 paper_main.py 取代 |
| InterceptMedium/Slow 详细分析 | 主表只含 Fast 变体 |

## 空间预算 (AAAI 8-page Limit)

| Section | 预估列数 | 说明 |
|---|---|---|
| 已有 Method | ~1.0 col | 压缩后 |
| Figure 1 (architecture) | ~0.8 col | 半页 |
| Experiments intro + protocol | ~0.5 col | §4.1 |
| Main table + curves | ~1.0 col | Table* + Figure* |
| Main results analysis | ~0.3 col | 正文讨论 |
| Write-rule ablation | ~0.3 col | 小表 + 段落 |
| Probes + routing | ~0.5 col | 图 + 段落 |
| Related Work | ~0.8 col | |
| Conclusion | ~0.3 col | |
| **总计** | **~5.5 col** | **留有 ~0.5 col 余量** |
