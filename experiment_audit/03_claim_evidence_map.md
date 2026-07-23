# 论断-证据映射 (Claim–Evidence Map)

## 核心论断 (Core Claims)

### C1: STRM 在 MIKASA-Robo 上显著优于所有基线
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| 主表 12 任务平均 51.4% vs GRU 30.4% | ✅ 已验证 | **强** | 21pp 差距, 10/12 最佳 |
| 3 种子, 统一协议 | ✅ 已验证 | 中 | 3种子偏少但可接受 |
| 曲线图可视化 | ✅ 已有 | 强 | 收敛特征清晰 |
| **弱点**：RC9 输给 GRU (25.6 vs 18.1)，InterceptFast 输给 LSTM (82.8 vs 80.9) | ✅ 已知 | — | 必须在正文诚实讨论 |

### C2: Memory-relative surprise 写入规则是性能的关键驱动力
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| RC5: STRM 55.3% vs w-random 21.9% vs w-fifo 19.1% | ⚠️ 部分 | 中→强 | w-random 3种子完整; w-fifo seed99 不可靠 |
| RS5: 仅 w-random seed33 | ❌ 不足 | 弱 | 需要补跑 |
| 对照：reader/buffer 相同，仅 write rule 不同 | ✅ 设计合理 | 强 | 因果检验干净 |

### C3: 情节缓冲区长期保持线索信息，LSTM 不能
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| Buffer 探针 ≥ 0.90 全 episode | ✅ 已有 | **强** | `memory_survival_rc5.png` |
| LSTM c state τ ≈ 8 步衰减 | ✅ 已有 | **强** | 探针曲线 |
| LSTM 检查点序列：训练降低初始回声记忆 | ✅ 已有 (exp3) | 中 | 需要更清晰图 |
| **GRU 探针** | ✅ 已有 | 补充 | 对比完整 |
| **FFM 探针** | ✅ 已有 | 补充 | 对比完整 |

### C4: 时间路由（LSTM 分支）对动态任务重要
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| NoLSTM Intercept: -11.8pp (InterceptFast), -13.2pp (InterceptGrabFast) | ✅ 完整 | **强** | 12任务×3种子 |
| NoLSTM Remember: 持平或更好 | ✅ 完整 | 反面有趣 | 说明 LSTM 非万能 |
| **故事**：LSTM 专注运动追踪，buffer 专注线索记忆 | ✅ 可构建 | 强叙事 | 互补分工 |

### C5: CRES 颜色残差对颜色任务关键
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| RC5: CRES-only 56.8% vs no-CRES 31.9% | ⚠️ 旧版数据 (14 evals) | 中 | 非 aaai_final config |
| 主表 crescaps = 55.3%，caps-only = 24.2% | ⚠️ 不同 eval 数 | 中 | 需谨慎对比 |
| 论点：DINOv2 本身不色盲（97.5% NCC）但 CRES 是唯一注入途径 | ✅ 已分析 | 补充 | memory 已有记录 |

### C6: CAPS 平滑对 success_at_end 关键
| 证据 | 状态 | 强度 | 备注 |
|---|---|---|---|
| RC5 at_end: CAPS 0.04→0.31 | ✅ 来自 memory | 中 | 非本轮验证 |
| CAPS λ=0.15 vs 0.0 单种子 | ⚠️ 仅 seed33 | 弱 | 不足以做表 |
| CAPS 应用于所有方法 | ✅ 已验证 | 前提 | 不是 unfair advantage |

## 证据充分性判断

| 论断 | 证据充分？ | 能否上表？ | 行动 |
|---|---|---|---|
| C1 主表 | ✅ 充分 | ✅ 已有表 | 补写正文分析 |
| C2 写入规则 | ⚠️ 部分 | ⚠️ 可做小表（RC5 only） | P1: 补 RS5; P1: 重跑 fifo seed99 |
| C3 记忆存活 | ✅ 充分 | ✅ 已有图 | 写正文 + 补 caption |
| C4 路由消融 | ✅ 充分 | ✅ 可做表 | 12任务数据齐全 |
| C5 CRES | ⚠️ 旧数据 | ⚠️ 附录可报告 | P0: 补 aaai_final config |
| C6 CAPS | ❌ 不足 | ❌ | 正文一句话提及即可 |
