# 行动清单 (Action Items)

## 紧急 (写正文前必须解决)

### A1. 指标矛盾：主表用 success_once 但 TODO 说 success_at_end
- **现状**：`paper_main.py` 用 `success_once`；论文 §4.1 TODO 提到报告 `success_at_end`
- **建议**：保持 success_once 作为主表指标（数值高、更稳定），在 §4.1 解释两个指标的区别，附录提供 success_at_end 补充表
- **行动**：决定后更新 TODO

### A2. ShellGamePush 种子数：图 caption 说 5 seeds，但主表仅用 3
- **现状**：paper_main.py 硬编码 `SEEDS = ("33","42","99")`，过滤掉 seed100/123
- **建议**：
  - 方案 A：图和表统一用 3 seeds（修 caption）
  - 方案 B：图和表统一用 5 seeds（修 paper_main.py）
  - 方案 C：图用 5 seeds（视觉更平滑），表用 3 seeds（表脚注说明）
- **行动**：选方案

### A3. Shape 任务也用了 CRES——需要解释
- **现状**：所有 Remember 任务（包括 Shape-only）都用 `--color-aug`
- **问题**：Shape 任务不需要颜色，为何开 CRES？
- **建议**：改为"所有 Remember 任务统一用 CRES+CAPS"（简单统一协议）或补充解释 CRES 对 Shape 无害
- **行动**：在 §4.1 说明统一协议的原因

## 高优先级 (论文质量关键)

### B1. Write-rule 消融表（Table 2）创建
- 数据：RC5 surprise/random/FIFO 已有
- 问题：FIFO seed99 仅 3 evals（需重跑或标注）
- 行动：创建 LaTeX 表格，seed99 完成前先用 2-seed 或标注 *

### B2. Figure 3 restyle
- 现有：`memory_survival_rc5.png`（matplotlib 原始）
- 需要：AAAI 格式，合适字号，可能合并 NoLSTM bar chart
- 行动：用 `scientific-figure-making` skill 重做

### B3. 正文分析段落（§4.2）
- 需写内容：整体优势、任务族差异、RC9/InterceptFast 失败分析
- 行动：用叙事骨架 `06_experiment_narrative_zh.md` 作为中文输入翻译

### B4. §4.1 基线/协议段落
- 当前状态：TODO 占位
- 行动：写正文（英文 ~200 words）

## 中优先级 (增强论文但非必须)

### C1. [实验] RC5 CRES-only 消融（aaai_final config）
- 预估：3 GPU-hours (3 seeds × 7M steps)
- 目的：干净地证明 CRES 贡献（当前旧数据仅 14 evals）
- 行动：准备 slurm，提交

### C2. [实验] Write-rule FIFO seed99 重跑
- 预估：1 GPU-hour
- 原因：当前仅 3 evals，数据不可用
- 行动：检查 slurm 状态，重新提交

### C3. [实验] Write-rule RS5 补完
- 预估：6 GPU-hours (2 variants × 3 seeds)
- 目的：第二个任务验证 write-rule 消融
- 行动：仅在 RC5 结论稳固后考虑

### C4. NoLSTM 消融表
- 数据：✅ 全 10 任务 × 3 seeds（缺 ShellGame 2 任务）
- 发现：**LSTM 对 Remember/Color 任务有害！**（RC3 +22.6pp, RC5 +10.3pp）
- 建议：附录完整表 + 主文讨论 Intercept 下降和 Remember 上升
- 行动：生成 LaTeX 表格

### C5. 附录 success_at_end 表
- 数据：✅ 已有 table_eval_last3.md
- 行动：转 LaTeX

## 低优先级 (收尾阶段)

### D1. Fig 1 caption
- 当前：`\todo{Caption for architecture figure.}`
- 行动：写 caption（需先看 figure 内容）

### D2. 附录任务描述补完
- 当前：缺 RememberShape, ShellGame, InterceptGrab 描述
- 行动：从 MIKASA-Robo 文档翻译

### D3. 探针方法论附录
- 当前：TODO 占位
- 行动：写探针协议（需参考 analysis/rnn_diagnosis/*.py）

### D4. Related Work 补完
- 当前：多个 TODO
- 行动：补引用和定位

## 关键发现总结（写正文时注意）

1. **NoLSTM 在 Color 任务上居然更好**——这是一个值得讨论的发现。叙事方向：LSTM 在 color 任务上引入干扰，因为运动信号对线索记忆无用且占用了学习容量。这反而强化了"两条路径服务不同功能"的论点。

2. **主表 success_once 是 last-3 eval mean**——NOT peak, NOT final single eval。这个定义比 peak 更保守（避免 cherry-pick），但比 single-eval 更稳定。

3. **FIFO 和 Random write rule 表现几乎一样** (19.1% vs 21.9%)——说明 "什么时候写入" 和 "写入什么" 一样不重要，重要的是 "根据内存相关性选择性写入"。

4. **FFM 和 SHM 作为高级记忆基线整体弱于简单 GRU**——FFM avg 15.1%, SHM 缺失太多无法比较。这说明在感知匹配条件下，简单循环网络已经是强基线。

5. **ShellGamePush 最大方差 (std=31.9%)**——论文需要诚实讨论。可能原因：力控制的高敏感性 + 遮挡后重识别的二元性（要么完全正确要么完全错误）。
