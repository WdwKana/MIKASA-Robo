# 实验方案核实 (Protocol Verification)

## 1. 指标定义

| 问题 | 答案 | 来源 |
|---|---|---|
| 主表指标是什么？ | `success_once`（非 `success_at_end`） | `paper_main.py` line 1, 89: `y = float(r["success_once"])` |
| 最终值怎么算的？ | 取最后 3 次 eval 的均值 | `paper_main.py` line 95: `e3 = s[1][-3:].mean()` |
| `success_once` 精确含义 | 在 episode 内任意时刻达成过成功判定即计 1 | MIKASA-Robo 定义 |
| `success_at_end` 精确含义 | 在 episode 结束瞬间仍保持成功状态 | MIKASA-Robo 定义 |
| 论文当前使用哪个？ | **主表用 `success_once`**，但论文正文 TODO 写 "report `success_at_end` with chance marked" | **矛盾！需决定** |

**关键矛盾**：论文 §Experiments TODO (line 324-325) 提到要报告 `success_at_end`，但 `paper_main.py` 和 `main_table_12.md` 实际使用的是 `success_once`。建议：保持 `success_once` 作为主表指标（因为更稳定、数值更高），但在正文说明两个指标的区别，并在 Appendix 提供 `success_at_end` 补充表。

## 2. 训练/评估协议

| 问题 | 答案 | 来源 |
|---|---|---|
| 总训练步数 | 7M env steps（所有任务统一） | `srbtr_l4main.slurm`: `--total-timesteps=7_000_000` |
| 并行环境数 | 256 train, 16 eval | `--num-envs=256 --num-eval-envs=16` |
| 评估频率（Remember） | 每 24 次 update（≈368K steps） | `--eval-freq=24 --num-steps=60` |
| 评估频率（Intercept） | 每 16 次 update（≈368K steps） | `remembershape_capsonly.slurm` vs Intercept config |
| 评估 episode 长度 | Remember: 720 steps, Intercept: 540 steps | `--num-eval-steps=720/540` |
| 种子 | 33, 42, 99（主表统一3种子） | `paper_main.py` line 64: `SEEDS = ("33","42","99")` |
| ShellGamePush 额外种子 | 100, 123（5种子） | 检查点目录含 `seed100`, `seed123` |
| 主表是否包含额外种子？ | **否**，`paper_main.py` 过滤为仅 33/42/99 | `if seed not in SEEDS: continue` |

**注意**：论文图 caption (line 341) 说 "5 for ShellGamePush"，但主表代码实际只取3个种子。需核实：是图用5种子、表用3种子？还是 caption 需修正？

## 3. 任务配置

| 任务族 | CRES | CAPS | Config 名 | 来源 |
|---|---|---|---|---|
| Remember 全系列 | YES | YES (λ=0.15) | `crescaps` | `srbtr_l4main.slurm` |
| ShellGame | YES | YES | `cres_caps` | `paper_main.py` TASKS 配置 |
| Intercept 全系列 | NO | NO | `plain` | `paper_main.py` TASKS 配置 |

**确认**：CRES 仅用于颜色相关任务（RememberColor, RememberShapeAndColor, ShellGame），Shape-only 任务也用了 CRES+CAPS。这符合 `remembershape_capsonly.slurm` 的注释 "CAPS-only (NO CRES)"——但主表配置用的是 `crescaps`，不是 `caps`。

**再次核实**：Shape 任务的 `crescaps` config 实际有 CRES 吗？
→ 查看 `srbtr_l4main.slurm` 中 RememberShape 的配置：所有 Remember 任务统一用 `--color-aug --caps-lambda-t=0.15`。所以 **Shape 任务也开了 CRES**。论文需明确这一点，或者解释为什么 Shape 也开了（因为 ShapeAndColor 需要？或是统一处理？）。

## 4. 基线配置

| 基线 | 实现来源 | 相同 DINOv2 | 相同 CRES/CAPS | 种子数 |
|---|---|---|---|---|
| MLP | `ppo_memtasks_dinov2_mlp_caps.py` | YES | YES (同规则) | 3 |
| GRU | `ppo_memtasks_dinov2_gru_caps.py` | YES | YES | 3 |
| LSTM | `ppo_memtasks_dinov2_lstm_caps.py` | YES | YES | 3 |
| FFM | `ppo_memtasks_dinov2_ffm.py` | YES | YES | 3 |
| SHM | `ppo_memtasks_dinov2_shm.py` | YES | YES | 3 |

**确认**：所有基线共享同一冻结 DINOv2 编码器和同一 CRES/CAPS 配置规则。感知匹配（perception-matched）。

## 5. FFM/SHM 缺失项

| 基线 | 完整覆盖 12 任务？ | 缺失 |
|---|---|---|
| FFM | 10/12 | 缺 ShellGameTouch, ShellGamePush（有数据但可能未运行）→ 实际 `ffm_shm_full.slurm` 不含 ShellGame |
| SHM | 8/12 | 缺 ShapeAndColor3x2, ShapeAndColor3x3, InterceptFast, InterceptGrabFast → 部分标记 diverged |

**注意**：SHM ShellGame 标 `†`（diverged），主表 SHM Average 显示 "—"。论文需说明 SHM 为何缺失。实际原因：SHM 在 ShellGame 上 diverge，在 SAC 复合任务上未运行。
