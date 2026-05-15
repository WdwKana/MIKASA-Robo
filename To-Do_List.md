# To-Do List

项目代号：**EBM-Robo / Episodic Buffer Memory for Partial-Observability Robot RL**
目标：指导 Coding Agent 从零完成实验实现与论文写作；所有中间产物按目录落盘以便后续自动写作与复现实验。

---

## 0. 使用约束与总原则

- [ ] 本项目定位为**记忆机制替换**（替代 GRU/LSTM），不得修改 PPO 算法本体或 mikasa_robo_suite 环境。
- [ ] 视觉编码器在主版本中**始终 frozen**；禁止反传到 ViT。
- [ ] 写入决策（push/evict）**禁止参与梯度反传**；保持其无参数性以确保训练稳定。
- [ ] 所有训练 run 必须保存：config YAML, seed, hardware info, training log, metrics JSON, 可复现脚本入口。
- [ ] 所有 ablation 必须使用与主表完全相同的 seed 集合与 env steps。
- [ ] 失败的硬阈值（Specification § 7.1）触发后，立即暂停，向人类汇报。
- [ ] 不修改、不扩展现有 `ppo_memtasks_gru.py` / `ppo_memtasks_lstm.py` / `ppo_memtasks_cvae.py`——这些是基线，对比时直接调用。

---

## 1. 仓库初始化与目录结构

- [ ] 在项目根 `/zfsstore/user/s4176650/MIKASA-Robo/` 下建立 EBM 专属子目录。
  - [ ] `baselines/ppo/modules/` —— EBM 各子模块（如不存在则创建）。
  - [ ] `baselines/ppo/configs/ebm/` —— 各任务 + 各 ablation 行的 YAML。
  - [ ] `final_results/ebm/{task}/{config_id}/seed_{N}/` —— 输出落盘。
  - [ ] `analysis/ebm/` —— 离线诊断脚本（saliency 可视化、linear probe、memory probe）。
  - [ ] `writing/ebm/` —— 论文写作目录（figures, tables, sections）。
- [ ] 在仓库根创建 `EBM_README.md`，简要说明 EBM 与原 baseline 的关系，并指向 `Specification.md`。

---

## 2. 环境与依赖

- [ ] 复用现有 `mikasa` conda env（位于 `/home/s4176650/.conda/envs/mikasa/`）。
- [ ] 安装新增依赖：
  - [ ] `transformers >= 4.40` （DINOv2 加载）
  - [ ] `timm >= 0.9` （ViT-Tiny 备选骨架）
  - [ ] 验证 `torch`, `gymnasium`, `mani_skill` 现有版本不需调整。
- [ ] 在 `analysis/ebm/env_check.py` 中写一个 30 行的脚本验证：能加载 `facebook/dinov2-small`，能在 GPU 上 forward 一张 128×128×3，输出维度正确。

---

## 3. Day 1 — Saliency Sanity Check（H1 假设验证）— **已完成，H1 证伪**

- [x] `analysis/ebm/visualize_dino_saliency.py` —— DINOv2 CLS attention on render_camera：失败（arm dominate）
- [x] `analysis/ebm/visualize_dino_saliency_armmask.py` —— HSV 抠 arm 后再看：仍被结构性边缘干扰
- [x] `analysis/ebm/visualize_dino_saliency_real.py` —— 修正 crop bug，改在真正的 base_camera 与 hand_camera 上跑：base 仍被 arm dominate；hand 显著好但被 gripper finger dominate
- [x] `analysis/ebm/visualize_dino_saliency_dual.py` —— 双目联合 top-K=8：5/5 帧仍未通过
- [x] **D1 结论**：H1 朴素版（DINOv2 CLS attention 即用作 saliency）证伪。但 hand_camera 仍 valuable（信息更丰富），决定切换双目 + 训 saliency head 替代 CLS attention
- [x] **副作用：决定切换到双目相机协议**（base + hand），重写 wrapper 在 [baselines/ppo/utils_ebm.py](baselines/ppo/utils_ebm.py)

---

## 3.5 Path A — 训练 Saliency Head（H1' 假设验证）— **已验证可行**

- [x] `analysis/ebm/path_a_collect_data.py` —— 用随机策略在 RememberColor9 上 rollout 20 episodes (~1200 frames)，存 (base_rgb, hand_rgb, target_color_idx)。saved → [analysis/ebm/path_a_data/](analysis/ebm/path_a_data/)
- [x] `analysis/ebm/path_a_train_head.py` (v1, MIL) —— 9-way 颜色分类经 attention pool；val_acc 17%，过拟合训练，small-target saliency 噪声大；**信号太间接，废弃**
- [x] `analysis/ebm/path_a_train_head_v2.py` (v2, per-patch BCE) —— 用 RGB 颜色匹配生成 per-patch 二分类标签；val_acc 0.97；**5/5 测试帧通过 top-K 命中评估**
  - 关键证据：InterceptFast step010 saliency 形成同心环 perfectly aligned with target；RememberColor9 step005 空桌 max sal=-4.84（自发预测无目标）
- [x] **D1' 结论**：DINOv2 patch features 本身编码足够信息，2 层 MLP head 可干净抽出。**Path A 验证可行**。
- [ ] **后续：监督来源的任务泛化**（RememberShape / ShellGame / TakeItBack 不能用颜色匹配）：
  - [ ] 写 `analysis/ebm/path_a_pose_label.py`：从 sim 拿对象 6D pose，用相机内/外参投影到图像空间，per-patch 二分类标签
  - [ ] 在 ShellGameTouch 上重复 Path A v2 流程，验证 pose-projection 监督也能产生干净 saliency
  - [ ] 形成"label generator per task family"的小工具集

---

## 4. Phase 0 — Linear Probe 诊断（H2 假设验证）

**目标**：在不训练任何 RL 的前提下，验证 frozen DINOv2 patch token 是否比项目现有 CNN 编码更可线性预测 oracle。

- [ ] 写脚本 `analysis/ebm/linear_probe.py`：
  - [ ] 离线收集数据：用现有 oracle policy / random policy 在 RememberColor9, ShellGameTouch, TakeItBack 各跑 100 episodes，保存 (image, joints, oracle_info)。
  - [ ] 对每个任务训两个 linear probe：
    - [ ] `cnn_features → oracle`：从项目现有 CNN 编码器输出（mean-pooled）训 linear。
    - [ ] `dinov2_features → oracle`：从 frozen DINOv2 输出（mean-pooled patches）训 linear。
  - [ ] 报告 R² / accuracy 对比。
- [ ] 重点检查：
  - [ ] RememberColor9 在 step ≤ 5（写入窗口）：DINOv2 → true_color_indices 的 top-1 acc。
  - [ ] ShellGameTouch 在 step ≤ 5：DINOv2 → cup_with_ball_number 的 top-1 acc。
- [ ] **决策点 D2**：
  - [ ] 若 DINOv2 probe 显著优于 CNN（≥ 15 pts top-1 acc 提升）→ 通过，进入 Phase 1。
  - [ ] 若两者相近 → 视觉编码不是主要瓶颈，主线方法可能不成立，停下来重新评估（参考 Specification § 7.1）。

---

## 5. Phase 1 — MVP 实现与首批训练（H3 假设验证）

**目标**：实现 EBM-Robo 主方法，在 RememberColor9 + ShellGameTouch 上跑通 PPO，对比 GRU baseline。

### 5.1 模块实现

- [ ] `baselines/ppo/modules/frozen_vit.py`：包装 `facebook/dinov2-small`，forward 输出 patch tokens + CLS attention，强制 `no_grad`。
- [ ] `baselines/ppo/modules/write_scorer.py`：实现 saliency + novelty 双过滤，返回 push 决定与候选 token 列表。
- [ ] `baselines/ppo/modules/episodic_buffer.py`：实现 K-V buffer 类，支持 push/evict/get_KV/reset，存储 timestamp 与 saliency。
- [ ] `baselines/ppo/modules/memory_reader.py`：单层 cross-attention，含 timestamp embedding 与 saliency-bias logits。
- [ ] `baselines/ppo/modules/ebm.py`：把上述四个组合成单一 `EBMMemory` nn.Module。
- [ ] `baselines/ppo/ppo_memtasks_ebm.py`：从 `ppo_memtasks_gru.py` 派生，唯一差异是 actor/critic 的视觉+记忆部分用 `EBMMemory` 替换 GRU。

### 5.2 单元测试

- [ ] `tests/test_episodic_buffer.py`：验证 push 在显著性低帧不触发；验证新颖性过滤拒绝重复；验证 eviction 优先级正确。
- [ ] `tests/test_memory_reader.py`：随机张量 forward，输出形状对，padding mask 生效。
- [ ] `tests/test_ebm_full_step.py`：mock obs，跑一遍 EBM forward，确认 retrieved_t 维度正确。

### 5.3 训练（行 1, 2, 3, 7）

- [ ] 准备 4 份 SLURM 配置（每行 × 2 任务 × 3 seeds 共 24 jobs，但分批提交）：
  - [ ] 行 1：`config_gru_baseline.yaml`（沿用现有 `ppo_memtasks_gru.py`，无需改动，但用相同的 logger 把结果写到 `final_results/ebm/...`）。
  - [ ] 行 2：`config_dinov2_gru.yaml`。
  - [ ] 行 3：`config_dinov2_ebm.yaml` —— 主方法。
  - [ ] 行 7：`config_dinov2_ebm_priv.yaml` —— 主方法 + 特权 critic。
- [ ] RememberColor9 + ShellGameTouch，每行 × 每任务 × 3 seeds，共 24 jobs。
- [ ] env steps：10M（Phase 1 节流）。

### 5.4 监控与诊断

- [ ] 训练期 wandb / tensorboard 至少记录：episode_return, success_rate, value_loss, policy_loss, entropy, kl, ev_explained, **buffer_size_mean, buffer_writes_per_episode**, **memory_probe_acc** (训练中每 100k steps 跑一次)。
- [ ] 训练完成后跑 `analysis/ebm/eval_memory_probe.py`：对每个 checkpoint 用 frozen linear probe 测 retrieved_t → oracle 的 top-1 acc（faithfulness 指标）。

### 5.5 决策点 D3

- [ ] 在 RememberColor9 + ShellGameTouch 上：
  - [ ] 主方法（行 3）SR ≥ GRU baseline（行 1）+ 10 pts，且 memory probe acc ≥ 0.85（在写入窗口结束后稳定）→ 通过，进入 Phase 2。
  - [ ] SR 提升不足但 memory probe 高 → 改进策略 head 或 query 构造；不要扩任务。
  - [ ] memory probe 也低 → 检查 saliency 与 novelty 过滤；可能需要训 saliency head（回到 Day 1 回退方案）。
  - [ ] SR 在 1.10× 以下且 probe 低 → 触发 Specification § 7.1 硬阈值，停下来。

---

## 6. Phase 2 — Ablation 与扩展（H4, H5, H6）

**目标**：扩到 4 个 MVP 任务（加 TakeItBack, ChainOfColors7），完成行 4–8 的 ablation。

- [ ] 添加 TakeItBack-v0, ChainOfColors7-v0 配置；写 task-specific oracle 字段处理（特权 critic 用）。
- [ ] 实现行 4（saliency-only）、行 5（novelty-only）、行 6（no filter）：在 `write_scorer.py` 中加 mode flag。
- [ ] 实现行 8（EBM-Hybrid）：在 `ebm.py` 中加可选 short-term GRU 路径。
- [ ] 提交所有 jobs：4 任务 × 行 4–8（5 行）× 3 seeds = 60 jobs，分批提交。
- [ ] env steps：30M（Phase 2 中度）。
- [ ] 写 `analysis/ebm/make_main_table.py`：自动从 `final_results/ebm/` 汇总成主表 LaTeX。
- [ ] 写 `analysis/ebm/make_ablation_table.py`：自动汇总 ablation 表。
- [ ] 决策点 D4：
  - [ ] 行 6（no filter）SR < 行 3 → H4 验证（filter 是 load-bearing），论文 story 成立。
  - [ ] 行 6 SR ≈ 行 3 → H4 失败，story 必须重写为"buffer 本身有效，filter 不重要"，重新评估贡献。

---

## 7. Phase 3 — 全任务扩展与论文级实验

**目标**：扩到 8–12 任务覆盖全部 4 类记忆能力，完成附录 ablation。

- [ ] 增加任务：InterceptMedium, SeqOfColors, RememberShape3, RememberShapeAndColor3x2, BunchOfColors3, RotateLenientPos。
- [ ] 主方法（行 3, 7）+ GRU baseline（行 1）：每任务 × 3 seeds（核心）+ 2 seeds（扩充至 5 seeds 用于附录）。
- [ ] 附录 ablation：
  - [ ] ViT 替换为 ViT-Tiny（轻量化版本）。
  - [ ] Buffer L 大小扫描：32 / 64 / 128。
  - [ ] τ_n 新颖性阈值扫描：0.90 / 0.95 / 0.98。
  - [ ] 是否 timestamp embedding（消融对 ChainOfColors / SeqOfColors 的影响）。
  - [ ] **双目相机扩展**（robustness 检查，非主线）：在 RememberColor9 + ShellGameTouch 两个任务上，加 gripper camera 作第二路 frozen ViT 编码，token 合并入同一 buffer（共享显著性/新颖性过滤），对比单目主版本是否有额外提升。
- [ ] env steps：100M（Phase 3 充分训练）。
- [ ] 把所有结果汇总到 `writing/ebm/tables/` 下的 LaTeX 文件。

---

## 8. 论文写作

- [ ] 调用 `ml-paper-writing` skill。
- [ ] 章节按 Specification § 11 组织。
- [ ] Figure 必有：
  - [ ] System diagram（按 Specification § 3.1 数据流图绘制）。
  - [ ] Day 1 saliency map 对比图（DINOv2 vs CNN 编码 attention）。
  - [ ] 主结果柱状图（4 任务 × 3 方法）。
  - [ ] Memory probe 时序图：训练过程中 retrieved_t → oracle 的 acc 曲线。
  - [ ] Buffer trace 可视化：单 episode 内 buffer 写入时刻与 SR 对应。
- [ ] Table 必有：
  - [ ] 主结果表（Specification § 5.4 行 1, 2, 3, 7）。
  - [ ] Ablation 表（行 4, 5, 6, 8）。
  - [ ] 全任务扩展表（Phase 3）。
  - [ ] 与 belief baselines（CVAE, VAE）的对照表（附录）。
- [ ] Limitations 章节必须诚实写入：
  - [ ] 依赖 frozen ViT 在合成图像上的前景敏感性（H1）。
  - [ ] Buffer 上限对超长 episode 任务的限制。
  - [ ] 顺序任务中 timestamp embedding 的设计敏感性。

---

## 9. 复现性与代码发布

- [ ] 在 `EBM_README.md` 中提供：
  - [ ] 一行命令复现 Phase 1 主结果。
  - [ ] checkpoint 下载链接。
  - [ ] 评测脚本入口。
- [ ] 提交 PR 合并到 main 分支前，确认所有 SLURM 脚本路径已更新（避免遗留 `/local/` 或 `/home/` 引用）。
- [ ] 论文接受后，发布 anonymized code。

---

## 10. 与人类的交互节点

以下节点必须暂停，向人类汇报：

1. **Day 1 决策点 D1 之后**：报告 saliency map 评估结果，确认是进入 Phase 0 还是回退到自监督 saliency head。
2. **Phase 0 决策点 D2 之后**：报告 linear probe 对比，确认是进入 Phase 1 还是停下来。
3. **Phase 1 决策点 D3 之后**：报告首批 SR + memory probe 对比，决定 Phase 2 范围。
4. **Phase 2 决策点 D4 之后**：报告 filter ablation 结果，确认论文 story 是否成立。
5. **任何触发 Specification § 7.1 硬阈值的情况**：立即停下来。

其他时刻可全自主推进。

---

## 11. 文件清单（终态）

完成 Phase 3 时，以下文件应全部存在并可运行：

- `Specification.md` ✓ (本规范)
- `To-Do_List.md` ✓ (本文件)
- `EBM_README.md`
- `baselines/ppo/ppo_memtasks_ebm.py`
- `baselines/ppo/modules/{frozen_vit,write_scorer,episodic_buffer,memory_reader,ebm}.py`
- `baselines/ppo/configs/ebm/*.yaml` (≥ 8 配置)
- `analysis/ebm/{visualize_dino_saliency,linear_probe,eval_memory_probe,make_main_table,make_ablation_table}.py`
- `tests/test_{episodic_buffer,memory_reader,ebm_full_step}.py`
- `final_results/ebm/{task}/{config_id}/seed_{N}/` (汇总结果)
- `writing/ebm/{figures,tables,sections}/` (论文素材)
