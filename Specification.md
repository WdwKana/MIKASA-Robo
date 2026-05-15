# Specification

项目代号：**EBM-Robo: Episodic Buffer Memory for Partial-Observability Robot RL**

版本：v1.0
适用对象：Coding Agent / 复现实验 Agent
目标投稿：NeurIPS / ICML / ICLR 主会，Robot Learning / Memory in RL 方向
论文定位：**RNN-free 记忆机制**，针对部分可观测机器人任务，主张 GRU/LSTM 在视觉记忆 RL 上的失败可以通过显式 episodic buffer + 写入端筛选直接绕开，而不是通过更强的 RNN 来缓解。

---

# 1. 核心定位与最终技术选择

## 1.1 论文主张

本文提出 **EBM-Robo**：一个用于机器人部分可观测记忆任务的 **episodic key-value buffer 记忆模块**，替代主流 PPO 流水线中的 GRU/LSTM。

核心叙事：

1. MIKASA-Robo benchmark 在 state 模式下 100% 可解，在 RGB+joints 模式下 GRU/LSTM 大幅退化；这是被反复观察到但未被诊断清楚的现象。
2. 我们诊断 GRU 失败的两个根本原因：
   - **固定向量压缩**：稀疏、离散的关键视觉事件被压进 d≈512 的 hidden vector，被后续观察持续覆写。
   - **每步强制更新**：vanilla GRU 在长延迟段（如空桌帧）仍然按 z_t 写入 hidden，从而擦除真正应被保留的信息。
3. 这两个失败模式不是通过"更强的 RNN/更深的 GRU"可以修补的——它们是 RNN 范式本身的结构问题。
4. 我们用一个 **frozen 预训练 ViT 视觉编码器 + 显著性 + 新颖性双过滤的写入门 + 学习的 cross-attention 读出** 替代 RNN：buffer 永远不存背景，决策时 cross-attend 检索；没有压缩，没有 hidden 覆写。

## 1.2 最终技术决策

**主方法采用：frozen 预训练 ViT (双目) + 轻量学习 saliency head + 单层 cross-attention 读出。**

- 视觉编码器：**冻结**，主版本用 **DINOv2-S/14**（22M params，成熟轻量）；推理时 `torch.no_grad()`。
- **相机配置：双目（base_camera + hand_camera）**。两路独立喂同一 frozen ViT，patch tokens 沿 token 维度 concat。理由：
  1. 与官方 MIKASA-Robo 协议对齐；
  2. hand_camera 提供顶视近景，目标物在它视野里更显著；
  3. 双视角让 saliency head 的特征池更丰富。
- 写入显著性来源：**学习的 2 层 MLP saliency head**，输入 frozen DINOv2 patch token + xy 位置编码，输出 per-patch importance score。
  - **不**用 DINOv2 CLS-to-patch attention（Day 1 实证：base_cam 被白色臂 dominate，hand_cam 被白色 gripper finger dominate；即使切双目，CLS attention 仍 misdirected）。
  - DINOv2 patch tokens 自身在 feature 层面包含物体信息，head 在 features 上学就能 bypass attention 的偏置。
  - head 通过单任务监督训练，输入离线 oracle rollout 数据，标签由 MIL attention-pool 生成。
- 时间新颖性过滤：**无参数**，与 buffer 现有条目最大余弦相似度，τ_n=0.95。
- 读出端：**单层 cross-attention**，全可微，PPO 端到端训练。
- 不引入 RNN；不修改 PPO 主算法。
- Critic 与 actor 输入完全相同（不使用特权信息）。

## 1.3 模块总览

将系统划分为 5 个组件：

1. **Visual Encoder**（frozen DINOv2-S/14，无训练，双目独立 forward）
2. **Saliency Head**（学习的 2 层 MLP on patch tokens，per-task 单独训练，对 PPO frozen）
3. **Write Filter**（saliency-head 输出 + 与 buffer 的余弦新颖性，无 RNN 端反传）
4. **Episodic Buffer**（有界 K-V 数组，FIFO + saliency-weighted eviction）
5. **Memory Reader**（学习 cross-attention） + **PPO Actor-Critic Heads**（小 MLP）

默认方法名：`EBM-Robo`
精简对照名：`EBM-Robo-NoRNN`（强调 RNN-free）
带短期记忆的扩展版：`EBM-Robo-Hybrid`（buffer + 小 GRU 双层）

---

# 2. 假设与可证伪点

为防止"故事先行、实验对齐"，列出本项目的核心假设与每条假设的可证伪实验。

| ID | 假设 | 证伪实验 | 失败时的行动 | 状态 |
|---|---|---|---|---|
| ~~H1~~ | ~~DINOv2 CLS attention 在 MIKASA 渲染图像上能高亮任务相关物体~~ | Day 1：attention map 可视化 | 回退到 H1' | **已证伪**（base_cam: arm dominate；hand_cam: gripper finger dominate） |
| **H1'** | 在 frozen DINOv2 patch tokens 之上训一个 2 层 MLP saliency head，能将 top-K=8 命中目标物的命中率提升到 ≥ 4/5 测试帧 | Path A：训练 head + 同一组帧的 top-K 命中评估 | 改用 SAM2 / Grounding-DINO；或加入 robot-mask prior 作为 head 输入；或扩大 head 容量 | **已验证（5/5 帧通过）**——v2 用 per-patch 颜色匹配监督，跨 3 任务测试干净命中；MIL 监督版（v1）信号太弱，废弃 |
| H2 | Frozen DINOv2 patch token 比项目现有 CNN 编码更可线性预测 oracle | Phase 0：linear probe `cup_with_ball_number` / `true_color_indices` / `xyz_initial` | 视觉编码不是瓶颈，方向需要重审 | 待 |
| H3 | Buffer + cross-attention 读出在 RememberColor9 / ShellGameTouch 上 SR ≥ GRU baseline + 10 pts | Phase 1：PPO 训练对比 | 启用 EBM-Robo-Hybrid（加 GRU 短期记忆） | 待 |
| H4 | 写入过滤（learned head + novelty）是 load-bearing 组件，不是 cross-attention 自身 | Phase 2 ablation：去掉过滤 + buffer 存全部 token，对比 | story 必须重新组织 | 待 |
| H5 | 该方法不需要修改 PPO 算法本体 | Phase 1：clip / gae / lr 全部沿用 ppo_memtasks_gru.py | 出现训练不稳定再考虑 PPO 端调整 | 待 |
| ~~H6~~ | ~~特权 critic 进一步提升样本效率~~ | — | — | **已撤销**：critic 与 actor 输入完全相同，不使用特权信息，简化 method story |
| ~~H7~~ | ~~单目 base camera 对 MIKASA 记忆任务足够~~ | — | — | **已撤销**：决定双目为主版本（协议合规 + 信息更丰富 + saliency head 特征池更大） |

**核心假设是 H1' + H3 + H4**。若 H1' 失败，整个写入端机制要重设计；若 H3 失败，方法不成立；若 H4 失败，story 必须重新组织。

---

# 3. 系统架构

## 3.1 数据流图

```
双目 base + hand cameras  I^B_t, I^H_t ∈ R^{128×128×3}
                         │              │
                         ▼              ▼
                [Frozen DINOv2-S/14] (共享权重，分别 forward)
                         │              │
                         │  T^B_t       │  T^H_t   ∈ R^{N_v × 384}   (N_v ≈ 64–81 patches/view)
                         └──────┬───────┘
                                ▼
                    T_t = concat([T^B_t, T^H_t])    ∈ R^{2 N_v × 384}
                                │
                                ▼
┌────────────────────────────────────────────────┐
│       Saliency Head (学习, per-task frozen)     │
│                                                │
│  s_{t,i} = MLP_sal(T_{t,i}, xy_embed_i)        │
│            ∈ R   (per-patch importance score)  │
└────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────┐
│       Write Decision (无 RNN 反传)              │
│                                                │
│  candidates = top-K(s_{t,*}, K=8)              │
│  if max_i s_{t,i} < τ_s: skip frame            │
│  else for each candidate v_i:                  │
│    if max_j cos(v_i, M_j) > τ_n=0.95: skip     │
│    else push (v_i, t, s_{t,i}) to M            │
│  if |M| > L=64: evict argmin(s_j × age_decay)  │
└────────────────────────────────────────────────┘
         │
         ▼
[Episodic Buffer M ∈ R^{L × d_vit}, L=64]
         │
         │  + per-entry metadata: timestamp, saliency
         ▼
┌────────────────────────────────────────────┐
│          Memory Reader (学习)               │
│                                            │
│  curr_t = cls_t  (单目 CLS token)          │
│  q_t = MLP_Q([proprio_t, curr_t,           │
│              learned_task_token])          │
│  K = M @ W_K + ts_embed(M.timestamps)      │
│  V = M @ W_V                                │
│  logits_i = (q_t · K_i)/√d                 │
│             + α · log(M.saliency_i)        │
│  mask  : padding 位 logit = -inf           │
│  retrieved_t = softmax(logits) @ V         │
└────────────────────────────────────────────┘
         │
         ▼
[s_t = MLP_fuse([proprio_t, curr_t, retrieved_t])]
         │
         ├─→ Actor MLP → action(8) + log_std
         └─→ Critic MLP (s_t) → V(s)        # 与 actor 同输入，无特权信息
```

## 3.2 维度规范（默认配置）

| 符号 | 含义 | 维度 / 取值 |
|---|---|---|
| H, W | 单视角图像高宽 | 128, 128 |
| d_vit | ViT patch token 维度 | 384 (DINOv2-S/14) |
| N_v | 单视角 patch 数（DINOv2-S/14 处理器默认 resize 224、patch=14 → grid 16×16 = 256） | 256 |
| N | 双目 concat 后总 patch 数 | 2 × N_v = 512 |
| d_sal | saliency head MLP 隐藏维度 | 256, 128 |
| xy_embed | per-patch 2D 位置编码 | 32 |
| K | 每帧候选 token 数 | 8 |
| L | buffer 上限 | 64 |
| d_q | query 投影维度 | 256 |
| d_v | value 投影维度 | 256 |
| τ_s | 显著性阈值 | 自适应：当前 batch saliency 中位数 |
| τ_n | 新颖性阈值（cos sim） | 0.95 |
| α | saliency-bias 系数 | 0.5（可调） |
| 隐藏层维度 | actor / critic MLP | 256 → 256 |
| 动作维度 | MIKASA-Robo 标准 | 8 |
| proprio | joints 维度 | 25 |

## 3.3 接口与现有代码集成

- 新文件：`baselines/ppo/ppo_memtasks_ebm.py`，由现有 `ppo_memtasks_gru.py` 派生。
- 新模块：`baselines/ppo/modules/ebm.py`，定义 `EpisodicBuffer`、`MemoryReader`、`WriteScorer`、`FrozenViT`。
- 不修改 `mikasa_robo_suite/`；环境 obs 接口保持现状。
- **相机配置**：双目（base_camera + hand_camera），通过 `baselines/ppo/utils_ebm.py:FlattenRGBDObservationWrapperMulti` 加载。`FrozenViT` 接受 `(B, 3, 128, 128)` 单视角输入，对两路分别 forward 后 token 维度 concat。
- **GRU baseline 必须重跑双目版本** 才能与 EBM-Robo 同口径对比；新增文件 `baselines/ppo/ppo_memtasks_gru_dual.py`，与 `ppo_memtasks_gru.py` 唯一差异是 wrapper 与第一卷积层输入通道（3→6）。
- 训练入口与现有 SLURM 脚本兼容；参数追加 `--memory ebm` flag。

---

# 4. 训练与推理协议

## 4.1 PPO 设置（与现有 baseline 严格对齐，禁止修改）

- 沿用现有 `ppo_memtasks_gru.py` 中：clip=0.2, γ=0.99, λ_GAE=0.95, lr=3e-4, ent_coef=0.01, vf_coef=0.5。
- num_envs / num_steps / total_timesteps 与对照实验同。
- 唯一允许变化的是 memory module 与 critic 输入。

## 4.2 训练期数据流

每个 env step：
1. ViT forward (no_grad)。
2. Write scorer 决定是否 push（不可微，hard 操作）。
3. Buffer 状态作为环境 internal state，跨 env step 持久化；每 episode 结束清空。
4. Memory reader cross-attention forward（可微）。
5. PPO rollout 收集 (s_t, a_t, r_t, log_prob_t)。
6. 反传只经过：MLP_Q, W_K, W_V, ts_embed, MLP_fuse, actor/critic heads。**不反传到 ViT、不反传到 buffer push 决策**。

## 4.3 推理期

- 关闭 critic、关闭 oracle；其余路径同训练期。
- Buffer 仍按相同规则维护。

## 4.4 Critic 输入（无特权信息）

- Critic 与 actor 输入**完全相同**：`s_t = MLP_fuse([proprio_t, curr_t, retrieved_t])`。
- 不使用 oracle / privileged information。理由：保持 method story 简洁；ablation 表干净；避免"critic 过强 → actor 跟不上"风险。
- 推理期与训练期 critic 输入一致（仅 critic 不参与控制）。

---

# 5. 评测协议

## 5.1 任务集（与项目现有 baseline 对齐）

| 任务 | 记忆能力类型 | 选择理由 |
|---|---|---|
| RememberColor9-v0 | object memory + delay | 空延迟段最严重，最考验 buffer 不被擦写 |
| ShellGameTouch-v0 | object memory + occlusion | 写入稀疏，遮挡后必须凭记忆决策 |
| InterceptFast-v0 | spatial / motion memory | 目标轨迹，需追踪运动 |
| RememberShapeAndColor3x2-v0 | object memory（多属性） | 颜色 + 形状联合记忆 |
| TakeItBack-v0 | spatial memory | 早期空间位置，后期回到初始点 |

ChainOfColors / SeqOfColors / BunchOfColors 等任务（前期实测过于困难、当前任何方法都无法 work）**不进入主战场**，仅作为附录中的 stress test。

## 5.3 主指标

- **Success Rate (SR)**：episode 结束时任务是否成功，对每个 task 独立报告，3 seeds 取 mean ± std。
- **Sample efficiency**：达到 SR=0.5 所需的 env steps。
- **Memory probe**：训练完成后，从 retrieved_t 通过 frozen linear probe 预测 oracle，作为 memory faithfulness 指标。
  - 这是 EBM-Robo 区别于 GRU baseline 的关键诊断指标，必须报告。

## 5.4 对照与 Ablation

### 主表（end-to-end 对照，每方法用其自然组件）

| 行 | Method | 视觉前端 | 记忆 | 角色 |
|---|---|---|---|---|
| M1 | **GRU**（项目现有，dual-cam 重训） | CNN | GRU(512) | 主对照 |
| M2 | **LSTM**（项目现有，dual-cam 重训） | CNN | LSTM(512) | 主对照 |
| M3 | **Mamba/SSM** *（建议新增；若工作量大可移到附录）* | CNN（与 GRU/LSTM 同处理） | Mamba(512) | 现代序列模型对照——审稿人最常问 |
| M4 | **EBM-Robo (ours)** | Frozen DINOv2 + saliency head | KV buffer + cross-attn | 主方法 |

每方法**用其自然的视觉前端**——不强制共享 perception，避免人为约束。

**已显式排除的 baseline**：
- **CVAE / VAE belief**：不同类方法（生成式状态推断 vs 显式存储），对比意义弱，不进入主表（项目现有 `ppo_memtasks_cvae.py` 仅作历史参考）。
- **Slot Attention + memory**：用户已实测在该 benchmark 上无法分离物体，不再尝试。
- **DreamerV3 / 模型基**：不同范式（模型基 vs 模型自由），不构成可比 baseline。

### Ablation 表（针对 ours 内部组件）

| 行 | 设置 | 验证的问题 |
|---|---|---|
| A1 | **Ours full**（DINOv2 + saliency head + buffer + cross-attn） | 完整方法 |
| A2 | – DINOv2，换为项目现有 CNN | perception backbone 贡献多少 |
| A3 | – saliency head（写所有 patches，仅靠 novelty 过滤） | 写入 saliency 的价值（验证 H4 一半） |
| A4 | – novelty filter | 时间去重的价值（验证 H4 另一半） |
| A5 | – cross-attn 读出，改成 mean-pool retrieved | 检索机制的价值 |

主表 + ablation 一共 9 行（M1–M4 + A1–A5；其中 A1 = M4，去重后 8 个独立 run），3 seeds × 5 任务 = ~120 jobs。可控。

## 5.5 报告与可复现要求

- 所有 run 至少保存：config YAML, seed, hardware info, training log, metrics JSON, checkpoint。
- 主表至少 3 seeds (e.g., 33, 42, 99)，扩展表 5 seeds。
- 所有 run 落盘到 `final_results/ebm/{task}/{config_id}/seed_{N}/`。
- 写作素材落盘到 `writing/figures/`、`writing/tables/`。

---

# 6. 实现细节与超参

## 6.1 ViT 选择 + 推理优化

主版本：`facebook/dinov2-small`（22M params, d=384）。

**推理 pipeline 优化（[modules/frozen_vit.py](baselines/ppo/modules/frozen_vit.py)）**：
- 输入：126×126（9×14，patch_size=14 native），用 `interpolate_pos_encoding=True`。每视角 81 个 patch（vs 224 输入下的 256）。
- 精度：**fp16**（model + activations）。输出 cast 回 fp32 保持下游兼容。
- 双相机：base 和 hand **沿 batch 维 concat 后单次 ViT forward**（vs 两次串行调用）。
- 实测加速（B=256）：1052ms → 95.7ms = **11x**。EBM 端到端 step：~1100ms → 172ms = **6.4x**。

Saliency head 必须按 ViT 配置匹配训练：
- v2 head（256 patches，224 输入，fp32）—— **deprecated**
- **v3 head（81 patches，126 输入，fp16）—— 主版本，[path_a_head_v3.pt](analysis/ebm/path_a_head_v3.pt)**，由 [analysis/ebm/path_a_train_head_v3.py](analysis/ebm/path_a_train_head_v3.py) 训出

## 6.2 Write Scorer 细节

- **Saliency 来源：learned 2 层 MLP head**，输入 frozen DINOv2 patch token + xy 位置编码 + view_id（base/hand）；输出 per-patch logit。Head 离线训练（用 per-patch 监督，PPO 阶段冻结）。详见 [analysis/ebm/path_a_train_head_v2.py](analysis/ebm/path_a_train_head_v2.py)。
  - 不使用 DINOv2 CLS attention（Day 1 实证失败）。
- 阈值 τ_s 用 sigmoid(logit) > 0.5 作为"该 patch 是 target"的判定；批内自适应回退：若 >0.5 的 patch 数 < K_min=4，取 logit top-K_min 作为候选。
- 新颖性距离：cos similarity，归一化前先 LayerNorm。
- Eviction 优先级：`priority = sigmoid(logit) × exp(-age / τ_age)`，τ_age = 30 steps；优先级最低被驱逐。

## 6.3 Memory Reader 细节

- 单 head cross-attention，dropout=0.0（PPO on-policy 不需要）。
- timestamp embedding: 正余弦位置编码，与 transformer 标准一致。
- query 中的 learned_task_token：单一 256-d 可学习向量；扩展到多任务时改为 task embedding。
- 输出 retrieved_t 经过 LayerNorm 后再 concat。

## 6.4 训练超参（与 [/home/s4176650/sphinx/experiments/scripts/main/_common.sh](file:///home/s4176650/sphinx/experiments/scripts/main/_common.sh) 同步——用户已验证的 working configuration）

| 项 | 值 | 备注 |
|---|---|---|
| optimizer | Adam | |
| lr | **1e-4** + anneal | 早期错用 3e-4 不收敛 |
| ViT lr | 0 (frozen) | |
| gamma | **0.99** | 早期错用 0.8 → 远期 reward 衰减到 0，记忆任务必死 |
| gae_lambda | 0.9 | |
| ent_coef | 0.001 | |
| target_kl | 0.05 | trust-region 保护 |
| update_epochs | **2** | 早期错用 8 过激进 |
| num_envs | 256 | |
| num_eval_envs | 16 | |
| num_minibatches | 32 | |
| num_eval_steps | 720 (RememberColor) / 1080 (Intercept) | |
| eval_freq | 48 (RememberColor) / 25 (Intercept) | |
| total env steps | 10M | |
| seeds | 33, 42, 99 (主表) +100, 123 (附录) | |
| `PYTHONUNBUFFERED=1` | 强制 | 否则 stdout 长跑期间不刷盘看不到进度 |

**警告**：未来不得直接复用 `run_scripts/ppo/ppo_{lstm,mlp}/.../*.sh` 旧模板——它们带的是项目早期未调优的 hyper（gamma=0.8, lr=3e-4, update_epochs=8）。如需新 baseline，统一从 [run_scripts/ppo/ppo_gru_dual/...](run_scripts/ppo/ppo_gru_dual/dense_reward/remember_color/rgb_joint/remember_color_9.slurm) 复制。

## 6.5 复现性（Reproducibility）

**所有随机性源必须显式设种子**——这是论文级 reproducibility 的硬要求，不可省略。

| 阶段 | 种子 | 设置位置 |
|---|---|---|
| **Saliency head 预训练**（[`analysis/ebm/path_a_train_head*.py`](analysis/ebm/path_a_train_head_v3.py)） | **0** | `set_seed(0)` 函数调用于 `main()` 顶部，覆盖 `random` / `np.random` / `torch.manual_seed` / `torch.cuda.manual_seed_all` / `cudnn.deterministic=True` / `cudnn.benchmark=False` |
| **离线 rollout 数据收集**（[`path_a_collect_data.py`](analysis/ebm/path_a_collect_data.py)） | CLI `--seed=42` | numpy default_rng 派生每个 episode 的 env reset seed |
| **PPO 训练**（所有 `baselines/ppo/ppo_memtasks_*.py`） | CLI `--seed=33/42/99` | 每个 SLURM array task 用 SEEDS=(33 42 99) 列表的对应索引 |
| **`torch.backends.cudnn.deterministic`** | `True` | 在 PPO 主 script 中显式设置 |

**3 seeds 选择理由**：33 / 42 / 99 来自项目历史"newhyper" preset，与 sphinx 阶段对照实验同集（便于跨阶段对比）。

**每个实验 SR 报告格式**：`mean ± std across 3 seeds`，主表至少 3 seeds，附录扩展可加 100 / 123。

## 6.6 计算资源

- ViT forward 是主要新增开销，但 frozen + no_grad，PPO rollout 吞吐预计下降 < 30%。
- buffer 操作 O(L × d) per step，L=64，可忽略。
- cross-attention O(L × d) per step，可忽略。
- 单任务 Phase 1 10M steps：单卡 L4-24G 约 8–12h；4 任务总计 < 48 GPU-hours。

---

# 7. 风险点与回退路径

| 风险 | 触发信号 | 回退 |
|---|---|---|
| DINOv2 saliency 在合成渲染上不显著 | Day 1 可视化失败 | 自监督训 saliency head；或换 V-JEPA / SAM2 |
| Buffer 在 long episode 溢出，重要条目被驱逐 | probe 在 long-horizon 任务上衰减 | 动态 L（按 saliency 总量调整）；分层 buffer |
| Cross-attention 学不出"跨任务-buffer"映射 | 训练 SR 不上升，probe 也低 | 加 auxiliary loss：retrieved_t 预测 oracle |
| PPO 反传通过 hard buffer push 失稳 | KL 暴增、value loss 振荡 | push 决策完全无参数（设计已规避）；若仍不稳，提高 ent_coef |
| 顺序任务依赖 timestamp，但 ts_embed 学不好 | ChainOfColors / SeqOfColors SR 低 | 改用 relative position bias（ALiBi 风格） |

## 7.1 失败的硬阈值

如果在 RememberColor9 + ShellGameTouch 两个任务上，**主方法（行 3）SR 在 3 seeds 平均上不超过 GRU baseline（行 1）的 1.10×**，则中止 Phase 2，重新评估假设。

---

# 8. 资源与时间估计

| 阶段 | 工作内容 | 预计时间 | 主要交付 |
|---|---|---|---|
| **Day 1** | DINOv2 saliency 可视化 + linear probe | 1 天 | 决定 H1, H2 是否成立 |
| **Phase 1 MVP** | 实现 EBM 模块 + RememberColor9, ShellGameTouch 训练 | 1 周 | 首张 SR 对比图 |
| **Phase 2 Ablation** | 4 任务，行 1–7 全部跑通，3 seeds | 2–3 周 | 主结果表 + ablation 表 |
| **Phase 3 Scale** | 扩到 8–12 任务，加附录 ablation | 3–4 周 | 论文实验完整结果 |
| **Writing** | 与 ml-paper-writing 协作 | 2 周 | 投稿草稿 |

---

# 9. 参考文献与思想来源

**未列入与本项目实现/写作不直接相关的论文。**

## 9.1 Benchmark

1. **MIKASA-Robo**（项目自身的 benchmark）
   - 用途：benchmark 协议、obs 接口、oracle 字段。

## 9.2 视觉骨架（frozen 视觉编码）

2. **DINOv2** (Oquab et al., 2024)
   - 用途：主方法的 frozen ViT。
   - 吸收：CLS-patch attention 作为前景显著性。
3. **DeiT / ViT-Tiny** (Touvron et al., 2021)
   - 用途：轻量对照版本。

## 9.3 Memory in RL / Sequence Modeling

4. **Differentiable Neural Computer (DNC)** (Graves et al., 2016)
   - 用途：相关工作；说明显式 K-V 记忆在 RL 中的先例。
5. **Compressive Transformer** (Rae et al., 2020)
   - 用途：相关工作；分层 memory tier 的灵感（用于 EBM-Hybrid）。
6. **Recurrent Memory Transformer** (Bulatov et al., 2022)
   - 用途：相关工作；memory token 范式。

## 9.4 Token Selection / Visual Compression

7. **DINO emergent attention** (Caron et al., 2021)
   - 用途：方法依赖的关键观察（无监督前景定位）。
8. **DynamicViT / EViT / ToMe** (Rao et al., 2021; Liang et al., 2022; Bolya et al., 2023)
   - 用途：相关工作；说明"哪些 token 重要"在视觉 Transformer 上是成熟问题；本文在 RL 记忆任务上把它纳入写入门。

## 9.5 现代序列模型（M3 baseline）

9. **Mamba** (Gu & Dao, 2023)
   - 用途：M3 行的记忆模块来源。说明现代 SSM 在长序列上的优势，论文中作为"非 RNN 替代品"对照。

---

# 10. 与项目现有 baseline 的关系

- 现有 `ppo_memtasks_gru.py` / `ppo_memtasks_lstm.py` 是 M1/M2 基线，重新在双目下训出 dual-cam 版本（[ppo_memtasks_gru_dual.py](baselines/ppo/ppo_memtasks_gru_dual.py)）。
- M3（Mamba/SSM）需新写 `ppo_memtasks_mamba_dual.py`（与 GRU 同骨架，仅记忆模块替换；可后置）。
- M4（EBM-Robo）= `ppo_memtasks_ebm.py`（Phase 1 待实现）。
- 现有 `ppo_memtasks_cvae.py` / `ppo_memtasks_vae.py`（belief 方法）**不进入论文 baseline 对比**——它们与本文方法属不同范式（生成式状态推断 vs 显式存储），同表对比意义弱。仅在 related work 中作为"我们考虑过但范式不同"的参考被简短提及。

---

# 11. 论文章节预定大纲

1. Introduction
2. Background：partial-observability robot RL，MIKASA-Robo benchmark，RNN memory in RL
3. Why GRU/LSTM fail on MIKASA-Robo（诊断章节，给出 H1, H2 的实验证据）
4. EBM-Robo Method（5 个组件）
5. Experimental Setup
6. Main Results（4 个 MVP 任务 + 8 个扩展任务）
7. Ablations（行 4–6, 8）
8. Memory Probe Analysis（faithfulness 指标）
9. Discussion: when does episodic memory help vs hurt?
10. Related Work
11. Conclusion + Limitations

---

# 12. 版本与交接

- 本规范是 Coding Agent 的唯一行为准则；如与口头指令冲突，以本文为准。
- 所有方法变更须先更新本文档，再实现。
- 失败的硬阈值（§ 7.1）触发后，必须暂停并请求人工重审。
