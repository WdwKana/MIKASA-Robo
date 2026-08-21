# STRM Architecture Implementation Fact Sheet

## A. Actual Entry and Version

**Training entry file (main table):**
- `baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py` -- used for STRM main results (`srbtr_l4main.slurm`)
  - Import at line 42: `from baselines.ppo.modules import EBMSRBTRCRESMemoryModule as EBMMemoryModule`
  - Confirmed by `/run_scripts/aaai_final/srbtr_l4main.slurm`

**STRM model class:**
- `EBMSRBTRCRESMemoryModule` in `baselines/ppo/modules/ebm_srb_tr_cres.py` (subclass of `EBMSRBTRMemoryModule`)
- Parent: `EBMSRBTRMemoryModule` in `baselines/ppo/modules/ebm_srb_tr.py`

**Multiple similar but unused implementations exist:** YES.
- `ebm_srb_tr.py` -- base SRB-TR (no CRES) -- used for non-CRES ablations
- `ebm_srb_tr_cres.py` -- CRES variant (main method for main table)
- `ebm_srb_tr_mv.py` -- MV scoring variant (NOT main table)
- `ebm_srb_tr_ccat.py` -- color concatenation variant
- `ebm_srb_tr_nolstm.py` -- no-LSTM ablation
- `ebm_srb_tr_cres_writeablate.py` -- write-rule ablation
- `ebm_srb.py`, `ebm_srb_ms.py` -- earlier versions without temporal routing
- `ebm.py`, `ebm_hybrid.py`, `ebm_hybrid_lstm.py` -- older predecessors

**Main table results correspond to:** `EBMSRBTRCRESMemoryModule` with CAPS (`caps_lambda_t=0.15`)

---

## B. End-to-End Data Flow

### Step-by-step execution in `EBMSRBTRCRESMemoryModule.step()` (ebm_srb_tr_cres.py:110-183):

1. **Camera images enter.** `rgb6: (B, H, W, 6)` = [base_RGB || hand_RGB]. `proprio: (B, 25)` from `obs["joints"]`.

2. **DINOv2 processes images.** `self.vit(rgb6)` (frozen_vit.py:82-95):
   - Splits channels: base=[:3], hand=[3:6]
   - Concatenates along batch: (2B, 3, H, W)
   - Resizes to 126x126, normalizes, runs in fp16
   - Produces: `tok_b (B,81,384)`, `tok_h (B,81,384)`, `cls_b (B,384)`, `cls_h (B,384)`

3. **Token concatenation.** `tokens_all = cat([tok_b, tok_h], dim=1)` → `(B, 162, 384)` (ebm_srb_tr.py:181)

4. **CRES color fusion** (ebm_srb_tr_cres.py:118):
   - `z = tokens_all + color_scale * ((meanRGB_patch - 0.5) @ R)` where R is fixed random (3→384)
   - z used for buffer ops; LSTM motion EMA stays on pure DINOv2

5. **Self-referential surprise** (under `torch.no_grad()`, ebm_srb_tr_cres.py:123-128):
   - `dist = _pairwise_sq_dist(z, buffer.features)` → squared L2
   - Invalid buffer entries masked to inf
   - `min_dist = dist.min(dim=-1)` per token
   - Empty buffer fallback: `||z||²`

6. **Temporal EMA** (ebm_srb_tr_cres.py:131-136):
   - `cur_change = ||tokens_all - p_prev||²` (per-patch, on PURE DINOv2)
   - `ema_change = 0.9 * ema_change + 0.1 * cur_change`

7. **Routing** (ebm_srb_tr_cres.py:139-149):
   - **Buffer (Branch A):** `buffer_score = s_raw / (1 + λ · ema_change / 384)` → top-K(8)
   - **LSTM (Branch B):** `top-K(8) of ema_change` → softmax-weighted pool → `pooled (B, 384)`

8. **Buffer push** (under `torch.no_grad()`):
   - Cosine novelty filter (thresh=0.95), priority eviction when full

9. **Current-frame summary:** `curr = curr_summary(cat([cls_b, cls_h]))` → `(B, 128)`

10. **LSTM step** (under no_grad during rollout; with grad during replay):
    - Input: `cat([pooled, curr, proprio])` → LSTM → `gru_hidden (B, 128)`

11. **Episodic retrieval:** `query = cat([proprio, curr])` → reader queries buffer ONLY → `retrieved (B, 256)`

12. **Policy fusion:** `s_t = fuse(cat([proprio, curr, retrieved, gru_hidden]))` → `(B, 256)`

13. **Actor/critic:** Both share s_t. Actor: Linear(256,512)+ReLU+Linear(512,act_dim)+Tanh. Critic: Linear(256,512)+ReLU+Linear(512,1).

---

## C. Per-Module Fact Audit

### Visual Encoding
- **One shared DINOv2 backbone** encoding both cameras (batched forward pass). frozen_vit.py:74
- **Fully frozen:** requires_grad_(False), @torch.no_grad(). frozen_vit.py:48-49,81
- **Uses patch tokens AND CLS tokens.** Patch → routing/buffer/LSTM. CLS → curr_summary.
- **81 tokens per camera** (9×9 from 126/14). 162 total.
- **No normalization/projection/positional encoding on tokens** before routing. Raw DINOv2 features.
- **Direct path exists:** `curr` (CLS) and `proprio` enter fuse directly, bypassing all memory.
- **All baselines use same encoder:** DinoV2SimpleEncoder wraps same FrozenDualDinoV2.

### Memory-Relative Novelty and Episodic Writing
- **Distance:** Squared L2 (`_pairwise_sq_dist`), NOT cosine. ebm_srb_tr.py:54-60
- **Tokens NOT normalized** for surprise distance.
- **Min distance** per token (ebm_srb_tr.py:193).
- **Empty buffer:** fallback = `||z||²` (ebm_srb_tr.py:194-195)
- **Motion suppression is routing/write-score, not novelty itself:** `buffer_score = s_raw / suppress`
- **K=8 tokens per step** (minus cosine dedup rejections)
- **Buffer eviction:** lowest `priority = saliency × exp(-age / τ_age)` (episodic_buffer.py:96-100)
- **Buffer stores:** features(384), timestamps(long), saliency(float=buffer_score), per-env used count. NO camera identity.
- **Writing under no_grad:** YES. Buffer features are detached clones (episodic_buffer.py:116-118).
- **"Gradient-isolated":** ACCURATE
- **"Parameter-free":** No learned params, but has hyperparams (K, λ, α, novelty_thresh, τ_age)
- **"Reward-free":** ACCURATE. Write score uses only distance/motion.

### Temporal Statistics and Routing
- `ema[n] = 0.9·ema[n] + 0.1·||tok[n] - tok_prev[n]||²`
- Per-patch, requires spatial index correspondence across frames
- Participates in BOTH episodic write score (suppression) and recurrent routing (direct)
- **NOT mutually exclusive:** same token CAN enter both top-K sets
- Hard top-K selection (not soft gating). Soft weighting for LSTM pooling only.

### Recurrent Register
- **Actually LSTM** (nn.LSTM, H=128, 1 layer). ebm_srb_tr.py:139
- **Variable name inconsistencies:** gru_state, gru_hidden_size, gru_input_dim — all legacy GRU names
- **Input:** `cat([pooled_high_motion, curr, proprio])` — single pooled vector
- **Reset:** `gru_state[:, done_mask] = 0.0`
- **Does NOT go through reader.** Enters fusion directly.

### Episodic Retrieval and Policy Fusion
- **Query:** `Q = W_Q(cat([proprio, curr])) + task_token` — additive bias, NOT concatenation (memory_reader.py:83)
- **Keys:** `K = W_K(buffer_features) + ts_embed(timestamps)` (memory_reader.py:84)
- **Values:** `V = W_V(buffer_features)` (memory_reader.py:85)
- **Saliency:** `logits += 0.5 · log(saliency)` (memory_reader.py:91)
- **Reader ONLY queries buffer.** Recurrent state excluded.
- **Fusion:** `fuse(cat([proprio, curr(128), retrieved(256), gru_hidden(128)]))`
- **Actor/critic share s_t**

---

## D. Routing Pseudocode

```python
# Under torch.no_grad():
tokens_all = cat(tok_base, tok_hand)  # (B, 162, 384)
z = tokens_all + color_residual       # CRES only

# Surprise: min squared L2 to buffer
s_raw[n] = min_j ||z[n] - buf[j]||²   # for occupied j
if buffer_empty: s_raw[n] = ||z[n]||²

# Motion EMA (pure DINOv2)
ema[n] = 0.9 * ema[n] + 0.1 * ||tok[n] - tok_prev[n]||²

# Buffer route: top-K of motion-suppressed surprise
buffer_score[n] = s_raw[n] / (1 + 1.0 * ema[n] / 384)
buf_idx = topk(buffer_score, K=8)
push candidates from z[buf_idx]       # cosine dedup inside push

# LSTM route: top-K of raw motion
lstm_idx = topk(ema, K=8)
pooled = softmax_weighted_pool(tokens_all[lstm_idx], weights=ema[lstm_idx])
```

---

## E. Gradient Path Table

| Module | Status | Reward gradient reaches? |
|--------|--------|--------------------------|
| DINOv2 backbone | Frozen (no_grad, requires_grad=False) | NO |
| curr_summary (Linear+GELU+Linear) | Learned by PPO | YES (via replay) |
| Novelty/surprise computation | Procedural (no_grad, no params) | NO |
| Episodic buffer (features/timestamps/saliency) | Procedural state (detached clones) | NO |
| Temporal EMA (ema_change, p_prev) | Procedural state (no_grad) | NO |
| LSTM (self.lstm) | Learned by PPO | YES (via replay) |
| MemoryReader (W_Q, W_K, W_V, task_token, ts_embed, out_norm) | Learned by PPO | YES (via replay) |
| Fuse MLP | Learned by PPO | YES (via replay) |
| Actor head | Learned by PPO | YES (direct) |
| Critic head | Learned by PPO | YES (direct) |
| CRES color projection R | Frozen buffer (fixed random) | NO |
| CRES color_scale | Auto-calibrated once then frozen | NO |

---

## F. Terminology Safety Table

| Term | Verdict | Rationale |
|------|---------|-----------|
| single visual backbone | SAFE | One DINOv2-S/14 instance, batched forward. frozen_vit.py:74,89 |
| dual-camera perception | SAFE | Base + hand cameras as 6-channel input. frozen_vit.py:73-74 |
| dual-path memory architecture | SAFE | Two distinct paths: episodic buffer + LSTM |
| multi-substrate memory | SAFE | Buffer (K-V store) and LSTM (recurrent) are structurally different |
| protected episodic slots | NEEDS QUALIFICATION | Entries are overwritten by eviction; "gradient-isolated" is more accurate |
| gradient-isolated episodic slots | SAFE | Detached clones, push under no_grad. episodic_buffer.py:116-118 |
| memory-relative surprise | SAFE | Min squared L2 to buffer contents |
| memory-relative novelty | NEEDS QUALIFICATION | "Novelty" used in two senses: surprise score (L2) vs. dedup filter (cosine) |
| reward-free writing | SAFE | Write score uses only geometry and motion EMA |
| parameter-free writing | NEEDS QUALIFICATION | No learned params, but has hyperparams. Say "no learned parameters" |
| non-parametric writing criterion | SAFE | No neural network weights in writing |
| temporal routing | SAFE | EMA of frame-to-frame change routes tokens |
| routes tokens into two memories | NEEDS QUALIFICATION | Both routes select independently; same token CAN go to both. Not strict partition |
| compact recurrent register | SAFE | Single-layer LSTM, H=128 |
| working memory | NEEDS QUALIFICATION | Could imply cognitive science claims |
| task-trained episodic retrieval | SAFE | W_Q, W_K, W_V, task_token learned via PPO |
| learned readout | SAFE | Reader has learned projections |
| complementary memory representations | SAFE | Buffer=sparse events, LSTM=dynamics |
| unified memory reader | UNSAFE | Reader ONLY queries buffer, NOT recurrent state |
| policy fusion | SAFE | fuse MLP concatenates all representations |

---

## G. Top 10 Points the Paper Is Most Likely to Get Wrong

1. **Main table uses CRES, not base SRB-TR.** If paper describes STRM as operating purely on DINOv2 features without mentioning CRES, this is inaccurate for main results.

2. **Variable naming: "GRU" everywhere, module is LSTM.** gru_state, gru_hidden_size, gru_input_dim — all legacy. Module is nn.LSTM.

3. **Surprise uses squared L2 distance, NOT cosine.** Dedup filter in push_batch uses cosine. Two different operations, different metrics.

4. **"Unified memory reader" is wrong.** Reader queries ONLY episodic buffer. LSTM state enters fusion via concatenation.

5. **Routing is NOT mutually exclusive.** Same token can enter both episodic buffer and LSTM input pool. Two independent top-K selections.

6. **CAPS and CRES applied to ALL baselines equally.** LSTM/MLP baselines also use color_aug and CAPS λ=0.15.

7. **"Parameter-free writing" has hyperparameters.** K=8, α=0.1, λ=1.0, thresh=0.95, τ_age=30.0.

8. **curr_summary uses CLS tokens, not patch tokens.** Patch tokens → routing/buffer/LSTM. CLS → curr_summary.

9. **Reader task_token is additive bias, not concatenation.** `Q = W_Q(query_in) + task_token` (memory_reader.py:83).

10. **CRES applies to buffer features but NOT to LSTM motion features.** Buffer uses color-fused z; LSTM motion EMA uses pure tokens_all.

---

## H. Questions Still Needing Confirmation

1. **CRES framing:** Is CRES part of "core method" or "experimental setup"? Main table uses CRES variant.

2. **joints_dim:** Default is 25 (ebm_srb_tr.py:77), actual value depends on environment.

3. **Buffer "saliency" field:** What is stored as "saliency" is actually buffer_score (motion-suppressed surprise), NOT a learned saliency prediction. Legacy naming.

4. **K overlap in practice:** Same token can theoretically be in both top-K sets. How often this happens is unknown from code.

5. **Sensitivity of color_frac=0.4 and caps_lambda_t=0.15:** Fixed in main slurm. Robustness requires ablation data.

---

## CRES/CAPS Details

**CRES (Color RESidual):**
- Perception/feature layer. Adds color info via additive random projection.
- Only affects buffer storage features, NOT LSTM motion features.
- Baselines: `DinoV2SimpleEncoder(color_aug=True)` concatenates per-patch mean RGB.
- Applied identically across all methods.

**CAPS (Continuous Action-smoothness Preservation System, Mysore et al. 2021):**
- Actor-head loss regularizer. `||π(s_t) - π(s_{t+1})||² · λ_t`
- Gradients flow ONLY through actor_mean. s_t is detached.
- Applied identically to all baselines with λ_t=0.15.

---

## Chinese Pure-Fact Architecture Summary

冻结DINOv2双目81+81token, 逐patch EMA运动量分流: 低运动高新奇top-K写入64-slot L2惊奇缓冲, 高运动top-K池化输入LSTM(H=128). 跨注意力读缓冲+LSTM隐层拼接proprio/CLS汇聚, PPO端到端训练reader/fuse/LSTM/actor/critic.
