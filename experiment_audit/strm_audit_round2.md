# STRM Audit Round 2: CRES + Routing

## A. CRES Implementation Comparison Table

### STRM CRES mechanism (ebm_srb_tr_cres.py)

- **Patch-level RGB:** Raw rgb6 (B,H,W,6) split into base/hand, each resized to 126, reshaped into (B,3,9,14,9,14), mean-pooled over (14,14) patch dims → (B,162,3)
- **Fixed random matrix R:** shape (3, 384), init `torch.randn(3,384,generator=g)/sqrt(384)`, seed=1234, registered as buffer
- **color_scale:** Auto-calibrated on first step: `s = color_frac * mean(||z_dino||) / mean(||rgb_proj||)`, color_frac=0.4, frozen after
- **Additive residual:** `z = tokens_all + color_scale * ((meanRGB - 0.5) @ R)`, output dim = 384
- **Zero learnable parameters.** R is fixed random, color_scale calibrated once.
- **Reward gradient cannot reach CRES.** Entire step() under no_grad.

### Where z (CRES-augmented) vs pure DINOv2 tokens go:

| Component | Uses z (CRES) | Uses pure DINOv2 |
|---|---|---|
| Surprise computation (L2 dist) | YES | NO |
| Buffer stored features | YES | NO |
| Cosine novelty filter in push | YES | NO |
| Reader cross-attention K,V | YES (buffer holds z) | NO |
| Motion EMA (ema_change) | NO | YES |
| LSTM route: high-motion top-K pool | NO | YES |
| curr_summary (CLS tokens) | NO | YES |
| LSTM input | NO | YES |
| Policy direct path (fuse) | Indirectly via reader | CLS via curr |

### Baseline color augmentation (DinoV2SimpleEncoder, color_aug=True)

- Per-patch mean RGB via same interpolate-reshape-mean pipeline
- Centers by -0.5 (same as CRES)
- Flattens to 1D: (B, 162*3) = (B, 486)
- **Concatenates** to [cls_summary(256), joints(25)] → total 767
- NO random projection R, NO fixed scale calibration
- Downstream linear layer (part of LSTM/GRU/MLP) is LEARNABLE

### Comparison Table

| Method | DINO input | RGB info form | RGB injection point | Learnable RGB params | Same as STRM? |
|---|---|---|---|---|---|
| **STRM** | Patch tokens (162×384) | Centered mean RGB → fixed random proj (162×384) | Additive to patch tokens before buffer ops | NONE | -- |
| **LSTM** | CLS only (2×384→256) | Centered mean RGB flat (486-d) | Concat to encoder output, into LSTM input | First layer weights | NO |
| **GRU** | CLS only (2×384→256) | Centered mean RGB flat (486-d) | Concat to encoder output, into GRU input | First layer weights | NO |
| **MLP** | CLS only (2×384→256) | Centered mean RGB flat (486-d) | Concat to encoder output, into policy MLP | First layer weights | NO |
| **FFM** | CLS only (2×384→256) | Centered mean RGB flat (486-d) | Concat to encoder output, into FFM cell | First layer weights | NO |
| **SHM** | CLS only (2×384→256) | Centered mean RGB flat (486-d) | Concat to encoder output, into SHM cell | First layer weights | NO |

Key structural differences:
1. STRM injects color into 162 individual patch tokens (spatial structure preserved). Baselines get flat 486-d vector (spatial collapsed).
2. STRM uses fixed random projection; baselines get raw RGB through learnable layers.
3. STRM color enters buffer pipeline (surprise, storage, retrieval). Baselines' color enters RNN/MLP network.
4. STRM has zero learnable color params. Baselines can freely adapt color processing.

---

## B. CRES Fairness Judgment

**A. "CRES is applied identically to all methods."**
→ **INACCURATE.** STRM uses fixed random projection additive residual on patch tokens. Baselines concatenate flat RGB to CLS summary. Mechanism, injection point, and dimensionality all differ.

**B. "All methods receive matched color information."**
→ **NEEDS QUALIFICATION.** Same underlying signal (per-patch centered mean RGB), same tasks. But form differs: STRM gets (162×384) additive residual; baselines get (486-d) concat. Information content equivalent but integration architecture not matched.

**C. "All methods use the same perception backbone and task-relevant color cues, although the color features are integrated according to each architecture."**
→ **ACCURATE AND DEFENSIBLE.** All share frozen DINOv2 and same centered-mean-RGB source. Integration method differs per architecture interface.

---

## C. Existing CRES Ablation Evidence

- **SRB-TR (no CRES, CAPS only) vs SRB-TR-CRES:** Partial data exists for RememberShape tasks only (not CRES target). **NOT AVAILABLE for RememberColor tasks (the critical ones).**
- **Baselines with/without color aug:** NOT AVAILABLE as controlled comparison.
- Main table uses crescaps for Remember tasks, plain for Intercept tasks.
- **Per-task CRES impact on RC5/RC9:** NOT AVAILABLE. This is the single most important missing experiment.

---

## D. Recommended CRES Placement

### Option A: Method section (core component)
- Faithful to code? Partially — easily separated module
- Faithful to results? Main claims work without CRES on shape/intercept
- Reviewer risk: HIGH — "what happens without CRES on RC5?" has no answer
- Needs: CRES ablation on RC5/RC9

### Option B: Experimental Setup (shared perception protocol) ← RECOMMENDED
- Faithful to code? YES
- Faithful to results? YES
- Reviewer risk: LOW — honest framing
- Suggested text: "DINOv2 features are color-insensitive on MIKASA's synthetic saturated objects. We supplement all methods with centered per-patch mean RGB on color-critical tasks. For STRM, this is an additive residual on stored patch features; for baselines, the 486-d color vector is concatenated to the encoder output."

### Option C: Method appendix note (benchmark adapter)
- Faithful? YES but undersells if CRES matters for RC5
- Reviewer risk: LOW but may look evasive

---

## E. Exact Routing Set Relationship

E_t = top-K(8) indices by `buffer_score = s_raw / (1 + λ·ema/384)`
R_t = top-K(8) indices by `ema_change`

- |E_t| = |R_t| = K = 8. Always same size.
- **E_t ∩ R_t can be non-empty.** NO exclusion logic, NO dedup, NO priority between them.
- Two independent `topk()` calls on same 162-token set with different scoring functions.
- Cosine dedup in push affects actual writes but NOT the selection itself (buffer-internal).
- CLS tokens feed curr_summary regardless of routing (separate information path).

Three parallel paths:
1. CLS → curr_summary → fuse (always on)
2. Buffer top-K → stored → reader → fuse (episodic)
3. LSTM top-K → pooled → LSTM → fuse (recurrent)
A single patch token can participate in paths 2 AND 3 simultaneously.

---

## F. Routing Overlap Evidence

**Existing logging:** No overlap metrics recorded. step() returns n_pushed, buffer_used, max_surprise, max_motion but not overlap.

**Instrumentation patch for step():**
```python
# After topk_idx_lstm computed (line 221), add:
with torch.no_grad():
    overlap_count = torch.zeros(B, device=tokens_all.device)
    for b_idx in range(B):
        set_e = set(topk_idx_buf[b_idx].tolist())
        set_r = set(topk_idx_lstm[b_idx].tolist())
        overlap_count[b_idx] = len(set_e & set_r)
    overlap_ratio = overlap_count / self.K
```

Add to return dict: `"overlap_count": overlap_count, "overlap_ratio": overlap_ratio.mean()`

**Expected:** Low overlap on static Remember tasks (cubes low-motion, gripper high-motion → different selections). Higher overlap on Intercept (ball is both novel and high-motion).

---

## G. Safe and Unsafe Routing Sentences

| Sentence | Verdict | Implies exclusive? | Notes |
|---|---|---|---|
| A. "routes each token either to episodic or recurrent" | INACCURATE | YES ("either...or") | Unsuitable anywhere |
| B. "partitions tokens between episodic and recurrent" | INACCURATE | YES ("partitions") | Unsuitable anywhere |
| C. "independently selects tokens for episodic writing and recurrent integration using complementary surprise and temporal criteria" | ACCURATE | NO | Best for method section |
| D. "uses surprise and temporal statistics to regulate two potentially overlapping memory pathways" | ACCURATE | NO | Best for abstract/intro |
| E. "directs low-motion surprising tokens toward episodic storage and emphasizes rapidly changing tokens for recurrent integration" | NEEDS QUALIFICATION | Implicit | Over-interprets; good for caption with "tends to" |

**Recommendation:**
- Abstract/intro: Sentence D
- Method: Sentence C + "The two selections may overlap, allowing a single token to inform both memory systems."
- Caption: Sentence E with "tends to direct"

---

## H. Three Remaining Decisions

1. **Run CRES ablation on RC5** (3 seeds, ~3 GPU-hours on L4). Most important missing experiment. Without it, reviewer can challenge any color-task result.

2. **Measure routing overlap** before finalizing method figure. If < 5%, diagram arrows can be distinct. If > 20%, figure must show overlapping sets.

3. **Method name:** "Routing" in ML does not universally imply mutual exclusion (cf. MoE soft routing). STRM is acceptable if text clarifies independent selection. Only consider renaming if overlap > 30%.

---

## Chinese Pure-Fact Summaries

**1. Core STRM (no CRES):**
冻结DINOv2 patch tokens经L2惊奇+运动EMA双独立top-K分流: 低动高新→64-slot梯度隔离缓冲, 高动→LSTM(H=128)池化输入, 两路可重叠; cross-attention仅读缓冲, LSTM隐层在融合层拼接, PPO训练reader/fuse/LSTM。

**2. Full main-table (with CRES):**
Remember任务加CRES: patch RGB均值居中→固定随机投影(3→384)→加性残差修改缓冲存储特征, scale首步校准后冻结; 基线同任务拼接486维平RGB至编码器输出; Intercept任务全方法无色彩增强。
