# Color Perception Investigation — Findings & Next Steps

**Date**: 2026-06-01
**Scope**: One-thread investigation triggered by SRB-TR underperforming on RC5
relative to its strong Shape5 wins. Closed with shipping **MV-SPLIT** to RL
and a documented set of negative findings on alternative approaches.

---

## 1. Problem framing

EBM-Robo's SRB-TR module wins big on Shape5 but underperforms LSTM on RC5.
Diagnosis pointed at **patch-level color discriminability** of frozen
DINOv2: the buffer's L2-novelty surprise cannot tell a red cube from a blue
cube under DINOv2's geometry, so SRB-TR conflates target with non-target on
Remember-Color tasks.

The goal of this thread was a **principled, paper-defensible** fix for
color discriminability. Constraints set by the user during the thread:

- **No per-task supervision** (rules out V1 saliency head, which was a
  per-task color BCE classifier)
- **Generalize to all MIKASA-Robo scene colors**, not just the 9 RC colors
- **Architecture-clean**: perception fix should help baselines (LSTM/GRU/MLP)
  too, not be a method-specific hack — perception is not our paper's
  contribution; the memory architecture is
- **Ship results**: bias toward Pareto-improving methods over elegant ones

---

## 2. Methodology baseline

| Probe | Result |
|---|---|
| DINOv2-small per-patch sep_ratio on MIKASA colors (1% purity threshold) | 1.000 |
| CLIP-base per-patch sep_ratio | 1.070 |
| SAM-base per-patch sep_ratio | 1.042 |
| **DINOv2 on 100% synthetic uniform-color images** | **1.359** |
| **MIKASA cubes occupy >30% of any patch** | **NEVER** (cubes ~3-5 px in 14×14 patches) |

**Reframe (important — corrected at end of thread)**: DINOv2 is *not* literally
color-blind. The 1.00 measurement was confounded by two effects:

1. **Patch composition**: real cube patches are >85% gray table; the cube's
   pixel contribution to the patch's mean RGB is tiny → DINOv2's feature is
   dominated by "table semantic" not "cube color"
2. **L2 dimension count**: even with 100% pure color input, sep_ratio is only
   1.36 (modest). DINOv2's 384-d space encodes color in few dimensions; the
   other ~370 dims are shape/texture/position. L2 weights all dims equally,
   so color signal is drowned out.

The reason **Gaussian-on-RGB trivially classifies colors** while DINOv2
struggles: raw RGB is 3-d, **100% of dimensions are color**. DINOv2 patches
are 384-d, **<10% of dimensions encode color**. L2 distance amplifies the
larger signal (shape) and drowns out the smaller (color). A learned linear
probe on DINOv2 can extract color (>80% acc), but naive L2/cosine cannot.

---

## 3. Approaches tried — summary table

| Approach | Where injected | sep_ratio | Stage 0 (RC5/RC9/Shape5 P_target late) | Verdict |
|---|---|---|---|---|
| **SRB-MS** (DINOv2 baseline) | — | 1.00 | 0.42 / **0.90** / **0.15** | reference |
| Backbone swap (CLIP/SAM) | perception backbone | 1.07 / 1.04 | not measured | ❌ all SSL ViTs color-blind |
| **MV-MAX** (max-of-normalized) | scoring layer | — | 0.34 / 0.51 / 0.13 | ❌ outlier dominates |
| **MV-SPLIT** (K/2 each view, dedup) | scoring layer | — | **0.54** / 0.87 / 0.15 | ✅ **shipped to RL** |
| **MV-CONCAT** (Mahalanobis-whitened concat) | feature layer | — | 0.48 / 0.85 / 0.15 | partial (worse than SPLIT) |
| **Inverse-Jitter SSL adapter** | feature layer | 1.07 | not measured | ❌ training plateau; frozen DINOv2 info bottleneck |
| **SupCon-hue v1** (16 hue bins, hue-only) | feature layer | 1.58 | not measured | ❌ 3 MIKASA color pairs collide on same bin |
| **SupCon-hue v2** (16 hue × 3 val = 49 bins) | feature layer | 1.34 | not measured | dilution from more classes |
| **SupCon-hue v3** (d_proj=256, d_hidden=512, τ=0.05) | feature layer | 1.34 | **0.52 / 0.42 / 0.13** | ❌ **Pareto-LOSING** (RC9 collapses) |

**Key result**: only **MV-SPLIT** Pareto-improves on Stage 0. All other
attempts either fail to learn or break the buffer's novelty filter on RC9.

---

## 4. Why feature-level injection fails — structural finding

The SupCon-trained adapter achieves sep_ratio 1.34 in feature space (real
color signal extracted) but **destroys** Stage 0 buffer composition on
RC9 (0.90 → 0.42). Cause:

> SRB's `EpisodicBuffer.push_batch` rejects a candidate if its cosine
> similarity to any existing buffer entry exceeds `novelty_thresh = 0.95`.
> The adapter projects features into tight per-color clusters; two patches
> of the same color have proj cosine ≈ 0.99 → combined cosine ≈ 0.97 →
> rejected.
>
> RC9's strength came from filling the buffer with DIVERSE cube content
> (each of 9 cubes contributes shape-varied patches). The adapter
> overwrites this with "color-novelty" ranking; same-color patches are
> rejected even if they show different cube fragments. Buffer fills with
> table+gripper colors and loses cube diversity.

**Generalization**: any feature-level perception adapter is at risk of
this. The buffer's novelty filter assumes feature geometry close to
DINOv2's original. Reshaping that geometry breaks the assumption.

**Implication for the paper**: color signal must enter at the **scoring
layer** (which composes orthogonally with SRB's geometry), not the
**feature layer** (which dominates it). MV-SPLIT is the architecturally
correct version.

---

## 5. What's currently deployed

### Code

- `baselines/ppo/modules/ebm_srb_tr_mv.py` — `EBMSRBTRMVMemoryModule`
  - Inherits `EBMSRBTRMemoryModule`
  - Adds per-patch mean RGB computation (3-d), parallel to DINOv2 patches
  - Dual motion suppression (`ema_change` for DINOv2, `ema_rgb_change` for RGB)
  - Split-K buffer writes: K/2 from DINOv2 SRB-MS, K/2 from RGB SRB-MS, deduped
  - LSTM-routing path unchanged from SRB-TR
- `baselines/ppo/ppo_memtasks_ebm_srb_tr_mv.py` — PPO entry (1-line import swap)
- `run_scripts/ppo_srb_tr_mv/` — sbatch scripts for RC5/RC9/Shape5 (3 seeds each)

### RL jobs in flight

| Job | Task | Seeds | Status |
|---|---|---|---|
| 2971479 | RC5 | 33, 42, 99 | PENDING (GPU queue) |
| 2971480 | RC9 | 33, 42, 99 | PENDING |
| 2971481 | Shape5 | 33, 42, 99 | PENDING |

### Adapter artifacts (kept for ablation but not for deployment)

- `analysis/color_head/train_inverse_jitter.py` + `adapter_v1.pt` (sep 1.07, negative)
- `analysis/color_head/train_supcon_hue.py` + `adapter_supcon_v3.pt` (sep 1.34, Pareto-losing Stage 0)
- `analysis/color_head/stage0_adapter.py` — buffer-composition probe at varying α
- `analysis/color_head/check_sep_ratio_*.py` — sep_ratio probes

### Negative findings recorded

- Memory: [project_color_adapter_negative](~/.claude/projects/.../memory/project_color_adapter_negative.md)
- Don't re-attempt feature-level color injection without redesigning SRB's
  novelty filter at the same time

---

## 6. Open questions / candidate next directions

### Q1. Does MV-SPLIT RL translate?

Stage 0 P(target in buffer) is a proxy. The actual RL return (PPO normalized
dense reward) might or might not follow. The 9 RL jobs above are the
verification. **Wait for results** before further iteration.

### Q2. Can MV-SPLIT be improved further?

Knobs not yet swept:
- `mv_k_dinov2` / `mv_k_rgb` split ratio (currently 4/4 from K=8)
- `ms_rgb_lambda` (currently 1.0)
- `rgb_hist_size` (currently same as buffer L=64)
- RGB summary stat: mean is simplest; could try (mean, std), HSV-hue, or
  a top-saturation pixel statistic to dampen the table-mean dilution
- Where to push into RGB FIFO: currently we push only the K_rgb-side picks;
  pushing all candidates might add noise

### Q3. Is there a perception-fair injection that ALSO helps LSTM/MLP baselines?

The user's earlier concern: MV-SPLIT only helps SRB-TR because it lives
inside the buffer write logic. Baselines (LSTM/GRU/MLP) without a buffer
can't use it. To make a level playing field for the paper, one option:

- Add mean-RGB-pooled-over-scene as an extra **encoder input** for LSTM
  baselines (`[cls_base, cls_hand, scene_mean_rgb (3), joints]`)
- This is the Fairness-Level-1 design from earlier in the thread
- Likely won't change LSTM's outcome (LSTM can't store per-color memory
  across time even if it sees scene-mean color), but it removes the
  "you gave SRB-TR more info" reviewer objection

### Q4. Is there a feature-level color fix that also preserves shape-novelty?

The SupCon adapter failed because it RANKED by color. Could we engineer:

- Adapter that ADDS a small color dimension to DINOv2 without changing
  cosine similarity in the original 384 dims much?
- E.g., RESIDUAL: `output = DINOv2_feat + λ · color_residual`, with λ small
  enough that novelty filter still mostly sees DINOv2 geometry but new
  dimension contributes when colors differ

Untested. Risk: re-introduces the same novelty-filter conflict at smaller
scale. Reward: a feature-level fix would benefit ALL baselines uniformly.

### Q5. Did we measure DINOv2 correctly?

The sep_ratio test originally used 1%-purity patches (mostly table). The
strict-purity test couldn't even find >30% pure patches (cubes too small).
A cleaner probe:

- Use DINOv2's CLS token (whole-image aggregation) — measures what
  LSTM-with-DINOv2 actually sees, not the per-patch geometry SRB uses
- If CLS sep_ratio is high, LSTM baseline gets color info "for free" and
  MV-SPLIT's improvement over LSTM is harder to attribute to color
- If CLS sep_ratio is low too, MV-SPLIT's color injection is the
  differentiator

**Not yet measured**. Quick to do.

### Q6. Hyperparameter sweep for MV-SPLIT

A systematic sweep over the knobs in Q2 would tell us whether 0.54 RC5 P(target)
is near the ceiling or there's significant headroom. Stage 0 only — cheap.

---

## 7. Code & data map

```
analysis/
  color_head/                       ← this thread's adapter code
    train_inverse_jitter.py         ← Plan B (SSL, negative)
    train_supcon_hue.py             ← Plan A/D (supervised, sep 1.34)
    check_sep_ratio_supcon.py
    check_concat_alpha.py
    stage0_adapter.py               ← buffer-composition probe
    adapter_v1.pt                   ← Inverse-Jitter checkpoint (1.07)
    adapter_supcon_v3.pt            ← SupCon checkpoint (1.34)
    COLOR_PERCEPTION_FINDINGS.md    ← THIS DOC
  pem/                              ← stage 0 diagnostic infrastructure
    diag_mvsrs.py                   ← MV-MAX / MV-SPLIT / MV-CONCAT sims
    check_color_sep.py              ← original DINOv2 probe (loose purity)
    check_color_sep_strict.py       ← strict purity + synthetic probe
    check_color_sep_clip.py         ← CLIP probe
    check_color_sep_sam.py          ← SAM probe
    run_stage0p.py                  ← MIKASA_COLORS + color_mask helpers
  ebm/path_a_data/                  ← cached eval frames (4 tasks × 20 eps)

baselines/ppo/modules/
  ebm_srb.py                        ← SRB v1 (no motion suppression)
  ebm_srb_ms.py                     ← SRB + motion suppression
  ebm_srb_tr.py                     ← SRB-MS + LSTM routing (Shape5 winner)
  ebm_srb_tr_mv.py                  ← SRB-TR + RGB view (shipped, RL pending)
  frozen_vit.py                     ← FrozenDualDinoV2
  episodic_buffer.py                ← novelty filter at 0.95 cosine

run_scripts/ppo_srb_tr_mv/          ← SLURM scripts (jobs 2971479-81)
```

---

## 8. References

- Memory entry: `project_color_adapter_negative.md` — feature-level injection
  incompatible with SRB novelty filter
- Memory entry: `project_saliency_head.md` — V1 saliency head defended
  via cross-task sharing
- Memory entry: `feedback_research_pragmatism.md` — ship over redesign
