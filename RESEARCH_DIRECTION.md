# Research Direction — Memory Architectures for Online Memory-RL

This document captures the current research direction and the load-bearing
observations behind it. It is intentionally general: it states the problem
we are trying to solve and the candidate framing, but does not prescribe a
specific implementation plan.

---

## 1. Context

Project: episodic-memory architecture for partially-observable robot RL on
the MIKASA-Robo benchmark. We are comparing memory mechanisms under a fixed
PPO backbone, dual-camera DINOv2 perception, and matched hyperparameters.

Methods currently in the comparison:
- **V1 (EBMMemoryModule)** — frozen DINOv2 → trained cross-task saliency head
  (color-BCE supervision) → top-K=8 patches/step into an episodic K-V buffer
  (L=64) with priority eviction `saliency × exp(-age / τ)` → 1-layer
  cross-attention reader with query `[proprio, curr_summary]` → fuse →
  actor/critic.
- **V2-Hybrid (EBMHybridMemoryModule)** — V1 plus a parallel GRU(128)
  branch. GRU input is `[pooled-top-K, curr_summary, proprio]`, hidden state
  goes directly into fuse alongside the buffer-retrieved feature.
- **Baselines** — A3 (V1 without saliency, K=8 random write), A6 family
  (DINOv2+saliency encoder + classical GRU/LSTM/MLP heads, no buffer),
  MemVLA-style (push-all-patches + cosine-pair-merge bank + 2-layer
  cross-attn, no saliency), CLIP-v1 perceptual backbone swap.

---

## 2. What the experiments revealed

Final-iter SR_once, matched seeds 33/42/99 unless noted.

- **Saliency-filtered write is the dominant factor on static recall.** With
  saliency removed, our V1 collapses to A3 (≈ 0.23 on RC5, ≈ 0.19 on
  Shape5), which is statistically indistinguishable from MemVLA (0.23 /
  0.15). At no-saliency, the write/eviction details we considered
  innovative do not separate from a published alternative.
- **V1 (full) wins decisively on static recall.** RC5 0.61, Shape5 0.68 —
  roughly 3× the nearest non-saliency competitor.
- **V1 is already near ceiling on the easy trajectory task we have data
  for.** IMed ≈ 0.87.
- **The parallel GRU in V2-Hybrid hurts on static recall.** RC9: V1 0.15
  vs V2-Hybrid 0.086. The GRU hidden dilutes the buffer-retrieved feature
  in fuse. Adding an integrator to a task that does not need integration
  is a net cost.
- **The drop from RC5 → RC9 is steeper than the information-theoretic
  difficulty change.** We do not yet know whether RC9 fails on the write
  side (buffer does not contain the target patches) or the read side
  (buffer contains them but the reader cannot discriminate).

---

## 3. The architectural question

Episodic K-V buffer with cross-attention is permutation-invariant set
retrieval. It captures *what was seen*; it is structurally weak at
maintaining continuous-state aggregates (*how things are evolving*).

Two design pressures pull in opposite directions:
- **Static recall** prefers a high-SNR buffer. Extra integrators (GRU) in
  parallel compete with the retrieved feature in the fuse layer and
  empirically dilute it (RC9 result above).
- **Trajectory tasks** need an explicit integrator whose state is directly
  readable by the actor/critic, because integrated quantities such as
  velocity do not live in any single buffer slot.

A "Hybrid" architecture that simply runs both in parallel pays the dilution
cost. The interesting question is whether the two systems can be made
**functionally complementary** rather than additively competitive.

---

## 4. Candidate direction — belief-state episodic memory

A recurrent belief state $b_t$ (e.g., a small GRU hidden) plays two roles:
- It is read **directly** by the actor/critic. This preserves the trajectory
  integration that the parallel V2 demonstrates on IMed.
- It also generates the **query** into the episodic buffer. The retrieval
  is steered by the agent's current belief about the task, not by the raw
  current frame as in V1.

The functional separation is:
- short-horizon continuous integration → recurrent belief
- long-horizon discrete events → episodic buffer
- belief drives retrieval; retrieved evidence informs the next belief
  update.

This is the deep-learning analog of a POMDP belief filter, with the buffer
providing the explicit long-context memory that a GRU alone cannot hold.

The dual-use is essential: a pure "GRU-as-query" design (without the direct
GRU → fuse path) would force trajectory-integrated state to be reconstructed
out of buffer slots that do not contain it, sacrificing the IMed benefit
that motivated the GRU in the first place.

---

## 5. What must be measured before committing

The framing above is only honest if the following can be shown empirically.
Until then, it is a candidate, not a contribution.

1. **V1's failure mode on RC9 must be diagnosed.** Buffer-content audit on
   failed rollouts plus linear-probe of `s_t → answer_id` will distinguish
   a write-side failure (target patches not in buffer) from a read-side
   failure (target patches present but reader cannot pick them). Belief-
   driven query only helps the second case; it is the wrong intervention
   for the first.
2. **V2-Hybrid must outperform V1 on IMed by a meaningful margin** (suggest
   ≥ 0.10 absolute SR). The trajectory-failure-of-K-V-buffer claim rests
   entirely on this.
3. **MemVLA-style must also fail on the same trajectory task.** Required
   to argue the limitation is a property of K-V buffer architectures in
   general, not a quirk of our V1.
4. **Cleaning the GRU input** (drop `pooled-top-K`, keep only
   `[proprio, curr]`) must preserve the IMed gain. If it does not, the
   "clean functional separation" framing is not supported by the data and
   we are back to ad-hoc parallel branches.

---

## 6. Risks per direction

| Direction                         | Strength                                | Main failure mode |
|-----------------------------------|-----------------------------------------|---|
| Ship V1 + saliency only           | Simplest, evidence already in           | Saliency-head supervision is the attack surface; contribution looks narrow next to MemVLA |
| Ship V2-Hybrid (parallel) as-is   | Empirical IMed gain (if confirmed)      | Hard to explain why parallel GRU is principled; dilution penalty on static tasks |
| Ship belief-state architecture    | Strongest framing, theoretical anchor   | Requires §5 evidence; if RC9 is a WRITE-side failure, belief-driven query is post-hoc rationalization |

---

## 7. Hard constraints (things we are deliberately not doing)

- **No MemVLA + saliency hybrid.** This would muddy attribution and weaken
  the comparison we already have.
- **No per-task saliency heads.** The shared cross-task head is part of the
  defense against the per-task-supervision critique and stays that way.
- **No in-place changes to V1.** V1 is the stable reference for all
  ablations. Any new architecture lives in new code with V1 preserved as-is.
- **No retraining the saliency head with a new objective on the critical
  path of this direction.** Replacing color-BCE with a self-supervised
  signal is a separate orthogonal improvement; it should not be entangled
  with the architecture work.

---

## 8. One-line summary

We are deciding between *V1 with saliency as the contribution* and *belief-
state episodic memory as the contribution*. The decision should be driven
by what the §5 diagnostics show, not by which framing reads better in an
abstract.
