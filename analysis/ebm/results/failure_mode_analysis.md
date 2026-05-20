# Path A v3 head — failure mode analysis & generalization plan

Saved 2026-05-06 after qualitative saliency check on new tasks.

## Verified working (from `path_a_v3_new_task_vis/`)
- RememberColor9 — clean hit on target color
- InterceptFast / InterceptMedium step010-030 — clean bullseye on target ring
- RememberShapeAndColor3x2 — fires on colored shape cluster

## Suspected weak (no demo frames available, inferred from task structure)
- **ShellGameTouch**: cups are gray/white plastic, NOT in 9 MIKASA colors → head likely fires only on the ball.
  - Buffer captures ball but not cup positions.
  - Mitigation candidate: spatial info via Option B (xy in buffer).
- **RememberShape variants**: if shapes are monochrome (e.g. plain cube without distinct color) → may miss.
- **InterceptMedium step050+**: when gripper occupies most of the frame, target ring < 1 patch → averaged out.

## Failure modes (in order of generality)

| # | Mode | Affected tasks | Mechanism |
|---|---|---|---|
| F1 | head silent on non-9-color objects | ShellGame, mono-color shapes | training label is "is patch in 9 MIKASA colors" |
| F2 | object too small to dominate a 14px patch | far / occluded targets | DINOv2 patch=14, object < 1 patch loses contrast |
| F3 | buffer stores features only, no spatial | all (especially position-sensitive) | reader cross-attn can't access "where in image" |
| F4 | head doesn't distinguish "task-relevant" from "any object" | sequential / multi-distractor | by design — head is an object detector, not a task-conditioned one |

## Improvement options (rank by generality vs cost)

### Option A — Multi-task augmented head training  ⭐ recommended if F1 hits
- random-rollout from 5-6 tasks, OR-combine labels: `is_mikasa_color | has_motion | is_foreground`
- single shared head, "one head no task knowledge" story preserved
- cost: 1-2 days

### Option B — Add spatial channel to buffer
- buffer entry: `(feature, timestamp, saliency, xy_pos, camera_id)` instead of `(feature, timestamp, saliency)`
- reader gets xy via K/V projection
- cost: ~half day, no head retrain needed
- fixes F3 in particular, helpful for ShellGame even if F1 still bites

### Option C — No-head saliency fallback
- write filter = `frame_diff * DINOv2_CLS_attention` (no learned head)
- catches transient events (ball appearing/disappearing) but misses static targets
- cost: ~2 hours
- value: ablation row showing learned head IS load-bearing (Spec H4)

### Option D — Per-task head
- avoid — reviewer-unfriendly story, high engineering cost

## Decision protocol — wait for data, then act

After first eval data on 8 tasks (~1 day):

| Observed pattern | Action |
|---|---|
| Only ShellGame fails | A only |
| ShellGame + Intercept both fail | A + B |
| All tasks fail (uniformly low SR) | system-level — reread Spec, debug ViT/buffer pipeline |
| ShellGame OK, Intercept fails | likely F2 — investigate ViT input size / patch_size |
| All tasks OK | celebrate; go to extended seeds + paper writing |
