# SRB-TR — Remote Experiment Specification

This document tells a second machine how to reproduce and extend our experiments.
Our method is an **online-RL** memory module for partial-observability robot
manipulation (MIKASA-Robo). Everything below is exact as of this commit.

---

## 0. TL;DR

- **Our method (the ONE file to run):**
  `baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py`
- **Run one job (pin to GPU 0):**
  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py \
      --env_id=ShellGameTouch-v0 --exp-name=srbtr-crescaps-seed33 \
      --capture-video --save-model \
      --num-steps=60 --num-eval-steps=720 --eval-freq=24 \
      --include-rgb --include-joints --seed=33 \
      --total-timesteps=7_000_000 --no-finite-horizon-gae \
      --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
      --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
      --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15
  ```
- **Or just use the helper:** `bash run_remote_cuda0.sh` (edits at top of file).
- **One run ≈ 16–17 GB VRAM, ≈ 8 h on an L4.** Run **one at a time** on a 24 GB card.

---

## 1. Environment setup

```bash
git clone git@github.com:WdwKana/MIKASA-Robo.git
cd MIKASA-Robo
git checkout <BRANCH>                      # see the prompt for the branch name
conda create -n mikasa python=3.10 -y && conda activate mikasa   # or venv
pip install -e .                           # uses setup.py + requirements.txt
pip install tyro tensorboard               # extras the PPO scripts need (if missing)
```

Key pins (from `requirements.txt`): `torch==2.2.1`, `mani_skill==3.0.0b15`,
`gymnasium==0.29.1`, `numpy==1.23.5`. Python ≥ 3.9.

**HARD REQUIREMENT — Vulkan / a real GPU.** MIKASA renders camera RGB through
SAPIEN (svulkan2). The machine must expose a **full GPU with graphics/Vulkan**.
**MIG slices do NOT work** — they are compute-only and fail at `gym.make` with
`svulkan2 ... Vulkan ... ErrorIncompatibleDriver`. Verify with `vulkaninfo` and a
1-minute smoke run before launching a batch.

**Single-GPU constraint (this machine).** Only GPU 0 is usable → every command
sets `CUDA_VISIBLE_DEVICES=0`. The entry scripts use the default CUDA device, so
this env var is the only thing needed to pin them. **Do not run two jobs on one
24 GB GPU** — each needs ~16–17 GB and the second will OOM on the camera buffer.
DINOv2 weights download automatically from HuggingFace on first run.

---

## 2. File map

### Our method
| Role | File |
|---|---|
| **Method entry (run this)** | `baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py` |
| **Method module (the actual mechanism)** | `baselines/ppo/modules/ebm_srb_tr_cres.py` |
| ↳ base SRB-TR (surprise + motion routing + LSTM) | `baselines/ppo/modules/ebm_srb_tr.py` |
| ↳ episodic K-V buffer (write/evict/novelty filter) | `baselines/ppo/modules/episodic_buffer.py` |
| ↳ cross-attention reader | `baselines/ppo/modules/memory_reader.py` |
| ↳ frozen DINOv2 perception | `baselines/ppo/modules/frozen_vit.py` |

### Baselines (perception-matched, same tricks — for fair comparison)
| Method | Entry | Encoder module |
|---|---|---|
| GRU | `baselines/ppo/ppo_memtasks_dinov2_gru_cres_caps.py` | `modules/dinov2_simple_encoder.py` |
| LSTM | `baselines/ppo/ppo_memtasks_dinov2_lstm_cres_caps.py` | `modules/dinov2_simple_encoder.py` |
| MLP (pure PPO, no memory) | `baselines/ppo/ppo_memtasks_dinov2_mlp_cres_caps.py` | `modules/dinov2_simple_encoder.py` |

All four share the **same frozen DINOv2 backbone** → any gap over baselines is
attributable to the memory mechanism, not perception.

---

## 3. The two "fair tricks" (applied to ALL methods equally)

These are auxiliary fairness mechanisms, NOT the contribution. They are given to
every method (ours + all baselines) so the comparison is clean.

1. **CRES — Color RESidual injection.** Frozen DINOv2 underweights color; CRES adds
   a dim-preserving color signal so color tasks aren't perception-limited.
   - For **ours**: built into `modules/ebm_srb_tr_cres.py` (`_fuse_color`): per-patch
     centered mean-RGB projected by a FIXED random matrix (seed 1234), scale
     auto-calibrated to `color_frac≈0.4·‖dino‖` then frozen. Dim-preserving.
   - For **baselines**: `modules/dinov2_simple_encoder.py` with `color_aug=True` —
     concatenates the per-patch centered mean-RGB map (162×3=486) to the feature.
   - Both hand the same color information to the model; nothing learned.

2. **CAPS — action smoothness.** Temporal `λ·‖π_mean(s_t)−π_mean(s_{t+1})‖²`,
   **actor-head only** (gradient never reaches the memory). Fixes "settling" so the
   arm holds still at episode end. Flag: **`--caps-lambda-t=0.15`** (same for all
   methods, all tasks).

---

## 4. Hyperparameters

### 4a. Command-line (identical across all tasks except the schedule below)
See the TL;DR command. The full canonical flag set lives in
`run_scripts/aaai_final/*.slurm` (those are SLURM wrappers; on this machine use
`run_remote_cuda0.sh` instead, which has the same flags minus SLURM).

### 4b. Per-task schedule (the ONLY thing that varies by task)
```
if env starts with "Intercept":  --num-steps 90  --num-eval-steps 540  --eval-freq 16
else (everything else):          --num-steps 60  --num-eval-steps 720  --eval-freq 24
```
ShellGame / Rotate / TakeItBack / capacity tasks all use the **else** branch.

### 4c. Memory-module hyperparameters (FIXED — do not tune per task)
Defaults in `modules/ebm_srb_tr.py.__init__`, used uniformly for every task. The
run commands do **not** override them, and that uniformity is a deliberate claim
("training-free write rule, 6 fixed hyperparameters held constant across all tasks").

| Param | Value | Meaning |
|---|---|---|
| `K` | 8 | tokens written to buffer per step |
| `L` | 64 | buffer capacity (slots) |
| `novelty_thresh` | 0.95 | cosine dedup threshold on write |
| `tau_age` | 30.0 | recency-eviction time constant |
| `ms_alpha` | 0.1 | motion EMA decay |
| `ms_lambda` | 1.0 | motion-suppression strength |

**Do not change these** unless explicitly running the K/L/tau ablation (Goal 4).

### 4d. Seeds & output
- Seeds: **33, 42, 99** (run all three per (method, task)).
- Output: written under `checkpoints/<...>/<env>/<exp-name>__<seed>__.../` with a
  per-run `training_metrics.csv` (this is the file to read; see §6).

---

## 5. What to run (priorities)

For **every task**, run **all 4 methods × 3 seeds** (srbtr, gru, lstm, mlp) so the
comparison is complete. ~8 h/run on one GPU → budget accordingly; if time is tight,
do `srbtr, gru, lstm` first and defer `mlp`.

### GOAL 1 — finish the rest of MIKASA-Robo
**Tier A (run first — single/few-cue, most informative):**
`ShellGameTouch-v0`, `ShellGamePush-v0`, `ShellGamePick-v0`,
`RotateStrictPos-v0`, `RotateStrictPosNeg-v0`,
`RotateLenientPos-v0`, `RotateLenientPosNeg-v0`, `TakeItBack-v0`

**Tier B (compositional extra):** `RememberShapeAndColor5x3-v0`

**Tier C (capacity — LOW value; all baselines tend to ~0 success, so this mostly
shows ours-vs-zero):** `SeqOfColors{3,5,7}-v0`, `BunchOfColors{3,5,7}-v0`,
`ChainOfColors{3,5,7}-v0`. Run only if Tier A/B are done and time remains.

### GOAL 2 — validate on a SECOND robot-memory benchmark
This needs light integration work, not turnkey. Pick a benchmark that exposes
**image/RGB observations under partial observability** so our `RGB → DINOv2 patches`
pipeline applies. Candidates, most→least compatible:
- **RoboMemArena** (arXiv:2605.10921) — robot memory benchmark, closest in spirit.
- **Memory-Gym** (Mortar Mayhem / Mystery Path) — image-based memory RL.
- **Memory-Maze** — 3D egocentric image memory RL.
Scope first: confirm it gives RGB frames + a Gym(nasium) API, then write a thin obs
adapter feeding `frozen_vit.py`. **If Goal 2 looks more decisive than Goal 1 for
the paper, do Goal 2 first** — order is not fixed.

### GOAL 3 (LOW priority — likely done on the other machine "alice")
Characterize **where LSTM fails** on these memory tasks (e.g., it collapses on
`success_at_end`, 9-way discrimination, shape tasks) to support the narrative.
Source data: the `training_metrics.csv` files (§6).

### GOAL 4 (LOW priority — likely done on "alice")
Ablations: (a) **write rule** `surprise vs FIFO vs random` at fixed L (the decisive
one), (b) **motion-routing on/off**, (c) **CRES on/off**, (d) **CAPS on/off**,
(e) **K/L/tau sensitivity**. These require small code switches; coordinate before
starting so we don't duplicate "alice".

---

## 6. Reading results

Each run writes `.../training_metrics.csv` with columns:
`iteration,total_env_steps,mode,timestamp,success_once,return,episode_len,reward,success_at_end`.
- Filter `mode=="eval"` rows.
- **Primary metric = eval-last3**: mean of `success_once`, `success_at_end`, `return`
  over the **last 3 eval rows**; then average across the 3 seeds (report mean ± std).
- A complete run has ~19 eval rows; fewer = still running / crashed (check the log).
- A 1–5 min crash with an empty run dir is usually the **camera-buffer OOM** from GPU
  contention — just rerun that (method, task, seed); it is not a config bug.

---

## 7. Coordination notes
- Use the **same output directory convention** and the **same exp-name pattern**
  (`<method>-crescaps-seed<SEED>`) so results merge cleanly with "alice".
- Do **not** change §4c memory hyperparameters or the §3 tricks — they must match
  "alice" exactly for the runs to be poolable.
- Push `training_metrics.csv` results back (or sync the `checkpoints/` tree) so both
  machines' numbers can be aggregated into one table.
