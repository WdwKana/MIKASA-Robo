# v1 Paper Snapshot — frozen 2026-05-15

This directory captures the version of results we treat as "Paper v1".
Source-of-truth CSVs remain in checkpoints/.../training_metrics.csv (not copied).
Plots and summary CSVs in this dir are a snapshot for reference.

## Main table (7 tasks × 4 methods × 3 seeds): COMPLETE
RC5, RC9, Shape5, Shape9, SAC3x2, SAC3x3, InterceptMedium — all done.

## Ablation status at snapshot time
- A6-GRU (DINOv2+sal + GRU): RC5 done (3 seeds), Shape5 partial (1/3 seeds)
- A6-LSTM, A6-MLP: PD in queue
- A3 (no_saliency): RC5 done (3 seeds), Shape5 partial
- A_backbone (CLIP): RC5 + Shape5 done

## Code files preserved (referenced by hash via git):
- baselines/ppo/ppo_memtasks_ebm.py
- baselines/ppo/modules/*  (frozen_vit, frozen_clip, saliency_head, episodic_buffer, memory_reader, ebm, dinov2_saliency_encoder)
- analysis/ebm/path_a_head_v3.pt, path_a_head_v3_clip.pt
- run_scripts/ppo_ablation/, run_scripts/ppo/

## DO NOT delete this dir. Future "v2" iterations write to other dirs.
