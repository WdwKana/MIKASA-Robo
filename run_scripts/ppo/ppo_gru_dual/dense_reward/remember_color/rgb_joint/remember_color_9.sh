#!/bin/bash
# Dual-camera GRU baseline on RememberColor9-v0.
# Identical hyper-params to ppo_memtasks_gru.py single-cam runs except for
# the wrapper (base_camera + hand_camera concatenated channel-wise -> 6-ch RGB).

# Sanity-check before launching: verify the dual wrapper actually produces 6-ch
# RGB. Run from project root:
#   /home/s4176650/.conda/envs/mikasa/bin/python -c "...smoke test as in chat..."

for seed in 123 231 321
do
    echo "[gru_dual] seed=$seed"
    python3 baselines/ppo/ppo_memtasks_gru_dual.py \
        --env_id=RememberColor9-v0 \
        --exp-name=ppo-gru-dual-dense-remember-color-9-v0 \
        --capture-video \
        --save-model \
        --num-steps=60 \
        --num-eval-steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --no-finite-horizon-gae \
        --eval-freq=25 \
        --num-envs=256 \
        --num-minibatches=32 \
        --update-epochs=8 \
        --num-eval-envs=16 \
        --gae-lambda=0.9 \
        --gamma=0.8 \
        --learning-rate=3e-4 \
        --anneal-lr
done
