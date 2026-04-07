#!/bin/bash
cd ~/work/RL-project
export PYTHONPATH=~/work/RL-project/src:$PYTHONPATH

# DQN baseline
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/dqn/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long &
done

# DQN greedy
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/extension/greedy/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/greedy &
done

# DQN security
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/extension/security/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/security &
done

# DQN dense
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/extension/dense/seed_${seed}/checkpoint_dqn_seed_${seed}_ep_7050.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/dense &
done

# Double DQN baseline
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/extension/double_dqn/baseline/seed_${seed}/model_double_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/double_dqn/baseline &
done

# Double DQN greedy
for seed in 0 1 2; do
    uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long/extension/double_dqn/greedy/seed_${seed}/model_double_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/double_dqn/greedy &
done

wait
echo "Toutes les evaluations sont finies !"