#!/bin/bash
#SBATCH --job-name=rl_all_eval
#SBATCH --partition=gpu_tp
#SBATCH --time=02:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project

# DQN baseline
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long_onyxia/dqn/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long &
done

# DQN greedy
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long_onyxia/extension/greedy/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/greedy &
done

# DQN security
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long_onyxia/extension/security/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/security &
done

# DQN dense
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/long_onyxia/extension/dense/seed_${seed}/checkpoint_dqn_seed_${seed}_ep_7050.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/dense &
done

# Double DQN baseline
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate_extension \
        --extension shared \
        --model-path data/training/long_onyxia/extension/double_dqn/baseline/seed_${seed}/model_double_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/double_dqn/baseline &
done

# Double DQN greedy
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate_extension \
        --extension shared \
        --model-path data/training/long_onyxia/extension/double_dqn/greedy/seed_${seed}/model_double_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/long/extension/double_dqn/greedy &
done

wait
echo "Toutes les evaluations sont finies !"