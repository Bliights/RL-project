#!/bin/bash
#SBATCH --job-name=rl_all_training
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project

# DQN baseline
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.train \
        --model dqn --seed $seed \
        --output-dir data/training/long &
done

# DQN greedy
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.extension_train \
        --extension greedy --model dqn --seed $seed \
        --output-dir data/training/long/extension &
done

# DQN security
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.extension_train \
        --extension security --model dqn --seed $seed \
        --output-dir data/training/long/extension &
done

# DQN dense
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.extension_train \
        --extension dense --model dqn --seed $seed \
        --output-dir data/training/long/extension &
done

# Double DQN baseline
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.train_double_dqn \
        --config baseline --seed $seed \
        --output-dir data/training/long/extension &
done

# Double DQN greedy
for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.train_double_dqn \
        --config greedy --seed $seed \
        --output-dir data/training/long/extension &
done

wait
echo "Tous les trainings sont finis !"