#!/bin/bash
#SBATCH --job-name=rl_double_dqn
#SBATCH --partition=gpu_tp
#SBATCH --time=02:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project

export CUDA_VISIBLE_DEVICES=""

for config in baseline greedy; do
    for seed in 0 1 2; do
        ~/.local/bin/uv run python -m src.scripts.training.train_double_dqn \
            --config $config \
            --seed $seed \
            --output-dir data/training/extension &
    done
done
wait
echo "Tous les trainings Double DQN sont finis"