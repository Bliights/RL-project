#!/bin/bash
#SBATCH --job-name=rl_extension_greedy
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project
export CUDA_VISIBLE_DEVICES=""

for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.training.extension_train \
        --extension dense \
        --model dqn \
        --seed $seed \
        --output-dir data/training/extension &
done
wait
echo "Tous les trainings greedy sont finis"