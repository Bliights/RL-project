#!/bin/bash
#SBATCH --job-name=rl_eval_greedy
#SBATCH --partition=gpu_tp
#SBATCH --time=02:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project
export CUDA_VISIBLE_DEVICES=""

for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.evaluate \
        --model-path data/training/extension/dense/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 \
        --output-dir data/evaluation/extension/dense &
done
wait
echo "Toutes les evaluations greedy sont finies"