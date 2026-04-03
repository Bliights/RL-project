#!/bin/bash
#SBATCH --job-name=rl_eval_extension
#SBATCH --partition=gpu_prod_long
#SBATCH --time=01:00:00
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_7/RL-project/logs/slurm-%j.err

cd ~/RL-project

for seed in 0 1 2; do
    ~/.local/bin/uv run python -m src.scripts.evaluation.extension_evaluate \
        --extension shared \
        --model-path data/training/extension/security/seed_${seed}/model_dqn_seed_${seed}.pt \
        --seed $seed \
        --n-episodes 10 &
done
wait
echo "Toutes les evaluations sont finies"