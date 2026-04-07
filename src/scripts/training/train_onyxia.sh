cd ~/work/RL-project
export PYTHONPATH=~/work/RL-project/src:$PYTHONPATH

for config in baseline greedy; do
    for seed in 0 1 2; do
        uv run python -m src.scripts.training.train_double_dqn \
            --config $config \
            --seed $seed \
            --output-dir data/training/long/extension &
    done
done

# DQN baseline
for seed in 0 1 2; do
    uv run python -m src.scripts.training.train \
        --model dqn \
        --seed $seed \
        --output-dir data/training/long &
done

# DQN greedy
for seed in 0 1 2; do
    uv run python -m src.scripts.training.extension_train \
        --extension greedy \
        --model dqn \
        --seed $seed \
        --output-dir data/training/long/extension &
done

# DQN security
for seed in 0 1 2; do
    uv run python -m src.scripts.training.extension_train \
        --extension security \
        --model dqn \
        --seed $seed \
        --output-dir data/training/long/extension &
done

# DQN dense
for seed in 0 1 2; do
    uv run python -m src.scripts.training.extension_train \
        --extension dense \
        --model dqn \
        --seed $seed \
        --output-dir data/training/long/extension &
done

wait
echo "Tous les trainings sont finis !"