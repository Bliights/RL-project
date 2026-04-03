for seed in 0 1 2; do
    uv run python -m src.scripts.training.train_extension \
        --extension greedy \
        --model dqn \
        --seed $seed \
        --output-dir data/training/extension &
done
wait
echo "Tous les trainings sont finis"