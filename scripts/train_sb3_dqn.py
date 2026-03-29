from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import EVAL_SEEDS, HighwayMetricsCallback, make_env

DQN_HPARAMS: dict = {
    "policy": "MlpPolicy",
    "learning_rate": 5e-4,
    "buffer_size": 50_000,
    "learning_starts": 2_000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [256, 256]},
    "verbose": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SB3 DQN on highway-v0")
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sb3_dqn"))
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Steps between evaluations")
    parser.add_argument("--n-envs", type=int, default=1, help="Nombre d'envs parallèles")
    return parser.parse_args()


def train(
    seed: int,
    total_timesteps: int,
    output_dir: Path,
    eval_freq: int,
    n_envs: int = 1,
) -> None:
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(seed=seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    train_env = VecMonitor(vec_env, str(run_dir / "monitor"))
    eval_env = DummyVecEnv([make_env(seed=EVAL_SEEDS[0])])

    model = DQN(
        env=train_env,
        seed=seed,
        tensorboard_log=str(run_dir / "tb_logs"),
        **DQN_HPARAMS,
    )

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=eval_freq * 2,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="dqn_highway",
            ),
            HighwayMetricsCallback(),
        ],
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(run_dir / "final_model"))

    hparams_to_save = dict(DQN_HPARAMS)
    hparams_to_save["seed"] = seed
    hparams_to_save["total_timesteps"] = total_timesteps
    hparams_to_save["n_envs"] = n_envs

    with (run_dir / "hparams.json").open("w") as f:
        json.dump(hparams_to_save, f, indent=2)

    print(f"Done. Results saved to {run_dir}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    train(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        eval_freq=args.eval_freq,
        n_envs=args.n_envs,
    )
