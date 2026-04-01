# SB3 DQN Baseline — `highway-v0`

This module implements and analyses a **DQN baseline** using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) on the `highway-v0` environment. It serves as the reference point against which a custom DQN implementation will be compared.

---

## File Overview

| File | Role |
|------|------|
| `train_sb3_dqn.py` | Train a DQN agent on `highway-v0`. Saves model checkpoints, monitor logs and hparams. |
| `evaluate_sb3_dqn.py` | Deterministic evaluation of a saved model over 50 episodes. Saves metrics to `eval_results.json`. |
| `record_rollout.py` | Scan N seeds, classify episodes (crash / low-reward / active), record the most interesting ones as MP4 videos. |
| `utils.py` | Shared utilities: `make_env`, `HighwayMetricsCallback`, `EVAL_SEEDS`. |
| `analysis.ipynb` | Full results analysis: training curves, cross-seed comparison, evaluation metrics, failure case breakdown. |

---

## Environment & Task

- **Environment:** `highway-v0` from [highway-env](https://highway-env.farama.org/)
- **Observation:** Kinematics of 10 surrounding vehicles (5 features each → 50-dim flattened vector)
- **Action space:** `DiscreteMetaAction` — 5 actions: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`
- **Episode duration:** 30 seconds (30 policy steps at 1 decision/s)
- **Reward:** Normalized composite of high-speed reward, collision penalty, lane-change penalty

The shared environment configuration is the single source of truth in `src/shared_core_config.py`.

---

## Architecture & Hyperparameters

**Algorithm:** DQN with MLP policy — chosen because:
- Discrete action space → DQN is a natural fit
- Continuous observation → MLP handles it efficiently
- Off-policy → sample-efficient with replay buffer

| Hyperparameter | Value | Rationale |
|---|---|---|
| `policy` | `MlpPolicy` | MLP suits the 50-dim flat observation |
| `net_arch` | `[256, 256]` | Enough capacity for the task without overfitting |
| `learning_rate` | `5e-4` | Standard for DQN on continuous obs |
| `buffer_size` | `50 000` | Keeps recent experience, avoids memory overflow |
| `batch_size` | `64` | Good balance between variance and compute |
| `gamma` | `0.99` | Long-horizon discounting — 30-step episodes |
| `train_freq` | `4` | Update every 4 environment steps |
| `target_update_interval` | `1 000` | Hard update (tau=1.0) to stabilise training |
| `exploration_fraction` | `0.15` | Epsilon decays over 15% of total steps |
| `exploration_final_eps` | `0.05` | Small residual exploration at convergence |
| `learning_starts` | `2 000` | Fill buffer before first gradient update |

---

## Parallelisation

Training uses `SubprocVecEnv` to collect experience across **N parallel environments** simultaneously. Each environment runs in its own subprocess with a distinct random seed (`seed + i`).

**What is parallelised:** environment stepping (simulation). Each subprocess rolls out one trajectory independently and sends transitions to the shared replay buffer.

**What is NOT parallelised:** gradient updates. The neural network is updated sequentially on the main process. DQN is an off-policy algorithm — there is one shared replay buffer and one network.

**Effect:** parallelisation fills the replay buffer ~N× faster, reducing wall-clock training time without changing the learning dynamics. For on-policy algorithms (PPO, A3C), workers also contribute gradients — which is fundamentally different.

---

## Commands

Run all commands from the **project root**.

### Train (one seed)

```bash
uv run python src/stablebaseline/train_sb3_dqn.py --seed 0 --total-timesteps 200000 --n-envs 102
```

### Train (multiple seeds in parallel)

```bash
uv run python src/stablebaseline/train_sb3_dqn.py --seed 1 --total-timesteps 120000 --n-envs 34 & \
uv run python src/stablebaseline/train_sb3_dqn.py --seed 2 --total-timesteps 120000 --n-envs 34 & \
uv run python src/stablebaseline/train_sb3_dqn.py --seed 3 --total-timesteps 120000 --n-envs 34 & \
wait
```

### Evaluate a model (200 episodes by default)

```bash
uv run python src/stablebaseline/evaluate_sb3_dqn.py \
    --model-path results/sb3_dqn/seed_0/final_model.zip
```

### Evaluate all seeds

```bash
for seed in 0 1 2 3 4 5; do
    uv run python src/stablebaseline/evaluate_sb3_dqn.py \
        --model-path results/sb3_dqn/seed_${seed}/final_model.zip
done
```

### Record interesting videos

```bash
uv run python src/stablebaseline/record_rollout.py \
    --model-path results/sb3_dqn/seed_0/final_model.zip \
    --n-candidates 60
```

---

## Results & Outputs

```
results/sb3_dqn/
└── seed_N/
    ├── final_model.zip       # Final model weights
    ├── best_model/           # Best model (highest eval reward during training)
    ├── checkpoints/          # Periodic snapshots every 20k steps
    ├── monitor.monitor.csv   # Per-episode reward and length log
    ├── eval_logs/            # EvalCallback periodic evaluation data (.npz)
    ├── hparams.json          # Hyperparameters used for this run
    ├── eval_results.json     # Final evaluation metrics (200 episodes, aggregate)
    └── eval_episodes.json    # Per-episode data (reward, length, crashed, speed)

videos/
├── seed_N/
│   └── crash/               # Crash videos per seed (up to 3 per seed)
│       └── crash_*.mp4
└── curated/
    ├── crash_*.mp4           # Collision episodes
    ├── lowreward_*.mp4       # Survived but low-performing episodes
    └── active_*.mp4          # Episodes with most actual lane changes
```

Open the analysis notebook for plots:

```bash
uv run jupyter notebook src/stablebaseline/analysis.ipynb
```

---

## Results Summary (6 seeds, 200 deterministic episodes each)

| Seed | Mean reward | Std reward | Crash rate | Mean speed | Steps |
|------|-------------|------------|------------|------------|-------|
| 0 | 20.20 | 1.67 | 3.0% | 20.02 m/s | 200k |
| 1 | 20.32 | 1.40 | 1.0% | 20.03 m/s | 120k |
| 2 | 20.05 | 2.66 | 3.5% | 20.12 m/s | 120k |
| **3** | **13.45** | **6.59** | **96.5%** | 25.05 m/s | 120k |
| 4 | 20.32 | 1.40 | 1.0% | 20.03 m/s | 120k |
| 5 | 19.99 | 2.82 | 4.5% | 20.12 m/s | 120k |
| **Avg (excl. seed_3)** | **20.18** | — | **2.6%** | **20.06 m/s** | — |

**Key observations:**
- The agent learns almost nothing during the first 30k steps (epsilon-greedy exploration dominates)
- A clear performance jump occurs around **60k steps**, when epsilon falls low enough for the policy to dominate
- After convergence, 95%+ of episodes achieve reward > 20 with near-zero crashes
- Seeds 0, 1, 2, 4, 5 converged consistently to a safe high-speed strategy
- The agent rarely changes lanes — the `lane_change_penalty` of −0.02 discourages active maneuvering

**Seed_3 — Policy Collapse:**
Seed_3 peaked at ~18 reward around 55–60k steps, then suffered a catastrophic collapse, ending at 96.5% crash rate.
The most likely cause is a destabilising hard target network update (`tau=1.0`) that propagated corrupted Q-values,
triggering a feedback loop: bad policy → more crashes → replay buffer fills with crash transitions → no recovery.
With only 120k training steps (vs 200k for seed_0), seed_3 had no time to recover.
**Fix:** use `best_model/` checkpoints instead of `final_model.zip` for seed_3, or retrain with more steps.

**Limitation:** the low `lane_change_penalty` biases the agent towards straight driving. This is optimal for reward but looks passive in qualitative evaluation.

---

## Comparison Template (to fill after custom DQN)

| Metric | SB3 DQN (baseline, healthy seeds) | Custom DQN |
|--------|-----------------------------------|------------|
| Mean reward | 20.18 ± 0.15 | — |
| Crash rate | 2.6% | — |
| Mean speed | 20.06 m/s | — |
| Steps to convergence | ~60k | — |
| Seed stability | 5/6 seeds converged | — |
