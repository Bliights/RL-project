from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium

if TYPE_CHECKING:
    from collections.abc import Callable
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import highway_env  # noqa: F401

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

EVAL_SEEDS: list[int] = [100, 101, 102, 103, 104]


def make_env(
    seed: int = 0,
    monitor_path: Path | None = None,
    render_mode: str = "rgb_array",
) -> Callable[[], gymnasium.Env]:
    """Return a callable that creates and seeds the highway env with Monitor."""

    def _init() -> gymnasium.Env:
        env = gymnasium.make(
            SHARED_CORE_ENV_ID,
            config=SHARED_CORE_CONFIG,
            render_mode=render_mode,
        )
        monitor_file = str(monitor_path) if monitor_path is not None else None
        wrapped: gymnasium.Env = Monitor(env, monitor_file)
        wrapped.reset(seed=seed)
        return wrapped

    return _init


class HighwayMetricsCallback(BaseCallback):
    """Logs crash_rate, offroad_rate and mean_speed (rolling window of 10 episodes)."""

    def __init__(self) -> None:
        super().__init__()
        self._crashes: list[float] = []
        self._offroads: list[float] = []
        self._speeds: list[float] = []
        self._step_speeds: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for info, done in zip(infos, dones):
            self._step_speeds.append(float(info.get("speed", 0.0)))
            if done:
                self._crashes.append(float(info.get("crashed", False)))
                self._offroads.append(float(not info.get("on_road", True)))
                if self._step_speeds:
                    self._speeds.append(float(np.mean(self._step_speeds)))
                self._step_speeds = []

        window = 10
        if len(self._crashes) >= window:
            self.logger.record("highway/crash_rate", float(np.mean(self._crashes[-window:])))
            self.logger.record("highway/offroad_rate", float(np.mean(self._offroads[-window:])))
            self.logger.record("highway/mean_speed", float(np.mean(self._speeds[-window:])))

        return True
