from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LinearRewardShapingCallback(BaseCallback):
    """Linearly anneal ``reward_shaping_coef`` from *start_coef* to *end_coef*
    over the course of training.

    This lets PPO bootstrap off dense shaping early, then gradually shift
    to the true sparse objective so the final policy optimises deliveries.
    """

    def __init__(
        self,
        total_timesteps: int,
        start_coef: float = 1.0,
        end_coef: float = 0.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps
        self.start_coef = start_coef
        self.end_coef = end_coef

    @staticmethod
    def _unwrap_to_rl(env):
        """Unwrap through Monitor/etc. to reach OvercookedRLWrapper."""
        while hasattr(env, "env") and not hasattr(env, "reward_shaping_coef"):
            env = env.env
        return env

    def _raw_envs(self) -> list:
        if hasattr(self.training_env, "envs"):
            return [self._unwrap_to_rl(e) for e in self.training_env.envs]
        return [self._unwrap_to_rl(self.training_env)]

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.total_timesteps_target))
        coef = self.start_coef + frac * (self.end_coef - self.start_coef)

        for env in self._raw_envs():
            env.reward_shaping_coef = coef

        self.logger.record("reward_shaping/coef", float(coef))
        return True


class EpisodeRewardLoggerCallback(BaseCallback):
    """Track sparse vs shaped rewards per episode so TensorBoard shows
    whether the agent is actually delivering soups (sparse > 0).

    Handles multiple parallel environments by keeping per-env accumulators.
    """

    def __init__(self, window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._acc_sparse: dict[int, float] = {}
        self._acc_shaped: dict[int, float] = {}
        self._acc_total: dict[int, float] = {}
        self._acc_len: dict[int, int] = {}

        self._history_sparse: deque[float] = deque(maxlen=window)
        self._history_total: deque[float] = deque(maxlen=window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            self._acc_sparse.setdefault(i, 0.0)
            self._acc_shaped.setdefault(i, 0.0)
            self._acc_total.setdefault(i, 0.0)
            self._acc_len.setdefault(i, 0)

            step_info = info.get("terminal_info", info) if dones[i] else info
            self._acc_sparse[i] += step_info.get("sparse_reward", 0.0)
            self._acc_shaped[i] += step_info.get("shaped_reward", 0.0)
            self._acc_total[i] += float(rewards[i]) if i < len(rewards) else 0.0
            self._acc_len[i] += 1

            if i < len(dones) and dones[i]:
                self._history_sparse.append(self._acc_sparse[i])
                self._history_total.append(self._acc_total[i])

                self.logger.record("episode/sparse_reward", self._acc_sparse[i])
                self.logger.record("episode/shaped_reward", self._acc_shaped[i])
                self.logger.record("episode/total_reward", self._acc_total[i])
                self.logger.record("episode/length", self._acc_len[i])

                if len(self._history_sparse) > 0:
                    self.logger.record(
                        "episode/mean_sparse_reward",
                        float(np.mean(self._history_sparse)),
                    )
                    self.logger.record(
                        "episode/mean_total_reward",
                        float(np.mean(self._history_total)),
                    )

                self._acc_sparse[i] = 0.0
                self._acc_shaped[i] = 0.0
                self._acc_total[i] = 0.0
                self._acc_len[i] = 0

        return True


class BestModelCheckpoint(BaseCallback):
    """Save the model whenever rolling mean sparse reward improves."""

    def __init__(self, save_dir: str, window: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.window = window
        self._history: deque[float] = deque(maxlen=window)
        self._best_mean: float = -float("inf")
        self._acc: dict[int, float] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            self._acc.setdefault(i, 0.0)
            step_info = info.get("terminal_info", info) if dones[i] else info
            self._acc[i] += step_info.get("sparse_reward", 0.0)

            if i < len(dones) and dones[i]:
                self._history.append(self._acc[i])
                self._acc[i] = 0.0

                if len(self._history) >= self.window:
                    current = float(np.mean(self._history))
                    if current > self._best_mean:
                        self._best_mean = current
                        self.save_dir.mkdir(parents=True, exist_ok=True)
                        path = self.save_dir / "best_model"
                        self.model.save(str(path))
                        if self.verbose:
                            print(f"[best-ckpt] New best mean_sparse={current:.1f} "
                                  f"at step {self.num_timesteps:,} -> {path}")
                        self.logger.record("best/mean_sparse_reward", current)

        return True
