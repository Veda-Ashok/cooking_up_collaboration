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

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.total_timesteps_target))
        coef = self.start_coef + frac * (self.end_coef - self.start_coef)

        if hasattr(self.training_env, "envs"):
            for env in self.training_env.envs:
                env.reward_shaping_coef = coef
        else:
            self.training_env.reward_shaping_coef = coef

        self.logger.record("reward_shaping/coef", float(coef))
        return True
