from __future__ import annotations
from collections import deque
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from imitation.models.lstm import LSTMPolicy
from imitation.preprocessing import _configure_planner_cache


class OvercookedRLWrapper(gym.Env):
    "Wrapper class to make overcooked-ai into gym env using same featurization as imitation learning"

    def __init__(
        self,
        layout_name: str,
        gym_partner: Any = None,
        gym_partners: list[Any] | None = None,
        planner_cache_dir: str = "./planner_cache",
        seq_len: int = 20,
        reward_shaping_coef: float = 0.0,
        debug: bool = False,
    ):

        super().__init__()
        _configure_planner_cache(planner_cache_dir)

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=4800)

        dummy_state = self.mdp.get_standard_start_state()
        dummy_feat = self.mdp.featurize_state(dummy_state, self.mlam)[0]

        self.action_space = gym.spaces.Discrete(6)  # up, down, right, left, stay, interact
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(dummy_feat),), dtype=np.float32)

        self.gym_partner = gym_partner
        self.gym_partners = list(gym_partners) if gym_partners else []
        self.seq_len = seq_len
        self.reward_shaping_coef = float(reward_shaping_coef)
        self.debug = bool(debug)

        self.latest_obs_p1 = None
        self.obs_history: deque[np.ndarray] | None = None  # rolling buffer for LSTM partner

    def _partner_is_lstm(self, partner: Any) -> bool:
        return isinstance(partner, LSTMPolicy) or hasattr(partner, "lstm")

    @property
    def _is_lstm(self) -> bool:
        return self.gym_partner is not None and self._partner_is_lstm(self.gym_partner)

    def _get_observations(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        feats = self.mdp.featurize_state(state, self.mlam)
        return np.array(feats[0], dtype=np.float32), np.array(feats[1], dtype=np.float32)

    def set_partner_pool(self, partners: list[Any] | None) -> None:
        self.gym_partners = list(partners) if partners else []

    def _choose_partner(self) -> None:
        if self.gym_partners:
            if hasattr(self, "np_random"):
                partner_idx = int(self.np_random.integers(low=0, high=len(self.gym_partners)))
            else:
                partner_idx = int(np.random.randint(0, len(self.gym_partners)))
            self.gym_partner = self.gym_partners[partner_idx]

        if self.gym_partner is not None and hasattr(self.gym_partner, "reset"):
            self.gym_partner.reset()

    @staticmethod
    def _extract_shaped_reward(info: dict[str, Any]) -> float:
        shaped_values = info.get("shaped_r_by_agent")
        if shaped_values is None:
            shaped_values = info.get("shaped_reward_by_agent")
        if shaped_values is None:
            return 0.0
        if isinstance(shaped_values, (list, tuple, np.ndarray)):
            return float(np.sum(shaped_values))
        return float(shaped_values)

    def _compute_team_reward(self, sparse_reward: float, info: dict[str, Any]) -> tuple[float, float]:
        shaped_reward = self._extract_shaped_reward(info)
        team_reward = float(sparse_reward + self.reward_shaping_coef * shaped_reward)
        return team_reward, shaped_reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._choose_partner()
        self.env.reset()
        obs_p0, obs_p1 = self._get_observations(self.env.state)
        self.latest_obs_p1 = obs_p1

        if self.gym_partner is not None and self._partner_is_lstm(self.gym_partner):
            # pre-fill history with the initial observation
            self.obs_history = deque([obs_p1] * self.seq_len, maxlen=self.seq_len)
        else:
            self.obs_history = None

        return obs_p0, {}

    def step(self, action_p0: int | np.ndarray):

        if isinstance(action_p0, np.ndarray):
            idx_p0 = int(action_p0.item())
        else:
            idx_p0 = int(action_p0)

        action_p0_conv = Action.INDEX_TO_ACTION[idx_p0]

        if self.gym_partner:
            with torch.no_grad():
                if self._partner_is_lstm(self.gym_partner):
                    if self.obs_history is None:
                        self.obs_history = deque([self.latest_obs_p1] * self.seq_len, maxlen=self.seq_len)
                    # stack rolling history into [1, seq_len, input_dim] for LSTM
                    seq = np.stack(list(self.obs_history), axis=0)
                    obs_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                else:
                    # single timestep [1, input_dim] for MLP
                    obs_tensor = torch.tensor(self.latest_obs_p1, dtype=torch.float32).unsqueeze(0)
                pred = self.gym_partner(obs_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                idx_p1 = torch.argmax(pred, dim=1).item()
            action_p1_conv = Action.INDEX_TO_ACTION[idx_p1]
        else:
            action_p1_conv = Action.STAY

        joint_action = (action_p0_conv, action_p1_conv)
        next_state, sparse_reward, term, info = self.env.step(joint_action)
        obs_p0, obs_p1 = self._get_observations(next_state)
        self.latest_obs_p1 = obs_p1

        if self.gym_partner and self._partner_is_lstm(self.gym_partner) and self.obs_history is not None:
            self.obs_history.append(obs_p1)

        trunc = False
        if not isinstance(info, dict):
            info = {"raw_info": info}
        team_reward, shaped_reward = self._compute_team_reward(float(sparse_reward), info)
        info["sparse_reward"] = float(sparse_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["team_reward"] = float(team_reward)

        if self.debug:
            print(
                f"P0 Action: {action_p0_conv} | P1 Action: {action_p1_conv} | "
                f"Sparse Reward: {sparse_reward} | Team Reward: {team_reward}"
            )
        return obs_p0, team_reward, term, trunc, info
