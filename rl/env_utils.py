import random
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
    """Gym wrapper for Overcooked with dual observation modes.

    obs_mode controls what the RL *learner* sees:
        "featurized" -> 96-D hand-crafted vector (same as BC training)
        "lossless"   -> (H, W, C) spatial tensor (CNN input)

    partner_obs_mode controls what the BC/self-play *partner* sees:
        always "featurized" in practice, since BC models expect the 96-D vector.
    """

    def __init__(
        self,
        layout_name: str,
        gym_partner: Any = None,
        gym_partners: list[Any] | None = None,
        planner_cache_dir: str = "./planner_cache",
        horizon: int = 400,
        seq_len: int = 20,
        reward_shaping_coef: float = 0.0,
        obs_mode: str = "featurized",
        partner_obs_mode: str = "featurized",
        debug: bool = False,
    ):

        super().__init__()
        _configure_planner_cache(planner_cache_dir)

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.horizon = horizon
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        self.obs_mode = obs_mode
        self.partner_obs_mode = partner_obs_mode

        dummy_state = self.mdp.get_standard_start_state()
        dummy_obs_p0, _ = self._encode_state(dummy_state, self.obs_mode)

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=dummy_obs_p0.shape, dtype=np.float32,
        )

        self.gym_partner = gym_partner
        self.gym_partners = list(gym_partners) if gym_partners else []
        self.seq_len = seq_len
        self.reward_shaping_coef = float(reward_shaping_coef)
        self.debug = bool(debug)

        self.latest_partner_obs_p1: np.ndarray | None = None
        self.obs_history: deque[np.ndarray] | None = None

        self._partner_pos_history: deque[tuple | None] = deque(maxlen=4)
        _STAY = (0, 0)
        self._UNSTICK_INDICES = [
            i for i, a in enumerate(Action.ALL_ACTIONS) if a != _STAY
        ]

        self._prev_phi: float | None = None
        self._phi_gamma = 0.99

    def _encode_state(self, state: Any, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode == "lossless":
            enc = self.mdp.lossless_state_encoding(state, horizon=self.horizon)
            return np.array(enc[0], dtype=np.float32), np.array(enc[1], dtype=np.float32)
        if mode == "featurized":
            feats = self.mdp.featurize_state(state, self.mlam)
            return np.array(feats[0], dtype=np.float32), np.array(feats[1], dtype=np.float32)
        raise ValueError(f"Unknown obs mode: {mode}")

    def _partner_is_lstm(self, partner: Any) -> bool:
        return isinstance(partner, LSTMPolicy) or hasattr(partner, "lstm")

    @property
    def _is_lstm(self) -> bool:
        return self.gym_partner is not None and self._partner_is_lstm(self.gym_partner)

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

    # Reward shaping
    _EVENT_PENALTIES: dict[str, float] = {
        "catastrophic_onion_potting": -0.20,
        "catastrophic_tomato_potting": -0.20,
        "useless_onion_potting": -0.10,
        "useless_tomato_potting": -0.10,
        "soup_drop": -0.30,
    }

    def _compute_phi(self, state: Any) -> float:
        """Compute potential phi(s) using the MDP's built-in potential function."""
        try:
            return float(self.mdp.potential_function(state, self.mlam, gamma=self._phi_gamma))
        except Exception:
            return 0.0

    @classmethod
    def _extract_event_penalties(cls, info: dict[str, Any]) -> float:
        agent_infos = info.get("agent_infos", [])
        total = 0.0
        for ai in agent_infos:
            if not isinstance(ai, dict):
                continue
            for event_key, penalty in cls._EVENT_PENALTIES.items():
                count = ai.get(event_key, 0)
                if isinstance(count, (int, float)) and count > 0:
                    total += penalty * count
        return total

    def _compute_team_reward(self, sparse_reward: float, info: dict[str, Any],
                             next_state: Any) -> tuple[float, float]:
        # Potential-based reward shaping: F(s,s') = phi(s') - phi(s)
        # When agents make progress toward correct recipes, phi increases -> positive shaping.
        # When agents waste actions or regress, phi decreases -> negative shaping.
        # Telescopes to phi(s_T) - phi(s_0) over the episode -- no systematic drift.
        phi_new = self._compute_phi(next_state)
        phi_delta = 0.0
        if self._prev_phi is not None:
            phi_delta = phi_new - self._prev_phi
        self._prev_phi = phi_new

        event_penalties = self._extract_event_penalties(info)
        shaped_reward = phi_delta + event_penalties
        team_reward = float(sparse_reward + self.reward_shaping_coef * shaped_reward)
        return team_reward, shaped_reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._choose_partner()
        if self.gym_partner is not None and hasattr(self.gym_partner, "on_episode_start"):
            self.gym_partner.on_episode_start()
        self._partner_pos_history.clear()
        self.env.reset()
        self._prev_phi = self._compute_phi(self.env.state)

        obs_p0, _ = self._encode_state(self.env.state, self.obs_mode)
        _, partner_obs_p1 = self._encode_state(self.env.state, self.partner_obs_mode)
        self.latest_partner_obs_p1 = partner_obs_p1

        if self.gym_partner is not None and self._partner_is_lstm(self.gym_partner):
            self.obs_history = deque([partner_obs_p1] * self.seq_len, maxlen=self.seq_len)
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
                        self.obs_history = deque(
                            [self.latest_partner_obs_p1] * self.seq_len, maxlen=self.seq_len,
                        )
                    seq = np.stack(list(self.obs_history), axis=0)
                    obs_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                else:
                    obs_tensor = torch.tensor(
                        self.latest_partner_obs_p1, dtype=torch.float32,
                    ).unsqueeze(0)
                pred = self.gym_partner(obs_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                logits = pred.squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                idx_p1 = torch.multinomial(probs, 1).item()

            p1_pos = None
            try:
                p1_pos = self.env.state.players[1].position
            except Exception:
                pass
            self._partner_pos_history.append(p1_pos)
            if (
                len(self._partner_pos_history) >= 3
                and p1_pos is not None
                and all(p == p1_pos for p in self._partner_pos_history)
            ):
                idx_p1 = random.choice(self._UNSTICK_INDICES)

            action_p1_conv = Action.INDEX_TO_ACTION[idx_p1]
        else:
            action_p1_conv = Action.STAY

        joint_action = (action_p0_conv, action_p1_conv)
        next_state, sparse_reward, term, info = self.env.step(joint_action)

        obs_p0, _ = self._encode_state(next_state, self.obs_mode)
        _, partner_obs_p1 = self._encode_state(next_state, self.partner_obs_mode)
        self.latest_partner_obs_p1 = partner_obs_p1

        if self.gym_partner and self._partner_is_lstm(self.gym_partner) and self.obs_history is not None:
            self.obs_history.append(partner_obs_p1)

        trunc = False
        if not isinstance(info, dict):
            info = {"raw_info": info}
        team_reward, shaped_reward = self._compute_team_reward(float(sparse_reward), info, next_state)
        info["sparse_reward"] = float(sparse_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["team_reward"] = float(team_reward)

        if self.debug:
            print(
                f"P0 Action: {action_p0_conv} | P1 Action: {action_p1_conv} | "
                f"Sparse Reward: {sparse_reward} | Team Reward: {team_reward}"
            )
        return obs_p0, team_reward, term, trunc, info
