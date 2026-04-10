import random
from collections import deque
from functools import partial
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from imitation.models.lstm import LSTMPolicy
from imitation.preprocessing import _configure_planner_cache


def make_overcooked_vec_env(
    n_envs: int = 8, **env_kwargs,
) -> DummyVecEnv:
    """Create a DummyVecEnv with *n_envs* ``OvercookedRLWrapper`` instances.

    Each sub-env is wrapped with SB3's ``Monitor`` so that episode reward /
    length stats (``ep_info_buffer``) are populated correctly.
    """

    def _make(idx: int):
        def _init():
            return Monitor(OvercookedRLWrapper(**env_kwargs))
        return _init

    return DummyVecEnv([_make(i) for i in range(n_envs)])


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
        reward_clip: float = 5.0,
        obs_mode: str = "featurized",
        partner_obs_mode: str = "featurized",
        player_idx: int | str = 0,
        debug: bool = False,
    ):
        """
        player_idx: which slot the RL learner occupies.
            0          -- always player 0 (default, original behavior)
            1          -- always player 1
            "alternate" -- randomly 0 or 1 each episode
        """
        super().__init__()
        _configure_planner_cache(planner_cache_dir)

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.horizon = horizon
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        self.obs_mode = obs_mode
        self.partner_obs_mode = partner_obs_mode

        self._player_idx_config = player_idx
        self.player_idx: int = 0 if player_idx == "alternate" else int(player_idx)

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
        self.reward_clip = float(reward_clip)
        self.debug = bool(debug)

        self.latest_partner_obs: np.ndarray | None = None
        self.obs_history: deque[np.ndarray] | None = None

        self._partner_pos_history: deque[tuple | None] = deque(maxlen=4)
        _STAY = (0, 0)
        self._UNSTICK_INDICES = [
            i for i, a in enumerate(Action.ALL_ACTIONS) if a != _STAY
        ]

        self._prev_phi: float | None = None
        self._phi_gamma = 0.99

    def _encode_state_raw(self, state: Any, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (player0_obs, player1_obs) without any index swapping."""
        if mode == "lossless":
            enc = self.mdp.lossless_state_encoding(state, horizon=self.horizon)
            return np.array(enc[0], dtype=np.float32), np.array(enc[1], dtype=np.float32)
        if mode == "featurized":
            feats = self.mdp.featurize_state(state, self.mlam)
            return np.array(feats[0], dtype=np.float32), np.array(feats[1], dtype=np.float32)
        raise ValueError(f"Unknown obs mode: {mode}")

    def _encode_state(self, state: Any, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (learner_obs, partner_obs) respecting self.player_idx."""
        p0, p1 = self._encode_state_raw(state, mode)
        if self.player_idx == 0:
            return p0, p1
        return p1, p0

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
        if self.reward_shaping_coef == 0.0:
            return float(sparse_reward), 0.0

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

        if self._player_idx_config == "alternate":
            if hasattr(self, "np_random"):
                self.player_idx = int(self.np_random.integers(0, 2))
            else:
                self.player_idx = int(np.random.randint(0, 2))

        self._choose_partner()
        if self.gym_partner is not None and hasattr(self.gym_partner, "on_episode_start"):
            self.gym_partner.on_episode_start()
        self._partner_pos_history.clear()
        self.env.reset()
        self._prev_phi = self._compute_phi(self.env.state)

        learner_obs, _ = self._encode_state(self.env.state, self.obs_mode)
        _, partner_obs = self._encode_state(self.env.state, self.partner_obs_mode)
        self.latest_partner_obs = partner_obs

        if self.gym_partner is not None and self._partner_is_lstm(self.gym_partner):
            self.obs_history = deque([partner_obs] * self.seq_len, maxlen=self.seq_len)
        else:
            self.obs_history = None

        return learner_obs, {}

    def step(self, learner_action: int | np.ndarray):

        if isinstance(learner_action, np.ndarray):
            learner_idx = int(learner_action.item())
        else:
            learner_idx = int(learner_action)

        learner_action_conv = Action.INDEX_TO_ACTION[learner_idx]

        partner_idx = 1 - self.player_idx

        if self.gym_partner:
            with torch.no_grad():
                if self._partner_is_lstm(self.gym_partner):
                    if self.obs_history is None:
                        self.obs_history = deque(
                            [self.latest_partner_obs] * self.seq_len, maxlen=self.seq_len,
                        )
                    seq = np.stack(list(self.obs_history), axis=0)
                    obs_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                else:
                    obs_tensor = torch.tensor(
                        self.latest_partner_obs, dtype=torch.float32,
                    ).unsqueeze(0)
                pred = self.gym_partner(obs_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                logits = pred.squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                partner_action_idx = torch.multinomial(probs, 1).item()

            partner_pos = None
            try:
                partner_pos = self.env.state.players[partner_idx].position
            except Exception:
                pass
            self._partner_pos_history.append(partner_pos)
            if (
                len(self._partner_pos_history) >= 3
                and partner_pos is not None
                and all(p == partner_pos for p in self._partner_pos_history)
            ):
                partner_action_idx = random.choice(self._UNSTICK_INDICES)

            partner_action_conv = Action.INDEX_TO_ACTION[partner_action_idx]
        else:
            partner_action_conv = Action.STAY

        if self.player_idx == 0:
            joint_action = (learner_action_conv, partner_action_conv)
        else:
            joint_action = (partner_action_conv, learner_action_conv)

        next_state, sparse_reward, term, info = self.env.step(joint_action)

        learner_obs, _ = self._encode_state(next_state, self.obs_mode)
        _, partner_obs = self._encode_state(next_state, self.partner_obs_mode)
        self.latest_partner_obs = partner_obs

        if self.gym_partner and self._partner_is_lstm(self.gym_partner) and self.obs_history is not None:
            self.obs_history.append(partner_obs)

        trunc = False
        if not isinstance(info, dict):
            info = {"raw_info": info}
        team_reward, shaped_reward = self._compute_team_reward(float(sparse_reward), info, next_state)

        if self.reward_clip > 0:
            team_reward = float(np.clip(team_reward, -self.reward_clip, self.reward_clip))

        info["sparse_reward"] = float(sparse_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["team_reward"] = float(team_reward)
        info["player_idx"] = self.player_idx

        if self.debug:
            print(
                f"Learner(P{self.player_idx}): {learner_action_conv} | "
                f"Partner(P{partner_idx}): {partner_action_conv} | "
                f"Sparse: {sparse_reward} | Team: {team_reward}"
            )
        return learner_obs, team_reward, term, trunc, info
