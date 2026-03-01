import numpy as np
import torch
import gymnasium as gym
from collections import deque
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from imitation.preprocessing import _configure_planner_cache
from imitation.models.lstm import LSTMPolicy


class OvercookedRLWrapper(gym.Env):
    "Wrapper class to make overcooked-ai into gym env using same featurization as imitation learning"

    def __init__(self, layout_name, gym_partner=None, planner_cache_dir='./planner_cache', seq_len=20):

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
        self.seq_len = seq_len
        self.latest_obs_p1 = None
        self.obs_history = None  # rolling buffer for LSTM partner

    @property
    def _is_lstm(self):
        return isinstance(self.gym_partner, LSTMPolicy)

    def _get_observations(self, state):
        feats = self.mdp.featurize_state(state, self.mlam)
        return np.array(feats[0], dtype=np.float32), np.array(feats[1], dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.env.reset()
        obs_p0, obs_p1 = self._get_observations(self.env.state)
        self.latest_obs_p1 = obs_p1

        if self._is_lstm:
            # pre-fill history with the initial observation
            self.obs_history = deque([obs_p1] * self.seq_len, maxlen=self.seq_len)

        return obs_p0, {}

    def step(self, action_p0):

        if hasattr(action_p0, '__len__'):
            idx_p0 = int(action_p0[0])
        else:
            idx_p0 = int(action_p0)

        action_p0_conv = Action.INDEX_TO_ACTION[idx_p0]

        if self.gym_partner:
            with torch.no_grad():
                if self._is_lstm:
                    # stack rolling history into [1, seq_len, input_dim] for LSTM
                    seq = np.stack(list(self.obs_history), axis=0)
                    obs_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                else:
                    # single timestep [1, input_dim] for MLP
                    obs_tensor = torch.tensor(self.latest_obs_p1, dtype=torch.float32).unsqueeze(0)
                pred = self.gym_partner(obs_tensor)
                idx_p1 = torch.argmax(pred, dim=1).item()
            action_p1_conv = Action.INDEX_TO_ACTION[idx_p1]
        else:
            action_p1_conv = Action.STAY

        combined_action = (action_p0_conv, action_p1_conv)
        next_state, reward, term, info = self.env.step(combined_action)
        obs_p0, obs_p1 = self._get_observations(next_state)
        self.latest_obs_p1 = obs_p1

        if self._is_lstm:
            self.obs_history.append(obs_p1)

        trunc = False

        print(f'P0 Action: {action_p0_conv} | P1 Action: {action_p1_conv} | Reward: {reward}')
        return obs_p0, reward, term, trunc, info
