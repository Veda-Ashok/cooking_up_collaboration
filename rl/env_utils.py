import numpy as np
import torch
import gymnasium as gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from imitation.preprocessing import _configure_planner_cache

class OvercookedRLWrapper(gym.Env):
    " Wrapper class to make overcooked-ai into gym env using same featurization as imitation learning"

    def __init__(self, layout_name, planner_cache_dir = './planner_cache'):

        super().__init__()
        _configure_planner_cache(planner_cache_dir)

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute = False)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon = 400)
        self.actions = gym.spaces.Discrete(6) # up, down, right, left, stay, interact

        dummy_state = self.mdp.get_standard_start_state()
        dummy_feat = self.mdp.featurize_state(dummy_state, self.mlam)[0]
        self.observations = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (len(dummy_feat), ), dtype = np.float32)
    
    def _get_observations(self, state):

        feats = self.mdp.featurize_state(state, self.mlam)

        return np.array(feats[0], dtype = np.float32), np.array(feats[1], dtype = np.float32)
    
    def reset(self):

        self.env.reset()
        obs_p0, obs_p1 = self._get_observations(self.env.state)

        return obs_p0, obs_p1
    
    def step(self, action_p0):

        if hasattr(action_p0, '__len__'):
            idx_p0 = int(action_p0[0])
        else:
            idx_p0 = int(action_p0) 

        action_p0_conv = Action.INDEX_TO_ACTION[idx_p0]
        action_p1_conv = Action.STAY

        combined_action = (action_p0_conv, action_p1_conv)
        next_state, reward, done, info = self.env.step(combined_action)
        obs_p0, obs_p1 = self._get_observations(next_state)
        
        return obs_p0, reward, done, info
