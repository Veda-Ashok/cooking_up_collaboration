""" Usage: From the root director, run python3 -m tests.rl.test_env"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from rl.env_utils import OvercookedRLWrapper
from rl.models.rl_mlp import RLActorCriticPolicy
from imitation.models.mlp import MLPPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = OvercookedRLWrapper("cramped_room")
din = env.observation_space.shape[0]
dout = env.action_space.n

# Load IL agent using MLPPolicy
gp = MLPPolicy(input_dim = din, num_actions = dout)
checkpoint = torch.load('trained_models/bc/mlp_partner/best.pt')
state_dict = checkpoint['model_state_dict']
if not isinstance(state_dict, dict):
    state_dict = state_dict.state_dict()
gp.load_state_dict(state_dict)
gp.eval()
env.gym_partner = gp

# env = OvercookedRLWrapper("cramped_room", gym_partner = gp)
obs, info = env.reset()
print('Observation Shape: ', obs.shape)

rl_agent = PPO.load('./rl/trained_models/mlp_rl_agent', policy = RLActorCriticPolicy)

gameplay_data = []

layout_grid = env.mdp.terrain_mtx

total_reward = 0
print('Sim starting: ')
for step in range(4800): # 1200 for test purposes. 
    curr_state_dict = env.env.state.to_dict()

    action, states = rl_agent.predict(obs, deterministic = True)
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc
    # total_reward += 1 # just to test the loop
    total_reward += reward
    gameplay_data.append({"step": step,
                          "state": json.dumps(curr_state_dict),
                          "layout": json.dumps(layout_grid),
                          "rl_action": int(action)})
    if step % 20 == 0:
        print(f'Step {step}: total reward {total_reward} -------')
    if done:
        print(f'Step {step}: Game Over, total reward {total_reward}')
        break
    if reward > 0:
        print(f'Step {step}: Team scored --> Reward: {reward}, Total Reward: {total_reward}')
        

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'./data/{timestamp}.csv'

rl_gameplay_data = pd.DataFrame(gameplay_data)
rl_gameplay_data.to_csv(csv_path, index = False)