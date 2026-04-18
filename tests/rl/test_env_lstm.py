""" Usage: From the root directory, run python3 -m tests.rl.test_env_lstm"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from rl.env_utils import OvercookedRLWrapper
from imitation.models.lstm import LSTMPolicy

SEQ_LEN = 20  # must match the seq_len used during LSTM training

env = OvercookedRLWrapper("cramped_room", seq_len=SEQ_LEN)
din = env.observation_space.shape[0]
dout = env.action_space.n

gp = LSTMPolicy(input_dim=din, num_actions=dout)
checkpoint = torch.load('trained_models/bc/lstm_partner/best.pt')
state_dict = checkpoint['model_state_dict']
if not isinstance(state_dict, dict):
    state_dict = state_dict.state_dict()
gp.load_state_dict(state_dict)
gp.eval()
env.gym_partner = gp  # env auto-detects LSTMPolicy and enables rolling buffer

obs, info = env.reset()
print('Observation Shape: ', obs.shape)

gameplay_data = []
layout_grid = env.mdp.terrain_mtx

total_reward = 0
print('Sim starting: ')
for step in range(4800):
    curr_state_dict = env.env.state.to_dict()
    gameplay_data.append({"step": step,
                          "state": json.dumps(curr_state_dict),
                          "layout": json.dumps(layout_grid)})

    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    total_reward += reward

    if step % 20 == 0:
        print(f'Step {step}: total reward {total_reward} -------')
    if term:
        print(f'Step {step}: Game Over, total reward {total_reward}')

    if reward > 0:
        print(f'Step {step}: Team scored --> Reward: {reward}, Total Reward: {total_reward}')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'./data/{timestamp}_lstm.csv'

gp_data = pd.DataFrame(gameplay_data)
gp_data.to_csv(csv_path, index=False)
