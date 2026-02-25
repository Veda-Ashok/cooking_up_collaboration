import numpy as np
import torch
from rl.env_utils import OvercookedRLWrapper
from imitation.models.mlp import MLPPolicy

env = OvercookedRLWrapper("cramped_room")
din = env.observation_space.shape[0]
dout = env.action_space.n

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

total_reward = 0
print('Sim starting: ')
for step in range(1200): # 1200 for test purposes. 
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    # total_reward += 1 # just to test the loop
    total_reward += reward

    if step % 20 == 0:
        print(f'Step {step}: total reward {total_reward} -------')
    if term:
        print(f'Step {step}: Game Over, total reward {total_reward}')

    if reward > 0:
        print(f'Step {step}: Team scored --> Reward: {reward}, Total Reward: {total_reward}')
        
