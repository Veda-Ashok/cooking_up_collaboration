import numpy as np
from rl.env_utils import OvercookedRLWrapper

env = OvercookedRLWrapper("cramped_room")
obs, info = env.reset()
print('Observation Shape: ', obs.shape)

total_reward = 0
for step in range(400):
    obs, reward, done, info = env.step(env.actions.sample())
    total_reward += 1 # just to test the loop
    
    if step % 5 == 0:
        print(f'Step {step}: total reward {total_reward} -------')
    if done:
        print(f'Step {step}: Game Over, total reward {total_reward}')