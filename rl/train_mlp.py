import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl.env_utils import OvercookedRLWrapper
from imitation.models.mlp import MLPPolicy
from rl.models.rl_mlp import RLActorCriticPolicy

# Setup
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

check_env(env)

# Using the SB3 with custom model.
model = PPO(
    policy = RLActorCriticPolicy,
    env = env,
    verbose = 1,
    learning_rate = 0.001,
    n_steps = 2048,
    batch_size = 64,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    tensorboard_log = './logs/mlp'
    )

model.learn(total_timesteps = 50000)

model.save('./rl/trained_models/mlp_rl_agent')