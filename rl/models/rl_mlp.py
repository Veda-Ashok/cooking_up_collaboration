""" Reference: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class RLMLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        # layers.append(nn.Linear(previous_dim, num_actions))
        self.network = nn.Sequential(*layers)

        self.latent_dim_pi = previous_dim
        self.latent_dim_vf = previous_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x), self.network(x)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
        ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _build_mlp_extractor(self):
        self.mlp_extractor = RLMLPPolicy(input_dim = self.features_dim, num_actions = self.action_space.n)