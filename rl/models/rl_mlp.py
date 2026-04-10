""" Reference: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


def _build_mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers)


class RLMLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.policy_net = _build_mlp(input_dim, hidden_dims)
        self.value_net = _build_mlp(input_dim, hidden_dims)

        self.latent_dim_pi = hidden_dims[-1]
        self.latent_dim_vf = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_net(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x)


class RLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = RLMLPPolicy(
            input_dim=self.features_dim, num_actions=self.action_space.n,
        )