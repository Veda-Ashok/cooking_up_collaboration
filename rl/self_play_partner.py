"""Adapter that wraps an SB3 PPO model so it can serve as the gym_partner
inside ``OvercookedRLWrapper``.

The wrapper receives the partner's featurized observation (96-D vector or
LSTM sequence) and returns action **logits** — matching the interface the
env expects from BC partners.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


class SB3SelfPlayPartner(nn.Module):
    """Use an SB3 PPO/RecurrentPPO model as the partner agent.

    The env calls ``partner(obs_tensor) -> logits`` where obs_tensor is
    (1, feat_dim) for MLP or (1, seq_len, feat_dim) for LSTM.
    """

    def __init__(self, model: PPO, obs_mode: str = "featurized"):
        super().__init__()
        self._model = model
        self._obs_mode = obs_mode
        self._is_recurrent = RecurrentPPO is not None and isinstance(model, RecurrentPPO)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        policy = self._model.policy
        obs = obs.to(policy.device)
        features = policy.extract_features(obs, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        return policy.action_net(latent_pi).cpu()
