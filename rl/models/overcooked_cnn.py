import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class OvercookedCNN(BaseFeaturesExtractor):
    """Paper-style CNN for lossless (H, W, C) Overcooked state encodings.

    Architecture: three same-padding conv layers followed by a two-layer
    MLP that outputs a ``features_dim``-sized embedding.
    SB3 feeds this embedding into the policy/value heads.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64,
                 n_filters: int = 25, fc_hidden: int | None = None):
        super().__init__(observation_space, features_dim)

        h, w, c = observation_space.shape
        if fc_hidden is None:
            fc_hidden = max(64, features_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(c, n_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)  # NHWC -> NCHW
            n_flatten = self.conv(sample).reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.conv(x)
        return self.fc(x)
