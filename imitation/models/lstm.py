import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_seq)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)

