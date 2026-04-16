from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..imitation_common import (
    ACTION_TO_ID,
    FeatureSchema,
    action_id_to_input_state,
    build_policy_model,
    featurize_state_for_player,
)
from .base import Agent


TRAINING_RUNS_DIR = Path(__file__).resolve().parents[1] / "training_runs"
DEFAULT_DEPLOY_DIR = TRAINING_RUNS_DIR / "imitation"
DEFAULT_CHECKPOINT_PATH = DEFAULT_DEPLOY_DIR / "best.pt"


def find_latest_checkpoint() -> Path:
    run_candidates = []
    if TRAINING_RUNS_DIR.is_dir():
        for run_dir in TRAINING_RUNS_DIR.iterdir():
            if not run_dir.is_dir() or run_dir.name == DEFAULT_DEPLOY_DIR.name:
                continue
            checkpoint_path = run_dir / "best.pt"
            if checkpoint_path.is_file():
                run_candidates.append((checkpoint_path.stat().st_mtime, checkpoint_path))

    if run_candidates:
        run_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return run_candidates[0][1]

    return DEFAULT_CHECKPOINT_PATH


class ImitationAgent(Agent):
    def __init__(
        self,
        player_num: int,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(player_num)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else find_latest_checkpoint()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Imitation checkpoint not found at {checkpoint_file}. "
                "Train a model first or pass an explicit checkpoint path."
            )

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.model_type = str(checkpoint.get("model_type", "mlp"))
        self.feature_schema = FeatureSchema.from_dict(checkpoint["feature_schema"])
        self.seq_len = int(checkpoint.get("seq_len", 1))
        self.model = build_policy_model(
            model_type=self.model_type,
            input_dim=int(checkpoint["input_dim"]),
            num_actions=int(checkpoint.get("num_actions", len(ACTION_TO_ID))),
            hidden_dims=checkpoint.get("hidden_dims"),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            num_layers=int(checkpoint.get("num_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._feature_history: deque[np.ndarray] = deque(maxlen=max(self.seq_len, 1))

    def update(self, delta_time: float, state: dict) -> dict:
        del delta_time

        features = featurize_state_for_player(state, self.player_num, self.feature_schema)
        with torch.no_grad():
            if self.model_type == "lstm":
                logits = self.model(self._build_lstm_input(features))
            else:
                logits = self.model(self._build_mlp_input(features))

        action_id = int(torch.argmax(logits, dim=1).item())
        return action_id_to_input_state(action_id)

    def _build_mlp_input(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_lstm_input(self, features: np.ndarray) -> torch.Tensor:
        self._feature_history.append(features)
        sequence = list(self._feature_history)
        if len(sequence) < self.seq_len:
            padding = [np.zeros_like(features) for _ in range(self.seq_len - len(sequence))]
            sequence = padding + sequence
        sequence_array = np.stack(sequence, axis=0)
        return torch.as_tensor(sequence_array, dtype=torch.float32, device=self.device).unsqueeze(0)
