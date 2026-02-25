import json
import logging
import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import overcooked_ai_py.data.planners as data_planners
import overcooked_ai_py.planning.planners as planning_planners
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS


LOGGER = logging.getLogger(__name__)


class TorchMLPPolicy(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchLSTMPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_actions))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x_seq)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class TorchBCAgent:
    """Runtime adapter for BC checkpoints trained in PyTorch."""

    IDX_TO_OVERCOOKED_ACTION = {
        0: Direction.NORTH,   # UP
        1: Direction.SOUTH,   # DOWN
        2: Direction.EAST,    # RIGHT
        3: Direction.WEST,    # LEFT
        4: Action.STAY,       # STAY
        5: Action.INTERACT,   # INTERACT
    }

    def __init__(self, agent_dir: str, agent_index: int, device: str = "cpu"):
        self.agent_dir = agent_dir
        self.agent_index = int(agent_index)
        self.manifest = self._load_manifest()
        self.model_type = self.manifest.get("model_type", "mlp").lower()
        manifest_player_idx = self.manifest.get("player_idx", None)
        self.player_idx = self.agent_index if manifest_player_idx is None else int(manifest_player_idx)
        self.num_actions = int(self.manifest.get("num_actions", 6))
        self.input_dim = int(self.manifest["input_dim"])
        self.seq_len = int(self.manifest.get("seq_len", 20))
        self.supported_layouts = set(self.manifest.get("supported_layouts", []))
        self._warned_layouts = set()
        self.deadlock_break_after = int(self.manifest.get("deadlock_break_after", 8))
        if self.deadlock_break_after < 0:
            self.deadlock_break_after = 0

        self.planner_cache_dir = os.path.abspath(
            os.path.join(self.agent_dir, self.manifest.get("planner_cache_dir", ".cache/overcooked_planners"))
        )
        self._set_planner_cache_dir(self.planner_cache_dir)

        requested_device = self.manifest.get("device", device)
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for %s but unavailable; using CPU.", self.agent_dir)
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        checkpoint_path = os.path.join(self.agent_dir, self.manifest.get("checkpoint", "best.pt"))
        self.model = self._build_model(self.manifest)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._mdp = None
        self._mlam = None
        self._layout_name = None
        self._history = deque(maxlen=self.seq_len)
        self._last_feature: np.ndarray | None = None
        self._stuck_count = 0

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = os.path.join(self.agent_dir, "agent_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing BC manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        if manifest.get("type") != "bc_torch":
            raise ValueError(f"Unsupported manifest type for BC torch agent: {manifest.get('type')}")
        return manifest

    def _build_model(self, manifest: dict[str, Any]) -> nn.Module:
        if self.model_type == "mlp":
            return TorchMLPPolicy(
                input_dim=self.input_dim,
                num_actions=self.num_actions,
                hidden_dims=[int(x) for x in manifest.get("mlp_hidden", [256, 128])],
                dropout=float(manifest.get("dropout", 0.1)),
            )
        if self.model_type == "lstm":
            return TorchLSTMPolicy(
                input_dim=self.input_dim,
                num_actions=self.num_actions,
                hidden_dim=int(manifest.get("hidden_dim", 128)),
                num_layers=int(manifest.get("num_layers", 1)),
                dropout=float(manifest.get("dropout", 0.1)),
            )
        raise ValueError(f"Unsupported model_type in manifest: {self.model_type}")

    def _set_planner_cache_dir(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        data_planners.PLANNERS_DIR = cache_dir
        planning_planners.PLANNERS_DIR = cache_dir

    def _clear_layout_planners(self, layout_name: str) -> None:
        if not os.path.isdir(self.planner_cache_dir):
            return
        for filename in os.listdir(self.planner_cache_dir):
            if filename.startswith(layout_name) and filename.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.planner_cache_dir, filename))
                except OSError:
                    pass

    def set_mdp_context(self, mdp, layout_name: str | None = None) -> None:
        self._mdp = mdp
        self._layout_name = layout_name or getattr(mdp, "layout_name", None)
        self._mlam = None
        if mdp is None:
            return

        custom_name = None
        if self._layout_name:
            custom_name = f"{self._layout_name}_am.pkl"

        try:
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                mdp,
                NO_COUNTERS_PARAMS,
                custom_filename=custom_name,
                force_compute=False,
            )
        except Exception:
            if self._layout_name:
                self._clear_layout_planners(self._layout_name)
            try:
                self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                    mdp,
                    NO_COUNTERS_PARAMS,
                    custom_filename=custom_name,
                    force_compute=True,
                )
            except Exception:
                self._mlam = None
                LOGGER.exception("Failed to initialize mlam for BC torch agent (%s).", self.agent_dir)

    def reset(self) -> None:
        self._history.clear()
        self._last_feature = None
        self._stuck_count = 0

    # breaks deadlock by choosing a random action if the agent is stuck, this happens often when both the players are BC cloned agents
    def _maybe_break_deadlock(self, action_idx: int, logits: torch.Tensor, feature: np.ndarray) -> int:
        if self.deadlock_break_after <= 0:
            return action_idx
        if action_idx != 4:
            self._stuck_count = 0
            return action_idx
        if self._last_feature is not None and np.array_equal(feature, self._last_feature):
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        if self._stuck_count < self.deadlock_break_after:
            return action_idx

        non_stay_indices = torch.tensor([0, 1, 2, 3, 5], device=logits.device)
        non_stay_logits = logits.index_select(dim=1, index=non_stay_indices)
        best_non_stay_local = int(torch.argmax(non_stay_logits, dim=1).item())
        self._stuck_count = 0
        return int(non_stay_indices[best_non_stay_local].item())

    def _featurize_state(self, state) -> np.ndarray | None:
        if self._mdp is None or self._mlam is None:
            return None
        feature_by_player = self._mdp.featurize_state(state, self._mlam)
        feature = np.asarray(feature_by_player[self.player_idx], dtype=np.float32)
        if feature.shape[0] != self.input_dim:
            LOGGER.error(
                "Feature size mismatch for %s: expected %s got %s",
                self.agent_dir,
                self.input_dim,
                feature.shape[0],
            )
            return None
        return feature

    @torch.no_grad()
    def _predict_action(self, feature: np.ndarray) -> int:
        if self.model_type == "mlp":
            x = torch.from_numpy(feature).unsqueeze(0).to(self.device)
            logits = self.model(x)
        else:
            self._history.append(feature)
            if len(self._history) == 1:
                while len(self._history) < self.seq_len:
                    self._history.append(feature)
            x_seq = np.stack(list(self._history), axis=0).astype(np.float32)
            x = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)
            logits = self.model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_idx = self._maybe_break_deadlock(pred_idx, logits, feature)
        self._last_feature = feature.copy()
        return pred_idx

    def action(self, state):
        layout = self._layout_name
        if self.supported_layouts and layout not in self.supported_layouts:
            if layout not in self._warned_layouts:
                LOGGER.warning(
                    "BC torch agent %s got unsupported layout=%s. Returning STAY.",
                    os.path.basename(self.agent_dir),
                    layout,
                )
                self._warned_layouts.add(layout)
            return Action.STAY, None

        try:
            feature = self._featurize_state(state)
            if feature is None:
                return Action.STAY, None
            action_idx = self._predict_action(feature)
            action = self.IDX_TO_OVERCOOKED_ACTION.get(action_idx, Action.STAY)
            return action, None
        except Exception:
            LOGGER.exception("BC torch inference failed for agent %s", os.path.basename(self.agent_dir))
            return Action.STAY, None
