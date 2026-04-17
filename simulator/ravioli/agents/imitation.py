from __future__ import annotations

from collections import deque
import math
from pathlib import Path

import numpy as np
import torch

from ..imitation_common import (
    ACTION_TO_ID,
    HOLDER_TYPES,
    ID_TO_ACTION,
    FeatureSchema,
    action_id_to_input_state,
    build_policy_model,
    featurize_state_for_player,
)
from .base import Agent


TRAINING_RUNS_DIR = Path(__file__).resolve().parents[1] / "training_runs"
DEFAULT_DEPLOY_DIR = TRAINING_RUNS_DIR / "imitation"
DEFAULT_CHECKPOINT_PATH = DEFAULT_DEPLOY_DIR / "best.pt"
BUTTON_ACTION_DISTANCE = 1.3
BUTTON_ACTION_ALIGNMENT_DOT = 0.2
INTERACTABLE_TYPES = {"chopping_board", "stove", "sink"}
STUCK_DISTANCE_EPSILON = 0.01
STUCK_ACTION_FRAMES = 20
BUTTON_REPEAT_COOLDOWNS = {
    "CARRY": 20,
    "INTERACT": 90,
}


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
        self._previous_position: tuple[float, float] | None = None
        self._previous_action_id: int | None = None
        self._stuck_action_frames = 0
        self._button_cooldowns = {action_name: 0 for action_name in BUTTON_REPEAT_COOLDOWNS}

    def update(self, delta_time: float, state: dict) -> dict:
        del delta_time

        self._tick_button_cooldowns()
        self._update_stuck_state(state)
        features = featurize_state_for_player(state, self.player_num, self.feature_schema)
        with torch.no_grad():
            if self.model_type == "lstm":
                logits = self.model(self._build_lstm_input(features))
            else:
                logits = self.model(self._build_mlp_input(features))

        action_id = self._select_valid_action_id(logits, state)
        action_name = ID_TO_ACTION[int(action_id)]
        if action_name in self._button_cooldowns:
            self._button_cooldowns[action_name] = BUTTON_REPEAT_COOLDOWNS[action_name]
        self._previous_action_id = action_id
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

    def _select_valid_action_id(self, logits: torch.Tensor, state: dict) -> int:
        ranked_action_ids = torch.argsort(logits, dim=1, descending=True)[0].tolist()
        for action_id in ranked_action_ids:
            if self._is_action_valid(int(action_id), state):
                return int(action_id)
        return int(torch.argmax(logits, dim=1).item())

    def _is_action_valid(self, action_id: int, state: dict) -> bool:
        action_name = ID_TO_ACTION[int(action_id)]
        if self._is_repeating_stuck_movement(action_id):
            return False
        if self._is_button_cooling_down(action_id):
            return False
        if action_name == "CARRY":
            return self._has_useful_carry_target(state)
        if action_name == "INTERACT":
            return self._has_useful_interact_target(state)
        return True

    def _tick_button_cooldowns(self) -> None:
        for action_name, frames_remaining in self._button_cooldowns.items():
            if frames_remaining > 0:
                self._button_cooldowns[action_name] = frames_remaining - 1

    def _update_stuck_state(self, state: dict) -> None:
        player = self._get_self_player(state)
        current_position = self._get_position(player) if player is not None else None
        if current_position is None:
            self._previous_position = None
            self._stuck_action_frames = 0
            return

        if self._previous_position is None or self._previous_action_id is None:
            self._previous_position = current_position
            self._stuck_action_frames = 0
            return

        previous_action = ID_TO_ACTION[int(self._previous_action_id)]
        if previous_action.startswith("MOVE_"):
            distance_moved = math.hypot(
                current_position[0] - self._previous_position[0],
                current_position[1] - self._previous_position[1],
            )
            if distance_moved <= STUCK_DISTANCE_EPSILON:
                self._stuck_action_frames += 1
            else:
                self._stuck_action_frames = 0
        else:
            self._stuck_action_frames = 0

        self._previous_position = current_position

    def _is_repeating_stuck_movement(self, action_id: int) -> bool:
        if self._previous_action_id is None or action_id != self._previous_action_id:
            return False
        if self._stuck_action_frames < STUCK_ACTION_FRAMES:
            return False
        return ID_TO_ACTION[int(action_id)].startswith("MOVE_")

    def _is_button_cooling_down(self, action_id: int) -> bool:
        action_name = ID_TO_ACTION[int(action_id)]
        return self._button_cooldowns.get(action_name, 0) > 0

    def _has_useful_carry_target(self, state: dict) -> bool:
        objects_by_id = self._objects_by_id(state)
        player = self._get_self_player(state)
        target = self._get_action_target(state, set(HOLDER_TYPES))
        if player is None or target is None:
            return False

        held_object = self._get_held_object(player, objects_by_id)
        if held_object is not None:
            return self._can_put_down_held_object(held_object, target, objects_by_id)

        target_held_object = self._get_held_object(target, objects_by_id)
        if target_held_object is not None:
            if target.get("name") == "chopping_board" and target_held_object.get("name") == "onion":
                return self._get_progress(target_held_object) >= 1.0
            return True

        try:
            return int(target.get("plate_count", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    def _can_put_down_held_object(self, held_object: dict, target: dict, objects_by_id: dict) -> bool:
        held_object_name = held_object.get("name")
        target_name = target.get("name")
        target_held_object = self._get_held_object(target, objects_by_id)

        if held_object_name == "onion":
            progress = self._get_progress(held_object)
            if progress >= 1.0:
                return self._can_put_chopped_onion_on_target(target, target_held_object, objects_by_id)
            return target_name == "chopping_board" and target_held_object is None

        if target_name in {"dispenser", "plate_return_station"}:
            return False
        if target_name == "drying_rack":
            return held_object_name == "plate"
        if target_name == "sink":
            return held_object_name in {"dirty_plate", "stacked_dirty_plates"} or target_held_object is None

        return target_held_object is None

    def _can_put_chopped_onion_on_target(
        self,
        target: dict,
        target_held_object: dict | None,
        objects_by_id: dict,
    ) -> bool:
        target_name = target.get("name")
        if target_name == "stove":
            return self._target_has_pot_space(target_held_object, objects_by_id)
        if target_name == "rubbish_bin":
            return True
        return target_name == "tabletop" and target_held_object is None

    def _target_has_pot_space(self, target_held_object: dict | None, objects_by_id: dict) -> bool:
        if target_held_object is None or target_held_object.get("name") != "pot":
            return False

        pot_contents = self._get_held_object(target_held_object, objects_by_id)
        if pot_contents is None:
            return True
        if pot_contents.get("name") != "soup":
            return False

        ingredients = pot_contents.get("ingredients")
        if isinstance(ingredients, list):
            return sum(ingredient is not None for ingredient in ingredients) < 3
        return True

    def _has_useful_interact_target(self, state: dict) -> bool:
        objects_by_id = self._objects_by_id(state)

        for target in self._get_action_targets(state, INTERACTABLE_TYPES):
            target_name = target.get("name")
            held_object = self._get_held_object(target, objects_by_id)

            if target_name == "chopping_board":
                if held_object is None or held_object.get("name") != "onion":
                    continue
                progress = self._get_progress(held_object)
                return progress <= 0.0

            if target_name == "sink":
                if held_object is None or held_object.get("name") != "stacked_dirty_plates":
                    continue
                try:
                    return int(held_object.get("plate_count", 0) or 0) > 0
                except (TypeError, ValueError):
                    return True

        return False

    def _get_action_target(self, state: dict, target_names: set[str]) -> dict | None:
        candidates = self._get_action_targets(state, target_names)
        return candidates[0] if candidates else None

    def _get_action_targets(self, state: dict, target_names: set[str]) -> list[dict]:
        player = self._get_self_player(state)
        if player is None:
            return []

        player_position = self._get_position(player)
        facing = self._get_facing(player)
        if player_position is None or facing is None:
            return []

        candidates: list[tuple[float, float, dict]] = []

        for obj in state.get("objects", []):
            if not isinstance(obj, dict) or obj.get("name") not in target_names:
                continue
            target_position = self._get_position(obj)
            if target_position is None:
                continue

            delta_x = target_position[0] - player_position[0]
            delta_y = target_position[1] - player_position[1]
            distance = math.hypot(delta_x, delta_y)
            if distance > BUTTON_ACTION_DISTANCE:
                continue

            if distance <= 0.0001:
                candidates.append((1.0, -distance, obj))
                continue

            alignment = ((delta_x / distance) * facing[0]) + ((delta_y / distance) * facing[1])
            if alignment >= BUTTON_ACTION_ALIGNMENT_DOT:
                candidates.append((alignment, -distance, obj))

        if not candidates:
            return []
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
        return [candidate[2] for candidate in candidates]

    @staticmethod
    def _get_held_object(entity: dict, objects_by_id: dict) -> dict | None:
        held_object_id = entity.get("held_object_id")
        if held_object_id is not None:
            held_object = objects_by_id.get(held_object_id)
            if held_object is not None:
                return held_object

        held_object_name = entity.get("held_object_name")
        if isinstance(held_object_name, str) and held_object_name:
            return {
                "id": held_object_id,
                "name": held_object_name,
                "plate_count": entity.get("plate_count"),
                "progress": entity.get("progress"),
            }
        return None

    @staticmethod
    def _objects_by_id(state: dict) -> dict:
        return {
            obj.get("id"): obj
            for obj in state.get("objects", [])
            if isinstance(obj, dict) and obj.get("id") is not None
        }

    @staticmethod
    def _get_progress(entity: dict) -> float:
        try:
            return float(entity.get("progress", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _get_self_player(self, state: dict) -> dict | None:
        player_id = f"player_{self.player_num}"
        for player in state.get("players", []):
            if isinstance(player, dict) and player.get("id") == player_id:
                return player
        return None

    @staticmethod
    def _get_position(entity: dict) -> tuple[float, float] | None:
        position = entity.get("position")
        if not isinstance(position, list) or len(position) != 2:
            return None
        return float(position[0]), float(position[1])

    @staticmethod
    def _get_facing(player: dict) -> tuple[float, float] | None:
        facing = player.get("facing")
        if not isinstance(facing, list) or len(facing) != 2:
            return None
        magnitude = math.hypot(float(facing[0]), float(facing[1]))
        if magnitude <= 0.0001:
            return None
        return float(facing[0]) / magnitude, float(facing[1]) / magnitude
