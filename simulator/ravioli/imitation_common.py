from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


CANONICAL_ACTIONS = (
    "STAY",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_UP_LEFT",
    "MOVE_UP_RIGHT",
    "MOVE_DOWN_LEFT",
    "MOVE_DOWN_RIGHT",
    "CARRY",
    "INTERACT",
)
ACTION_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_ACTIONS)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}

_MOVE_ACTION_TO_VECTOR = {
    "MOVE_UP": (0.0, -1.0),
    "MOVE_DOWN": (0.0, 1.0),
    "MOVE_LEFT": (-1.0, 0.0),
    "MOVE_RIGHT": (1.0, 0.0),
    "MOVE_UP_LEFT": (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
    "MOVE_UP_RIGHT": (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
    "MOVE_DOWN_LEFT": (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
    "MOVE_DOWN_RIGHT": (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
}

_MOVE_ACTION_NAMES = tuple(_MOVE_ACTION_TO_VECTOR.keys())
_MOVE_ACTION_VECTORS = tuple(_MOVE_ACTION_TO_VECTOR[action_name] for action_name in _MOVE_ACTION_NAMES)

HOLDER_TYPES = (
    "tabletop",
    "dispenser",
    "chopping_board",
    "stove",
    "delivery_station",
    "plate_return_station",
    "drying_rack",
    "sink",
    "rubbish_bin",
)

ROOT_OBJECT_TYPES = (
    "none",
    "onion",
    "plate",
    "pot",
    "soup",
    "stacked_dirty_plates",
    "fire_extinguisher",
)

COOKING_STATES = ("raw", "cooked", "burnt")

COUNT_SCALE = 5.0


@dataclass
class FeatureSchema:
    holder_specs: list[dict[str, str]]
    position_scale_x: float
    position_scale_y: float
    distance_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "holder_specs": self.holder_specs,
            "position_scale_x": self.position_scale_x,
            "position_scale_y": self.position_scale_y,
            "distance_scale": self.distance_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSchema":
        return cls(
            holder_specs=[
                {"id": str(holder_spec["id"]), "name": str(holder_spec["name"])}
                for holder_spec in payload.get("holder_specs", [])
            ],
            position_scale_x=float(payload.get("position_scale_x", 1.0)),
            position_scale_y=float(payload.get("position_scale_y", 1.0)),
            distance_scale=float(payload.get("distance_scale", 1.0)),
        )


class MLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
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


def build_policy_model(
    model_type: str,
    input_dim: int,
    num_actions: int,
    *,
    hidden_dims: list[int] | None = None,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> nn.Module:
    if model_type == "mlp":
        return MLPPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if model_type == "lstm":
        return LSTMPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_type={model_type}")


def input_state_to_action_name(input_state: dict[str, Any] | None) -> str:
    if not isinstance(input_state, dict):
        return "STAY"

    if bool(input_state.get("interact", False)):
        return "INTERACT"
    if bool(input_state.get("carry", False)):
        return "CARRY"

    move_x = float(input_state.get("move_x", 0.0))
    move_y = float(input_state.get("move_y", 0.0))
    magnitude = math.hypot(move_x, move_y)
    if magnitude <= 1e-4:
        return "STAY"

    normalized_x = move_x / magnitude
    normalized_y = move_y / magnitude
    best_index = max(
        range(len(_MOVE_ACTION_VECTORS)),
        key=lambda idx: (_MOVE_ACTION_VECTORS[idx][0] * normalized_x) + (_MOVE_ACTION_VECTORS[idx][1] * normalized_y),
    )
    return _MOVE_ACTION_NAMES[best_index]


def input_state_to_action_id(input_state: dict[str, Any] | None) -> int:
    return ACTION_TO_ID[input_state_to_action_name(input_state)]


def action_id_to_input_state(action_id: int) -> dict[str, Any]:
    action_name = ID_TO_ACTION[int(action_id)]
    input_state = {
        "move_x": 0.0,
        "move_y": 0.0,
        "carry": False,
        "interact": False,
    }

    if action_name in _MOVE_ACTION_TO_VECTOR:
        move_x, move_y = _MOVE_ACTION_TO_VECTOR[action_name]
        input_state["move_x"] = move_x
        input_state["move_y"] = move_y
    elif action_name == "CARRY":
        input_state["carry"] = True
    elif action_name == "INTERACT":
        input_state["interact"] = True

    return input_state


def build_feature_schema(states: list[dict[str, Any]]) -> FeatureSchema:
    holder_specs_by_id: dict[str, dict[str, str]] = {}
    max_x = 1.0
    max_y = 1.0

    for state in states:
        for player in state.get("players", []):
            if not isinstance(player, dict):
                continue
            position = player.get("position", [0.0, 0.0])
            if isinstance(position, list) and len(position) == 2:
                max_x = max(max_x, abs(float(position[0])))
                max_y = max(max_y, abs(float(position[1])))

        for obj in state.get("objects", []):
            if not isinstance(obj, dict):
                continue
            position = obj.get("position", [0.0, 0.0])
            if isinstance(position, list) and len(position) == 2:
                max_x = max(max_x, abs(float(position[0])))
                max_y = max(max_y, abs(float(position[1])))

            object_name = str(obj.get("name", ""))
            object_id = str(obj.get("id", ""))
            if object_name in HOLDER_TYPES and object_id:
                holder_specs_by_id[object_id] = {
                    "id": object_id,
                    "name": object_name,
                }

    holder_specs = sorted(
        holder_specs_by_id.values(),
        key=lambda holder_spec: (holder_spec["name"], holder_spec["id"]),
    )
    return FeatureSchema(
        holder_specs=holder_specs,
        position_scale_x=max(max_x, 1.0),
        position_scale_y=max(max_y, 1.0),
        distance_scale=max(math.hypot(max_x, max_y), 1.0),
    )


def featurize_state_for_player(
    state: dict[str, Any],
    player_num: int,
    schema: FeatureSchema,
) -> np.ndarray:
    players = state.get("players", [])
    objects = state.get("objects", [])
    players_by_id = {
        player.get("id"): player
        for player in players
        if isinstance(player, dict) and player.get("id") is not None
    }
    objects_by_id = {
        obj.get("id"): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    holders_by_id = {
        obj.get("id"): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("name") in HOLDER_TYPES and obj.get("id") is not None
    }

    self_player = players_by_id.get(f"player_{player_num}")
    if self_player is None:
        raise ValueError(f"State does not contain player_{player_num}")
    other_player = next(
        (
            player for player in players
            if isinstance(player, dict) and player.get("id") != f"player_{player_num}"
        ),
        None,
    )

    self_position = _get_position(self_player)
    features: list[float] = []
    features.extend(_encode_slot(player_num))
    features.extend(_encode_absolute_position(self_position, schema))
    features.extend(_encode_facing(self_player))
    features.extend(_encode_entity_summary(self_player, objects_by_id, objects, allow_player_fallback=True))

    if other_player is None:
        features.extend([0.0, 0.0, 0.0])
        features.extend([0.0, 0.0])
        features.extend(_encode_empty_summary())
    else:
        features.extend(_encode_relative_position(_get_position(other_player), self_position, schema))
        features.extend(_encode_facing(other_player))
        features.extend(_encode_entity_summary(other_player, objects_by_id, objects, allow_player_fallback=True))

    for holder_spec in schema.holder_specs:
        holder = holders_by_id.get(holder_spec["id"])
        features.extend(_encode_holder(holder, holder_spec, self_position, objects_by_id, objects, schema))

    return np.asarray(features, dtype=np.float32)


def _encode_slot(player_num: int) -> list[float]:
    return [
        1.0 if player_num == 0 else 0.0,
        1.0 if player_num == 1 else 0.0,
    ]


def _encode_absolute_position(position: tuple[float, float], schema: FeatureSchema) -> list[float]:
    return [
        _safe_div(position[0], schema.position_scale_x),
        _safe_div(position[1], schema.position_scale_y),
    ]


def _encode_relative_position(
    target_position: tuple[float, float],
    reference_position: tuple[float, float],
    schema: FeatureSchema,
) -> list[float]:
    delta_x = target_position[0] - reference_position[0]
    delta_y = target_position[1] - reference_position[1]
    return [
        _safe_div(delta_x, schema.position_scale_x),
        _safe_div(delta_y, schema.position_scale_y),
        _safe_div(math.hypot(delta_x, delta_y), schema.distance_scale),
    ]


def _encode_facing(entity: dict[str, Any]) -> list[float]:
    facing = entity.get("facing")
    if not isinstance(facing, list) or len(facing) != 2:
        return [0.0, 0.0]
    return [float(facing[0]), float(facing[1])]


def _encode_holder(
    holder: dict[str, Any] | None,
    holder_spec: dict[str, str],
    self_position: tuple[float, float],
    objects_by_id: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    schema: FeatureSchema,
) -> list[float]:
    if holder is None:
        return [0.0, 0.0, 0.0, 0.0] + _encode_one_hot(holder_spec["name"], HOLDER_TYPES) + _encode_empty_summary() + [0.0]

    features = [1.0]
    features.extend(_encode_relative_position(_get_position(holder), self_position, schema))
    features.extend(_encode_one_hot(holder_spec["name"], HOLDER_TYPES))
    features.extend(_encode_entity_summary(holder, objects_by_id, objects))
    features.append(_safe_div(float(holder.get("plate_count", 0.0)), COUNT_SCALE))
    return features


def _encode_entity_summary(
    entity: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    allow_player_fallback: bool = False,
) -> list[float]:
    object_summary = _summarize_entity(entity, objects_by_id, objects, allow_player_fallback=allow_player_fallback)
    return _encode_summary(object_summary)


def _encode_empty_summary() -> list[float]:
    return _encode_summary(_empty_summary())


def _summarize_entity(
    entity: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    allow_player_fallback: bool = False,
) -> dict[str, Any]:
    entity_id = entity.get("id")
    held_object = None

    held_object_id = entity.get("held_object_id")
    if held_object_id is not None:
        held_object = objects_by_id.get(held_object_id)

    if held_object is None and entity_id is not None:
        held_object = next(
            (
                obj for obj in objects
                if isinstance(obj, dict) and obj.get("parent_id") == entity_id
            ),
            None,
        )

    if held_object is None and str(entity.get("name", "")) == "drying_rack" and int(entity.get("plate_count", 0) or 0) > 0:
        return _summarize_object(None, objects_by_id, objects, fallback_name="plate")

    if held_object is None and allow_player_fallback:
        held_object_name = entity.get("held_object_name")
        if isinstance(held_object_name, str):
            return _summarize_object(None, objects_by_id, objects, fallback_name=held_object_name)

    if held_object is None:
        held_object_name = entity.get("held_object_name")
        if isinstance(held_object_name, str):
            return _summarize_object(None, objects_by_id, objects, fallback_name=held_object_name)
        return _empty_summary()

    return _summarize_object(held_object, objects_by_id, objects)


def _summarize_object(
    obj: dict[str, Any] | None,
    objects_by_id: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    fallback_name: str | None = None,
) -> dict[str, Any]:
    summary = _empty_summary()
    root_name = _canonical_object_type(fallback_name or (obj.get("name") if obj is not None else None))
    summary["root_type"] = root_name

    if obj is None:
        if root_name == "stacked_dirty_plates":
            summary["dirty_plate_count"] = _safe_div(1.0, COUNT_SCALE)
        return summary

    if root_name == "onion":
        summary["onion_progress"] = float(obj.get("progress", 0.0))
    if root_name == "stacked_dirty_plates":
        summary["dirty_plate_count"] = _safe_div(float(obj.get("plate_count", 0.0)), COUNT_SCALE)
    if root_name == "soup":
        _populate_soup_summary(summary, obj)

    if root_name in {"plate", "pot"}:
        child_object = _resolve_child_object(obj, objects_by_id, objects)
        if child_object is not None and _canonical_object_type(child_object.get("name")) == "soup":
            if root_name == "plate":
                summary["plate_has_soup"] = 1.0
            else:
                summary["pot_has_soup"] = 1.0
            _populate_soup_summary(summary, child_object)
        elif obj.get("held_object_name") == "soup":
            if root_name == "plate":
                summary["plate_has_soup"] = 1.0
            else:
                summary["pot_has_soup"] = 1.0

    return summary


def _resolve_child_object(
    obj: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    child_id = obj.get("held_object_id")
    if child_id is not None:
        child_object = objects_by_id.get(child_id)
        if child_object is not None:
            return child_object

    parent_id = obj.get("id")
    if parent_id is None:
        return None
    return next(
        (
            child for child in objects
            if isinstance(child, dict) and child.get("parent_id") == parent_id
        ),
        None,
    )


def _populate_soup_summary(summary: dict[str, Any], soup: dict[str, Any]) -> None:
    ingredients = soup.get("ingredients", [])
    if isinstance(ingredients, list):
        ingredient_count = sum(1 for ingredient in ingredients if ingredient is not None)
        summary["soup_ingredient_count"] = _safe_div(float(ingredient_count), 3.0)

    cooking_state = str(soup.get("cooking_state", "raw")).lower()
    if cooking_state not in COOKING_STATES:
        return
    summary[f"soup_is_{cooking_state}"] = 1.0


def _encode_summary(summary: dict[str, Any]) -> list[float]:
    features = _encode_one_hot(summary["root_type"], ROOT_OBJECT_TYPES)
    features.extend(
        [
            float(summary["onion_progress"]),
            float(summary["dirty_plate_count"]),
            float(summary["soup_ingredient_count"]),
            float(summary["soup_is_raw"]),
            float(summary["soup_is_cooked"]),
            float(summary["soup_is_burnt"]),
            float(summary["plate_has_soup"]),
            float(summary["pot_has_soup"]),
        ]
    )
    return features


def _empty_summary() -> dict[str, Any]:
    return {
        "root_type": "none",
        "onion_progress": 0.0,
        "dirty_plate_count": 0.0,
        "soup_ingredient_count": 0.0,
        "soup_is_raw": 0.0,
        "soup_is_cooked": 0.0,
        "soup_is_burnt": 0.0,
        "plate_has_soup": 0.0,
        "pot_has_soup": 0.0,
    }


def _canonical_object_type(name: Any) -> str:
    if not isinstance(name, str):
        return "none"
    lowered_name = name.lower()
    if lowered_name == "dirty_plate":
        return "stacked_dirty_plates"
    if lowered_name in ROOT_OBJECT_TYPES:
        return lowered_name
    return "none"


def _encode_one_hot(value: str, options: tuple[str, ...]) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


def _get_position(entity: dict[str, Any]) -> tuple[float, float]:
    position = entity.get("position", [0.0, 0.0])
    if not isinstance(position, list) or len(position) != 2:
        return 0.0, 0.0
    return float(position[0]), float(position[1])


def _safe_div(num: float, den: float) -> float:
    if abs(den) <= 1e-6:
        return 0.0
    return float(num / den)
