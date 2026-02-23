import ast
from typing import Any


CANONICAL_ACTIONS = ["UP", "DOWN", "RIGHT", "LEFT", "STAY", "INTERACT"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_ACTIONS)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}

_VECTOR_TO_ACTION = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (1, 0): "RIGHT",
    (-1, 0): "LEFT",
    (0, 0): "STAY",
}

_TEXT_TO_ACTION = {
    "UP": "UP",
    "DOWN": "DOWN",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "STAY": "STAY",
    "INTERACT": "INTERACT",
    "SPACE": "INTERACT",
}


def normalize_action_token(action: Any) -> str:
    """Normalize action token from CSV/json into one of CANONICAL_ACTIONS."""
    if isinstance(action, (list, tuple)) and len(action) == 2:
        try:
            key = (int(action[0]), int(action[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid vector action {action}") from exc
        if key in _VECTOR_TO_ACTION:
            return _VECTOR_TO_ACTION[key]
        raise ValueError(f"Unknown vector action {action}")

    if isinstance(action, str):
        cleaned = action.strip().strip("'").strip('"')
        upper = cleaned.upper()
        if upper in _TEXT_TO_ACTION:
            return _TEXT_TO_ACTION[upper]

        if cleaned.startswith("[") or cleaned.startswith("("):
            try:
                parsed = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Could not parse action string {action}") from exc
            return normalize_action_token(parsed)

    raise ValueError(f"Unknown action token: {action}")

