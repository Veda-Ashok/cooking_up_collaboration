import re
from pathlib import Path
from typing import Any
import torch
from imitation.models.lstm import LSTMPolicy


def load_checkpoint_state_dict(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    return state_dict


def infer_lstm_architecture(
    state_dict: dict[str, Any],
    default_input_dim: int | None = None,
    default_hidden_dim: int = 128,
    default_num_layers: int = 1,
) -> tuple[int, int, int]:
    input_dim = default_input_dim
    hidden_dim = int(default_hidden_dim)
    layer_ids: set[int] = set()

    pattern = re.compile(r"^lstm\.weight_ih_l(\d+)$")
    for key, value in state_dict.items():
        match = pattern.match(key)
        if not match:
            continue
        layer_id = int(match.group(1))
        layer_ids.add(layer_id)
        if layer_id == 0:
            if hasattr(value, "shape") and len(value.shape) == 2:
                hidden_dim = int(value.shape[0] // 4)
                input_dim = int(value.shape[1])

    num_layers = max(layer_ids) + 1 if layer_ids else int(default_num_layers)
    if input_dim is None:
        raise ValueError("Could not infer LSTM input_dim; provide default_input_dim explicitly.")
    return int(input_dim), int(hidden_dim), int(num_layers)


def load_lstm_policy_from_checkpoint(
    checkpoint_path: str | Path,
    num_actions: int,
    input_dim: int | None = None,
    default_hidden_dim: int = 128,
    default_num_layers: int = 1,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> LSTMPolicy:
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=device)
    inferred_input_dim, hidden_dim, num_layers = infer_lstm_architecture(
        state_dict=state_dict,
        default_input_dim=input_dim,
        default_hidden_dim=default_hidden_dim,
        default_num_layers=default_num_layers,
    )

    if input_dim is not None and inferred_input_dim != int(input_dim):
        raise ValueError(
            f"Checkpoint input_dim mismatch for {checkpoint_path}: "
            f"inferred {inferred_input_dim}, expected {input_dim}"
        )

    model = LSTMPolicy(
        input_dim=inferred_input_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def transfer_bc_lstm_to_recurrent_policy(
    recurrent_policy: Any,
    bc_state_dict: dict[str, Any],
) -> dict[str, Any]:
    policy_state = recurrent_policy.state_dict()
    loaded_keys: list[dict[str, str]] = []
    skipped_keys: list[dict[str, str]] = []

    for policy_key in list(policy_state.keys()):
        if not policy_key.startswith("lstm_actor."):
            continue
        suffix = policy_key[len("lstm_actor.") :]
        bc_key = f"lstm.{suffix}"
        if bc_key not in bc_state_dict:
            continue
        if tuple(policy_state[policy_key].shape) != tuple(bc_state_dict[bc_key].shape):
            skipped_keys.append(
                {
                    "policy_key": policy_key,
                    "bc_key": bc_key,
                    "reason": "shape_mismatch",
                }
            )
            continue
        policy_state[policy_key] = bc_state_dict[bc_key].detach().clone()
        loaded_keys.append({"policy_key": policy_key, "bc_key": bc_key})

    action_mappings = [
        ("action_net.weight", "head.1.weight"),
        ("action_net.bias", "head.1.bias"),
        ("action_net.weight", "head.weight"),
        ("action_net.bias", "head.bias"),
    ]
    seen_policy_action_key: set[str] = set()
    for policy_key, bc_key in action_mappings:
        if policy_key in seen_policy_action_key:
            continue
        if policy_key not in policy_state or bc_key not in bc_state_dict:
            continue
        if tuple(policy_state[policy_key].shape) != tuple(bc_state_dict[bc_key].shape):
            skipped_keys.append(
                {
                    "policy_key": policy_key,
                    "bc_key": bc_key,
                    "reason": "shape_mismatch",
                }
            )
            continue
        policy_state[policy_key] = bc_state_dict[bc_key].detach().clone()
        loaded_keys.append({"policy_key": policy_key, "bc_key": bc_key})
        seen_policy_action_key.add(policy_key)

    recurrent_policy.load_state_dict(policy_state, strict=False)
    return {
        "loaded_count": len(loaded_keys),
        "skipped_count": len(skipped_keys),
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
    }
