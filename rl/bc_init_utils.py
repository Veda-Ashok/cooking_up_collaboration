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


def transfer_bc_mlp_to_sb3_policy(
    sb3_policy: Any,
    bc_state_dict: dict[str, Any],
    hidden_dims: list[int],
) -> dict[str, Any]:
    """Transfer BC MLPPolicy weights into an SB3 ActorCriticPolicy.

    Assumes the SB3 policy was created with
        net_arch=dict(pi=hidden_dims, vf=hidden_dims)
    so the layer sizes match the BC checkpoint exactly.

    BC MLPPolicy.network layout (with dropout):
        [Linear, ReLU, Dropout, Linear, ReLU, Dropout, ..., Linear(out)]
    SB3 MlpExtractor.policy_net layout (no dropout):
        [Linear, ReLU, Linear, ReLU, ...]
    """
    ppo_state = sb3_policy.state_dict()
    loaded: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    bc_linears: list[tuple[str, str]] = []
    for key in sorted(bc_state_dict.keys()):
        if key.startswith("network.") and key.endswith(".weight"):
            idx = key.split(".")[1]
            bc_linears.append((f"network.{idx}.weight", f"network.{idx}.bias"))

    n_hidden = len(hidden_dims)

    # SB3 MlpExtractor with net_arch=dict(pi=[h1,h2], vf=[h1,h2]):
    #   policy_net: Linear(in, h1) at .0, Linear(h1, h2) at .2, ...
    #   value_net:  same
    # BC network: Linear at .0, .3, .6, ... (stride 3 because of ReLU+Dropout)
    # Hidden layers: first n_hidden bc_linears -> policy_net/value_net
    # Output layer: last bc_linear -> action_net

    for layer_i in range(n_hidden):
        bc_w_key, bc_b_key = bc_linears[layer_i]
        sb3_idx = layer_i * 2  # stride 2 in SB3 (Linear, ReLU)

        for net_name in ("mlp_extractor.policy_net", "mlp_extractor.value_net"):
            for suffix, bc_key in [("weight", bc_w_key), ("bias", bc_b_key)]:
                ppo_key = f"{net_name}.{sb3_idx}.{suffix}"
                if ppo_key in ppo_state and bc_key in bc_state_dict:
                    if ppo_state[ppo_key].shape == bc_state_dict[bc_key].shape:
                        ppo_state[ppo_key] = bc_state_dict[bc_key].detach().clone()
                        loaded.append({"ppo_key": ppo_key, "bc_key": bc_key})
                    else:
                        skipped.append({"ppo_key": ppo_key, "bc_key": bc_key,
                                        "reason": "shape_mismatch"})

    if len(bc_linears) > n_hidden:
        bc_out_w, bc_out_b = bc_linears[n_hidden]
        for suffix, bc_key in [("weight", bc_out_w), ("bias", bc_out_b)]:
            ppo_key = f"action_net.{suffix}"
            if ppo_key in ppo_state and bc_key in bc_state_dict:
                if ppo_state[ppo_key].shape == bc_state_dict[bc_key].shape:
                    ppo_state[ppo_key] = bc_state_dict[bc_key].detach().clone()
                    loaded.append({"ppo_key": ppo_key, "bc_key": bc_key})
                else:
                    skipped.append({"ppo_key": ppo_key, "bc_key": bc_key,
                                    "reason": "shape_mismatch"})

    sb3_policy.load_state_dict(ppo_state)
    return {
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
    }


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
