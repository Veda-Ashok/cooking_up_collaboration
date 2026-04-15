import json
import hashlib
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from results.utils.config import agent_dir, layout_keys, layout_labels


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_SERVER = REPO_ROOT / "webapp" / "server"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WEBAPP_SERVER) not in sys.path:
    sys.path.insert(0, str(WEBAPP_SERVER))

@dataclass(frozen=True)
class PairSpec:
    layout: str
    p0_key: str
    p1_key: str

    @property
    def cache_key(self) -> str:
        return f"{self.layout}::{self.p0_key}+{self.p1_key}"


class RandomOvercookedAgent:
    """Minimal random policy with the same action/reset interface as webapp agents."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def set_mdp_context(self, mdp, layout_name: str | None = None) -> None:
        return None

    def reset(self) -> None:
        return None

    def action(self, state):
        from overcooked_ai_py.mdp.actions import Action

        return self._rng.choice(Action.ALL_ACTIONS), None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _final_methods(config: dict[str, Any]) -> list[tuple[str, tuple[str, str]]]:
    return [(entry["label"], (entry["p0"], entry["p1"])) for entry in config.get("final_methods", [])]


def _heatmap_agents(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [(entry["label"], entry["key"]) for entry in config.get("heatmap_agents", [])]


def _role_swap_series(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (entry["label"], entry["p0"], entry["p1"])
        for entry in config.get("role_swap", {}).get("series", [])
    ]


def _config_signature(config: dict[str, Any]) -> str:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    encoded = json.dumps(serializable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_dir_for(config: dict[str, Any], layout: str, agent_key: str) -> Path:
    resolved = agent_dir(config, layout, agent_key)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing webapp agent directory: {resolved}")
    checkpoint_path = config["agents"][layout][agent_key].get("checkpoint_path")
    if checkpoint_path:
        from results.utils.config import resolve_repo_path

        resolved_checkpoint = resolve_repo_path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Missing configured checkpoint: {resolved_checkpoint}")
    return resolved


def _load_webapp_agent_classes():
    """Import webapp agent classes lazily to avoid editor/runtime import noise."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(WEBAPP_SERVER) not in sys.path:
        sys.path.insert(0, str(WEBAPP_SERVER))
    bc_module = importlib.import_module("bc_torch_agent")
    rl_module = importlib.import_module("rl_torch_agent")
    return bc_module.TorchBCAgent, rl_module.TorchRLAgent


def make_agent(config: dict[str, Any], layout: str, agent_key: str, agent_index: int, seed: int, device: str = "cpu"):
    if agent_key == "random":
        return RandomOvercookedAgent(seed=seed + 1009 * agent_index)
    resolved_agent_dir = _agent_dir_for(config, layout, agent_key)
    manifest_path = resolved_agent_dir / "agent_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for {resolved_agent_dir}")
    with open(manifest_path, "r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)
    TorchBCAgent, TorchRLAgent = _load_webapp_agent_classes()
    if manifest.get("type") == "bc_torch":
        return TorchBCAgent(agent_dir=str(resolved_agent_dir), agent_index=agent_index, device=device)
    if manifest.get("type") == "rl_torch":
        return TorchRLAgent(agent_dir=str(resolved_agent_dir), agent_index=agent_index, device=device)
    raise ValueError(f"Unsupported manifest type in {manifest_path}: {manifest.get('type')}")


def _summarize_episode_values(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "stderr": 0.0,
            "min": 0.0,
            "max": 0.0,
            "values": [],
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "stderr": float(np.std(arr) / np.sqrt(max(1, arr.size))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(x) for x in arr.tolist()],
    }


def evaluate_pair(
    config: dict[str, Any],
    pair: PairSpec,
    n_episodes: int,
    horizon: int,
    seed: int,
    device: str = "cpu",
    soup_reward: float = 20.0,
) -> dict[str, Any]:
    """Evaluate a fixed player-0/player-1 pairing for one layout."""
    ordered_layouts = layout_keys(config)
    labels = layout_labels(config)
    if pair.layout not in ordered_layouts:
        raise ValueError(f"Unsupported layout for results pipeline: {pair.layout}")

    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    set_global_seed(seed)
    mdp = OvercookedGridworld.from_layout_name(pair.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

    agent_p0 = make_agent(config, pair.layout, pair.p0_key, agent_index=0, seed=seed, device=device)
    agent_p1 = make_agent(config, pair.layout, pair.p1_key, agent_index=1, seed=seed + 7919, device=device)
    for agent in (agent_p0, agent_p1):
        if hasattr(agent, "set_mdp_context"):
            agent.set_mdp_context(mdp, pair.layout)

    rewards: list[float] = []
    deliveries: list[float] = []
    delivery_episode_count = 0

    for ep in range(n_episodes):
        set_global_seed(seed + ep)
        env.reset()
        if hasattr(agent_p0, "reset"):
            agent_p0.reset()
        if hasattr(agent_p1, "reset"):
            agent_p1.reset()

        ep_reward = 0.0
        for _ in range(horizon):
            state = env.state
            action0, _ = agent_p0.action(state)
            action1, _ = agent_p1.action(state)
            _, sparse_reward, done, _ = env.step((action0, action1))
            ep_reward += float(sparse_reward)
            if done:
                break
        ep_deliveries = ep_reward / soup_reward if soup_reward > 0 else 0.0
        rewards.append(ep_reward)
        deliveries.append(ep_deliveries)
        if ep_reward > 0:
            delivery_episode_count += 1

    reward_stats = _summarize_episode_values(rewards)
    delivery_stats = _summarize_episode_values(deliveries)
    return {
        "layout": pair.layout,
        "layout_label": labels.get(pair.layout, pair.layout),
        "p0_key": pair.p0_key,
        "p1_key": pair.p1_key,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "mean_reward": reward_stats["mean"],
        "std_reward": reward_stats["std"],
        "stderr_reward": reward_stats["stderr"],
        "mean_deliveries": delivery_stats["mean"],
        "std_deliveries": delivery_stats["std"],
        "stderr_deliveries": delivery_stats["stderr"],
        "delivery_rate": delivery_episode_count / max(1, n_episodes),
        "per_episode_rewards": reward_stats["values"],
        "per_episode_deliveries": delivery_stats["values"],
    }


def required_pair_specs(config: dict[str, Any], include_heatmap: bool = True, include_role_swap: bool = True) -> list[PairSpec]:
    """Build the deduplicated set of pairings required for all result plots."""
    specs: dict[str, PairSpec] = {}
    for layout in layout_keys(config):
        for _, (p0_key, p1_key) in _final_methods(config):
            pair = PairSpec(layout, p0_key, p1_key)
            specs[pair.cache_key] = pair
        if include_heatmap:
            for _, row_key in _heatmap_agents(config):
                for _, col_key in _heatmap_agents(config):
                    pair = PairSpec(layout, row_key, col_key)
                    specs[pair.cache_key] = pair
        if include_role_swap:
            for _, p0_key, p1_key in _role_swap_series(config):
                pair = PairSpec(layout, p0_key, p1_key)
                specs[pair.cache_key] = pair
    return list(specs.values())


def evaluate_required_pairs(
    config: dict[str, Any],
    n_episodes: int,
    horizon: int,
    seed: int,
    device: str,
    cache_path: str | Path,
    force: bool = False,
    include_heatmap: bool = True,
    include_role_swap: bool = True,
) -> dict[str, Any]:
    """Evaluate or load cached rollout results for all requested pairings."""
    output = Path(cache_path)
    signature = _config_signature(config)
    if output.exists() and not force:
        with open(output, "r", encoding="utf-8") as file_handle:
            cached = json.load(file_handle)
        if (
            cached.get("n_episodes") == n_episodes
            and cached.get("horizon") == horizon
            and cached.get("seed") == seed
            and cached.get("config_signature") == signature
        ):
            return cached

    pair_results: dict[str, Any] = {}
    for idx, pair in enumerate(required_pair_specs(config, include_heatmap, include_role_swap), start=1):
        print(f"[eval {idx:02d}] {pair.layout}: {pair.p0_key} + {pair.p1_key}")
        pair_results[pair.cache_key] = evaluate_pair(
            config=config,
            pair=pair,
            n_episodes=n_episodes,
            horizon=horizon,
            seed=seed + idx * 17,
            device=device,
        )

    payload = {
        "n_episodes": n_episodes,
        "horizon": horizon,
        "seed": seed,
        "config_signature": signature,
        "pair_results": pair_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return payload


def final_bar_data(config: dict[str, Any], evaluation_payload: dict[str, Any], metric: str) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    """Return groups, series, values, and stderr arrays for final performance bars."""
    pair_results = evaluation_payload["pair_results"]
    final_methods = _final_methods(config)
    labels = layout_labels(config)
    series_names = [name for name, _ in final_methods]
    groups = [labels.get(layout, layout) for layout in layout_keys(config)]
    values: list[list[float]] = []
    errors: list[list[float]] = []
    err_metric = metric.replace("mean_", "stderr_")
    for _, pair_keys in final_methods:
        method_values = []
        method_errors = []
        for layout in layout_keys(config):
            key = PairSpec(layout, pair_keys[0], pair_keys[1]).cache_key
            row = pair_results[key]
            method_values.append(float(row[metric]))
            method_errors.append(float(row.get(err_metric, 0.0)))
        values.append(method_values)
        errors.append(method_errors)
    return groups, series_names, values, errors


def heatmap_data(config: dict[str, Any], evaluation_payload: dict[str, Any], metric: str) -> dict[str, list[list[float]]]:
    pair_results = evaluation_payload["pair_results"]
    labels = layout_labels(config)
    heatmap_agents = _heatmap_agents(config)
    matrices = {}
    for layout in layout_keys(config):
        matrix = []
        for _, row_key in heatmap_agents:
            row = []
            for _, col_key in heatmap_agents:
                pair = PairSpec(layout, row_key, col_key)
                row.append(float(pair_results[pair.cache_key][metric]))
            matrix.append(row)
        matrices[labels.get(layout, layout)] = matrix
    return matrices


def role_swap_bar_data(config: dict[str, Any], evaluation_payload: dict[str, Any], metric: str) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    pair_results = evaluation_payload["pair_results"]
    labels = layout_labels(config)
    groups = [labels.get(layout, layout) for layout in layout_keys(config)]
    role_series = _role_swap_series(config)
    series_names = [name for name, _, _ in role_series]
    err_metric = metric.replace("mean_", "stderr_")
    values = []
    errors = []
    for _, p0_key, p1_key in role_series:
        row_values = []
        row_errors = []
        for layout in layout_keys(config):
            key = PairSpec(layout, p0_key, p1_key).cache_key
            row = pair_results[key]
            row_values.append(float(row[metric]))
            row_errors.append(float(row.get(err_metric, 0.0)))
        values.append(row_values)
        errors.append(row_errors)
    return groups, series_names, values, errors
