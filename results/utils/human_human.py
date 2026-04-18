import json
from pathlib import Path
from typing import Any
import numpy as np
from results.utils.config import layout_keys, layout_labels, resolve_repo_path
from results.utils.evaluation import PairSpec


def collect_human_human_reference(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect human-human episode scores from data/trajectories JSON files."""
    human_config = config.get("human_human", {})
    trajectories_dir = resolve_repo_path(human_config.get("trajectories_dir", "data/trajectories"))
    labels = layout_labels(config)
    wanted_layouts = set(layout_keys(config))
    rows_by_layout: dict[str, list[dict[str, float]]] = {layout: [] for layout in wanted_layouts}

    for path in sorted(trajectories_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            continue
        trajectory = payload.get("trajectory", [])
        if not trajectory:
            continue
        first_step = trajectory[0]
        layout = first_step.get("layout_name") or payload.get("layout_name")
        if layout not in wanted_layouts:
            continue

        rewards = [float(step.get("reward", 0.0) or 0.0) for step in trajectory]
        scores = [float(step.get("score", 0.0) or 0.0) for step in trajectory]
        total_reward = float(sum(rewards))
        final_score = float(max(scores)) if scores else total_reward
        rows_by_layout[layout].append(
            {
                "reward": total_reward,
                "score": final_score,
                "deliveries": total_reward / 20.0,
                "steps": float(len(trajectory)),
            }
        )

    rows = []
    for layout in layout_keys(config):
        episodes = rows_by_layout.get(layout, [])
        rewards = np.asarray([ep["reward"] for ep in episodes], dtype=float)
        deliveries = np.asarray([ep["deliveries"] for ep in episodes], dtype=float)
        steps = np.asarray([ep["steps"] for ep in episodes], dtype=float)
        n = int(len(episodes))
        if n == 0:
            rows.append(
                {
                    "layout": layout,
                    "layout_label": labels.get(layout, layout),
                    "n_episodes": 0,
                    "mean_reward": 0.0,
                    "stderr_reward": 0.0,
                    "mean_deliveries": 0.0,
                    "stderr_deliveries": 0.0,
                    "mean_steps": 0.0,
                    "min_steps": 0.0,
                    "max_steps": 0.0,
                }
            )
            continue
        rows.append(
            {
                "layout": layout,
                "layout_label": labels.get(layout, layout),
                "n_episodes": n,
                "mean_reward": float(np.mean(rewards)),
                "stderr_reward": float(np.std(rewards) / np.sqrt(n)),
                "mean_deliveries": float(np.mean(deliveries)),
                "stderr_deliveries": float(np.std(deliveries) / np.sqrt(n)),
                "mean_steps": float(np.mean(steps)),
                "min_steps": float(np.min(steps)),
                "max_steps": float(np.max(steps)),
            }
        )
    return rows


def human_reference_bar_data(
    config: dict[str, Any],
    evaluation_payload: dict[str, Any],
    human_rows: list[dict[str, Any]],
    metric: str,
) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    """Build grouped bar data for agents plus human-human recorded episodes."""
    labels = layout_labels(config)
    layouts = layout_keys(config)
    groups = [labels.get(layout, layout) for layout in layouts]
    human_label = config.get("human_human", {}).get("label", "Human-Human")
    agent_methods = [
        (entry["label"], (entry["p0"], entry["p1"]))
        for entry in config.get("final_methods", [])
        if entry["label"] != "Random"
    ]
    series_names = [name for name, _ in agent_methods] + [human_label]
    err_metric = metric.replace("mean_", "stderr_")
    pair_results = evaluation_payload["pair_results"]
    human_by_layout = {row["layout"]: row for row in human_rows}

    values = []
    errors = []
    for _, pair_keys in agent_methods:
        series_values = []
        series_errors = []
        for layout in layouts:
            row = pair_results[PairSpec(layout, pair_keys[0], pair_keys[1]).cache_key]
            series_values.append(float(row[metric]))
            series_errors.append(float(row.get(err_metric, 0.0)))
        values.append(series_values)
        errors.append(series_errors)

    human_values = []
    human_errors = []
    for layout in layouts:
        row = human_by_layout.get(layout, {})
        human_values.append(float(row.get(metric, 0.0)))
        human_errors.append(float(row.get(err_metric, 0.0)))
    values.append(human_values)
    errors.append(human_errors)
    return groups, series_names, values, errors
