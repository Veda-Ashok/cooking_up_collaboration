import json
from pathlib import Path
from typing import Any
import numpy as np
from results.utils.bc_metrics import LAYOUT_LABELS
from results.utils.config import layout_labels, resolve_repo_path


DEFAULT_TRAINING_CURVES = {
    "cramped_room": "logs/ppo_curriculum/ppo_cramped_v1_1",
    "asymmetric_advantages": "logs/ppo_curriculum/ppo_asymmetric_v1_1",
    "coordination_ring": "logs/ppo_curriculum/ppo_coordination_v1_1",
}


def _load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return None
    return event_accumulator


def read_scalar_series(log_dir: str | Path, tag: str) -> list[dict[str, float]]:
    """Read a scalar series from all TensorBoard event files under log_dir."""
    event_accumulator = _load_event_accumulator()
    if event_accumulator is None:
        return []
    path = Path(log_dir)
    if not path.exists():
        return []
    accumulator = event_accumulator.EventAccumulator(str(path))
    try:
        accumulator.Reload()
    except Exception:
        return []
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        return []
    series = []
    for event in accumulator.Scalars(tag):
        series.append(
            {
                "step": float(event.step),
                "value": float(event.value),
                "wall_time": float(event.wall_time),
            }
        )
    deduped = {}
    for point in series:
        deduped[point["step"]] = point
    return [deduped[step] for step in sorted(deduped)]


def collect_training_curves(
    log_map: dict[str, str | Path] | None = None,
    tag: str = "episode/mean_sparse_reward",
) -> dict[str, list[dict[str, float]]]:
    """Collect default PPO learning curves from TensorBoard logs."""
    curves = {}
    for layout, log_dir in (log_map or DEFAULT_TRAINING_CURVES).items():
        series = read_scalar_series(log_dir, tag=tag)
        if series:
            curves[layout] = series
    return curves


def collect_training_curves_from_config(config: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    """Collect PPO learning curves from explicit log paths in results/config.json."""
    curve_config = config.get("training_curves", {})
    tag = curve_config.get("tag", "episode/mean_sparse_reward")
    curves = {}
    for run in curve_config.get("runs", []):
        layout = run["layout"]
        series = read_scalar_series(resolve_repo_path(run["log_dir"]), tag=tag)
        if series:
            curves[layout] = series
    return curves


def smooth_series(values: list[float], window: int = 5) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(x) for x in smoothed[: len(values)]]


def save_json(data: Any, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)


def plot_training_curves(
    curves: dict[str, list[dict[str, float]]],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    stem: str = "ppo_training_curves",
    tag_label: str = "Mean sparse reward",
) -> list[Path]:
    if not curves:
        return []
    import matplotlib.pyplot as plt

    from results.utils.plot_style import PALETTE, apply_publication_style, save_figure

    apply_publication_style(font_size=15, axes_linewidth=2.2)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    colors = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["teal"]]
    labels = layout_labels(config) if config is not None else LAYOUT_LABELS
    for idx, (layout, series) in enumerate(curves.items()):
        steps = [point["step"] for point in series]
        values = smooth_series([point["value"] for point in series], window=5)
        ax.plot(
            steps,
            values,
            linewidth=2.8,
            color=colors[idx % len(colors)],
            label=labels.get(layout, layout),
        )
    ax.set_title("PPO Training Curves", pad=14, weight="bold")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(tag_label)
    ax.tick_params(axis="both", width=2.0, length=6)
    ax.legend(loc="best")
    return save_figure(fig, output_dir, stem)
