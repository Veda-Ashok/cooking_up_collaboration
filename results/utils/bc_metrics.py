import json
from pathlib import Path
from typing import Any

from results.utils.config import layout_keys, layout_labels, resolve_repo_path


LAYOUT_ORDER = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
LAYOUT_LABELS = {
    "cramped_room": "Cramped",
    "asymmetric_advantages": "Asymmetric",
    "coordination_ring": "Coordination",
}
MODEL_LABELS = {
    "mlp": "BC MLP",
    "lstm": "BC LSTM",
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def collect_bc_metrics(bc_root: str | Path) -> list[dict[str, Any]]:
    """Collect validation metrics from trained_models/bc/*/metrics.json."""
    root = Path(bc_root)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(root.glob("bc_*_v1")):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        best_metrics = metrics.get("best_val_metrics", {})
        model_type = str(config.get("model", "")).lower()
        layout = str(config.get("layout_name", ""))
        rows.append(
            {
                "run_name": run_dir.name,
                "model_type": model_type,
                "model_label": MODEL_LABELS.get(model_type, model_type.upper()),
                "layout": layout,
                "layout_label": LAYOUT_LABELS.get(layout, layout),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_loss": float(metrics.get("best_val_loss", 0.0)),
                "accuracy": float(best_metrics.get("accuracy", 0.0)),
                "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
                "weighted_f1": float(best_metrics.get("weighted_f1", 0.0)),
            }
        )
    return rows


def collect_bc_metrics_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect BC validation metrics from the explicit paths in results/config.json."""
    labels = layout_labels(config)
    rows: list[dict[str, Any]] = []
    for entry in config.get("bc_metrics", []):
        metrics_path = resolve_repo_path(entry["metrics_path"])
        config_path = resolve_repo_path(entry["config_path"])
        if not metrics_path.exists() or not config_path.exists():
            continue
        run_config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        best_metrics = metrics.get("best_val_metrics", {})
        layout = str(entry["layout"])
        model_label = str(entry["model_label"])
        rows.append(
            {
                "run_name": run_config.get("run_name", config_path.parent.name),
                "model_type": str(run_config.get("model", "")).lower(),
                "model_label": model_label,
                "layout": layout,
                "layout_label": labels.get(layout, layout),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_loss": float(metrics.get("best_val_loss", 0.0)),
                "accuracy": float(best_metrics.get("accuracy", 0.0)),
                "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
                "weighted_f1": float(best_metrics.get("weighted_f1", 0.0)),
            }
        )
    return rows


def bc_metric_panels(rows: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Convert BC metric rows to bar-panel data structures."""
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    series = ["BC MLP", "BC LSTM"]
    if config is None:
        ordered_layouts = LAYOUT_ORDER
        labels = LAYOUT_LABELS
    else:
        ordered_layouts = layout_keys(config)
        labels = layout_labels(config)
    groups = [labels.get(layout, layout) for layout in ordered_layouts]
    panels = []
    for metric_key, metric_name in metrics:
        values = []
        for model_label in series:
            model_values = []
            for layout in ordered_layouts:
                match = next(
                    (
                        row
                        for row in rows
                        if row["layout"] == layout and row["model_label"] == model_label
                    ),
                    None,
                )
                model_values.append(float(match[metric_key]) if match else 0.0)
            values.append(model_values)
        panels.append(
            {
                "title": metric_name,
                "ylabel": metric_name,
                "groups": groups,
                "series": series,
                "values": values,
            }
        )
    return panels
