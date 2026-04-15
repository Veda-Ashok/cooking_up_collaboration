import argparse
import csv
import json
from pathlib import Path
from typing import Any

from results.utils.bc_metrics import bc_metric_panels, collect_bc_metrics_from_config
from results.utils.config import load_results_config, resolve_repo_path
from results.utils.evaluation import (
    final_bar_data,
    heatmap_data,
    evaluate_required_pairs,
    role_swap_bar_data,
)
from results.utils.plot_bars import plot_grouped_bars, plot_small_multipanel_bars
from results.utils.plot_heatmap import plot_heatmap_grid
from results.utils.tensorboard_curves import (
    collect_training_curves_from_config,
    plot_training_curves,
    save_json,
)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)


def _write_final_summary_csv(evaluation_payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, row in sorted(evaluation_payload.get("pair_results", {}).items()):
        rows.append(
            {
                "pair_key": key,
                "layout": row["layout"],
                "p0": row["p0_key"],
                "p1": row["p1_key"],
                "n_episodes": row["n_episodes"],
                "horizon": row["horizon"],
                "mean_reward": row["mean_reward"],
                "stderr_reward": row["stderr_reward"],
                "mean_deliveries": row["mean_deliveries"],
                "stderr_deliveries": row["stderr_deliveries"],
                "delivery_rate": row["delivery_rate"],
            }
        )
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_label(metric: str) -> str:
    if metric == "mean_deliveries":
        return "Mean deliveries per 400-step episode"
    if metric == "mean_reward":
        return "Mean sparse reward per 400-step episode"
    if metric == "episode/mean_sparse_reward":
        return "Mean sparse reward per episode"
    if metric == "episode/mean_total_reward":
        return "Mean total reward per episode"
    return metric.replace("_", " ").title()


def generate_all(args: argparse.Namespace) -> None:
    config = load_results_config(args.config)
    output_root = resolve_repo_path(args.output_dir)
    data_dir = output_root / "data"
    plot_dir = output_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] {config['_config_path']}")
    print(f"[output] {output_root}")

    bc_rows = collect_bc_metrics_from_config(config)
    _write_json(bc_rows, data_dir / "bc_validation_metrics.json")
    if bc_rows:
        plot_small_multipanel_bars(
            panels=bc_metric_panels(bc_rows, config=config),
            output_dir=plot_dir,
            stem="bc_validation_metrics",
            title="BC Validation Quality",
        )
        print("[plot] bc_validation_metrics")
    else:
        print("[skip] No BC metric rows found from config.")

    curves = collect_training_curves_from_config(config)
    save_json(curves, data_dir / "ppo_training_curves.json")
    curve_tag = config.get("training_curves", {}).get("tag", "episode/mean_sparse_reward")
    curve_paths = plot_training_curves(
        curves,
        output_dir=plot_dir,
        config=config,
        stem="ppo_training_curves",
        tag_label=_metric_label(curve_tag),
    )
    if curve_paths:
        print("[plot] ppo_training_curves")
    else:
        print("[skip] No TensorBoard training curves found. Check results/config.json log_dir values or tensorboard install.")

    if args.skip_rollouts:
        print("[skip] Rollout evaluation skipped.")
        return

    rollout_cache = data_dir / "rollout_evaluation.json"
    evaluation_payload = evaluate_required_pairs(
        config=config,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        cache_path=rollout_cache,
        force=args.force_eval,
        include_heatmap=not args.skip_heatmap,
        include_role_swap=not args.skip_role_swap,
    )
    _write_final_summary_csv(evaluation_payload, data_dir / "rollout_summary.csv")

    for metric in ("mean_reward", "mean_deliveries"):
        groups, series, values, errors = final_bar_data(config, evaluation_payload, metric=metric)
        keep_indices = [idx for idx, name in enumerate(series) if name != "Random"]
        series = [series[idx] for idx in keep_indices]
        values = [values[idx] for idx in keep_indices]
        errors = [errors[idx] for idx in keep_indices]
        plot_grouped_bars(
            groups=groups,
            series_names=series,
            values=values,
            errors=errors,
            ylabel=_metric_label(metric),
            title="Final 400-Step Team Performance",
            output_dir=plot_dir,
            stem=f"final_performance_{metric}",
            annotate=False,
            figsize=(14.5, 7.2),
            legend_inside=True,
        )
        print(f"[plot] final_performance_{metric}")

    if not args.skip_heatmap:
        heatmap_metric = args.heatmap_metric
        matrices = heatmap_data(config, evaluation_payload, metric=heatmap_metric)
        labels = [entry["label"] for entry in config.get("heatmap_agents", [])]
        plot_heatmap_grid(
            matrices=matrices,
            row_labels=labels,
            col_labels=labels,
            output_dir=plot_dir,
            stem=f"cross_partner_heatmap_{heatmap_metric}",
            title="Cross-Partner Coordination Matrix",
            value_label=_metric_label(heatmap_metric),
        )
        print(f"[plot] cross_partner_heatmap_{heatmap_metric}")

    if not args.skip_role_swap:
        groups, series, values, errors = role_swap_bar_data(
            config,
            evaluation_payload,
            metric=args.role_swap_metric,
        )
        plot_grouped_bars(
            groups=groups,
            series_names=series,
            values=values,
            errors=errors,
            ylabel=_metric_label(args.role_swap_metric),
            title="PPO Role-Swap Evaluation",
            output_dir=plot_dir,
            stem=f"ppo_role_swap_{args.role_swap_metric}",
            annotate=False,
            figsize=(10.8, 6.8),
            legend_inside=True,
        )
        print(f"[plot] ppo_role_swap_{args.role_swap_metric}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate course-project result plots for BC and PPO Overcooked agents."
    )
    parser.add_argument("--config", default="results/config.json", help="Path to results config JSON.")
    parser.add_argument("--output-dir", default="results", help="Directory for generated data and plots.")
    parser.add_argument("--n-episodes", type=int, default=50, help="Episodes per evaluated pairing.")
    parser.add_argument("--horizon", type=int, default=400, help="Overcooked rollout horizon.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device for model inference.")
    parser.add_argument("--force-eval", action="store_true", help="Ignore cached rollout_evaluation.json.")
    parser.add_argument("--skip-rollouts", action="store_true", help="Only generate BC metric and training-curve plots.")
    parser.add_argument("--skip-heatmap", action="store_true", help="Skip cross-partner heatmap evaluations and plots.")
    parser.add_argument("--skip-role-swap", action="store_true", help="Skip PPO role-swap evaluations and plots.")
    parser.add_argument(
        "--heatmap-metric",
        default="mean_reward",
        choices=["mean_reward", "mean_deliveries"],
        help="Metric used in the cross-partner heatmap.",
    )
    parser.add_argument(
        "--role-swap-metric",
        default="mean_reward",
        choices=["mean_reward", "mean_deliveries"],
        help="Metric used in the PPO role-swap bar plot.",
    )
    return parser


if __name__ == "__main__":
    generate_all(build_parser().parse_args())
