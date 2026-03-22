import argparse
import os
import torch
from imitation.trainer import BCConfig, run_training


def _parse_hidden_dims(hidden_dims_raw: str) -> list[int]:
    values = [chunk.strip() for chunk in hidden_dims_raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("mlp hidden dims cannot be empty")
    hidden_dims = [int(value) for value in values]
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("mlp hidden dims must be positive integers")
    return hidden_dims


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior Cloning training for Overcooked CSV trajectories")

    parser.add_argument("--data-csv", type=str, default="data/2019_hh_trials_plus_local.csv")
    parser.add_argument("--model", type=str, choices=["lstm", "mlp"], default="mlp")
    parser.add_argument("--player-mode", type=str, choices=["both", "single"], default="both")
    parser.add_argument("--player-idx", type=int, default=None, choices=[0, 1])
    parser.add_argument("--layout-name", type=str, default="cramped_room",
                        help="Train on a single layout only (e.g. cramped_room)")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)  # LSTM hidden dims
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-hidden", type=str, default="64,64")

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)

    parser.add_argument("--min-episode-steps", type=int, default=800)
    parser.add_argument("--min-total-reward", type=float, default=80)
    parser.add_argument("--save-every-epoch", action="store_true")

    parser.add_argument("--outdir", type=str, default="trained_models/bc")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    if args.seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    if args.player_mode == "single" and args.player_idx is None:
        raise ValueError("--player-idx is required when --player-mode=single")
    if args.player_mode == "both" and args.player_idx is not None:
        raise ValueError("--player-idx must not be set when --player-mode=both")

    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"CSV not found: {args.data_csv}")

    config = BCConfig(
        data_csv=args.data_csv,
        model=args.model,
        player_mode=args.player_mode,
        player_idx=args.player_idx,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=_resolve_device(args.device),
        outdir=args.outdir,
        run_name=args.run_name,
        planner_cache_dir=args.planner_cache_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
        mlp_hidden=_parse_hidden_dims(args.mlp_hidden),
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        layout_name=args.layout_name,
        min_episode_steps=args.min_episode_steps,
        min_total_reward=args.min_total_reward,
        save_every_epoch=args.save_every_epoch,
    )

    run_dir = run_training(config)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

