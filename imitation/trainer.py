import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from imitation.constants import ACTION_TO_ID
from imitation.datasets import build_dataloaders
from imitation.io_utils import create_run_dir, save_checkpoint, save_json
from imitation.metrics import compute_classification_metrics
from imitation.models import LSTMPolicy, MLPPolicy
from imitation.preprocessing import (
    build_trial_records,
    featurize_trials_with_overcooked,
    load_csv_rows,
    split_by_trial_id,
)


@dataclass
class BCConfig:
    data_csv: str
    model: str
    player_idx: int
    train_ratio: float
    val_ratio: float
    seed: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    device: str
    outdir: str
    run_name: str | None
    planner_cache_dir: str
    hidden_dim: int
    num_layers: int
    dropout: float
    seq_len: int
    mlp_hidden: list[int]
    grad_clip: float
    early_stop_patience: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros_like(counts, dtype=np.float64)

    for idx, count in enumerate(counts):
        if count > 0:
            weights[idx] = total / (num_classes * count)

    nonzero = weights[weights > 0]
    if nonzero.size > 0:
        weights = weights / nonzero.mean()
    else:
        weights = np.ones(num_classes, dtype=np.float64)

    return torch.tensor(weights, dtype=torch.float32)


def _build_model(config: BCConfig, input_dim: int, num_actions: int) -> nn.Module:
    if config.model == "mlp":
        return MLPPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=config.mlp_hidden,
            dropout=config.dropout,
        )
    if config.model == "lstm":
        return LSTMPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model {config.model}")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None,
) -> tuple[float, dict]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = y_batch.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        y_true.append(y_batch.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred, axis=0) if y_pred else np.array([], dtype=np.int64)
    metrics = compute_classification_metrics(y_true_np, y_pred_np, num_classes=len(ACTION_TO_ID))
    mean_loss = running_loss / max(sample_count, 1)
    return mean_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        y_true.append(y_batch.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred, axis=0) if y_pred else np.array([], dtype=np.int64)
    metrics = compute_classification_metrics(y_true_np, y_pred_np, num_classes=len(ACTION_TO_ID))
    mean_loss = running_loss / max(sample_count, 1)
    return mean_loss, metrics


def run_training(config: BCConfig) -> Path:
    _set_seed(config.seed)
    run_dir = create_run_dir(config.outdir, config.model, config.run_name)

    rows = load_csv_rows(config.data_csv)
    cleaned_trials, clean_report = build_trial_records(rows, player_idx=config.player_idx)
    if not cleaned_trials:
        raise RuntimeError("No valid trials found after CSV cleaning")

    featurized_trials, feat_report = featurize_trials_with_overcooked(
        cleaned_trials,
        player_idx=config.player_idx,
        planner_cache_dir=config.planner_cache_dir,
    )
    if not featurized_trials:
        raise RuntimeError("No valid featurized trials found")

    train_trials, val_trials, test_trials = split_by_trial_id(
        featurized_trials,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    if not train_trials or not val_trials or not test_trials:
        raise RuntimeError(
            "One or more data splits are empty after split_by_trial_id. "
            "Adjust train_ratio/val_ratio or inspect preprocessing filters."
        )

    pin_memory = config.device.startswith("cuda")
    dataloaders, datasets = build_dataloaders(
        train_trials,
        val_trials,
        test_trials,
        model_type=config.model,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    input_dim = int(train_trials[0].x.shape[1])
    num_actions = len(ACTION_TO_ID)
    model = _build_model(config, input_dim, num_actions)
    device = torch.device(config.device)
    model.to(device)

    class_weights = _compute_class_weights(datasets["train"].targets_np, num_classes=num_actions).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        grad_clip = config.grad_clip if config.model == "lstm" else None
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        current_macro_f1 = float(val_metrics["macro_f1"])
        if current_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_macro_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
            )
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    save_checkpoint(
        last_path,
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )

    split_summary = {
        "train_trials": len(train_trials),
        "val_trials": len(val_trials),
        "test_trials": len(test_trials),
        "train_rows": int(sum(trial.x.shape[0] for trial in train_trials)),
        "val_rows": int(sum(trial.x.shape[0] for trial in val_trials)),
        "test_rows": int(sum(trial.x.shape[0] for trial in test_trials)),
        "train_trial_ids": [trial.trial_id for trial in train_trials],
        "val_trial_ids": [trial.trial_id for trial in val_trials],
        "test_trial_ids": [trial.trial_id for trial in test_trials],
    }

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history": history,
    }
    report_payload = {
        "clean_report": clean_report,
        "featurization_report": feat_report,
    }
    config_payload = asdict(config)
    label_payload = ACTION_TO_ID

    save_json(run_dir / "metrics.json", metrics_payload)
    save_json(run_dir / "split_summary.json", split_summary)
    save_json(run_dir / "preprocessing_report.json", report_payload)
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "label_mapping.json", label_payload)

    print(f"Training complete. Best epoch: {best_epoch}, val_macro_f1={best_val_macro_f1:.4f}")
    print(f"Test macro_f1={test_metrics['macro_f1']:.4f}, accuracy={test_metrics['accuracy']:.4f}")
    print(f"Artifacts saved to: {run_dir}")
    return run_dir

