from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ravioli.imitation_common import (  # type: ignore[import-not-found]
        ACTION_TO_ID,
        ID_TO_ACTION,
        FeatureSchema,
        build_feature_schema,
        build_policy_model,
        featurize_state_for_player,
        input_state_to_action_id,
    )
else:
    from .imitation_common import (
        ACTION_TO_ID,
        ID_TO_ACTION,
        FeatureSchema,
        build_feature_schema,
        build_policy_model,
        featurize_state_for_player,
        input_state_to_action_id,
    )


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "exports"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "training_runs"
DEFAULT_DEPLOY_DIR = DEFAULT_OUTPUT_DIR / "imitation"


@dataclass
class BCConfig:
    data_path: str
    model: str
    data_source: str
    player_mode: str
    player_idx: int | None
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
    hidden_dim: int
    num_layers: int
    dropout: float
    seq_len: int
    trial_chunk_size: int
    mlp_hidden: list[int]
    grad_clip: float
    early_stop_patience: int


@dataclass
class RawTrial:
    trial_id: str
    slot_id: int
    rows: list[dict[str, Any]]


@dataclass
class FeaturizedTrial:
    trial_id: str
    slot_id: int
    x: np.ndarray
    y: np.ndarray
    slot_ids: np.ndarray


class MLPTimestepDataset(Dataset):
    def __init__(self, trials: list[FeaturizedTrial]) -> None:
        if not trials:
            raise ValueError("Cannot build MLPTimestepDataset with no trials")

        x = np.concatenate([trial.x for trial in trials], axis=0).astype(np.float32)
        y = np.concatenate([trial.y for trial in trials], axis=0).astype(np.int64)
        slot_ids = np.concatenate([trial.slot_ids for trial in trials], axis=0).astype(np.int64)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.targets_np = y
        self.slot_ids_np = slot_ids

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMWindowDataset(Dataset):
    def __init__(self, trials: list[FeaturizedTrial], seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.trials = trials
        self.seq_len = seq_len
        self.index: list[tuple[int, int]] = []
        targets: list[int] = []
        slot_ids: list[int] = []

        for trial_idx, trial in enumerate(trials):
            num_steps = int(trial.x.shape[0])
            if num_steps < seq_len:
                continue
            for start_idx in range(0, num_steps - seq_len + 1):
                self.index.append((trial_idx, start_idx))
                targets.append(int(trial.y[start_idx + seq_len - 1]))
                slot_ids.append(int(trial.slot_ids[start_idx + seq_len - 1]))

        self.targets_np = np.asarray(targets, dtype=np.int64)
        self.slot_ids_np = np.asarray(slot_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trial_idx, start_idx = self.index[idx]
        trial = self.trials[trial_idx]
        x_seq = trial.x[start_idx : start_idx + self.seq_len]
        y_target = trial.y[start_idx + self.seq_len - 1]
        return torch.from_numpy(x_seq).float(), torch.tensor(y_target, dtype=torch.long)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_name


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


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            confusion[target, pred] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = _safe_div(correct, total)

    per_class: dict[str, dict[str, Any]] = {}
    precision_values = []
    recall_values = []
    f1_values = []
    supports = []

    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        class_name = ID_TO_ACTION.get(class_id, str(class_id))
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        supports.append(support)

    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    support_arr = np.asarray(supports, dtype=np.float64)
    if support_arr.sum() == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weight = support_arr / support_arr.sum()
        weighted_precision = float(np.dot(weight, np.asarray(precision_values)))
        weighted_recall = float(np.dot(weight, np.asarray(recall_values)))
        weighted_f1 = float(np.dot(weight, np.asarray(f1_values)))

    return {
        "num_samples": int(y_true.shape[0]),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _compute_metrics_by_slot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slot_ids: np.ndarray,
    num_actions: int,
) -> dict[str, dict[str, Any]]:
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != slot_ids.shape[0]:
        raise RuntimeError(
            "Cannot compute slot metrics: length mismatch "
            f"(y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}, slot_ids={slot_ids.shape[0]})"
        )

    out: dict[str, dict[str, Any]] = {}
    for slot in (0, 1):
        mask = slot_ids == slot
        out[f"player_{slot}"] = compute_classification_metrics(
            y_true[mask],
            y_pred[mask],
            num_classes=num_actions,
        )
    return out


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    train_trials: list[FeaturizedTrial],
    val_trials: list[FeaturizedTrial],
    test_trials: list[FeaturizedTrial],
    *,
    model_type: str,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[dict[str, DataLoader], dict[str, Dataset]]:
    if model_type == "mlp":
        train_dataset = MLPTimestepDataset(train_trials)
        val_dataset = MLPTimestepDataset(val_trials)
        test_dataset = MLPTimestepDataset(test_trials)
    elif model_type == "lstm":
        train_dataset = LSTMWindowDataset(train_trials, seq_len=seq_len)
        val_dataset = LSTMWindowDataset(val_trials, seq_len=seq_len)
        test_dataset = LSTMWindowDataset(test_trials, seq_len=seq_len)
    else:
        raise ValueError(f"Unsupported model_type {model_type}")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after preprocessing")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty after preprocessing")

    dataloaders = {
        "train": _make_loader(train_dataset, batch_size, True, num_workers, pin_memory),
        "val": _make_loader(val_dataset, batch_size, False, num_workers, pin_memory),
        "test": _make_loader(test_dataset, batch_size, False, num_workers, pin_memory),
    }
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    return dataloaders, datasets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None,
) -> tuple[float, dict[str, Any]]:
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
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray]:
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
    return mean_loss, metrics, y_true_np, y_pred_np


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def create_run_dir(base_dir: str | Path, model_name: str, run_name: str | None) -> Path:
    base = ensure_dir(base_dir)
    final_name = run_name if run_name else f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return ensure_dir(base / final_name)


def save_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def resolve_export_paths(data_path: str) -> list[Path]:
    path = Path(data_path)
    if path.is_dir():
        paths = sorted(path.glob("*.jsonl"))
    elif path.is_file():
        paths = [path]
    else:
        raise FileNotFoundError(f"Could not find export data at {data_path}")

    if not paths:
        raise FileNotFoundError(f"No .jsonl files found at {data_path}")
    return paths


def load_raw_trials(
    export_paths: list[Path],
    *,
    data_source: str,
    player_mode: str,
    player_idx: int | None,
) -> tuple[list[RawTrial], dict[str, Any]]:
    selected_slots = [player_idx] if player_mode == "single" else [0, 1]
    raw_trials: list[RawTrial] = []
    total_frames = 0
    invalid_frames = 0
    empty_trials = 0
    source_skipped_frames = 0
    kept_source_counts = {"human": 0, "synthetic": 0}

    for export_path in export_paths:
        rows = []
        with export_path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                row = json.loads(stripped_line)
                if not isinstance(row, dict):
                    continue
                rows.append(row)
                total_frames += 1

        for slot in selected_slots:
            valid_rows = []
            for row in rows:
                players = row.get("players", [])
                input_states = row.get("input_states", [])
                if not isinstance(players, list) or not isinstance(input_states, list):
                    invalid_frames += 1
                    continue
                if len(players) <= slot or len(input_states) <= slot:
                    invalid_frames += 1
                    continue
                input_state = input_states[slot]
                is_human = bool(input_state.get("is_human", False))
                if data_source == "human" and not is_human:
                    source_skipped_frames += 1
                    continue
                if data_source == "synthetic" and is_human:
                    source_skipped_frames += 1
                    continue
                kept_source_counts["human" if is_human else "synthetic"] += 1
                valid_rows.append(row)

            if valid_rows:
                raw_trials.append(
                    RawTrial(
                        trial_id=f"{export_path.stem}:player_{slot}",
                        slot_id=slot,
                        rows=valid_rows,
                    )
                )
            else:
                empty_trials += 1

    report = {
        "num_files": len(export_paths),
        "file_paths": [str(path) for path in export_paths],
        "total_frames": total_frames,
        "invalid_frames": invalid_frames,
        "source_skipped_frames": source_skipped_frames,
        "raw_trial_count": len(raw_trials),
        "empty_trials": empty_trials,
        "selected_slots": selected_slots,
        "data_source": data_source,
        "kept_source_counts": kept_source_counts,
    }
    return raw_trials, report


def featurize_trials(
    raw_trials: list[RawTrial],
    schema: FeatureSchema,
) -> tuple[list[FeaturizedTrial], dict[str, Any]]:
    featurized_trials: list[FeaturizedTrial] = []
    skipped_rows = 0
    action_counts = {action_name: 0 for action_name in ACTION_TO_ID}

    for raw_trial in raw_trials:
        x_rows: list[np.ndarray] = []
        y_rows: list[int] = []

        for row in raw_trial.rows:
            input_state = row["input_states"][raw_trial.slot_id]
            try:
                features = featurize_state_for_player(row, raw_trial.slot_id, schema)
                action_id = input_state_to_action_id(input_state)
            except Exception:
                skipped_rows += 1
                continue

            x_rows.append(features)
            y_rows.append(action_id)
            action_counts[ID_TO_ACTION[action_id]] += 1

        if not x_rows:
            continue

        targets = np.asarray(y_rows, dtype=np.int64)
        featurized_trials.append(
            FeaturizedTrial(
                trial_id=raw_trial.trial_id,
                slot_id=raw_trial.slot_id,
                x=np.stack(x_rows, axis=0).astype(np.float32),
                y=targets,
                slot_ids=np.full(targets.shape[0], raw_trial.slot_id, dtype=np.int64),
            )
        )

    feature_dim = int(featurized_trials[0].x.shape[1]) if featurized_trials else 0
    report = {
        "featurized_trial_count": len(featurized_trials),
        "skipped_rows": skipped_rows,
        "feature_dim": feature_dim,
        "holder_count": len(schema.holder_specs),
        "action_counts": action_counts,
    }
    return featurized_trials, report


def chunk_featurized_trials(
    trials: list[FeaturizedTrial],
    *,
    chunk_size: int,
) -> tuple[list[FeaturizedTrial], dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunked_trials: list[FeaturizedTrial] = []
    num_chunks_created = 0
    num_source_trials_chunked = 0

    for trial in trials:
        num_steps = int(trial.x.shape[0])
        if num_steps <= chunk_size:
            chunked_trials.append(trial)
            continue

        num_source_trials_chunked += 1
        chunk_start = 0
        chunk_index = 0
        while chunk_start < num_steps:
            chunk_end = min(chunk_start + chunk_size, num_steps)
            chunked_trials.append(
                FeaturizedTrial(
                    trial_id=f"{trial.trial_id}:chunk_{chunk_index}",
                    slot_id=trial.slot_id,
                    x=trial.x[chunk_start:chunk_end].copy(),
                    y=trial.y[chunk_start:chunk_end].copy(),
                    slot_ids=trial.slot_ids[chunk_start:chunk_end].copy(),
                )
            )
            num_chunks_created += 1
            chunk_start = chunk_end
            chunk_index += 1

    report = {
        "trial_chunk_size": chunk_size,
        "source_trial_count": len(trials),
        "chunked_trial_count": len(chunked_trials),
        "source_trials_chunked": num_source_trials_chunked,
        "chunks_created": num_chunks_created,
    }
    return chunked_trials, report


def split_by_trial_id(
    trials: list[FeaturizedTrial],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[FeaturizedTrial], list[FeaturizedTrial], list[FeaturizedTrial]]:
    if len(trials) < 3:
        raise RuntimeError("Need at least 3 featurized trials to create train/val/test splits")

    shuffled_trials = list(trials)
    random.Random(seed).shuffle(shuffled_trials)

    train_count = max(1, int(round(len(shuffled_trials) * train_ratio)))
    val_count = max(1, int(round(len(shuffled_trials) * val_ratio)))
    if train_count + val_count >= len(shuffled_trials):
        val_count = max(1, len(shuffled_trials) - train_count - 1)
    train_count = max(1, min(train_count, len(shuffled_trials) - val_count - 1))

    train_trials = shuffled_trials[:train_count]
    val_trials = shuffled_trials[train_count : train_count + val_count]
    test_trials = shuffled_trials[train_count + val_count :]

    if not train_trials or not val_trials or not test_trials:
        raise RuntimeError(
            "One or more data splits are empty after split_by_trial_id. "
            "Adjust train_ratio/val_ratio or add more export files."
        )

    return train_trials, val_trials, test_trials


def _build_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: BCConfig,
    schema: FeatureSchema,
    input_dim: int,
    num_actions: int,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "model_type": config.model,
        "input_dim": input_dim,
        "num_actions": num_actions,
        "hidden_dims": config.mlp_hidden,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "seq_len": config.seq_len,
        "feature_schema": schema.to_dict(),
        "action_to_id": ACTION_TO_ID,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def run_training(config: BCConfig) -> Path:
    if config.data_source not in {"all", "human", "synthetic"}:
        raise ValueError(f"Unsupported data_source={config.data_source}")
    if config.player_mode not in {"both", "single"}:
        raise ValueError(f"Unsupported player_mode={config.player_mode}")
    if config.player_mode == "single" and config.player_idx not in (0, 1):
        raise ValueError(
            "player_idx must resolve to 0 or 1 for player_mode=single "
            f"(use --player-slot 1 or 2), got {config.player_idx}"
        )
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {config.train_ratio}")
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {config.val_ratio}")
    if config.train_ratio + config.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    config.device = _resolve_device(config.device)
    _set_seed(config.seed)
    run_dir = create_run_dir(config.outdir, config.model, config.run_name)

    export_paths = resolve_export_paths(config.data_path)
    raw_trials, clean_report = load_raw_trials(
        export_paths,
        data_source=config.data_source,
        player_mode=config.player_mode,
        player_idx=config.player_idx,
    )
    if not raw_trials:
        raise RuntimeError("No valid trials found in the export data")

    schema_states = [row for trial in raw_trials for row in trial.rows]
    feature_schema = build_feature_schema(schema_states)
    featurized_trials, feat_report = featurize_trials(raw_trials, feature_schema)
    if not featurized_trials:
        raise RuntimeError("No valid featurized trials found")
    featurized_trials, chunk_report = chunk_featurized_trials(
        featurized_trials,
        chunk_size=max(config.trial_chunk_size, config.seq_len),
    )
    if len(featurized_trials) < 3:
        raise RuntimeError(
            "Need at least 3 featurized trial chunks to create train/val/test splits. "
            f"Current chunked trial count: {len(featurized_trials)}. "
            "Increase data volume or reduce --trial-chunk-size."
        )

    train_trials, val_trials, test_trials = split_by_trial_id(
        featurized_trials,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
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
    model = build_policy_model(
        model_type=config.model,
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dims=config.mlp_hidden,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    device = torch.device(config.device)
    model.to(device)

    class_weights = _compute_class_weights(datasets["train"].targets_np, num_actions).to(device)
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
        val_loss, val_metrics, _, _ = evaluate(
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
                _build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    schema=feature_schema,
                    input_dim=input_dim,
                    num_actions=num_actions,
                    metrics=val_metrics,
                ),
            )
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    save_checkpoint(
        last_path,
        _build_checkpoint_payload(
            epoch=history[-1]["epoch"] if history else 0,
            model=model,
            optimizer=optimizer,
            config=config,
            schema=feature_schema,
            input_dim=input_dim,
            num_actions=num_actions,
        ),
    )

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    best_val_loss, best_val_metrics, best_val_y_true, best_val_y_pred = evaluate(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device,
    )
    test_loss, test_metrics, test_y_true, test_y_pred = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )
    best_val_metrics_by_slot = _compute_metrics_by_slot(
        y_true=best_val_y_true,
        y_pred=best_val_y_pred,
        slot_ids=np.asarray(datasets["val"].slot_ids_np, dtype=np.int64),
        num_actions=num_actions,
    )
    test_metrics_by_slot = _compute_metrics_by_slot(
        y_true=test_y_true,
        y_pred=test_y_pred,
        slot_ids=np.asarray(datasets["test"].slot_ids_np, dtype=np.int64),
        num_actions=num_actions,
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
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "best_val_metrics_by_slot": best_val_metrics_by_slot,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_metrics_by_slot": test_metrics_by_slot,
        "history": history,
    }
    report_payload = {
        "data_loading_report": clean_report,
        "featurization_report": feat_report,
        "chunking_report": chunk_report,
    }
    config_payload = asdict(config)

    save_json(run_dir / "metrics.json", metrics_payload)
    save_json(run_dir / "split_summary.json", split_summary)
    save_json(run_dir / "preprocessing_report.json", report_payload)
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "label_mapping.json", ACTION_TO_ID)
    save_json(run_dir / "feature_schema.json", feature_schema.to_dict())

    deploy_dir = ensure_dir(DEFAULT_DEPLOY_DIR)
    save_checkpoint(deploy_dir / "best.pt", best_checkpoint)
    save_json(deploy_dir / "feature_schema.json", feature_schema.to_dict())
    save_json(deploy_dir / "label_mapping.json", ACTION_TO_ID)

    print(f"Training complete. Best epoch: {best_epoch}, val_macro_f1={best_val_macro_f1:.4f}")
    print(
        f"Best-checkpoint val macro_f1={best_val_metrics['macro_f1']:.4f}, "
        f"accuracy={best_val_metrics['accuracy']:.4f}"
    )
    print(f"Test macro_f1={test_metrics['macro_f1']:.4f}, accuracy={test_metrics['accuracy']:.4f}")
    print(f"Artifacts saved to: {run_dir}")
    return run_dir


def _parse_hidden_dims(raw_value: str) -> list[int]:
    hidden_dims = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not hidden_dims:
        raise ValueError("mlp_hidden must contain at least one integer")
    return hidden_dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Ravioli imitation policy from JSONL exports.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Directory or .jsonl export file path.")
    parser.add_argument("--model", choices=("mlp", "lstm"), default="mlp", help="Policy architecture.")
    parser.add_argument("--data-source", choices=("all", "human", "synthetic"), default="all", help="Which exported demonstrations to train from.")
    parser.add_argument("--player-mode", choices=("both", "single"), default="both", help="Train on both players or one slot.")
    parser.add_argument("--player-idx", type=int, default=None, help="Zero-based player index for single-player training: 0 or 1.")
    parser.add_argument("--player-slot", type=int, choices=(1, 2), default=None, help="Human-facing player slot for single-player training: 1 or 2.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of trials for training.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of trials for validation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for training artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=1, help="LSTM layer count.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for both models.")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for LSTM training.")
    parser.add_argument("--trial-chunk-size", type=int, default=4096, help="Maximum contiguous frames per pseudo-trial before train/val/test splitting.")
    parser.add_argument("--mlp-hidden", default="256,128", help="Comma-separated hidden sizes for the MLP.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip value for LSTM.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop after this many non-improving epochs.")
    return parser


def config_from_args(args: argparse.Namespace) -> BCConfig:
    if args.player_idx is not None and args.player_slot is not None:
        raise ValueError("Use only one of --player-idx or --player-slot.")

    resolved_player_idx = args.player_idx
    if args.player_slot is not None:
        resolved_player_idx = args.player_slot - 1

    return BCConfig(
        data_path=args.data_path,
        model=args.model,
        data_source=args.data_source,
        player_mode=args.player_mode,
        player_idx=resolved_player_idx,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        outdir=args.outdir,
        run_name=args.run_name,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
        trial_chunk_size=args.trial_chunk_size,
        mlp_hidden=_parse_hidden_dims(args.mlp_hidden),
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    run_training(config)


if __name__ == "__main__":
    main()
