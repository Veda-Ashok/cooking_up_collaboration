from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from imitation.preprocessing import FeaturizedTrial


class MLPTimestepDataset(Dataset):
    def __init__(self, trials: list[FeaturizedTrial]):
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
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMWindowDataset(Dataset):
    def __init__(self, trials: list[FeaturizedTrial], seq_len: int):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.trials = trials
        self.seq_len = seq_len
        self.index: list[tuple[int, int]] = []
        targets = []
        slot_ids = []

        for trial_idx, trial in enumerate(trials):
            steps = trial.x.shape[0]
            if steps < seq_len:
                continue
            for start in range(0, steps - seq_len + 1):
                self.index.append((trial_idx, start))
                targets.append(int(trial.y[start + seq_len - 1]))
                slot_ids.append(int(trial.slot_ids[start + seq_len - 1]))

        self.targets_np = np.asarray(targets, dtype=np.int64)
        self.slot_ids_np = np.asarray(slot_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trial_idx, start = self.index[idx]
        trial = self.trials[trial_idx]
        x_seq = trial.x[start : start + self.seq_len]
        y_target = trial.y[start + self.seq_len - 1]
        return torch.from_numpy(x_seq).float(), torch.tensor(y_target, dtype=torch.long)


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
    model_type: str,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
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
        "train": _make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        "val": _make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": _make_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    return dataloaders, datasets
