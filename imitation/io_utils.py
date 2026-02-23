import json
from datetime import datetime
from pathlib import Path
from typing import Any
import torch


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base_dir: str | Path, model_name: str, run_name: str | None) -> Path:
    base = ensure_dir(base_dir)
    if run_name:
        final_name = run_name
    else:
        final_name = f"{model_name}_{timestamp_string()}"
    run_dir = ensure_dir(base / final_name)
    return run_dir


def save_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)

