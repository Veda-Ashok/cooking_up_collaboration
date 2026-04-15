import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "results" / "config.json"


def load_results_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    config["_config_path"] = str(path)
    return config


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def layout_keys(config: dict[str, Any]) -> list[str]:
    return [layout["key"] for layout in config.get("layouts", [])]


def layout_labels(config: dict[str, Any]) -> dict[str, str]:
    return {layout["key"]: layout.get("label", layout["key"]) for layout in config.get("layouts", [])}


def agent_dir(config: dict[str, Any], layout: str, agent_key: str) -> Path:
    try:
        path = config["agents"][layout][agent_key]["agent_dir"]
    except KeyError as exc:
        raise KeyError(f"Missing agent_dir for layout={layout}, agent={agent_key}") from exc
    return resolve_repo_path(path)
