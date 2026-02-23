import ast
import csv
import json
import os
import random
from hashlib import md5
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import overcooked_ai_py.data.planners as data_planners
import overcooked_ai_py.planning.planners as planning_planners
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from imitation.constants import ACTION_TO_ID, normalize_action_token


@dataclass
class StepRecord:
    timestep: int
    state_dict: dict[str, Any]
    action_id: int


@dataclass
class TrialRecord:
    trial_id: str
    layout_name: str
    layout_grid: list[str] | None
    steps: list[StepRecord]


@dataclass
class FeaturizedTrial:
    trial_id: str
    layout_name: str
    x: np.ndarray
    y: np.ndarray


def load_csv_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def parse_joint_action(raw_joint_action: str) -> tuple[str, str]:
    parsed = ast.literal_eval(raw_joint_action)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"Invalid joint_action: {raw_joint_action}")
    action_0 = normalize_action_token(parsed[0])
    action_1 = normalize_action_token(parsed[1])
    return action_0, action_1


def extract_player_action(raw_joint_action: str, player_idx: int) -> str:
    action_0, action_1 = parse_joint_action(raw_joint_action)
    if player_idx == 0:
        return action_0
    if player_idx == 1:
        return action_1
    raise ValueError(f"player_idx must be 0 or 1, got {player_idx}")


def build_trial_records(rows: list[dict[str, str]], player_idx: int) -> tuple[list[TrialRecord], dict[str, int]]:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    report = {
        "raw_rows": len(rows),
        "raw_trials": 0,
        "missing_trial_id": 0,
        "missing_layout": 0,
        "layout_mismatch_rows": 0,
        "bad_layout_rows": 0,
        "bad_action_rows": 0,
        "bad_state_rows": 0,
        "empty_trials_after_cleaning": 0,
        "kept_rows": 0,
        "kept_trials": 0,
    }

    for row in rows:
        trial_id = row.get("trial_id")
        if not trial_id:
            report["missing_trial_id"] += 1
            continue
        grouped_rows[trial_id].append(row)

    report["raw_trials"] = len(grouped_rows)
    trials: list[TrialRecord] = []

    for trial_id, trial_rows in grouped_rows.items():
        def _row_sort_key(row: dict[str, str]) -> float:
            try:
                return float(row.get("cur_gameloop", "0") or 0)
            except (TypeError, ValueError):
                return 0.0

        sorted_rows = sorted(trial_rows, key=_row_sort_key)

        layout_name = None
        layout_grid = None
        steps: list[StepRecord] = []

        for row in sorted_rows:
            row_layout = row.get("layout_name")
            if not row_layout:
                report["missing_layout"] += 1
                continue

            if layout_name is None:
                layout_name = row_layout
            elif row_layout != layout_name:
                report["layout_mismatch_rows"] += 1
                continue

            if layout_grid is None:
                parsed_layout = _parse_layout_grid(row.get("layout"))
                if parsed_layout is not None:
                    layout_grid = parsed_layout
                elif row.get("layout") is not None:
                    report["bad_layout_rows"] += 1

            try:
                action_name = extract_player_action(row["joint_action"], player_idx=player_idx)
                action_id = ACTION_TO_ID[action_name]
            except (KeyError, ValueError, SyntaxError):
                report["bad_action_rows"] += 1
                continue

            try:
                state_dict = json.loads(row["state"])
            except (KeyError, TypeError, json.JSONDecodeError):
                report["bad_state_rows"] += 1
                continue

            timestep_raw = row.get("cur_gameloop")
            try:
                timestep = int(float(timestep_raw))
            except (TypeError, ValueError):
                timestep = len(steps)

            steps.append(StepRecord(timestep=timestep, state_dict=state_dict, action_id=action_id))

        if not steps or layout_name is None:
            report["empty_trials_after_cleaning"] += 1
            continue

        steps.sort(key=lambda step: step.timestep)
        trials.append(
            TrialRecord(
                trial_id=trial_id,
                layout_name=layout_name,
                layout_grid=layout_grid,
                steps=steps,
            )
        )

    report["kept_trials"] = len(trials)
    report["kept_rows"] = sum(len(trial.steps) for trial in trials)
    return trials, report


def _configure_planner_cache(planner_cache_dir: str) -> None:
    planner_cache_dir = os.path.abspath(planner_cache_dir)
    os.makedirs(planner_cache_dir, exist_ok=True)
    data_planners.PLANNERS_DIR = planner_cache_dir
    planning_planners.PLANNERS_DIR = planner_cache_dir


def _clear_planner_cache_for_layout(planner_cache_dir: str, layout_prefix: str) -> None:
    if not os.path.isdir(planner_cache_dir):
        return
    for filename in os.listdir(planner_cache_dir):
        if filename.startswith(layout_prefix) and filename.endswith(".pkl"):
            try:
                os.remove(os.path.join(planner_cache_dir, filename))
            except OSError:
                pass


def _parse_layout_grid(raw_layout: str | None) -> list[str] | None:
    if raw_layout is None:
        return None
    try:
        parsed = ast.literal_eval(raw_layout)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed, list) and parsed:
        if all(isinstance(row, str) for row in parsed):
            return [str(row) for row in parsed]
        if all(isinstance(row, list) for row in parsed):
            out = []
            for row in parsed:
                if not all(isinstance(cell, str) for cell in row):
                    return None
                out.append("".join(row))
            return out
    return None


def _layout_signature(layout_name: str, layout_grid: list[str] | None) -> str:
    if not layout_grid:
        return layout_name
    return f"{layout_name}::{ '|'.join(layout_grid) }"


def featurize_trials_with_overcooked(
    trials: list[TrialRecord],
    player_idx: int,
    planner_cache_dir: str,
) -> tuple[list[FeaturizedTrial], dict[str, int]]:
    planner_cache_dir = os.path.abspath(planner_cache_dir)
    _configure_planner_cache(planner_cache_dir)

    layout_to_artifacts: dict[str, tuple[OvercookedGridworld, MediumLevelActionManager]] = {}
    report = {
        "input_trials": len(trials),
        "input_rows": sum(len(trial.steps) for trial in trials),
        "featurization_errors": 0,
        "empty_trials_after_featurization": 0,
        "kept_trials": 0,
        "kept_rows": 0,
    }

    featurized_trials: list[FeaturizedTrial] = []
    expected_feature_dim = None

    for trial in trials:
        signature = _layout_signature(trial.layout_name, trial.layout_grid)
        if signature not in layout_to_artifacts:
            signature_hash = md5(signature.encode("utf-8")).hexdigest()[:10]
            planner_filename = f"{trial.layout_name}_{signature_hash}_am.pkl"
            try:
                mdp = OvercookedGridworld.from_layout_name(trial.layout_name)
            except Exception:
                if not trial.layout_grid:
                    raise RuntimeError(
                        f"Failed to initialize layout '{trial.layout_name}' and no valid layout "
                        f"grid was found in CSV for trial_id={trial.trial_id}."
                    )
                mdp = OvercookedGridworld.from_grid(
                    trial.layout_grid,
                    base_layout_params={"layout_name": f"{trial.layout_name}_{signature_hash}"},
                )
            try:
                mlam = MediumLevelActionManager.from_pickle_or_compute(
                    mdp,
                    NO_COUNTERS_PARAMS,
                    custom_filename=planner_filename,
                    force_compute=False,
                )
            except Exception as exc:
                _clear_planner_cache_for_layout(planner_cache_dir, trial.layout_name)
                try:
                    mlam = MediumLevelActionManager.from_pickle_or_compute(
                        mdp,
                        NO_COUNTERS_PARAMS,
                        custom_filename=planner_filename,
                        force_compute=True,
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Failed to initialize featurizer for layout '{trial.layout_name}' "
                        f"(trial_id={trial.trial_id})."
                    ) from retry_exc
            layout_to_artifacts[signature] = (mdp, mlam)

        mdp, mlam = layout_to_artifacts[signature]
        x_rows = []
        y_rows = []

        for step in trial.steps:
            try:
                state = OvercookedState.from_dict(step.state_dict)
                feat_by_player = mdp.featurize_state(state, mlam)
                feat = np.asarray(feat_by_player[player_idx], dtype=np.float32)
            except Exception:
                report["featurization_errors"] += 1
                continue

            if expected_feature_dim is None:
                expected_feature_dim = int(feat.shape[0])
            elif int(feat.shape[0]) != expected_feature_dim:
                raise RuntimeError(
                    f"Inconsistent feature size for trial {trial.trial_id}: "
                    f"expected {expected_feature_dim}, found {feat.shape[0]}"
                )

            x_rows.append(feat)
            y_rows.append(step.action_id)

        if not x_rows:
            report["empty_trials_after_featurization"] += 1
            continue

        x = np.stack(x_rows, axis=0).astype(np.float32)
        y = np.asarray(y_rows, dtype=np.int64)
        featurized_trials.append(
            FeaturizedTrial(
                trial_id=trial.trial_id,
                layout_name=trial.layout_name,
                x=x,
                y=y,
            )
        )

    report["kept_trials"] = len(featurized_trials)
    report["kept_rows"] = int(sum(trial.x.shape[0] for trial in featurized_trials))
    return featurized_trials, report


def split_by_trial_id(
    trials: list[FeaturizedTrial],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[FeaturizedTrial], list[FeaturizedTrial], list[FeaturizedTrial]]:
    if not trials:
        return [], [], []
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    trials_by_id = {trial.trial_id: trial for trial in trials}
    trial_ids = list(trials_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(trial_ids)

    n_total = len(trial_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    if n_train <= 0:
        n_train = 1
    if n_train >= n_total:
        n_train = n_total - 1
    if n_val < 0:
        n_val = 0
    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)

    train_ids = set(trial_ids[:n_train])
    val_ids = set(trial_ids[n_train : n_train + n_val])
    test_ids = set(trial_ids[n_train + n_val :])

    train_trials = [trials_by_id[trial_id] for trial_id in trial_ids if trial_id in train_ids]
    val_trials = [trials_by_id[trial_id] for trial_id in trial_ids if trial_id in val_ids]
    test_trials = [trials_by_id[trial_id] for trial_id in trial_ids if trial_id in test_ids]

    return train_trials, val_trials, test_trials
