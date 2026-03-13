import json
import logging
import os
import types
from typing import Any

import numpy as np
import overcooked_ai_py.data.planners as data_planners
import overcooked_ai_py.planning.planners as planning_planners
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

try:
    from rl.sampling_utils import (
        configure_model_sampling_temperature,
        normalize_sampling_mode,
        validate_sampling_temperature,
    )
except Exception:
    def normalize_sampling_mode(mode, default="sample"):
        if mode is None:
            mode = default
        value = str(mode).strip().lower()
        if value not in {"sample", "argmax"}:
            raise ValueError(f"Unsupported sampling_mode={mode}")
        return value

    def validate_sampling_temperature(value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"sampling_temperature must be > 0, got {value}")
        return value

    def configure_model_sampling_temperature(model, sampling_temperature):
        sampling_temperature = validate_sampling_temperature(sampling_temperature)
        policy = getattr(model, "policy", None)
        if policy is None:
            raise ValueError("Model does not expose a `policy` attribute.")

        original_method = getattr(policy, "_sampling_original_get_action_dist_from_latent", None)
        if original_method is None:
            if not hasattr(policy, "_get_action_dist_from_latent"):
                raise ValueError("Unsupported policy type for temperature sampling.")
            original_method = policy._get_action_dist_from_latent
            policy._sampling_original_get_action_dist_from_latent = original_method

        if sampling_temperature == 1.0:
            policy._get_action_dist_from_latent = original_method
            policy._sampling_patch_applied = False
            policy._sampling_temperature = 1.0
            return {"patched": False, "reason": "identity_temperature", "sampling_temperature": 1.0}

        def _patched_get_action_dist_from_latent(self, *args, **kwargs):
            if not args:
                return original_method(*args, **kwargs)
            latent_pi = args[0]
            if not hasattr(self, "action_net") or not hasattr(self, "action_dist"):
                return original_method(*args, **kwargs)
            if not hasattr(self.action_dist, "proba_distribution"):
                return original_method(*args, **kwargs)
            action_logits = self.action_net(latent_pi) / float(self._sampling_temperature)
            return self.action_dist.proba_distribution(action_logits=action_logits)

        policy._get_action_dist_from_latent = types.MethodType(_patched_get_action_dist_from_latent, policy)
        policy._sampling_patch_applied = True
        policy._sampling_temperature = float(sampling_temperature)
        return {"patched": True, "reason": "patched", "sampling_temperature": float(sampling_temperature)}


LOGGER = logging.getLogger(__name__)


class TorchRLAgent:
    IDX_TO_OVERCOOKED_ACTION = {
        0: Direction.NORTH,   # UP
        1: Direction.SOUTH,   # DOWN
        2: Direction.EAST,    # RIGHT
        3: Direction.WEST,    # LEFT
        4: Action.STAY,       # STAY
        5: Action.INTERACT,   # INTERACT
    }

    def __init__(self, agent_dir: str, agent_index: int, device: str = "cpu"):
        self.agent_dir = agent_dir
        self.agent_index = int(agent_index)
        self.manifest = self._load_manifest()

        manifest_player_idx = self.manifest.get("player_idx", None)
        self.player_idx = self.agent_index if manifest_player_idx is None else int(manifest_player_idx)
        self.supported_layouts = set(self.manifest.get("supported_layouts", []))
        self._warned_layouts = set()

        legacy_deterministic = bool(self.manifest.get("deterministic", True))
        default_sampling_mode = "argmax" if legacy_deterministic else "sample"
        self.sampling_mode = normalize_sampling_mode(
            self.manifest.get("sampling_mode"),
            default=default_sampling_mode,
        )
        self.sampling_temperature = validate_sampling_temperature(
            self.manifest.get("sampling_temperature", 1.0)
        )
        self.deterministic = self.sampling_mode == "argmax"

        self.planner_cache_dir = os.path.abspath(
            os.path.join(self.agent_dir, self.manifest.get("planner_cache_dir", ".cache/overcooked_planners"))
        )
        self._set_planner_cache_dir(self.planner_cache_dir)

        requested_device = self.manifest.get("device", device)
        if requested_device == "auto":
            requested_device = "cuda" if self._is_cuda_available() else "cpu"
        if requested_device == "cuda" and not self._is_cuda_available():
            LOGGER.warning("CUDA requested for %s but unavailable; using CPU.", self.agent_dir)
            requested_device = "cpu"
        self.device = requested_device

        self.algo = str(self.manifest.get("algo", "recurrent_ppo")).lower()
        checkpoint = self.manifest.get("checkpoint", "best_model.zip")
        checkpoint_path = os.path.join(self.agent_dir, checkpoint)
        self._is_recurrent = self.algo in {"recurrent_ppo", "ppo_lstm", "lstm_ppo"}
        if self._is_recurrent:
            if RecurrentPPO is None:
                raise ImportError(
                    "sb3-contrib is required to load recurrent PPO agents in webapp runtime."
                )
            self.model = RecurrentPPO.load(checkpoint_path, device=self.device)
        elif self.algo == "ppo":
            self.model = PPO.load(checkpoint_path, device=self.device)
        else:
            raise ValueError(f"Unsupported rl_torch algo: {self.algo}")
        self._sampling_report = configure_model_sampling_temperature(self.model, self.sampling_temperature)

        self._mdp = None
        self._mlam = None
        self._layout_name = None
        self._lstm_state = None
        self._episode_start = np.array([True], dtype=bool)

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = os.path.join(self.agent_dir, "agent_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing RL manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("type") != "rl_torch":
            raise ValueError(f"Unsupported manifest type for RL torch agent: {manifest.get('type')}")
        return manifest

    def _set_planner_cache_dir(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        data_planners.PLANNERS_DIR = cache_dir
        planning_planners.PLANNERS_DIR = cache_dir

    def _clear_layout_planners(self, layout_name: str) -> None:
        if not os.path.isdir(self.planner_cache_dir):
            return
        for filename in os.listdir(self.planner_cache_dir):
            if filename.startswith(layout_name) and filename.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.planner_cache_dir, filename))
                except OSError:
                    pass

    def set_mdp_context(self, mdp, layout_name: str | None = None) -> None:
        self._mdp = mdp
        self._layout_name = layout_name or getattr(mdp, "layout_name", None)
        self._mlam = None
        if mdp is None:
            return

        custom_name = None
        if self._layout_name:
            custom_name = f"{self._layout_name}_am.pkl"

        try:
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                mdp,
                NO_COUNTERS_PARAMS,
                custom_filename=custom_name,
                force_compute=False,
            )
        except Exception:
            if self._layout_name:
                self._clear_layout_planners(self._layout_name)
            try:
                self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                    mdp,
                    NO_COUNTERS_PARAMS,
                    custom_filename=custom_name,
                    force_compute=True,
                )
            except Exception:
                self._mlam = None
                LOGGER.exception("Failed to initialize planner for rl_torch agent (%s).", self.agent_dir)

    def reset(self) -> None:
        self._lstm_state = None
        self._episode_start = np.array([True], dtype=bool)

    def _featurize_state(self, state) -> np.ndarray | None:
        if self._mdp is None or self._mlam is None:
            return None
        feature_by_player = self._mdp.featurize_state(state, self._mlam)
        feature = np.asarray(feature_by_player[self.player_idx], dtype=np.float32)
        return feature

    def action(self, state):
        layout = self._layout_name
        if self.supported_layouts and layout not in self.supported_layouts:
            if layout not in self._warned_layouts:
                LOGGER.warning(
                    "rl_torch agent %s got unsupported layout=%s. Returning STAY.",
                    os.path.basename(self.agent_dir),
                    layout,
                )
                self._warned_layouts.add(layout)
            return Action.STAY, None

        try:
            feature = self._featurize_state(state)
            if feature is None:
                return Action.STAY, None

            if self._is_recurrent:
                action_idx, self._lstm_state = self.model.predict(
                    feature,
                    state=self._lstm_state,
                    episode_start=self._episode_start,
                    deterministic=(self.sampling_mode == "argmax"),
                )
                self._episode_start = np.array([False], dtype=bool)
            else:
                action_idx, _ = self.model.predict(
                    feature,
                    deterministic=(self.sampling_mode == "argmax"),
                )
            idx = int(np.asarray(action_idx).item())
            action = self.IDX_TO_OVERCOOKED_ACTION.get(idx, Action.STAY)
            return action, None
        except Exception:
            LOGGER.exception("rl_torch inference failed for agent %s", os.path.basename(self.agent_dir))
            return Action.STAY, None
