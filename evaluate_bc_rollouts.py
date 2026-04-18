"""Evaluate a trained BC checkpoint via environment rollouts.

Runs the BC policy as both players in self-play, collecting behavioral
statistics that classification metrics alone cannot reveal.

Pass --render to watch live, --save-video to record MP4s.
"""
import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from imitation.models import LSTMPolicy, MLPPolicy
from imitation.preprocessing import _configure_planner_cache


IDX_TO_ACTION = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (1, 0),    # RIGHT
    3: (-1, 0),   # LEFT
    4: (0, 0),    # STAY
    5: "interact",
}

DIRECTIONAL_ACTION_INDICES = [0, 1, 2, 3, 5]


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _idx_to_overcooked_action(idx: int):
    raw = IDX_TO_ACTION.get(idx, (0, 0))
    if raw == "interact":
        return Action.INTERACT
    return raw


def _load_bc_model(
    checkpoint_path: str,
    config_path: str | None,
    device: str,
) -> tuple[nn.Module, dict]:
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        parent = Path(checkpoint_path).parent
        candidate = parent / "config.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(
                f"No config.json found alongside {checkpoint_path}. "
                "Provide --config-path explicitly."
            )

    model_type = config.get("model", "mlp")
    input_dim = config["mlp_hidden"]  # will infer from checkpoint
    num_actions = 6

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    first_key = next(iter(state_dict))
    first_weight = state_dict[first_key]
    input_dim = first_weight.shape[1] if first_weight.ndim == 2 else first_weight.shape[0]

    if model_type == "mlp":
        hidden_dims = config.get("mlp_hidden", [64, 64])
        dropout = config.get("dropout", 0.0)
        model = MLPPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif model_type == "lstm":
        model = LSTMPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


class BCRolloutAgent:
    """Wraps a trained BC model for rollout inference."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        player_idx: int,
        seq_len: int,
        device: str,
        sampling_mode: str = "sample",
        temperature: float = 1.0,
        deadlock_break_after: int = 3,
    ):
        self.model = model
        self.model_type = model_type
        self.player_idx = player_idx
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.sampling_mode = sampling_mode
        self.temperature = temperature
        self.deadlock_break_after = deadlock_break_after
        self._history: deque[np.ndarray] = deque(maxlen=seq_len)
        self._last_feature: np.ndarray | None = None
        self._stuck_count = 0

    def reset(self):
        self._history.clear()
        self._last_feature = None
        self._stuck_count = 0

    @torch.no_grad()
    def act(self, feature: np.ndarray) -> int:
        if self.model_type == "mlp":
            x = torch.from_numpy(feature).unsqueeze(0).to(self.device)
            logits = self.model(x)
        else:
            self._history.append(feature)
            if len(self._history) == 1:
                while len(self._history) < self.seq_len:
                    self._history.appendleft(feature.copy())
            x_seq = np.stack(list(self._history), axis=0).astype(np.float32)
            x = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)
            logits = self.model(x)

        if self.sampling_mode == "argmax":
            action_idx = int(torch.argmax(logits, dim=1).item())
        else:
            scaled = logits / self.temperature
            probs = torch.softmax(scaled, dim=1)
            action_idx = int(torch.distributions.Categorical(probs=probs).sample().item())

        action_idx = self._maybe_break_deadlock(action_idx, feature)
        self._last_feature = feature.copy()
        return action_idx

    def _maybe_break_deadlock(self, action_idx: int, feature: np.ndarray) -> int:
        if self.deadlock_break_after <= 0:
            return action_idx
        if action_idx != 4:
            self._stuck_count = 0
            return action_idx
        if self._last_feature is not None and np.array_equal(feature, self._last_feature):
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        if self._stuck_count >= self.deadlock_break_after:
            self._stuck_count = 0
            return random.choice(DIRECTIONAL_ACTION_INDICES)
        return action_idx


def _state_has_cooking(state: OvercookedState) -> bool:
    """Check if any pot has items cooking."""
    if not hasattr(state, "objects") or state.objects is None:
        return False
    for obj in state.objects.values():
        if hasattr(obj, "name") and obj.name == "soup":
            if hasattr(obj, "is_cooking") and obj.is_cooking:
                return True
            if hasattr(obj, "_cooking_tick") and obj._cooking_tick > 0:
                return True
    return False


class _Visualizer:
    """Lazy-loaded rendering context (only imported when --render or --save-video is used)."""

    def __init__(self, fps: int, save_video_dir: str | None, render_live: bool):
        import cv2 as cv
        import pygame.surfarray
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        self._cv = cv
        self._pygame_surfarray = pygame.surfarray
        self._vis = StateVisualizer()
        self.fps = max(1, fps)
        self.render_live = render_live
        self.save_video_dir = save_video_dir
        self._writer = None
        self._current_ep = -1

    def _start_episode(self, ep: int) -> None:
        self._close_writer()
        self._current_ep = ep
        if self.save_video_dir:
            Path(self.save_video_dir).mkdir(parents=True, exist_ok=True)

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def render_step(
        self,
        mdp: OvercookedGridworld,
        env: OvercookedEnv,
        step: int,
        ep: int,
        score: float,
        reward: float,
    ) -> bool:
        """Render one frame. Returns False if the user pressed 'q' to quit."""
        cv = self._cv
        surface = self._vis.render_state(env.state, grid=mdp.terrain_mtx)
        frame_rgb = self._pygame_surfarray.array3d(surface)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        overlay = f"ep={ep} step={step} score={score:.0f} r={reward:.0f}"
        cv.putText(frame_bgr, overlay, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2, cv.LINE_AA)

        if self.save_video_dir:
            if self._writer is None or self._current_ep != ep:
                self._close_writer()
                self._current_ep = ep
                h, w = frame_bgr.shape[:2]
                out_path = str(Path(self.save_video_dir) / f"episode_{ep:03d}.mp4")
                self._writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._writer.write(frame_bgr)

        if self.render_live:
            cv.imshow("BC Self-Play Evaluation", frame_bgr)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
        return True

    def close(self) -> None:
        self._close_writer()
        if self.render_live:
            self._cv.destroyAllWindows()


def evaluate_policy_in_env(
    model: nn.Module,
    config: dict,
    layout_name: str,
    n_episodes: int = 20,
    horizon: int = 400,
    sampling_mode: str = "sample",
    temperature: float = 1.0,
    deadlock_break_after: int = 3,
    planner_cache_dir: str = ".cache/overcooked_planners",
    device: str = "cpu",
    seq_len: int = 20,
    visualizer: _Visualizer | None = None,
) -> dict:
    _configure_planner_cache(planner_cache_dir)
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=False,
    )
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    model_type = config.get("model", "mlp")

    agent_p0 = BCRolloutAgent(
        model, model_type, player_idx=0, seq_len=seq_len,
        device=device, sampling_mode=sampling_mode,
        temperature=temperature, deadlock_break_after=deadlock_break_after,
    )
    agent_p1 = BCRolloutAgent(
        model, model_type, player_idx=1, seq_len=seq_len,
        device=device, sampling_mode=sampling_mode,
        temperature=temperature, deadlock_break_after=deadlock_break_after,
    )

    episode_rewards = []
    episode_deliveries = []
    cook_started_count = 0
    delivery_count = 0
    stuck_episodes = 0

    quit_early = False
    for ep in range(n_episodes):
        if quit_early:
            break
        env.reset()
        agent_p0.reset()
        agent_p1.reset()

        if visualizer is not None:
            visualizer._start_episode(ep)

        ep_reward = 0.0
        ep_deliveries = 0
        ep_cook_started = False
        position_history: deque[tuple] = deque(maxlen=10)
        ep_stuck_steps = 0
        frame_budget = 1.0 / visualizer.fps if visualizer else 0.0

        for t in range(horizon):
            tick_start = time.time()
            state = env.state
            feats = mdp.featurize_state(state, mlam)
            feat_p0 = np.asarray(feats[0], dtype=np.float32)
            feat_p1 = np.asarray(feats[1], dtype=np.float32)

            a0_idx = agent_p0.act(feat_p0)
            a1_idx = agent_p1.act(feat_p1)

            a0 = _idx_to_overcooked_action(a0_idx)
            a1 = _idx_to_overcooked_action(a1_idx)

            next_state, sparse_reward, done, info = env.step((a0, a1))
            ep_reward += float(sparse_reward)

            if float(sparse_reward) > 0:
                ep_deliveries += int(float(sparse_reward) / 20) if float(sparse_reward) >= 20 else 1

            if not ep_cook_started and _state_has_cooking(next_state):
                ep_cook_started = True

            p0_pos = tuple(next_state.player_positions[0]) if hasattr(next_state, "player_positions") else None
            p1_pos = tuple(next_state.player_positions[1]) if hasattr(next_state, "player_positions") else None
            current_positions = (p0_pos, p1_pos)
            if len(position_history) >= 5 and all(p == current_positions for p in position_history):
                ep_stuck_steps += 1
            position_history.append(current_positions)

            if visualizer is not None:
                keep_going = visualizer.render_step(
                    mdp=mdp, env=env, step=t, ep=ep,
                    score=ep_reward, reward=float(sparse_reward),
                )
                if not keep_going:
                    quit_early = True
                    break
                elapsed = time.time() - tick_start
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_deliveries.append(ep_deliveries)
        if ep_cook_started:
            cook_started_count += 1
        if ep_deliveries > 0:
            delivery_count += 1
        if ep_stuck_steps > horizon * 0.3:
            stuck_episodes += 1

    if visualizer is not None:
        visualizer.close()

    completed = len(episode_rewards)
    stats = {
        "layout_name": layout_name,
        "n_episodes": completed,
        "n_episodes_requested": n_episodes,
        "horizon": horizon,
        "sampling_mode": sampling_mode,
        "mean_reward": float(np.mean(episode_rewards)) if completed else 0.0,
        "std_reward": float(np.std(episode_rewards)) if completed else 0.0,
        "min_reward": float(np.min(episode_rewards)) if completed else 0.0,
        "max_reward": float(np.max(episode_rewards)) if completed else 0.0,
        "mean_deliveries": float(np.mean(episode_deliveries)) if completed else 0.0,
        "cook_started_rate": cook_started_count / max(completed, 1),
        "delivery_rate": delivery_count / max(completed, 1),
        "stuck_rate": stuck_episodes / max(completed, 1),
        "per_episode_rewards": [float(r) for r in episode_rewards],
        "per_episode_deliveries": episode_deliveries,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy via environment rollouts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or epoch_XXX.pt")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config.json (auto-detected if next to checkpoint)")
    parser.add_argument("--layout-name", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--sampling-mode", type=str, choices=["sample", "argmax"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--deadlock-break-after", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default=None, help="Save stats JSON to this path")

    parser.add_argument("--render", action="store_true",
                        help="Show live visualization window (press q to quit early)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Directory to save per-episode MP4 videos (e.g. outputs/videos)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Playback FPS for --render / --save-video")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, config = _load_bc_model(args.checkpoint, args.config_path, device)

    visualizer = None
    if args.render or args.save_video:
        visualizer = _Visualizer(
            fps=args.fps,
            save_video_dir=args.save_video,
            render_live=args.render,
        )

    print(f"Evaluating {args.checkpoint} on {args.layout_name} ({args.n_episodes} episodes)...")
    stats = evaluate_policy_in_env(
        model=model,
        config=config,
        layout_name=args.layout_name,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        sampling_mode=args.sampling_mode,
        temperature=args.temperature,
        deadlock_break_after=args.deadlock_break_after,
        planner_cache_dir=args.planner_cache_dir,
        device=device,
        seq_len=args.seq_len,
        visualizer=visualizer,
    )

    print(f"\n{'='*50}")
    print(f"Layout: {stats['layout_name']}")
    print(f"Episodes: {stats['n_episodes']}")
    print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"Mean deliveries: {stats['mean_deliveries']:.2f}")
    print(f"Delivery rate: {stats['delivery_rate']:.1%}")
    print(f"Cook started rate: {stats['cook_started_rate']:.1%}")
    print(f"Stuck rate: {stats['stuck_rate']:.1%}")
    print(f"{'='*50}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {args.output}")

    if args.save_video:
        print(f"Videos saved to {args.save_video}/")


if __name__ == "__main__":
    main()
