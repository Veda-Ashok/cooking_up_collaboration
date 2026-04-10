"""Evaluate a trained PPO checkpoint via environment rollouts.

Supports RL self-play (same model as both players) and RL + BC partner.
Produces the same metrics as evaluate_bc_rollouts.py for direct comparison.

Usage
-----
# RL self-play
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --layout-name cramped_room --n-episodes 20

# RL + BC partner
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --partner-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout-name cramped_room --n-episodes 20
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from stable_baselines3 import PPO

from rl.env_utils import OvercookedRLWrapper
from rl.self_play_partner import SB3SelfPlayPartner
from rl.models.overcooked_cnn import OvercookedCNN


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_train_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.parent / "train_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_ppo_model(checkpoint_path: str, device: str) -> PPO:
    custom_objects = {"OvercookedCNN": OvercookedCNN}
    return PPO.load(checkpoint_path, device=device, custom_objects=custom_objects)


def _load_bc_partner(
    checkpoint_path: str, input_dim: int, num_actions: int, device: str,
) -> torch.nn.Module:
    cfg = {}
    config_candidate = Path(checkpoint_path).parent / "config.json"
    if config_candidate.exists():
        with open(config_candidate, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    model_type = cfg.get("model", "mlp")

    if model_type == "lstm":
        from rl.bc_init_utils import load_lstm_policy_from_checkpoint
        partner = load_lstm_policy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            input_dim=input_dim, num_actions=num_actions,
            default_hidden_dim=cfg.get("hidden_dim", 128),
            default_num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.0), device=device,
        )
        partner.eval()
        return partner

    from rl.bc_init_utils import load_checkpoint_state_dict
    from imitation.models.mlp import MLPPolicy
    partner = MLPPolicy(
        input_dim=input_dim, num_actions=num_actions,
        hidden_dims=list(cfg.get("mlp_hidden", [64, 64])),
        dropout=cfg.get("dropout", 0.0),
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=device)
    partner.load_state_dict(state_dict)
    partner.to(device)
    partner.eval()
    return partner


def _state_has_cooking(state: OvercookedState) -> bool:
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

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def render_step(self, env: OvercookedRLWrapper, step: int, ep: int,
                    score: float, reward: float) -> bool:
        cv = self._cv
        surface = self._vis.render_state(env.env.state, grid=env.mdp.terrain_mtx)
        frame_rgb = self._pygame_surfarray.array3d(surface)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        mode_label = "self-play" if isinstance(env.gym_partner, SB3SelfPlayPartner) else "RL+BC"
        overlay = f"ep={ep} step={step} score={score:.0f} r={reward:.0f} [{mode_label}]"
        cv.putText(frame_bgr, overlay, (10, 24), cv.FONT_HERSHEY_SIMPLEX,
                   0.55, (40, 220, 40), 2, cv.LINE_AA)

        if self.save_video_dir:
            if self._writer is None or self._current_ep != ep:
                self._close_writer()
                self._current_ep = ep
                Path(self.save_video_dir).mkdir(parents=True, exist_ok=True)
                h, w = frame_bgr.shape[:2]
                out_path = str(Path(self.save_video_dir) / f"episode_{ep:03d}.mp4")
                self._writer = cv.VideoWriter(
                    out_path, cv.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._writer.write(frame_bgr)

        if self.render_live:
            cv.imshow("RL Rollout Evaluation", frame_bgr)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
        return True

    def close(self) -> None:
        self._close_writer()
        if self.render_live:
            self._cv.destroyAllWindows()


def evaluate_rl_rollouts(
    model: PPO,
    env: OvercookedRLWrapper,
    n_episodes: int = 20,
    deterministic: bool = False,
    visualizer: _Visualizer | None = None,
) -> dict:
    episode_rewards = []
    episode_deliveries = []
    cook_started_count = 0
    delivery_count = 0
    stuck_episodes = 0
    horizon = env.horizon

    quit_early = False
    for ep in range(n_episodes):
        if quit_early:
            break

        obs, _ = env.reset()

        if visualizer is not None:
            visualizer._start_episode(ep)

        ep_sparse = 0.0
        ep_deliveries = 0
        ep_cook_started = False
        position_history: deque[tuple] = deque(maxlen=10)
        ep_stuck_steps = 0
        frame_budget = 1.0 / visualizer.fps if visualizer else 0.0

        for t in range(horizon):
            tick_start = time.time()

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(action)

            sparse = info.get("sparse_reward", 0.0)
            ep_sparse += sparse

            if sparse > 0:
                ep_deliveries += max(1, int(sparse / 20))

            if not ep_cook_started and _state_has_cooking(env.env.state):
                ep_cook_started = True

            try:
                p0_pos = tuple(env.env.state.player_positions[0])
                p1_pos = tuple(env.env.state.player_positions[1])
                current_positions = (p0_pos, p1_pos)
            except Exception:
                current_positions = None
            if current_positions and len(position_history) >= 5:
                if all(p == current_positions for p in position_history):
                    ep_stuck_steps += 1
            if current_positions:
                position_history.append(current_positions)

            if visualizer is not None:
                keep_going = visualizer.render_step(
                    env=env, step=t, ep=ep, score=ep_sparse, reward=sparse)
                if not keep_going:
                    quit_early = True
                    break
                elapsed = time.time() - tick_start
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)

            if term or trunc:
                break

        episode_rewards.append(ep_sparse)
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
    return {
        "n_episodes": completed,
        "n_episodes_requested": n_episodes,
        "horizon": horizon,
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL PPO policy via environment rollouts")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SB3 PPO .zip checkpoint")
    parser.add_argument("--layout-name", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax actions (default: stochastic)")
    parser.add_argument("--player-idx", type=str, default="alternate",
                        help="0, 1, or alternate")
    parser.add_argument("--planner-cache-dir", type=str,
                        default=".cache/overcooked_planners")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda"])

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-play", action="store_true", default=True,
                       help="RL model plays both slots (default)")
    group.add_argument("--partner-checkpoint", type=str, default=None,
                       help="BC checkpoint for partner (disables self-play)")

    parser.add_argument("--output", type=str, default=None,
                        help="Save stats JSON to this path")
    parser.add_argument("--render", action="store_true",
                        help="Show live visualization (press q to quit)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Directory for per-episode MP4 videos")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_config = _load_train_config(checkpoint_path)
    obs_mode = train_config.get("obs_mode", "featurized")

    player_idx: int | str = args.player_idx
    if player_idx not in ("alternate",):
        player_idx = int(player_idx)

    model = _load_ppo_model(str(checkpoint_path), device=device)

    env = OvercookedRLWrapper(
        layout_name=args.layout_name,
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        obs_mode=obs_mode,
        partner_obs_mode="featurized",
        player_idx=player_idx,
        reward_shaping_coef=0.0,
    )

    if args.partner_checkpoint:
        bc_input_dim = 96
        if obs_mode != "featurized":
            dummy = env.mdp.get_standard_start_state()
            bc_input_dim = len(env.mdp.featurize_state(dummy, env.mlam)[0])
        else:
            bc_input_dim = int(env.observation_space.shape[0])
        num_actions = int(env.action_space.n)
        env.gym_partner = _load_bc_partner(
            args.partner_checkpoint, bc_input_dim, num_actions, "cpu")
        mode_str = f"RL + BC partner ({Path(args.partner_checkpoint).name})"
    else:
        env.gym_partner = SB3SelfPlayPartner(model, obs_mode=obs_mode)
        mode_str = "RL self-play"

    visualizer = None
    if args.render or args.save_video:
        visualizer = _Visualizer(
            fps=args.fps, save_video_dir=args.save_video,
            render_live=args.render)

    print(f"Evaluating {args.checkpoint}")
    print(f"  Layout: {args.layout_name}  |  Mode: {mode_str}")
    print(f"  Episodes: {args.n_episodes}  |  Player idx: {player_idx}")
    print(f"  Deterministic: {args.deterministic}")

    stats = evaluate_rl_rollouts(
        model=model, env=env, n_episodes=args.n_episodes,
        deterministic=args.deterministic, visualizer=visualizer)

    stats["layout_name"] = args.layout_name
    stats["checkpoint"] = str(args.checkpoint)
    stats["mode"] = mode_str
    stats["deterministic"] = args.deterministic
    stats["player_idx"] = str(player_idx)

    print(f"\n{'='*50}")
    print(f"Layout: {stats['layout_name']}")
    print(f"Mode: {mode_str}")
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
