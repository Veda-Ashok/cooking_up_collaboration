from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2 as cv
import numpy as np
import pygame.surfarray
import torch
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from rl.bc_init_utils import load_lstm_policy_from_checkpoint
from rl.env_utils import OvercookedRLWrapper
from rl.sampling_utils import (
    configure_model_sampling_temperature,
    normalize_sampling_mode,
    validate_sampling_temperature,
)
from imitation.models.mlp import MLPPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live rollout viewer for Overcooked RL agents.")
    parser.add_argument("--agent-dir", type=str, default=None, help="Webapp agent folder containing agent_manifest.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to RL checkpoint (.zip)")
    parser.add_argument("--algo", type=str, default=None, choices=["recurrent_ppo", "ppo"])
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--partner-checkpoint", type=str, default=None, help="Optional BC checkpoint for partner")
    parser.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save-video", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sampling-mode", type=str, choices=["sample", "argmax"], default="sample")
    parser.add_argument("--sampling-temperature", type=float, default=1.3)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--obs-mode", type=str, choices=["featurized", "lossless", "auto"],
                        default="auto", help="Observation mode for the RL learner (auto reads train_config.json)")

    parser.add_argument("--bc-hidden-dim", type=int, default=128)
    parser.add_argument("--bc-num-layers", type=int, default=1)
    parser.add_argument("--bc-dropout", type=float, default=0.1)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_agent_manifest(agent_dir: Path) -> dict:
    manifest_path = agent_dir / "agent_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint_and_algo(args: argparse.Namespace) -> tuple[Path, str, dict]:
    if args.agent_dir is not None:
        agent_dir = Path(args.agent_dir)
        manifest = _load_agent_manifest(agent_dir)
        checkpoint_name = manifest.get("checkpoint", "best_model.zip")
        checkpoint_path = agent_dir / checkpoint_name
        algo = args.algo or str(manifest.get("algo", "recurrent_ppo")).lower()
        return checkpoint_path, algo, manifest

    if args.checkpoint is None:
        raise ValueError("Provide either --agent-dir or --checkpoint")
    checkpoint_path = Path(args.checkpoint)
    if args.algo is None:
        algo = "recurrent_ppo"
    else:
        algo = args.algo
    return checkpoint_path, algo, {}


def _load_train_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.parent / "train_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _try_infer_partner_checkpoint(checkpoint_path: Path) -> str | None:
    config = _load_train_config(checkpoint_path)
    partner_paths = config.get("partner_checkpoints", [])
    if partner_paths:
        return str(partner_paths[0])
    bc_ckpt = config.get("bc_checkpoint")
    if bc_ckpt:
        return str(bc_ckpt)
    return None


def _infer_obs_mode(checkpoint_path: Path) -> str:
    config = _load_train_config(checkpoint_path)
    return config.get("obs_mode", "featurized")


def _load_model(checkpoint_path: Path, algo: str, device: str):
    if algo in {"recurrent_ppo", "ppo_lstm", "lstm_ppo"}:
        if RecurrentPPO is None:
            raise ImportError(
                "sb3-contrib is required for recurrent PPO rollouts. Install with `pip install sb3-contrib`."
            )
        model = RecurrentPPO.load(str(checkpoint_path), device=device)
        is_recurrent = True
    elif algo == "ppo":
        model = PPO.load(str(checkpoint_path), device=device)
        is_recurrent = False
    else:
        raise ValueError(f"Unsupported algo={algo}")
    return model, is_recurrent


def _load_bc_partner(
    checkpoint_path: str,
    input_dim: int,
    num_actions: int,
    bc_hidden_dim: int,
    bc_num_layers: int,
    bc_dropout: float,
    device: str,
) -> torch.nn.Module:
    """Load a BC partner, auto-detecting MLP vs LSTM from config.json."""
    import json as _json

    config_candidate = Path(checkpoint_path).parent / "config.json"
    model_type = "lstm"
    mlp_hidden = [256, 128]
    dropout = bc_dropout

    if config_candidate.exists():
        with open(config_candidate, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        model_type = cfg.get("model", "lstm")
        mlp_hidden = cfg.get("mlp_hidden", mlp_hidden)
        dropout = cfg.get("dropout", dropout)

    if model_type == "mlp":
        from rl.bc_init_utils import load_checkpoint_state_dict
        state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=device)
        partner = MLPPolicy(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=list(mlp_hidden),
            dropout=dropout,
        )
        partner.load_state_dict(state_dict)
        partner.to(device)
        partner.eval()
        return partner

    return load_lstm_policy_from_checkpoint(
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        num_actions=num_actions,
        default_hidden_dim=bc_hidden_dim,
        default_num_layers=bc_num_layers,
        dropout=bc_dropout,
        device=device,
    )


def _render_frame(visualizer: StateVisualizer, env: OvercookedRLWrapper, step: int, score: float, reward: float) -> np.ndarray:
    surface = visualizer.render_state(env.env.state, grid=env.mdp.terrain_mtx)
    frame_rgb = pygame.surfarray.array3d(surface)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    overlay = f"step={step} score={score:.1f} reward={reward:.2f}"
    cv.putText(frame_bgr, overlay, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2, cv.LINE_AA)
    return frame_bgr


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint_path, algo, manifest = _resolve_checkpoint_and_algo(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {checkpoint_path}")

    legacy_deterministic = bool(manifest.get("deterministic", False))
    default_sampling_mode = "argmax" if legacy_deterministic else "sample"
    if args.sampling_mode is not None:
        sampling_mode = normalize_sampling_mode(args.sampling_mode, default=default_sampling_mode)
    elif args.deterministic:
        sampling_mode = "argmax"
    else:
        sampling_mode = normalize_sampling_mode(manifest.get("sampling_mode"), default=default_sampling_mode)
    if args.sampling_temperature is not None:
        sampling_temperature = validate_sampling_temperature(args.sampling_temperature)
    else:
        sampling_temperature = validate_sampling_temperature(manifest.get("sampling_temperature", 1.0))

    obs_mode = args.obs_mode
    if obs_mode == "auto":
        obs_mode = _infer_obs_mode(checkpoint_path)
    print(f"[live_rollout] obs_mode={obs_mode}")

    env = OvercookedRLWrapper(
        layout_name=args.layout,
        planner_cache_dir=args.planner_cache_dir,
        seq_len=args.seq_len,
        reward_shaping_coef=0.0,
        obs_mode=obs_mode,
        partner_obs_mode="featurized",
    )

    # Get featurized input dim for the BC partner (always 96-D regardless of learner obs mode)
    if obs_mode != "featurized":
        dummy_state = env.mdp.get_standard_start_state()
        bc_input_dim = len(env.mdp.featurize_state(dummy_state, env.mlam)[0])
    else:
        bc_input_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    partner_checkpoint = args.partner_checkpoint or _try_infer_partner_checkpoint(checkpoint_path)
    if partner_checkpoint:
        if not Path(partner_checkpoint).exists():
            raise FileNotFoundError(f"Partner checkpoint not found: {partner_checkpoint}")
        env.gym_partner = _load_bc_partner(
            checkpoint_path=partner_checkpoint,
            input_dim=bc_input_dim,
            num_actions=num_actions,
            bc_hidden_dim=args.bc_hidden_dim,
            bc_num_layers=args.bc_num_layers,
            bc_dropout=args.bc_dropout,
            device="cpu",
        )

    model, is_recurrent = _load_model(checkpoint_path=checkpoint_path, algo=algo, device=device)
    configure_model_sampling_temperature(model, sampling_temperature)

    obs, _ = env.reset()
    done = False
    visualizer = StateVisualizer()
    writer = None
    lstm_state = None
    episode_start = np.array([True], dtype=bool)
    total_reward = 0.0
    score = 0.0
    steps_taken = 0

    for step in range(int(args.max_steps)):
        tick_start = time.time()
        steps_taken = step + 1
        if is_recurrent:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=(sampling_mode == "argmax"),
            )
        else:
            action, _ = model.predict(obs, deterministic=(sampling_mode == "argmax"))

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        episode_start = np.array([done], dtype=bool)
        total_reward += float(reward)
        score = float(info.get("sparse_reward", 0.0) + score)

        frame = _render_frame(visualizer=visualizer, env=env, step=step, score=score, reward=float(reward))
        if writer is None and args.save_video:
            save_path = Path(args.save_video)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv.VideoWriter(str(save_path), cv.VideoWriter_fourcc(*"mp4v"), max(1, args.fps), (w, h))
        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv.imshow("Overcooked RL Live Rollout", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if done:
            break

        elapsed = time.time() - tick_start
        frame_budget = 1.0 / max(1, args.fps)
        if elapsed < frame_budget:
            time.sleep(frame_budget - elapsed)

    if writer is not None:
        writer.release()
    if not args.no_display:
        cv.destroyAllWindows()

    print(
        f"Finished rollout | steps={steps_taken} | total_team_reward={total_reward:.2f} "
        f"| sampling_mode={sampling_mode} | sampling_temperature={sampling_temperature}"
    )
    if args.save_video:
        print(f"Saved video to: {args.save_video}")


if __name__ == "__main__":
    main()
