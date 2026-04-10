import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from rl.bc_init_utils import (
    load_checkpoint_state_dict,
    load_lstm_policy_from_checkpoint,
    transfer_bc_lstm_to_recurrent_policy,
)
from rl.env_utils import OvercookedRLWrapper
from rl.sampling_utils import (
    configure_model_sampling_temperature,
    normalize_sampling_mode,
    validate_sampling_temperature,
)


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return values


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class TrainConfig:
    layout: str
    planner_cache_dir: str
    bc_init_checkpoint: str
    partner_checkpoints: list[str]
    seq_len: int
    reward_shaping_coef: float
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    lstm_hidden_size: int
    lstm_layers: int
    seed: int
    device: str
    eval_freq: int
    eval_episodes: int
    outdir: str
    run_name: str
    export_agent_name: str | None
    supported_layouts: list[str]
    sampling_mode: str
    sampling_temperature: float
    bc_hidden_dim: int
    bc_num_layers: int
    bc_dropout: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Recurrent PPO for Overcooked from BC-LSTM initialization.")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    parser.add_argument("--bc-init-checkpoint", type=str, required=True)
    parser.add_argument("--partner-checkpoints", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--reward-shaping-coef", type=float, default=0.2)

    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)

    parser.add_argument("--bc-hidden-dim", type=int, default=128)
    parser.add_argument("--bc-num-layers", type=int, default=1)
    parser.add_argument("--bc-dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--outdir", type=str, default="trained_models/rl")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--export-agent-name", type=str, default=None)
    parser.add_argument("--supported-layouts", type=str, default=None)
    parser.add_argument("--sampling-mode", type=str, choices=["sample", "argmax"], default="sample")
    parser.add_argument("--sampling-temperature", type=float, default=1.3)
    return parser


def _load_partners(
    partner_paths: list[str],
    input_dim: int,
    num_actions: int,
    default_hidden_dim: int,
    default_num_layers: int,
    dropout: float,
) -> list[torch.nn.Module]:
    partners = []
    for path in partner_paths:
        partner = load_lstm_policy_from_checkpoint(
            checkpoint_path=path,
            input_dim=input_dim,
            num_actions=num_actions,
            default_hidden_dim=default_hidden_dim,
            default_num_layers=default_num_layers,
            dropout=dropout,
            device="cpu",
        )
        partners.append(partner)
    return partners


def _save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _export_webapp_agent(
    run_dir: Path,
    export_agent_name: str,
    supported_layouts: list[str],
    input_dim: int,
    num_actions: int,
    sampling_mode: str,
    sampling_temperature: float,
) -> Path:
    agent_root = Path("webapp/server/static/assets/agents")
    agent_dir = agent_root / export_agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.zip"
    if not best_model_path.exists():
        best_model_path = run_dir / "last_model.zip"
    checkpoint_name = "best_model.zip"
    shutil.copy2(best_model_path, agent_dir / checkpoint_name)

    manifest = {
        "type": "rl_torch",
        "algo": "recurrent_ppo",
        "policy": "MlpLstmPolicy",
        "checkpoint": checkpoint_name,
        "supported_layouts": supported_layouts,
        "input_dim": int(input_dim),
        "num_actions": int(num_actions),
        "sampling_mode": str(sampling_mode),
        "sampling_temperature": float(sampling_temperature),
        "deterministic": sampling_mode == "argmax",
        "planner_cache_dir": ".cache/overcooked_planners",
    }
    _save_json(agent_dir / "agent_manifest.json", manifest)
    return agent_dir


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if RecurrentPPO is None:
        raise ImportError(
            "sb3-contrib is required for RecurrentPPO. Install with `pip install sb3-contrib`."
        )

    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not Path(args.bc_init_checkpoint).exists():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_init_checkpoint}")

    run_name = args.run_name or f"lstm_bc_ppo_{_timestamp()}"
    outdir = Path(args.outdir)
    run_dir = outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    partner_paths = _parse_csv_list(args.partner_checkpoints)
    if not partner_paths:
        partner_paths = [args.bc_init_checkpoint]
    for partner_path in partner_paths:
        if not Path(partner_path).exists():
            raise FileNotFoundError(f"Partner checkpoint not found: {partner_path}")

    resolved_device = _resolve_device(args.device)
    supported_layouts = _parse_csv_list(args.supported_layouts) or [args.layout]
    sampling_mode = normalize_sampling_mode(args.sampling_mode, default="sample")
    sampling_temperature = validate_sampling_temperature(args.sampling_temperature)
    if sampling_mode == "argmax" and sampling_temperature == 1.0:
        # Keep argmax mode near-deterministic during rollouts by default.
        sampling_temperature = 1e-3

    train_env = OvercookedRLWrapper(
        layout_name=args.layout,
        planner_cache_dir=args.planner_cache_dir,
        seq_len=args.seq_len,
        reward_shaping_coef=args.reward_shaping_coef,
    )
    input_dim = int(train_env.observation_space.shape[0])
    num_actions = int(train_env.action_space.n)

    partners = _load_partners(
        partner_paths=partner_paths,
        input_dim=input_dim,
        num_actions=num_actions,
        default_hidden_dim=args.bc_hidden_dim,
        default_num_layers=args.bc_num_layers,
        dropout=args.bc_dropout,
    )
    train_env.set_partner_pool(partners)

    eval_partner = _load_partners(
        partner_paths=[partner_paths[0]],
        input_dim=input_dim,
        num_actions=num_actions,
        default_hidden_dim=args.bc_hidden_dim,
        default_num_layers=args.bc_num_layers,
        dropout=args.bc_dropout,
    )[0]
    eval_env = OvercookedRLWrapper(
        layout_name=args.layout,
        gym_partner=eval_partner,
        planner_cache_dir=args.planner_cache_dir,
        seq_len=args.seq_len,
        reward_shaping_coef=args.reward_shaping_coef,
    )

    policy_kwargs = {
        "lstm_hidden_size": int(args.lstm_hidden_size),
        "n_lstm_layers": int(args.lstm_layers),
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=resolved_device,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    sampling_patch_report = configure_model_sampling_temperature(model, sampling_temperature)

    bc_state_dict = load_checkpoint_state_dict(args.bc_init_checkpoint, map_location="cpu")
    transfer_report = transfer_bc_lstm_to_recurrent_policy(model.policy, bc_state_dict)
    print(f"BC warm start: loaded {transfer_report['loaded_count']} parameters into recurrent PPO policy")
    print(
        f"Sampling config: mode={sampling_mode}, temperature={sampling_temperature}, "
        f"patch={sampling_patch_report['reason']}"
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(1, int(args.eval_freq)),
        n_eval_episodes=max(1, int(args.eval_episodes)),
        deterministic=(sampling_mode == "argmax"),
        render=False,
    )

    model.learn(total_timesteps=int(args.total_timesteps), callback=eval_callback, progress_bar=False)
    model.save(str(run_dir / "last_model"))

    best_model = run_dir / "best_model.zip"
    if not best_model.exists():
        model.save(str(run_dir / "best_model"))

    config = TrainConfig(
        layout=args.layout,
        planner_cache_dir=args.planner_cache_dir,
        bc_init_checkpoint=args.bc_init_checkpoint,
        partner_checkpoints=partner_paths,
        seq_len=args.seq_len,
        reward_shaping_coef=args.reward_shaping_coef,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        seed=args.seed,
        device=resolved_device,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        outdir=args.outdir,
        run_name=run_name,
        export_agent_name=args.export_agent_name,
        supported_layouts=supported_layouts,
        sampling_mode=sampling_mode,
        sampling_temperature=sampling_temperature,
        bc_hidden_dim=args.bc_hidden_dim,
        bc_num_layers=args.bc_num_layers,
        bc_dropout=args.bc_dropout,
    )
    _save_json(run_dir / "train_config.json", asdict(config))
    _save_json(run_dir / "bc_transfer_report.json", transfer_report)
    _save_json(run_dir / "sampling_report.json", sampling_patch_report)

    eval_npz_path = run_dir / "eval" / "evaluations.npz"
    if eval_npz_path.exists():
        eval_blob = np.load(eval_npz_path, allow_pickle=False)
        eval_metrics = {key: eval_blob[key].tolist() for key in eval_blob.files}
        _save_json(run_dir / "metrics.json", eval_metrics)

    if args.export_agent_name:
        agent_dir = _export_webapp_agent(
            run_dir=run_dir,
            export_agent_name=args.export_agent_name,
            supported_layouts=supported_layouts,
            input_dim=input_dim,
            num_actions=num_actions,
            sampling_mode=sampling_mode,
            sampling_temperature=sampling_temperature,
        )
        print(f"Exported webapp agent to: {agent_dir}")

    print(f"RL training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
