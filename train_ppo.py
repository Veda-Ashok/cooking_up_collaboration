"""
PPO fine-tuning from BC initialisation (featurized 96-D observations).

The PPO policy is initialised from a BC checkpoint so it starts competent
rather than random.  Training against the BC partner then refines the policy
via RL reward.  Supports player-index alternation for symmetric play.

Usage
-----
python train_ppo.py \\
    --bc-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \\
    --layout cramped_room \\
    --run-name curriculum_v1
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from imitation.models.mlp import MLPPolicy
from rl.bc_init_utils import (
    load_checkpoint_state_dict,
    load_lstm_policy_from_checkpoint,
    transfer_bc_mlp_to_sb3_policy,
)
from rl.callbacks import (
    BestModelCheckpoint,
    EpisodeRewardLoggerCallback,
    LinearRewardShapingCallback,
)
from rl.env_utils import OvercookedRLWrapper, make_overcooked_vec_env


def load_bc_partner(
    checkpoint_path: str,
    input_dim: int,
    num_actions: int,
    hidden_dims: list[int] | None = None,
    config_path: str | None = None,
) -> nn.Module:
    """Load a BC partner, auto-detecting MLP vs LSTM from config.json."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        candidate = Path(checkpoint_path).parent / "config.json"
        cfg = {}
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                cfg = json.load(f)

    model_type = cfg.get("model", "mlp")

    if model_type == "lstm":
        partner = load_lstm_policy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            input_dim=input_dim,
            num_actions=num_actions,
            default_hidden_dim=cfg.get("hidden_dim", 128),
            default_num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.0),
            device="cpu",
        )
        for p in partner.parameters():
            p.requires_grad = False
        print(f"[curriculum] Loaded LSTM BC partner from {checkpoint_path}")
        return partner

    if hidden_dims is None:
        hidden_dims = cfg.get("mlp_hidden", [64, 64])
    dropout = cfg.get("dropout", 0.0)

    partner = MLPPolicy(
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dims=list(hidden_dims),
        dropout=dropout,
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    partner.load_state_dict(state_dict)
    partner.eval()
    for p in partner.parameters():
        p.requires_grad = False
    print(f"[curriculum] Loaded MLP BC partner from {checkpoint_path}")
    return partner


class FrozenSB3Partner(nn.Module):
    """Frozen snapshot of an SB3 PPO policy's feature extractor + action head."""

    def __init__(self, sb3_policy: nn.Module):
        super().__init__()
        self.features_extractor: nn.Module | None = None
        self.mlp_extractor_policy: nn.Module | None = None
        self.action_net: nn.Module | None = None
        self.update_from(sb3_policy)

    def update_from(self, sb3_policy: nn.Module) -> None:
        def _clone(module: nn.Module) -> nn.Module:
            clone = type(module).__new__(type(module))
            nn.Module.__init__(clone)
            clone.__dict__.update({
                k: v for k, v in module.__dict__.items()
                if k not in ("_parameters", "_buffers", "_modules")
            })
            for k, v in module._parameters.items():
                clone.register_parameter(k, nn.Parameter(v.detach().clone()) if v is not None else None)
            for k, v in module._buffers.items():
                clone.register_buffer(k, v.detach().clone() if v is not None else None)
            for k, v in module._modules.items():
                clone.add_module(k, _clone(v))
            return clone

        self.features_extractor = _clone(sb3_policy.features_extractor).cpu().eval()
        self.mlp_extractor_policy = _clone(sb3_policy.mlp_extractor).cpu().eval()
        self.action_net = _clone(sb3_policy.action_net).cpu().eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor_policy(features)
        return self.action_net(latent_pi)


class CurriculumPartner(nn.Module):
    """Picks BC or self-play snapshot each episode based on curriculum phase."""

    def __init__(
        self,
        bc_partner: nn.Module,
        bc_prob_start: float = 0.10,
        bc_prob_end: float = 0.80,
        anneal_steps: int = 150_000,
    ):
        super().__init__()
        self.bc_partner = bc_partner
        self.selfplay_partner: nn.Module | None = None
        self.active_partner: nn.Module = bc_partner

        self._bc_is_lstm = hasattr(bc_partner, "lstm")
        if self._bc_is_lstm:
            self.lstm = True

        self.phase = "self_play"
        self.bc_prob_start = bc_prob_start
        self.bc_prob_end = bc_prob_end
        self.anneal_steps = anneal_steps
        self.anneal_progress = 0

    def set_selfplay_partner(self, partner: nn.Module) -> None:
        self.selfplay_partner = partner
        if self.phase == "self_play":
            self.active_partner = partner

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def set_anneal_progress(self, steps: int) -> None:
        self.anneal_progress = max(0, steps)

    def current_bc_prob(self) -> float:
        if self.phase != "anneal":
            return 0.0
        frac = min(1.0, self.anneal_progress / max(1, self.anneal_steps))
        return self.bc_prob_start + frac * (self.bc_prob_end - self.bc_prob_start)

    def on_episode_start(self) -> None:
        if self.phase == "self_play":
            self.active_partner = self.selfplay_partner or self.bc_partner
            return
        if self.phase == "anneal":
            p_bc = self.current_bc_prob()
            use_bc = (self.selfplay_partner is None) or (random.random() < p_bc)
            self.active_partner = self.bc_partner if use_bc else self.selfplay_partner
            return
        self.active_partner = self.bc_partner

    def _active_is_lstm(self) -> bool:
        return hasattr(self.active_partner, "lstm")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._active_is_lstm():
            if obs.ndim == 2:
                obs = obs.unsqueeze(1)
            return self.active_partner(obs)
        if obs.ndim == 3:
            obs = obs[:, -1, :]
        return self.active_partner(obs)


class CurriculumCallback(BaseCallback):
    """SB3 callback that drives the curriculum schedule."""

    def __init__(
        self,
        curriculum_partner: CurriculumPartner,
        self_play_steps: int = 0,
        anneal_steps: int = 500_000,
        snapshot_freq: int = 25_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_partner = curriculum_partner
        self.self_play_steps = self_play_steps
        self.anneal_steps = anneal_steps
        self.snapshot_freq = snapshot_freq
        self.last_snapshot_step = 0

    @staticmethod
    def _unwrap(env):
        while hasattr(env, "env") and not isinstance(env, OvercookedRLWrapper):
            env = env.env
        return env

    def _raw_envs(self) -> list:
        env = self.training_env
        if hasattr(env, "envs"):
            return [self._unwrap(e) for e in env.envs]
        return [self._unwrap(env)]

    def _on_training_start(self) -> None:
        snap = FrozenSB3Partner(self.model.policy)
        self.curriculum_partner.set_selfplay_partner(snap)
        self.last_snapshot_step = 0
        for env in self._raw_envs():
            env.gym_partner = self.curriculum_partner

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.self_play_steps:
            self.curriculum_partner.set_phase("self_play")
        else:
            self.curriculum_partner.set_phase("anneal")
            self.curriculum_partner.set_anneal_progress(
                self.num_timesteps - self.self_play_steps
            )

        if (self.num_timesteps - self.last_snapshot_step) >= self.snapshot_freq:
            snap = FrozenSB3Partner(self.model.policy)
            self.curriculum_partner.set_selfplay_partner(snap)
            self.last_snapshot_step = self.num_timesteps

        self.logger.record("curriculum/bc_prob", float(self.curriculum_partner.current_bc_prob()))
        self.logger.record("curriculum/phase", self.curriculum_partner.phase)

    def _on_step(self) -> bool:
        return True


def _load_bc_config(checkpoint_path: str, config_path: str | None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    candidate = Path(checkpoint_path).parent / "config.json"
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO fine-tuning from BC init, with optional curriculum")
    p.add_argument("--bc-checkpoint", type=str, required=True,
                   help="Path to BC best.pt (used as partner AND to init PPO weights)")
    p.add_argument("--bc-config", type=str, default=None,
                   help="Path to BC config.json (auto-detected if next to checkpoint)")
    p.add_argument("--no-bc-init", action="store_true",
                   help="Skip BC weight init (train PPO from scratch)")
    p.add_argument("--layout", type=str, default="cramped_room")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")

    p.add_argument("--reward-shaping-start", type=float, default=0.0,
                   help="Reward shaping coef at start (0=sparse only, recommended for BC init)")
    p.add_argument("--reward-shaping-end", type=float, default=0.0)
    p.add_argument("--reward-clip", type=float, default=5.0)

    p.add_argument("--self-play-steps", type=int, default=0,
                   help="Steps of self-play before BC anneal (0=skip)")
    p.add_argument("--anneal-steps", type=int, default=1_000_000)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--snapshot-freq", type=int, default=25_000)
    p.add_argument("--bc-prob-start", type=float, default=1.0,
                   help="BC partner probability at start of anneal phase")
    p.add_argument("--bc-prob-end", type=float, default=1.0)

    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (linear decay to 0 over training)")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.02,
                   help="Entropy coef (explore slightly beyond BC)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=5,
                   help="Fewer epochs per update to prevent overfit per batch")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--player-idx", type=str, default="alternate",
                   help="Which player slot the learner occupies: 0, 1, or 'alternate' (random each episode)")

    p.add_argument("--outdir", type=str, default="trained_models/rl")
    p.add_argument("--run-name", type=str, default="ppo_curriculum")
    p.add_argument("--tensorboard-log", type=str, default="./logs/ppo_curriculum")
    p.add_argument("--export-agent-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    total_timesteps = args.total_timesteps or (args.self_play_steps + args.anneal_steps)

    player_idx: int | str = args.player_idx
    if player_idx not in ("alternate",):
        player_idx = int(player_idx)

    bc_cfg = _load_bc_config(args.bc_checkpoint, args.bc_config)
    bc_hidden = list(bc_cfg.get("mlp_hidden", [64, 64]))

    probe_env = OvercookedRLWrapper(
        layout_name=args.layout, horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
    )
    obs_dim = probe_env.observation_space.shape[0]
    num_actions = probe_env.action_space.n
    del probe_env

    bc_partner = load_bc_partner(
        checkpoint_path=args.bc_checkpoint, input_dim=obs_dim,
        num_actions=num_actions, hidden_dims=bc_hidden,
        config_path=args.bc_config,
    )

    curriculum_partner = CurriculumPartner(
        bc_partner=bc_partner,
        bc_prob_start=args.bc_prob_start,
        bc_prob_end=args.bc_prob_end,
        anneal_steps=args.anneal_steps,
    )

    if args.n_envs > 1:
        env = make_overcooked_vec_env(
            n_envs=args.n_envs, layout_name=args.layout,
            gym_partner=curriculum_partner, horizon=args.horizon,
            planner_cache_dir=args.planner_cache_dir,
            reward_shaping_coef=args.reward_shaping_start,
            reward_clip=args.reward_clip,
            player_idx=player_idx,
        )
    else:
        env = OvercookedRLWrapper(
            layout_name=args.layout, gym_partner=curriculum_partner,
            horizon=args.horizon, planner_cache_dir=args.planner_cache_dir,
            reward_shaping_coef=args.reward_shaping_start,
            reward_clip=args.reward_clip,
            player_idx=player_idx,
        )
        check_env(env)

    run_dir = Path(args.outdir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def linear_lr(progress_remaining: float) -> float:
        return progress_remaining

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=lambda prog: args.lr * linear_lr(prog),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=bc_hidden, vf=bc_hidden),
        ),
    )

    if not args.no_bc_init:
        bc_state_dict = load_checkpoint_state_dict(
            args.bc_checkpoint, map_location="cpu")
        report = transfer_bc_mlp_to_sb3_policy(
            model.policy, bc_state_dict, hidden_dims=bc_hidden)
        print(f"[bc-init] Transferred {report['loaded_count']} weight tensors "
              f"from BC checkpoint ({report['skipped_count']} skipped)")
        if report["skipped_keys"]:
            for sk in report["skipped_keys"]:
                print(f"  SKIP: {sk}")
    else:
        print("[bc-init] Skipped (--no-bc-init)")

    curriculum_cb = CurriculumCallback(
        curriculum_partner=curriculum_partner,
        self_play_steps=args.self_play_steps,
        anneal_steps=args.anneal_steps,
        snapshot_freq=args.snapshot_freq,
    )
    shaping_cb = LinearRewardShapingCallback(
        total_timesteps=total_timesteps,
        start_coef=args.reward_shaping_start,
        end_coef=args.reward_shaping_end,
    )
    reward_logger = EpisodeRewardLoggerCallback()
    best_ckpt = BestModelCheckpoint(save_dir=str(run_dir))

    print(f"Training PPO on {args.layout}")
    print(f"  BC init:              {'YES' if not args.no_bc_init else 'no'}")
    print(f"  BC hidden dims:       {bc_hidden}")
    print(f"  Player idx:           {player_idx}")
    print(f"  Parallel envs:        {args.n_envs}")
    print(f"  Self-play steps:      {args.self_play_steps:,}")
    print(f"  Anneal steps:         {args.anneal_steps:,} "
          f"(BC prob {args.bc_prob_start:.0%} -> {args.bc_prob_end:.0%})")
    print(f"  Reward shaping:       {args.reward_shaping_start} -> {args.reward_shaping_end}")
    print(f"  LR / clip / ent:      {args.lr} / {args.clip_range} / {args.ent_coef}")
    print(f"  Total:                {total_timesteps:,} steps")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, shaping_cb, reward_logger, best_ckpt],
        tb_log_name=args.run_name,
    )
    save_path = run_dir / "final_model.zip"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    train_config = {
        "layout": args.layout,
        "bc_checkpoint": args.bc_checkpoint,
        "bc_init": not args.no_bc_init,
        "bc_hidden": bc_hidden,
        "player_idx": str(player_idx),
        "self_play_steps": args.self_play_steps,
        "anneal_steps": args.anneal_steps,
        "snapshot_freq": args.snapshot_freq,
        "bc_prob_start": args.bc_prob_start,
        "bc_prob_end": args.bc_prob_end,
        "total_timesteps": total_timesteps,
        "reward_shaping_start": args.reward_shaping_start,
        "reward_shaping_end": args.reward_shaping_end,
        "reward_clip": args.reward_clip,
        "n_envs": args.n_envs,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    if args.export_agent_name:
        agent_dir = Path("webapp/server/static/assets/agents") / args.export_agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(save_path), str(agent_dir / "final_model.zip"))
        manifest = {
            "type": "rl_torch",
            "algo": "ppo",
            "policy": "MlpPolicy",
            "checkpoint": "final_model.zip",
            "supported_layouts": [args.layout],
            "input_dim": obs_dim,
            "num_actions": num_actions,
            "sampling_mode": "sample",
            "sampling_temperature": 1.0,
            "deterministic": False,
            "planner_cache_dir": ".cache/overcooked_planners",
        }
        with open(agent_dir / "agent_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Exported webapp agent to {agent_dir}")


if __name__ == "__main__":
    main()
