"""
PPO Curriculum Training: Self-Play followed by Annealing to a BC Partner

Overview
--------
This curriculum consists of two main stages:

Stage 1 - Self-Play:
    - The learner trains against a *frozen* snapshot of itself.
    - The frozen snapshot is periodically refreshed every N steps.

Stage 2 - Annealing:
    - For each new episode, the partner is randomly chosen to be either:
        * The current frozen self-play snapshot, or
        * A pre-trained BC (behavioral cloning) partner.
    - The probability of facing the BC partner increases linearly
      from ``bc_prob_start`` to ``bc_prob_end``.

Usage
-----
python -m rl.train_ppo_curriculum \
    --bc-checkpoint trained_models/bc/bc_cramped_v1/best.pt \
    --layout cramped_room \
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
from rl.bc_init_utils import load_checkpoint_state_dict, load_lstm_policy_from_checkpoint
from rl.env_utils import OvercookedRLWrapper
from rl.models.rl_mlp import RLActorCriticPolicy


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



# Frozen snapshot of an SB3 PPO policy's feature extractor + action head,
# wrapped so the env can call partner(obs) -> logits.
class FrozenSB3Partner(nn.Module):
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
            sd = {k: v.detach().clone() for k, v in module.state_dict().items()}
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



# Curriculum partner: picks BC or self-play snapshot each episode
class CurriculumPartner(nn.Module):
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
            self.lstm = True  # expose so env wrapper detects LSTM-style partner

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
            # Active partner is LSTM -- expects [batch, seq_len, input_dim]
            if obs.ndim == 2:
                obs = obs.unsqueeze(1)
            return self.active_partner(obs)
        # Active partner is MLP/SB3 -- expects [batch, input_dim]
        if obs.ndim == 3:
            obs = obs[:, -1, :]
        return self.active_partner(obs)



# SB3 callback that drives the curriculum schedule
class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        curriculum_partner: CurriculumPartner,
        self_play_steps: int = 150_000,
        anneal_steps: int = 150_000,
        snapshot_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_partner = curriculum_partner
        self.self_play_steps = self_play_steps
        self.anneal_steps = anneal_steps
        self.snapshot_freq = snapshot_freq
        self.last_snapshot_step = 0

    def _raw_env(self) -> OvercookedRLWrapper:
        env = self.training_env
        if hasattr(env, "envs"):
            return env.envs[0]
        return env

    def _on_training_start(self) -> None:
        self.curriculum_partner.set_selfplay_partner(
            FrozenSB3Partner(self.model.policy)
        )
        self.last_snapshot_step = 0
        self._raw_env().gym_partner = self.curriculum_partner

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.self_play_steps:
            self.curriculum_partner.set_phase("self_play")
        else:
            self.curriculum_partner.set_phase("anneal")
            self.curriculum_partner.set_anneal_progress(
                self.num_timesteps - self.self_play_steps
            )

        if (self.num_timesteps - self.last_snapshot_step) >= self.snapshot_freq:
            self.curriculum_partner.set_selfplay_partner(
                FrozenSB3Partner(self.model.policy)
            )
            self.last_snapshot_step = self.num_timesteps

        self.logger.record("curriculum/bc_prob", float(self.curriculum_partner.current_bc_prob()))
        self.logger.record("curriculum/phase", self.curriculum_partner.phase)

    def _on_step(self) -> bool:
        return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO curriculum: self-play then BC anneal")
    p.add_argument("--bc-checkpoint", type=str, required=True,
                   help="Path to BC best.pt for the partner")
    p.add_argument("--bc-config", type=str, default=None,
                   help="Path to BC config.json (auto-detected if next to checkpoint)")
    p.add_argument("--bc-hidden", type=str, default=None,
                   help="BC hidden dims, e.g. '64,64' (auto-detected from config)")
    p.add_argument("--layout", type=str, default="cramped_room")
    p.add_argument("--horizon", type=int, default=400,
                   help="Max timesteps per episode (default 400, paper uses 400)")
    p.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    p.add_argument("--reward-shaping-coef", type=float, default=3,
                   help="Multiplier for event-based shaped reward (0=sparse only, 1.0=full shaping)")

    p.add_argument("--self-play-steps", type=int, default=150_000)
    p.add_argument("--anneal-steps", type=int, default=150_000)
    p.add_argument("--snapshot-freq", type=int, default=10_000)
    p.add_argument("--bc-prob-start", type=float, default=0.10)
    p.add_argument("--bc-prob-end", type=float, default=0.80)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--outdir", type=str, default="trained_models/rl")
    p.add_argument("--run-name", type=str, default="ppo_curriculum")
    p.add_argument("--tensorboard-log", type=str, default="./logs/ppo_curriculum")
    p.add_argument("--export-agent-name", type=str, default=None,
                   help="If set, copy final model into webapp agents folder")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    total_timesteps = args.self_play_steps + args.anneal_steps

    env = OvercookedRLWrapper(
        layout_name=args.layout,
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        reward_shaping_coef=args.reward_shaping_coef,
    )
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    bc_hidden = None
    if args.bc_hidden:
        bc_hidden = [int(x.strip()) for x in args.bc_hidden.split(",")]

    bc_partner = load_bc_partner(
        checkpoint_path=args.bc_checkpoint,
        input_dim=obs_dim,
        num_actions=num_actions,
        hidden_dims=bc_hidden,
        config_path=args.bc_config,
    )

    curriculum_partner = CurriculumPartner(
        bc_partner=bc_partner,
        bc_prob_start=args.bc_prob_start,
        bc_prob_end=args.bc_prob_end,
        anneal_steps=args.anneal_steps,
    )
    env.gym_partner = curriculum_partner

    check_env(env)

    model = PPO(
        policy=RLActorCriticPolicy,
        env=env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
    )

    callback = CurriculumCallback(
        curriculum_partner=curriculum_partner,
        self_play_steps=args.self_play_steps,
        anneal_steps=args.anneal_steps,
        snapshot_freq=args.snapshot_freq,
        verbose=1,
    )

    print(f"Training PPO curriculum on {args.layout}")
    print(f"  Stage 1 (self-play):  {args.self_play_steps:,} steps")
    print(f"  Stage 2 (anneal):     {args.anneal_steps:,} steps "
          f"(BC prob {args.bc_prob_start:.0%} -> {args.bc_prob_end:.0%})")
    print(f"  Snapshot refresh:     every {args.snapshot_freq:,} steps")
    print(f"  Total:                {total_timesteps:,} steps")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    run_dir = Path(args.outdir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "final_model.zip"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    train_config = {
        "layout": args.layout,
        "bc_checkpoint": args.bc_checkpoint,
        "self_play_steps": args.self_play_steps,
        "anneal_steps": args.anneal_steps,
        "snapshot_freq": args.snapshot_freq,
        "bc_prob_start": args.bc_prob_start,
        "bc_prob_end": args.bc_prob_end,
        "total_timesteps": total_timesteps,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "seed": args.seed,
        "reward_shaping_coef": args.reward_shaping_coef,
    }
    config_path = run_dir / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)
    print(f"Config saved to {config_path}")

    if args.export_agent_name:
        agent_dir = Path("webapp/server/static/assets/agents") / args.export_agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(save_path), str(agent_dir / "final_model.zip"))
        manifest = {
            "type": "rl_torch",
            "algo": "ppo",
            "policy": "RLActorCriticPolicy",
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
