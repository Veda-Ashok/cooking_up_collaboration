"""
PPO with CNN feature extractor (lossless spatial observations).

The PPO learner sees the lossless (H, W, C) spatial tensor while the BC
partner receives the 96-D featurized vector.  Reward shaping is annealed
from dense to sparse over training.

Usage
-----
python train_ppo_cnn.py \\
    --bc-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \\
    --layout cramped_room \\
    --run-name cnn_v1
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from rl.callbacks import BestModelCheckpoint, EpisodeRewardLoggerCallback, LinearRewardShapingCallback
from rl.env_utils import OvercookedRLWrapper, make_overcooked_vec_env
from rl.models.overcooked_cnn import OvercookedCNN


def _load_bc_partner(
    checkpoint_path: str, input_dim: int, num_actions: int,
) -> nn.Module:
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
            dropout=cfg.get("dropout", 0.0), device="cpu",
        )
        for p in partner.parameters():
            p.requires_grad = False
        print(f"[cnn-curriculum] Loaded LSTM BC partner from {checkpoint_path}")
        return partner

    from rl.bc_init_utils import load_checkpoint_state_dict
    from imitation.models.mlp import MLPPolicy
    partner = MLPPolicy(
        input_dim=input_dim, num_actions=num_actions,
        hidden_dims=list(cfg.get("mlp_hidden", [64, 64])),
        dropout=cfg.get("dropout", 0.0),
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    partner.load_state_dict(state_dict)
    partner.eval()
    for p in partner.parameters():
        p.requires_grad = False
    print(f"[cnn-curriculum] Loaded MLP BC partner from {checkpoint_path}")
    return partner


class FrozenCNNPartner(nn.Module):
    """Frozen snapshot of the CNN PPO policy.

    The env partner stream sends featurized obs, but this partner needs
    lossless obs.  It grabs the env state directly via a reference to the
    wrapper and encodes it on the fly.
    """

    def __init__(self, sb3_policy: nn.Module):
        super().__init__()
        self.features_extractor: nn.Module | None = None
        self.mlp_extractor_policy: nn.Module | None = None
        self.action_net: nn.Module | None = None
        self._env_ref: OvercookedRLWrapper | None = None
        self.update_from(sb3_policy)

    def set_env_ref(self, env) -> None:
        while not isinstance(env, OvercookedRLWrapper) and hasattr(env, "env"):
            env = env.env
        self._env_ref = env

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
        if self._env_ref is not None:
            enc = self._env_ref.mdp.lossless_state_encoding(
                self._env_ref.env.state, horizon=self._env_ref.horizon)
            partner_slot = 1 - self._env_ref.player_idx
            lossless_partner = np.array(enc[partner_slot], dtype=np.float32)
            obs = torch.as_tensor(lossless_partner, dtype=torch.float32).unsqueeze(0)
        elif obs.ndim == 1:
            obs = obs.unsqueeze(0)

        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor_policy(features)
        return self.action_net(latent_pi)


class CurriculumPartner(nn.Module):
    """Picks BC or frozen self-play each episode."""

    def __init__(
        self, bc_partner: nn.Module,
        bc_prob_start: float = 0.80, bc_prob_end: float = 1.0,
        anneal_steps: int = 500_000,
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


class CNNCurriculumCallback(BaseCallback):
    """Curriculum schedule + snapshot refresh for CNN PPO."""

    def __init__(
        self, curriculum_partner: CurriculumPartner,
        self_play_steps: int = 0, anneal_steps: int = 500_000,
        snapshot_freq: int = 25_000, verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_partner = curriculum_partner
        self.self_play_steps = self_play_steps
        self.anneal_steps = anneal_steps
        self.snapshot_freq = snapshot_freq
        self.last_snapshot_step = 0

    @staticmethod
    def _unwrap(env) -> OvercookedRLWrapper:
        while not isinstance(env, OvercookedRLWrapper) and hasattr(env, "env"):
            env = env.env
        return env

    def _raw_envs(self) -> list:
        env = self.training_env
        if hasattr(env, "envs"):
            return [self._unwrap(e) for e in env.envs]
        return [self._unwrap(env)]

    def _make_snapshot(self) -> FrozenCNNPartner:
        snap = FrozenCNNPartner(self.model.policy)
        snap.set_env_ref(self._raw_envs()[0])
        return snap

    def _on_training_start(self) -> None:
        snap = self._make_snapshot()
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
                self.num_timesteps - self.self_play_steps)

        if (self.num_timesteps - self.last_snapshot_step) >= self.snapshot_freq:
            snap = self._make_snapshot()
            self.curriculum_partner.set_selfplay_partner(snap)
            self.last_snapshot_step = self.num_timesteps

        self.logger.record("curriculum/bc_prob",
                           float(self.curriculum_partner.current_bc_prob()))
        self.logger.record("curriculum/is_anneal_phase",
                           1.0 if self.curriculum_partner.phase == "anneal" else 0.0)

    def _on_step(self) -> bool:
        return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="PPO + CNN learner + self-play curriculum + BC anneal")
    p.add_argument("--bc-checkpoint", type=str, required=True)
    p.add_argument("--layout", type=str, default="cramped_room")
    p.add_argument("--run-name", type=str, default="cnn_curriculum_v1")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    p.add_argument("--features-dim", type=int, default=64)

    p.add_argument("--self-play-steps", type=int, default=0,
                   help="Steps of self-play before BC anneal (0=skip)")
    p.add_argument("--anneal-steps", type=int, default=1_000_000)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--snapshot-freq", type=int, default=50_000)
    p.add_argument("--bc-prob-start", type=float, default=1.0,
                   help="Always use BC partner (stable learning signal)")
    p.add_argument("--bc-prob-end", type=float, default=1.0)

    p.add_argument("--reward-shaping-start", type=float, default=1.0,
                   help="Dense shaping needed since CNN starts from scratch")
    p.add_argument("--reward-shaping-end", type=float, default=0.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--reward-transform", type=str, default="symlog",
                   choices=OvercookedRLWrapper.REWARD_TRANSFORMS,
                   help="How to scale rewards: clip (hard), symlog (smooth log compression), none")

    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak LR (linear decay to 0 over training)")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05,
                   help="Higher entropy for CNN (learning from scratch)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=5,
                   help="Fewer epochs to prevent overfitting per batch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--player-idx", type=str, default="alternate",
                   help="Which player slot the learner occupies: 0, 1, or 'alternate'")
    args = p.parse_args()

    total_timesteps = args.total_timesteps or (args.self_play_steps + args.anneal_steps)

    player_idx: int | str = args.player_idx
    if player_idx not in ("alternate",):
        player_idx = int(player_idx)

    temp_env = OvercookedRLWrapper(
        layout_name=args.layout, obs_mode="featurized",
        partner_obs_mode="featurized", horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
    )
    bc_input_dim = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    del temp_env

    bc_partner = _load_bc_partner(
        checkpoint_path=args.bc_checkpoint,
        input_dim=bc_input_dim, num_actions=num_actions,
    )

    curriculum_partner = CurriculumPartner(
        bc_partner=bc_partner,
        bc_prob_start=args.bc_prob_start,
        bc_prob_end=args.bc_prob_end,
        anneal_steps=args.anneal_steps,
    )

    env_kwargs = dict(
        layout_name=args.layout,
        gym_partner=curriculum_partner,
        obs_mode="lossless",
        partner_obs_mode="featurized",
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        reward_shaping_coef=args.reward_shaping_start,
        reward_clip=args.reward_clip,
        reward_transform=args.reward_transform,
        player_idx=player_idx,
    )

    if args.n_envs > 1:
        env = make_overcooked_vec_env(n_envs=args.n_envs, **env_kwargs)
    else:
        env = OvercookedRLWrapper(**env_kwargs)
        check_env(env)

    print(f"[cnn-curriculum] Learner obs: lossless")
    print(f"[cnn-curriculum] BC partner input dim: {bc_input_dim}")

    run_dir = Path("trained_models/rl") / args.run_name
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
        tensorboard_log="./logs/ppo_cnn_curriculum",
        policy_kwargs=dict(
            features_extractor_class=OvercookedCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            normalize_images=False,
        ),
    )

    curriculum_cb = CNNCurriculumCallback(
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

    print(f"[cnn-curriculum] Training on {args.layout}")
    print(f"  Player idx:           {player_idx}")
    print(f"  Parallel envs:        {args.n_envs}")
    print(f"  Self-play steps:      {args.self_play_steps:,}")
    print(f"  Anneal steps:         {args.anneal_steps:,} "
          f"(BC prob {args.bc_prob_start:.0%} -> {args.bc_prob_end:.0%})")
    print(f"  Snapshot refresh:     every {args.snapshot_freq:,} steps")
    print(f"  Reward shaping:       {args.reward_shaping_start} -> {args.reward_shaping_end}")
    print(f"  Reward transform:     {args.reward_transform} (clip={args.reward_clip})")
    print(f"  LR / clip / ent:      {args.lr} (decay) / {args.clip_range} / {args.ent_coef}")
    print(f"  Features dim:         {args.features_dim}")
    print(f"  n_epochs:             {args.n_epochs}")
    print(f"  Total:                {total_timesteps:,} steps")

    model.learn(total_timesteps=total_timesteps,
                callback=[curriculum_cb, shaping_cb, reward_logger, best_ckpt],
                tb_log_name=args.run_name)

    save_path = run_dir / "final_model"
    model.save(str(save_path))
    print(f"[cnn-curriculum] Model saved to {save_path}")

    config = {
        "bc_checkpoint": args.bc_checkpoint,
        "layout": args.layout,
        "horizon": args.horizon,
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
        "reward_transform": args.reward_transform,
        "features_dim": args.features_dim,
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
        "obs_mode": "lossless",
        "partner_obs_mode": "featurized",
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
