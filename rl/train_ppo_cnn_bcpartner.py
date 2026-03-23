"""
PPO with CNN feature extractor, self-play curriculum, and BC anneal.

The PPO learner sees the lossless (H, W, C) spatial tensor.  The BC partner
receives the 96-D featurized vector.  During self-play, a frozen snapshot of
the PPO policy is used -- it encodes its own lossless observation from the
env state directly, bypassing the partner observation stream.

Stages
------
1. Self-play:  Learner trains against a frozen snapshot of itself.
2. BC anneal:  Partner switches to BC with increasing probability.

Reward shaping is annealed independently via LinearRewardShapingCallback.

Usage
-----
python -m rl.train_ppo_cnn_bcpartner \
    --bc-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
    --layout cramped_room \
    --run-name cnn_curriculum_v1
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

from rl.callbacks import LinearRewardShapingCallback
from rl.env_utils import OvercookedRLWrapper
from rl.models.overcooked_cnn import OvercookedCNN


# Load frozen BC partner (MLP or LSTM, auto-detected)
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


# Frozen snapshot of the CNN PPO policy.
# The env partner stream sends featurized obs, but this partner needs
# lossless obs.  It grabs the env state directly via a reference to the
# wrapper and encodes it on the fly.
class FrozenCNNPartner(nn.Module):
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
        # obs is the featurized partner stream -- ignore it.
        # Encode player 1's lossless observation from the live env state.
        if self._env_ref is not None:
            enc = self._env_ref.mdp.lossless_state_encoding(
                self._env_ref.env.state, horizon=self._env_ref.horizon)
            lossless_p1 = np.array(enc[1], dtype=np.float32)
            obs = torch.as_tensor(lossless_p1, dtype=torch.float32).unsqueeze(0)
        elif obs.ndim == 1:
            obs = obs.unsqueeze(0)

        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor_policy(features)
        return self.action_net(latent_pi)


# Curriculum partner: picks BC or frozen self-play each episode
class CurriculumPartner(nn.Module):
    def __init__(
        self, bc_partner: nn.Module,
        bc_prob_start: float = 0.10, bc_prob_end: float = 0.80,
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


# Callback: curriculum schedule + snapshot refresh
class CNNCurriculumCallback(BaseCallback):
    def __init__(
        self, curriculum_partner: CurriculumPartner,
        self_play_steps: int = 150_000, anneal_steps: int = 150_000,
        snapshot_freq: int = 10_000, verbose: int = 1,
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
            env = env.envs[0]
        while not isinstance(env, OvercookedRLWrapper) and hasattr(env, "env"):
            env = env.env
        return env

    def _make_snapshot(self) -> FrozenCNNPartner:
        snap = FrozenCNNPartner(self.model.policy)
        snap.set_env_ref(self._raw_env())
        return snap

    def _on_training_start(self) -> None:
        snap = self._make_snapshot()
        self.curriculum_partner.set_selfplay_partner(snap)
        self.last_snapshot_step = 0
        self._raw_env().gym_partner = self.curriculum_partner

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
    p.add_argument("--features-dim", type=int, default=32)

    p.add_argument("--self-play-steps", type=int, default=150_000)
    p.add_argument("--anneal-steps", type=int, default=150_000)
    p.add_argument("--snapshot-freq", type=int, default=10_000)
    p.add_argument("--bc-prob-start", type=float, default=0.10)
    p.add_argument("--bc-prob-end", type=float, default=0.80)

    p.add_argument("--reward-shaping-start", type=float, default=0.3)
    p.add_argument("--reward-shaping-end", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    total_timesteps = args.self_play_steps + args.anneal_steps

    # Get BC partner's featurized input dim
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

    env = OvercookedRLWrapper(
        layout_name=args.layout,
        gym_partner=curriculum_partner,
        obs_mode="lossless",
        partner_obs_mode="featurized",
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        reward_shaping_coef=args.reward_shaping_start,
    )

    check_env(env)
    print(f"[cnn-curriculum] Learner obs: {env.observation_space.shape}")
    print(f"[cnn-curriculum] BC partner input dim: {bc_input_dim}")

    run_dir = Path("trained_models/rl") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
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
        tensorboard_log="./logs/ppo_cnn_curriculum",
        policy_kwargs=dict(
            features_extractor_class=OvercookedCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=dict(pi=[], vf=[]),
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

    print(f"[cnn-curriculum] Training on {args.layout}")
    print(f"  Stage 1 (self-play):  {args.self_play_steps:,} steps")
    print(f"  Stage 2 (anneal):     {args.anneal_steps:,} steps "
          f"(BC prob {args.bc_prob_start:.0%} -> {args.bc_prob_end:.0%})")
    print(f"  Snapshot refresh:     every {args.snapshot_freq:,} steps")
    print(f"  Reward shaping:       {args.reward_shaping_start} -> {args.reward_shaping_end}")
    print(f"  Total:                {total_timesteps:,} steps")

    model.learn(total_timesteps=total_timesteps,
                callback=[curriculum_cb, shaping_cb],
                tb_log_name=args.run_name)

    save_path = run_dir / "final_model"
    model.save(str(save_path))
    print(f"[cnn-curriculum] Model saved to {save_path}")

    config = {
        "bc_checkpoint": args.bc_checkpoint,
        "layout": args.layout,
        "horizon": args.horizon,
        "self_play_steps": args.self_play_steps,
        "anneal_steps": args.anneal_steps,
        "snapshot_freq": args.snapshot_freq,
        "bc_prob_start": args.bc_prob_start,
        "bc_prob_end": args.bc_prob_end,
        "total_timesteps": total_timesteps,
        "reward_shaping_start": args.reward_shaping_start,
        "reward_shaping_end": args.reward_shaping_end,
        "features_dim": args.features_dim,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "seed": args.seed,
        "obs_mode": "lossless",
        "partner_obs_mode": "featurized",
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
