"""
Alternating PPO fine-tuning from BC initialization.

This trains two role-specific PPO policies:
  - player 0 policy trains while player 1 is frozen
  - player 1 policy trains while player 0 is frozen

The frozen partner is refreshed after each update chunk.  Optional BC and older
snapshot partners keep the policies from overfitting only to the latest partner.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.bc_init_utils import load_checkpoint_state_dict, transfer_bc_mlp_to_sb3_policy
from rl.callbacks import (
    BestModelCheckpoint,
    EpisodeRewardLoggerCallback,
    LinearRewardShapingCallback,
)
from rl.env_utils import OvercookedRLWrapper
from train_ppo import FrozenSB3Partner, _load_bc_config, load_bc_partner


class PartnerPool(nn.Module):
    """Episode-level weighted partner sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[tuple[str, nn.Module, float]] = []
        self.active_partner: nn.Module | None = None

    def set_candidates(self, candidates: list[tuple[str, nn.Module, float]]) -> None:
        self.candidates = [
            (name, partner, float(weight))
            for name, partner, weight in candidates
            if weight > 0.0
        ]
        if not self.candidates:
            raise ValueError("PartnerPool needs at least one positive-weight partner.")
        self.active_partner = self.candidates[0][1]

        has_lstm = any(hasattr(partner, "lstm") for _, partner, _ in self.candidates)
        if has_lstm:
            self.lstm = True
        elif hasattr(self, "lstm"):
            delattr(self, "lstm")

    def on_episode_start(self) -> None:
        total_weight = sum(weight for _, _, weight in self.candidates)
        threshold = random.random() * total_weight
        running = 0.0
        for _, partner, weight in self.candidates:
            running += weight
            if threshold <= running:
                self.active_partner = partner
                return
        self.active_partner = self.candidates[-1][1]

    def _active_is_lstm(self) -> bool:
        return self.active_partner is not None and hasattr(self.active_partner, "lstm")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.active_partner is None:
            raise RuntimeError("PartnerPool has no active partner.")
        if self._active_is_lstm():
            if obs.ndim == 2:
                obs = obs.unsqueeze(1)
            return self.active_partner(obs)
        if obs.ndim == 3:
            obs = obs[:, -1, :]
        return self.active_partner(obs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Role-specific alternating PPO fine-tuning from BC initialization."
    )
    p.add_argument("--bc-checkpoint", type=str, required=True)
    p.add_argument("--bc-config", type=str, default=None)
    p.add_argument("--no-bc-init", action="store_true")
    p.add_argument("--layout", type=str, default="asymmetric_advantages")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")

    p.add_argument("--updates-per-player", type=int, default=5,
                   help="Number of alternating PPO update chunks for each player.")
    p.add_argument("--steps-per-update", type=int, default=50_000,
                   help="PPO timesteps per player update chunk.")
    p.add_argument("--start-player", type=int, default=0, choices=(0, 1))

    p.add_argument("--bc-partner-prob", type=float, default=0.10,
                   help="Approximate probability of training against the original BC partner.")
    p.add_argument("--old-snapshot-prob", type=float, default=0.20,
                   help="Approximate probability mass assigned to older partner snapshots.")
    p.add_argument("--snapshot-pool-size", type=int, default=4,
                   help="How many frozen snapshots per player to keep.")

    p.add_argument("--reward-shaping-start", type=float, default=1.0,
                   help="Dense shaping is useful here because hard layouts may get no sparse deliveries initially.")
    p.add_argument("--reward-shaping-end", type=float, default=0.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--reward-transform", type=str, default="symlog",
                   choices=OvercookedRLWrapper.REWARD_TRANSFORMS)

    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--outdir", type=str, default="trained_models/rl")
    p.add_argument("--run-name", type=str, default="ppo_alternate")
    p.add_argument("--tensorboard-log", type=str, default="./logs/ppo_alternate")
    p.add_argument("--export-agent-prefix", type=str, default=None,
                   help="Optional webapp export prefix; creates <prefix>_p0 and <prefix>_p1.")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(
    args: argparse.Namespace,
    player_idx: int,
    partner_pools: list[PartnerPool],
) -> DummyVecEnv:
    def _make(idx: int):
        def _init():
            env = OvercookedRLWrapper(
                layout_name=args.layout,
                gym_partner=partner_pools[idx],
                horizon=args.horizon,
                planner_cache_dir=args.planner_cache_dir,
                reward_shaping_coef=args.reward_shaping_start,
                reward_clip=args.reward_clip,
                reward_transform=args.reward_transform,
                player_idx=player_idx,
            )
            return Monitor(env)
        return _init

    return DummyVecEnv([_make(i) for i in range(args.n_envs)])


def _make_probe_env(args: argparse.Namespace) -> OvercookedRLWrapper:
    return OvercookedRLWrapper(
        layout_name=args.layout,
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        reward_transform=args.reward_transform,
    )


def _build_model(
    args: argparse.Namespace,
    env: DummyVecEnv,
    bc_hidden: list[int],
    seed: int,
) -> PPO:
    return PPO(
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
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        seed=seed,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=dict(net_arch=dict(pi=bc_hidden, vf=bc_hidden)),
    )


def _init_from_bc(
    model: PPO,
    bc_state_dict: dict[str, Any] | None,
    bc_hidden: list[int],
    label: str,
) -> None:
    if bc_state_dict is None:
        print(f"[bc-init:{label}] skipped")
        return
    report = transfer_bc_mlp_to_sb3_policy(model.policy, bc_state_dict, hidden_dims=bc_hidden)
    print(
        f"[bc-init:{label}] transferred {report['loaded_count']} tensors "
        f"({report['skipped_count']} skipped)"
    )
    for skipped in report["skipped_keys"]:
        print(f"  SKIP {label}: {skipped}")


def _snapshot(model: PPO) -> FrozenSB3Partner:
    snap = FrozenSB3Partner(model.policy)
    snap.eval()
    return snap


def _trim_snapshots(snapshots: list[FrozenSB3Partner], max_count: int) -> list[FrozenSB3Partner]:
    max_count = max(1, int(max_count))
    return snapshots[-max_count:]


def _partner_candidates(
    partner_id: int,
    snapshots: dict[int, list[FrozenSB3Partner]],
    bc_partner: nn.Module,
    bc_prob: float,
    old_snapshot_prob: float,
) -> list[tuple[str, nn.Module, float]]:
    bc_prob = min(max(float(bc_prob), 0.0), 1.0)
    old_snapshot_prob = min(max(float(old_snapshot_prob), 0.0), 1.0 - bc_prob)

    partner_snaps = snapshots[partner_id]
    latest = partner_snaps[-1]
    older = partner_snaps[:-1]

    candidates: list[tuple[str, nn.Module, float]] = []
    if bc_prob > 0.0:
        candidates.append(("bc", bc_partner, bc_prob))

    if older:
        latest_weight = max(0.0, 1.0 - bc_prob - old_snapshot_prob)
        candidates.append((f"p{partner_id}_latest", latest, latest_weight))
        per_old = old_snapshot_prob / len(older)
        for idx, snap in enumerate(older):
            candidates.append((f"p{partner_id}_old_{idx}", snap, per_old))
    else:
        candidates.append((f"p{partner_id}_latest", latest, 1.0 - bc_prob))

    return candidates


def _set_partner_pools(
    pools: list[PartnerPool],
    candidates: list[tuple[str, nn.Module, float]],
) -> None:
    for pool in pools:
        pool.set_candidates(candidates)


def _save_models(run_dir: Path, models: dict[int, PPO], filename: str) -> None:
    for player_idx, model in models.items():
        player_dir = run_dir / f"player{player_idx}"
        player_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(player_dir / filename))


def _write_manifest(
    agent_dir: Path,
    checkpoint_name: str,
    args: argparse.Namespace,
    obs_dim: int,
    num_actions: int,
    player_idx: int,
) -> None:
    manifest = {
        "type": "rl_torch",
        "algo": "ppo",
        "policy": "MlpPolicy",
        "checkpoint": checkpoint_name,
        "supported_layouts": [args.layout],
        "input_dim": obs_dim,
        "num_actions": num_actions,
        "player_idx": player_idx,
        "sampling_mode": "sample",
        "sampling_temperature": 1.0,
        "deterministic": False,
        "learning_rate": args.lr,
        "clip_range": args.clip_range,
        "planner_cache_dir": ".cache/overcooked_planners",
    }
    with open(agent_dir / "agent_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _export_webapp_agents(
    args: argparse.Namespace,
    run_dir: Path,
    obs_dim: int,
    num_actions: int,
) -> None:
    if not args.export_agent_prefix:
        return
    base_dir = Path("webapp/server/static/assets/agents")
    for player_idx in (0, 1):
        source = run_dir / f"player{player_idx}" / "final_model.zip"
        agent_dir = base_dir / f"{args.export_agent_prefix}_p{player_idx}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source), str(agent_dir / "final_model.zip"))
        _write_manifest(
            agent_dir=agent_dir,
            checkpoint_name="final_model.zip",
            args=args,
            obs_dim=obs_dim,
            num_actions=num_actions,
            player_idx=player_idx,
        )
        print(f"[export] player {player_idx} -> {agent_dir}")


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)

    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")
    if args.updates_per_player < 1:
        raise ValueError("--updates-per-player must be >= 1")
    if args.steps_per_update < 1:
        raise ValueError("--steps-per-update must be >= 1")

    bc_cfg = _load_bc_config(args.bc_checkpoint, args.bc_config)
    bc_hidden = list(bc_cfg.get("mlp_hidden", [64, 64]))

    probe_env = _make_probe_env(args)
    obs_dim = int(probe_env.observation_space.shape[0])
    num_actions = int(probe_env.action_space.n)
    if args.n_envs == 1:
        check_env(probe_env)
    del probe_env

    bc_partner = load_bc_partner(
        checkpoint_path=args.bc_checkpoint,
        input_dim=obs_dim,
        num_actions=num_actions,
        hidden_dims=bc_hidden,
        config_path=args.bc_config,
    )

    pools = {
        0: [PartnerPool() for _ in range(args.n_envs)],
        1: [PartnerPool() for _ in range(args.n_envs)],
    }
    initial_candidates = [("bc", bc_partner, 1.0)]
    _set_partner_pools(pools[0], initial_candidates)
    _set_partner_pools(pools[1], initial_candidates)

    envs = {
        0: _make_env(args, player_idx=0, partner_pools=pools[0]),
        1: _make_env(args, player_idx=1, partner_pools=pools[1]),
    }

    models = {
        0: _build_model(args, envs[0], bc_hidden, seed=args.seed),
        1: _build_model(args, envs[1], bc_hidden, seed=args.seed + 1),
    }

    bc_state_dict = None
    if not args.no_bc_init:
        bc_state_dict = load_checkpoint_state_dict(args.bc_checkpoint, map_location="cpu")
    _init_from_bc(models[0], bc_state_dict, bc_hidden, "p0")
    _init_from_bc(models[1], bc_state_dict, bc_hidden, "p1")

    snapshots = {
        0: [_snapshot(models[0])],
        1: [_snapshot(models[1])],
    }

    run_dir = Path(args.outdir) / args.run_name
    (run_dir / "player0").mkdir(parents=True, exist_ok=True)
    (run_dir / "player1").mkdir(parents=True, exist_ok=True)

    per_player_total = args.updates_per_player * args.steps_per_update
    callbacks = {
        player_idx: [
            LinearRewardShapingCallback(
                total_timesteps=per_player_total,
                start_coef=args.reward_shaping_start,
                end_coef=args.reward_shaping_end,
            ),
            EpisodeRewardLoggerCallback(),
            BestModelCheckpoint(save_dir=str(run_dir / f"player{player_idx}")),
        ]
        for player_idx in (0, 1)
    }

    print(f"Alternating PPO on {args.layout}")
    print(f"  Updates/player:       {args.updates_per_player}")
    print(f"  Steps/update:         {args.steps_per_update:,}")
    print(f"  Parallel envs:        {args.n_envs}")
    print(f"  BC init:              {'YES' if not args.no_bc_init else 'no'}")
    print(f"  BC hidden dims:       {bc_hidden}")
    print(f"  Partner mix:          BC={args.bc_partner_prob:.0%}, old snapshots={args.old_snapshot_prob:.0%}")
    print(f"  Snapshot pool size:   {args.snapshot_pool_size}")
    print(f"  Reward shaping:       {args.reward_shaping_start} -> {args.reward_shaping_end}")
    print(f"  Reward transform:     {args.reward_transform} (clip={args.reward_clip})")

    update_order = [args.start_player, 1 - args.start_player]
    for update_idx in range(args.updates_per_player):
        for active_player in update_order:
            partner_player = 1 - active_player
            candidates = _partner_candidates(
                partner_id=partner_player,
                snapshots=snapshots,
                bc_partner=bc_partner,
                bc_prob=args.bc_partner_prob,
                old_snapshot_prob=args.old_snapshot_prob,
            )
            _set_partner_pools(pools[active_player], candidates)
            mix = ", ".join(f"{name}:{weight:.2f}" for name, _, weight in candidates)
            print(
                f"[alternate] update {update_idx + 1}/{args.updates_per_player} "
                f"train P{active_player} vs P{partner_player} pool [{mix}]"
            )

            models[active_player].learn(
                total_timesteps=args.steps_per_update,
                callback=callbacks[active_player],
                tb_log_name=f"{args.run_name}_p{active_player}",
                reset_num_timesteps=False,
            )
            snapshots[active_player].append(_snapshot(models[active_player]))
            snapshots[active_player] = _trim_snapshots(
                snapshots[active_player],
                max_count=args.snapshot_pool_size,
            )
            models[active_player].save(
                str(run_dir / f"player{active_player}" / "latest_model.zip")
            )

    _save_models(run_dir, models, "final_model.zip")

    train_config = vars(args).copy()
    train_config.update({
        "bc_hidden": bc_hidden,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "per_player_total_timesteps": per_player_total,
        "total_training_timesteps": 2 * per_player_total,
    })
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    _export_webapp_agents(args, run_dir, obs_dim, num_actions)
    print(f"Saved alternating PPO run to {run_dir}")


if __name__ == "__main__":
    main()
