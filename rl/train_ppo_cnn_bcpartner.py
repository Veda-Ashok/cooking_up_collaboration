"""
PPO with CNN feature extractor and a frozen BC partner.

The PPO learner sees the lossless (H, W, C) spatial tensor while the
BC partner continues to receive the 96-D featurized vector it was
trained on.  Reward shaping is annealed from full to zero over training.

Usage
-----
python -m rl.train_ppo_cnn_bcpartner \
    --bc-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
    --layout cramped_room \
    --run-name cnn_bc_v1
"""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl.callbacks import LinearRewardShapingCallback
from rl.env_utils import OvercookedRLWrapper
from rl.models.overcooked_cnn import OvercookedCNN


def _load_bc_partner(
    checkpoint_path: str,
    input_dim: int,
    num_actions: int,
) -> torch.nn.Module:
    """Auto-detect MLP vs LSTM from config.json next to the checkpoint."""
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
            input_dim=input_dim,
            num_actions=num_actions,
            default_hidden_dim=cfg.get("hidden_dim", 128),
            default_num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.0),
            device="cpu",
        )
        for p in partner.parameters():
            p.requires_grad = False
        print(f"[cnn-bc] Loaded LSTM BC partner from {checkpoint_path}")
        return partner

    from rl.bc_init_utils import load_checkpoint_state_dict
    from imitation.models.mlp import MLPPolicy

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
    print(f"[cnn-bc] Loaded MLP BC partner from {checkpoint_path}")
    return partner


def main() -> None:
    p = argparse.ArgumentParser(description="PPO + CNN learner + frozen BC partner")
    p.add_argument("--bc-checkpoint", type=str, required=True)
    p.add_argument("--layout", type=str, default="cramped_room")
    p.add_argument("--run-name", type=str, default="cnn_bc_v1")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--planner-cache-dir", type=str, default=".cache/overcooked_planners")
    p.add_argument("--total-timesteps", type=int, default=1000_000)
    p.add_argument("--reward-shaping-start", type=float, default=1.0)
    p.add_argument("--reward-shaping-end", type=float, default=0.0)
    p.add_argument("--features-dim", type=int, default=32,
                   help="Output dim of the CNN feature extractor")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # First create a featurized-only env to get the BC partner's input dim
    temp_env = OvercookedRLWrapper(
        layout_name=args.layout,
        obs_mode="featurized",
        partner_obs_mode="featurized",
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
    )
    bc_input_dim = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    del temp_env

    bc_partner = _load_bc_partner(
        checkpoint_path=args.bc_checkpoint,
        input_dim=bc_input_dim,
        num_actions=num_actions,
    )

    env = OvercookedRLWrapper(
        layout_name=args.layout,
        gym_partner=bc_partner,
        obs_mode="lossless",
        partner_obs_mode="featurized",
        horizon=args.horizon,
        planner_cache_dir=args.planner_cache_dir,
        reward_shaping_coef=args.reward_shaping_start,
    )

    check_env(env)
    print(f"[cnn-bc] Learner obs space: {env.observation_space.shape}")
    print(f"[cnn-bc] BC partner input dim: {bc_input_dim}")

    run_dir = Path("rl/trained_models") / args.run_name
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
        tensorboard_log=f"./logs/ppo_cnn_bc",
        policy_kwargs=dict(
            features_extractor_class=OvercookedCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=dict(pi=[], vf=[]),
            normalize_images=False,
        ),
    )

    callback = LinearRewardShapingCallback(
        total_timesteps=args.total_timesteps,
        start_coef=args.reward_shaping_start,
        end_coef=args.reward_shaping_end,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback,
                tb_log_name=args.run_name)

    save_path = run_dir / "final_model"
    model.save(str(save_path))
    print(f"[cnn-bc] Model saved to {save_path}")

    config = {
        "bc_checkpoint": args.bc_checkpoint,
        "layout": args.layout,
        "horizon": args.horizon,
        "total_timesteps": args.total_timesteps,
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
