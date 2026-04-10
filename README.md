# Cooking Up Collaboration: Training Agents for Human-AI Cooperation

Agents for the [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai) cooperative cooking game, trained via **Behavioural Cloning (BC)** on human gameplay and refined with **Proximal Policy Optimisation (PPO)**.

## Project Structure

```
cooking_up_collaboration/
├── train_bc.py                 # BC training (MLP / LSTM)
├── train_ppo.py                # PPO fine-tuning from MLP BC init (featurized obs)
├── train_ppo_cnn.py            # PPO with CNN (lossless spatial obs)
├── train_ppo_lstm.py           # Recurrent PPO from BC LSTM init
├── evaluate_bc_rollouts.py     # BC self-play evaluation & video
├── evaluate_rl_rollouts.py     # RL self-play / RL+BC evaluation & video
├── live_rollout.py             # RL live viewer & self-play
│
├── rl/                         # RL utilities
│   ├── env_utils.py            #   Gymnasium wrapper, vec-env factory
│   ├── callbacks.py            #   Reward shaping, episode logger, best-model ckpt
│   ├── bc_init_utils.py        #   Transfer BC weights to SB3 policies
│   ├── sampling_utils.py       #   Temperature-scaled sampling for SB3
│   ├── self_play_partner.py    #   SB3 model as gym partner (RL self-play)
│   └── models/
│       ├── overcooked_cnn.py   #   CNN feature extractor for SB3
│       └── rl_mlp.py           #   MLP actor-critic for SB3
│
├── imitation/                  # BC data pipeline & model utilities
│   ├── datasets.py             #   Dataset / DataLoader from CSV
│   ├── preprocessing.py        #   Featurization, filtering, splits
│   ├── trainer.py              #   Training loop
│   ├── constants.py
│   ├── metrics.py
│   ├── io_utils.py
│   └── models/
│       ├── mlp.py              #   MLP policy
│       └── lstm.py             #   LSTM policy
│
├── data/                       # Human gameplay data & exploration
│   └── data_exploration.ipynb
├── tests/                      # Environment smoke tests
├── utils/                      # Misc utilities
└── webapp/                     # Flask webapp for human-AI play
```

---

## Installation

```bash
pip install -r requirements.txt
```

Download human gameplay data from [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57) and place it under `data/`.

You can follow the steps in the data_exploration.ipynb under data/ folder to check steps on how to inlcude custom collected data into training the BC models.

---

## 1. Behavioural Cloning (BC)

Train a policy to imitate human gameplay from CSV trajectories.

### Train

```bash
# MLP (recommended starting point)
python train_bc.py --layout-name cramped_room --run-name bc_mlp_cramped_v1

# LSTM
python train_bc.py --model lstm --layout-name cramped_room --run-name bc_lstm_cramped_v1
```

Both player perspectives are used by default (`--player-mode both`).

### Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `--model` | `mlp` | `mlp` or `lstm` |
| `--mlp-hidden` | `64,64` | MLP hidden layer sizes |
| `--layout-name` | `None` | Filter to one layout (recommended) |
| `--epochs` | `50` | |
| `--lr` | `1e-3` | |
| `--early-stop-patience` | `5` | On validation loss |

### Evaluate

```bash
# Headless (stats only)
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 20

# Live viewer
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 5 --render --fps 10

# Save videos
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 10 --save-video outputs/bc_videos --fps 10
```

### Outputs

Saved under `trained_models/bc/<run-name>/`:

- `best.pt` — best checkpoint (by val loss)
- `config.json` — architecture & hyperparameters
- `metrics.json`, `split_summary.json`

---

## 2. PPO Fine-Tuning from BC (`train_ppo.py`)

**Recommended RL path.** Initialises PPO weights from a trained BC MLP checkpoint, then fine-tunes against the BC partner using sparse reward only. The agent alternates between player 0 and player 1 each episode so a single model can play either slot.

### Train

```bash
python train_ppo.py \
  --bc-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout cramped_room \
  --run-name ppo_cramped_v1
```

### Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `--bc-checkpoint` | (required) | BC model for partner & weight init |
| `--no-bc-init` | off | Skip weight init (train from scratch) |
| `--player-idx` | `alternate` | `0`, `1`, or `alternate` |
| `--lr` | `1e-4` | Linear decay to 0 |
| `--clip-range` | `0.15` | |
| `--ent-coef` | `0.02` | |
| `--n-epochs` | `5` | |
| `--reward-shaping-start` | `0.0` | Sparse only (BC init doesn't need shaping) |
| `--total-timesteps` | `1000000` | |
| `--n-envs` | `8` | Parallel environments |

### Why BC init works

The BC policy already knows how to pick up onions, place them, and deliver soup. PPO refines this competent starting point, discovering better coordination strategies that BC's offline imitation misses. Without BC init, PPO from scratch takes much longer and often fails to find the sparse delivery reward.

### Outputs

Saved under `trained_models/rl/<run-name>/`:

- `best_model.zip` — best by rolling mean sparse reward
- `final_model.zip` — end of training
- `train_config.json` — full hyperparameters

TensorBoard: `logs/ppo_curriculum/<run-name>/`

### Evaluate

```bash
# RL self-play (same model as both players)
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --layout-name cramped_room \
  --n-episodes 20

# RL + BC partner
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --partner-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 20

# With live rendering
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --layout-name cramped_room \
  --n-episodes 5 --render --fps 10

# Save videos + stats JSON
python evaluate_rl_rollouts.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --layout-name cramped_room \
  --n-episodes 10 \
  --save-video outputs/rl_videos \
  --output results/rl_stats.json
```

Produces the same metrics as `evaluate_bc_rollouts.py`: mean reward, delivery rate, cook started rate, stuck rate, per-episode breakdowns.

---

## 3. PPO with CNN (`train_ppo_cnn.py`)

Uses lossless spatial observations `(H, W, C)` instead of the 96-D featurized vector. The BC partner still receives featurized observations. Starts from scratch (no weight transfer possible), so reward shaping is annealed from dense to sparse.

### Train

```bash
python train_ppo_cnn.py \
  --bc-checkpoint trained_models/bc/bc_mlp_cramped_v1/best.pt \
  --layout cramped_room \
  --run-name cnn_v1
```

### Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `--features-dim` | `64` | CNN embedding dimension |
| `--reward-shaping-start` | `1.0` | Dense shaping (needed from scratch) |
| `--reward-shaping-end` | `0.0` | Anneals to sparse |
| `--lr` | `3e-4` | Linear decay |
| `--ent-coef` | `0.05` | Higher exploration |
| `--total-timesteps` | `1000000` | |

### Outputs

Same structure as `train_ppo.py`. TensorBoard: `logs/ppo_cnn_curriculum/<run-name>/`

---

## Reward Design

Overcooked gives a **sparse +20 reward** per soup delivery and nothing otherwise. To help RL agents learn, the `overcooked_ai` library provides a **potential-based shaping function** that gives dense feedback for subtask progress (moving toward ingredients, picking up items, starting cooking, etc.). This potential function computes `phi(s') - phi(s)` each step, its positive when the agent moves toward useful states and negative when it moves away.

### BC-initialised PPO (`train_ppo.py`) — sparse reward only

When the policy starts from BC weights, it already knows the full task sequence. We found that adding potential-based shaping to an already-competent policy is **actively harmful**: the shaping function generates negative deltas for any deviation from the optimal path, which overwhelms the +20 delivery reward and causes the policy to degrade. The fix was simple — **disable shaping entirely** (`reward_shaping_start=0.0`) and let PPO refine the BC policy using only the sparse delivery signal.

### CNN PPO (`train_ppo_cnn.py`) — annealed shaping

The CNN policy starts from random weights and must learn visual features from scratch. Here the potential-based shaping **is** useful as it provides the dense gradient signal needed to discover that picking up onions, placing them in pots, and delivering soup are rewarding subtasks. Shaping starts at `1.0` and linearly anneals to `0.0` over training, so the final policy optimises for actual deliveries.

### Reward clipping

Per-step rewards are clipped to `[-5, +5]` by default (`--reward-clip`) to prevent large potential deltas from destabilising training. When shaping is off (BC-init path), this only clips the sparse reward which is already bounded.

### Event penalties

The wrapper also applies small penalties for wasteful actions that the potential function doesn't catch:

| Event | Penalty |
|---|---|
| Catastrophic onion/tomato potting | -0.20 |
| Useless onion/tomato potting | -0.10 |
| Soup drop | -0.30 |

These are scaled by `reward_shaping_coef`, so they're inactive when shaping is off.

---

## 4. Recurrent PPO from BC LSTM (`train_ppo_lstm.py`)

Fine-tunes a RecurrentPPO (LSTM) policy initialised from a BC LSTM checkpoint. Requires `sb3-contrib`.

```bash
python train_ppo_lstm.py \
  --bc-init-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
  --partner-checkpoints trained_models/bc/bc_lstm_cramped_v1/best.pt \
  --layout cramped_room \
  --run-name ppo_lstm_v1
```

---

## 5. Live Rollout Viewer

Watch trained RL agents play in real time with optional video recording.

### RL agent + BC partner

```bash
python live_rollout.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --algo ppo \
  --layout cramped_room \
  --fps 10
```

The partner checkpoint is auto-detected from `train_config.json`. Override with `--partner-checkpoint`.

### RL self-play (same model controls both players)

```bash
python live_rollout.py \
  --checkpoint trained_models/rl/ppo_cramped_v1/best_model.zip \
  --algo ppo \
  --layout cramped_room \
  --self-play \
  --player-idx alternate \
  --fps 10
```

### Options

| Flag | Description |
|---|---|
| `--self-play` | RL model plays both slots |
| `--player-idx` | `0`, `1`, or `alternate` |
| `--sampling-mode` | `sample` (stochastic) or `argmax` (deterministic) |
| `--sampling-temperature` | Logit temperature (default 1.3) |
| `--save-video PATH` | Save MP4 |
| `--no-display` | Headless (combine with `--save-video`) |
| `--obs-mode` | `auto`, `featurized`, or `lossless` |

---

## Player Index Alternation

All PPO training scripts default to `--player-idx alternate`, which randomly assigns the learner to player 0 or player 1 each episode. This means:

- A single trained model can play **either** player slot at evaluation time
- Enables **RL self-play** evaluation (same model as both players)
- The observation encoding from `overcooked_ai` is perspective-aware — each player sees "self" and "partner" features from their own viewpoint

Set `--player-idx 0` or `--player-idx 1` to fix the learner to one slot.

---

## Webapp

A Flask webapp for human-AI play, based on [overcooked-demo](https://github.com/HumanCompatibleAI/overcooked-demo).

```bash
cd webapp
./up.sh            # start
./up.sh --build    # rebuild
./down.sh          # stop
```

Opens at [http://localhost](http://localhost). Collect human trajectories at [http://localhost/psiturk](http://localhost/psiturk).

### Deploying trained agents

Place agent files under `webapp/server/static/assets/agents/<AgentName>/`:

- `best.pt` or `best_model.zip`
- `agent_manifest.json`

Example manifest (BC MLP):

```json
{
  "type": "bc_torch",
  "model_type": "mlp",
  "checkpoint": "best.pt",
  "sampling_mode": "sample",
  "supported_layouts": ["cramped_room"],
  "input_dim": 96,
  "num_actions": 6,
  "mlp_hidden": [64, 64],
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

Example manifest (RL PPO):

```json
{
  "type": "rl_torch",
  "algo": "ppo",
  "policy": "MlpPolicy",
  "checkpoint": "best_model.zip",
  "supported_layouts": ["cramped_room"],
  "sampling_mode": "sample",
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

---

## `rl/` Module Reference

| File | Purpose |
|---|---|
| `env_utils.py` | `OvercookedRLWrapper` (Gymnasium env), `make_overcooked_vec_env` factory. Handles dual obs modes, player-idx alternation, reward shaping, partner integration. |
| `callbacks.py` | `LinearRewardShapingCallback` (anneal shaping coef), `EpisodeRewardLoggerCallback` (per-episode sparse/shaped/total), `BestModelCheckpoint` (save on best sparse reward). |
| `bc_init_utils.py` | `transfer_bc_mlp_to_sb3_policy` (load BC weights into PPO), checkpoint loading utilities. |
| `sampling_utils.py` | Temperature-scaled logit sampling for SB3 policies. |
| `self_play_partner.py` | `SB3SelfPlayPartner` — wraps an SB3 PPO model to act as gym partner. |
| `models/overcooked_cnn.py` | `OvercookedCNN` — CNN feature extractor for lossless observations. |
| `models/rl_mlp.py` | `RLActorCriticPolicy` — MLP actor-critic with separate policy/value heads. |
