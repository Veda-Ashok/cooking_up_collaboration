# Cooking Up Collaboration: Training Agents for Human-AI Cooperation

Our goal for this project is to explore state-of-the-art approaches to agent systems in cooperative games and apply them to Overcooked AI to improve generalized cooperative adaptability across changing teammates. The main focus is on developing effective agents that enhance human-AI collaboration.

# About the repo

## Installation and Setup

This repo currently only contains webapp which is modifed UI for visulaizing the agents trained for overcooked ai based on this [repo](https://github.com/HumanCompatibleAI/overcooked-demo).

In order to build and run the development webapp, which includes a deterministic scheduler and helpful debugging logs, run

```bash
cd webapp
./up.sh
```

By default this now reuses existing Docker images.  
If you need to rebuild:

```bash
./up.sh --build      # incremental rebuild with cache
./up.sh --rebuild    # full no-cache rebuild
```

If you prefer the old silent/background behavior:

```bash
./up.sh --detach
```

If needed use the below, if Docker's build cache is corrupted

```bash
docker system prune -f
```

this should open this webapp under [http://localhost](http://localhost), by creating a docker build. **I cannot seem to install it anyother way easily, cause the dependences are a mess!! like really bad took 1 day to fix it.**

Note: I tested this on WSL so Mac guys you might need to test this thing out yourselves.

To spin it down, use

```bash
./down.sh
```

## Collecting Human Data for Imitation Learning

Currently the webapp is configured for Amazon Mechanical Turk (but I modified it for localhost), and to test it out open [http://localhost/psiturk](http://localhost/psiturk). It will automatically start the game and collect trajectory data under `webapp/server/data/trajectories/`.

**Note: To configure the game settings, you can check the config.json file under webapp/server/ folder.**

### Trajectory Data Format

Files are saved with the naming convention: `game_{game_id}_{session_id}_{game_type}_{timestamp}.json`

Example filenames:

- `game_0_a3f2b1c4_human-human_20260208_175230.json`
- `game_1_b7e9d2f1_human-ai_20260208_175315.json`

Each JSON file contains:

```json
{
    "uid": "unique_identifier_timestamp",
    "session_id": "a3f2b1c4",
    "game_id": 0,
    "game_type": "human-human",
    "timestamp": "20260208_175230",
    "trajectory": [
        {
            "state": "<OvercookedState JSON>",
            "joint_action": "<tuple of actions>",
            "reward": "<int>",
            "time_left": "<float>",
            "score": "<float>",
            "time_elapsed": "<float>",
            "cur_gameloop": "<int>",
            "layout": "<JSON terrain matrix>",
            "layout_name": "<string>",
            "trial_id": "<string>",
            "player_0_id": "<string>",
            "player_1_id": "<string>",
            "player_0_is_human": "<bool>",
            "player_1_is_human": "<bool>"
        }
    ]
}
```

The `trajectory` array contains all state-action transitions for the entire game session.

### Screenshot Data

Screenshots are automatically saved alongside trajectory data in the `data/screenshots/` directory.

Directory naming convention: `game_{game_id}_{session_id}_{game_type}_{timestamp}/`

Example screenshot directories:

- `game_0_a3f2b1c4_human-human_20260208_175230/`
  - `frame_0000.png` (first frame)
  - `frame_0010.png`
  - `frame_0020.png`
  - `frame_0030.png`
  - ...
  - `frame_0900.png` (last frame)

**Screenshot Sampling:**

- The game runs at 30 FPS (frames per second)
- Screenshots are saved every 10 frames
- This means one screenshot is captured every ~0.33 seconds of gameplay
- A 30-second game will generate approximately 90 screenshots

## Behaviour Cloning

You can download the original data collected in the overcooked AI repo [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57) and then place it under `data/` in the root folder of this repo.

BC training reads from CSV (`data/2019_hh_trials.csv`) and supports two model architectures:

- **MLP** (default)
- **LSTM**

### Quick start (recommended)

Train on a single layout with the deliberately simple defaults:

```bash
python train_bc.py --layout-name cramped_room --run-name bc_mlp_cramped_v1
```

This uses MLP with `[64, 64]` hidden layers, no dropout, no weight decay, plain cross-entropy loss, Adam optimizer, and checkpoints by **validation loss** (not macro-F1). The idea is to get a boring-but-correct baseline first.

To train LSTM instead:

```bash
python train_bc.py --model lstm --layout-name cramped_room --run-name bc_lstm_cramped_v1
```

Default training uses both player perspectives (`--player-mode both`), but each player's trajectory is kept as a **separate sequence** so LSTM windows are never contaminated across players.

To train single-player BC for ablations:

```bash
python train_bc.py --model mlp --layout-name cramped_room --player-mode single --player-idx 0 --run-name bc_mlp_p0
```

### What data is actually used as input?

For each timestep row in CSV:

- `state` is parsed into `OvercookedState`
- Overcooked featurizer (`mdp.featurize_state`) produces per-player feature vectors (typically 96-dim)
- In `--player-mode both`, player 0 and player 1 views are collected into **separate per-player streams** (not interleaved)
- Each stream becomes its own `FeaturizedTrial` with a unique `trial_id` (e.g. `abc_p0`, `abc_p1`) but a shared `split_group_id` so both halves of the same game always land in the same train/val split

Label/target is from `joint_action`:

- Default: both players are used (shared policy training)
- Optional ablation: only one selected player with `--player-mode single --player-idx {0|1}`

Action space is 6 classes: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `INTERACT`

### Layout filtering

By default `--layout-name` is `None`, which means all layouts in the CSV are used. **For a focused BC baseline, always specify a single layout:**

```bash
python train_bc.py --layout-name cramped_room
```

Start with `cramped_room`, then try `asymmetric_advantages` if time permits.

### Trajectory quality filters

Weak/short episodes (wandering, idle, failed games) are filtered out before training:

- `--min-episode-steps 50` (default) -- drop trials shorter than 50 timesteps
- `--min-total-reward 1.0` (default) -- drop trials with total sparse reward < 1.0

These defaults match the spirit of the original [human_aware_rl](https://github.com/HumanCompatibleAI/human_aware_rl) paper which trained BC on cleaned human trajectories.

### Training defaults


| Parameter               | Default | Notes                                  |
| ----------------------- | ------- | -------------------------------------- |
| `--model`               | `mlp`   | Use LSTM only after MLP baseline works |
| `--mlp-hidden`          | `64,64` | Small network, less overfitting        |
| `--dropout`             | `0.0`   | Keep it simple                         |
| `--weight-decay`        | `0.0`   | No regularization by default           |
| `--train-ratio`         | `0.85`  | 85% train, 15% val, no test set        |
| `--val-ratio`           | `0.15`  |                                        |
| `--lr`                  | `1e-3`  |                                        |
| `--batch-size`          | `256`   |                                        |
| `--epochs`              | `50`    |                                        |
| `--early-stop-patience` | `5`     | Early stopping on val loss             |


Checkpointing is by **validation loss** (lower is better), not macro-F1. Macro-F1 can reward action balancing that looks nice in logs but still produces terrible behavior in rollout.

Loss is plain `CrossEntropyLoss()` (unweighted). Optimizer is `Adam`.

### Training outputs

Each run saves:

- `best.pt` -- best checkpoint by val loss
- `last.pt` -- final epoch checkpoint
- `epoch_NNN.pt` -- per-epoch checkpoints (if `--save-every-epoch` is passed)
- `metrics.json`
- `config.json`
- `split_summary.json`
- `preprocessing_report.json`

Default output dir:

```bash
trained_models/bc/
```

### Evaluating BC via rollouts

Classification metrics (accuracy, F1) are not enough -- you need behavior-side evaluation. Use the rollout evaluation script to run BC self-play in the Overcooked environment:

```bash
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 20 \
  --horizon 400
```

This reports:

- **Mean episode reward** and std
- **Delivery rate** -- fraction of episodes with at least one delivery
- **Cook started rate** -- fraction where cooking begins
- **Mean deliveries** per episode
- **Stuck rate** -- fraction of episodes where agents get stuck

**Watch self-play live** with `--render` (press `q` to quit early):

```bash
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 5 \
  --render --fps 10
```

**Save per-episode MP4 videos** with `--save-video`:

```bash
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout-name cramped_room \
  --n-episodes 10 \
  --save-video outputs/bc_cramped_videos \
  --fps 10
```

This creates `outputs/bc_cramped_videos/episode_000.mp4`, `episode_001.mp4`, etc. You can combine `--render` and `--save-video` to watch live and record at the same time.

Save stats to JSON:

```bash
python evaluate_bc_rollouts.py \
  --checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout-name cramped_room \
  --output results/cramped_rollout_stats.json
```

You can also compare `best.pt` vs `last.pt`, or use `--save-every-epoch` during training and evaluate each `epoch_NNN.pt` to pick the best by delivery rate.

## Running BC Models in the Webapp (PyTorch)

Now webapp can load BC PyTorch models directly (no tensorflow conversion needed).

It works by adding an agent folder under:

```bash
webapp/server/static/assets/agents/<YourAgentName>/
```

Put these files inside:

- `best.pt`
- `agent_manifest.json`

Then restart webapp:

```bash
cd webapp
./up.sh
```

Your folder name shows up in agent dropdown automatically.

### Example manifest (MLP, single layout)

```json
{
  "type": "bc_torch",
  "model_type": "mlp",
  "checkpoint": "best.pt",
  "sampling_mode": "sample",
  "sampling_temperature": 1.0,
  "deadlock_break_after": 3,
  "supported_layouts": ["cramped_room"],
  "input_dim": 96,
  "num_actions": 6,
  "mlp_hidden": [64, 64],
  "dropout": 0.0,
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

### Example manifest (LSTM, single layout)

```json
{
  "type": "bc_torch",
  "model_type": "lstm",
  "checkpoint": "best.pt",
  "sampling_mode": "sample",
  "sampling_temperature": 1.0,
  "deadlock_break_after": 3,
  "supported_layouts": ["cramped_room"],
  "input_dim": 96,
  "num_actions": 6,
  "seq_len": 20,
  "hidden_dim": 128,
  "num_layers": 1,
  "dropout": 0.0,
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

Note:

- If selected layout is not in `supported_layouts`, BC agent logs an **error** and returns `STAY`.
- `player_idx` in manifest is optional. If omitted, the agent uses its runtime slot (player 0 or player 1).
- Default inference uses `sampling_mode="sample"`, which helps avoid BC-vs-BC symmetry lock at spawn.
- **Deadlock breaker is on by default** (`deadlock_break_after: 3`). When the agent observes the same state for 3 consecutive steps, it forces a random action from `{UP, DOWN, LEFT, RIGHT, INTERACT}`. This works in both `sample` and `argmax` modes.
- For deterministic behavior, set `sampling_mode` to `"argmax"`.
- LSTM inference now **zero-pads** the history on cold start (first few steps of an episode) instead of duplicating the first observation. History is properly reset between episodes.
- Since server requirements now include `torch`, do a docker rebuild if needed.

## BC -> RL LSTM Fine-Tuning Baseline

There is now a PyTorch RL baseline that starts from a trained BC LSTM checkpoint instead of random init.
It trains a recurrent PPO policy against BC partner policies and supports collaboration shaping through reward shaping.

### Train RL from BC init

```bash
python -m rl.train_lstm_bc_ppo \
  --bc-init-checkpoint trained_models/bc/lstm_20260224_153831/best.pt \
  --partner-checkpoints trained_models/bc/lstm_20260224_153831/best.pt \
  --layout cramped_room \
  --run-name rl_lstm_bc_v1 \
  --sampling-mode sample \
  --sampling-temperature 0.8 \
  --reward-shaping-coef 0.2 \
  --export-agent-name RLTorchLSTM_v1
```

Outputs are saved under:

```bash
trained_models/rl/<run_name>/
```

Key files:

- `best_model.zip`
- `last_model.zip`
- `train_config.json`
- `bc_transfer_report.json`
- `metrics.json` (if eval callback produced logs)

If `--export-agent-name` is set, a webapp-ready folder is created under:

```bash
webapp/server/static/assets/agents/<export_agent_name>/
```

### RL manifest type for webapp

```json
{
  "type": "rl_torch",
  "algo": "recurrent_ppo",
  "policy": "MlpLstmPolicy",
  "checkpoint": "best_model.zip",
  "supported_layouts": ["cramped_room"],
  "input_dim": 96,
  "num_actions": 6,
  "sampling_mode": "sample",
  "sampling_temperature": 0.8,
  "deterministic": false,
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

## PPO Curriculum: Self-Play then BC Anneal

This is the recommended RL training path on top of the BC baseline. Instead of training PPO against the BC partner from step 0 (which often leads to "let-partner-do-everything" collapse), the curriculum has two stages:

1. **Self-play** (stage 1): The PPO learner (player 0) trains against a *frozen snapshot* of its own policy (player 1). The snapshot is refreshed every N steps. This teaches basic game mechanics without coupling to the BC partner's quirks.
2. **BC anneal** (stage 2): Each episode, the partner is randomly chosen -- either the frozen self-play snapshot or the BC partner. The BC probability ramps linearly from `bc_prob_start` (10%) to `bc_prob_end` (80%). This gradually exposes the learner to the BC partner's style.

### Why this helps on top of BC

BC alone produces agents that imitate human actions well in classification but often fail in closed-loop rollout -- they don't recover from novel states, get stuck, or fail to coordinate. PPO with curriculum fixes this:

- **Self-play stage** teaches the agent to actually complete tasks (pick up, cook, deliver) through trial-and-error reward, rather than just mimicking.
- **BC anneal stage** adapts the self-play policy to cooperate with the specific BC partner it will be paired with at deployment, without catastrophically forgetting its basic skills.
- The frozen snapshot (not online self-play) keeps training stable -- both players aren't changing simultaneously.

### Quick start

```bash
python -m rl.train_ppo_curriculum \
  --bc-checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout cramped_room \
  --run-name curriculum_v1
```

### Full options

```bash
python -m rl.train_ppo_curriculum \
  --bc-checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --layout cramped_room \
  --self-play-steps 150000 \
  --anneal-steps 150000 \
  --snapshot-freq 10000 \
  --bc-prob-start 0.10 \
  --bc-prob-end 0.80 \
  --lr 3e-4 \
  --ent-coef 0.01 \
  --reward-shaping-coef 0.0 \
  --run-name curriculum_v1 \
  --export-agent-name PPOCurriculum_v1
```

If rewards stay at zero during early self-play, reduce `--self-play-steps` to 75000 and increase `--anneal-steps` to 225000 to expose the learner to the BC partner sooner.

### Outputs

Saved under `trained_models/rl/<run-name>/`:
- `final_model.zip` -- the trained PPO policy
- `train_config.json` -- all hyperparameters

TensorBoard logs go to `./logs/ppo_curriculum/` by default. Track `curriculum/bc_prob` and `curriculum/phase` to see the schedule.

If `--export-agent-name` is set, a webapp-ready agent folder is created automatically.

### Recommended ablations

Run these three for a clean comparison:

| Ablation | Command |
|----------|---------|
| BC partner only | `python -m rl.train_mlp` (existing simple PPO vs BC) |
| Self-play only | `--anneal-steps 0` (never introduces BC partner) |
| Full curriculum | Default `--self-play-steps 150000 --anneal-steps 150000` |

## PPO with CNN + Lossless Observations (Paper-Style)

This approach moves the PPO learner from the 96-D featurized vector to the **lossless spatial tensor** `(H, W, C)` that the original Overcooked paper uses, while the BC partner continues to receive the 96-D vector it was trained on.

### Why dual observation modes?

The featurized vector is a compact hand-crafted summary -- it works well for BC but may lose spatial information that PPO could exploit (e.g. relative positions, pot contents). The lossless encoding preserves the full grid state as a spatial tensor, which a CNN can learn from directly. Keeping the BC partner on featurized observations means you don't need to retrain BC.

### Architecture

- **Learner**: PPO with `OvercookedCNN` feature extractor (3 conv layers + small MLP), sees `(5, 4, 26)` spatial tensor on `cramped_room`
- **Partner**: Frozen BC model (MLP or LSTM), sees `(96,)` featurized vector
- **Reward shaping**: Annealed linearly from full (`1.0`) to zero (`0.0`) over training via `LinearRewardShapingCallback`

### Quick start

```bash
python -m rl.train_ppo_cnn_bcpartner \
  --bc-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
  --layout cramped_room \
  --run-name cnn_bc_v1
```

### Full options

```bash
python -m rl.train_ppo_cnn_bcpartner \
  --bc-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
  --layout cramped_room \
  --total-timesteps 200000 \
  --reward-shaping-start 1.0 \
  --reward-shaping-end 0.0 \
  --features-dim 32 \
  --lr 3e-4 \
  --ent-coef 0.01 \
  --run-name cnn_bc_v1
```

### Key files

| File | Purpose |
|------|---------|
| `rl/env_utils.py` | `OvercookedRLWrapper` now accepts `obs_mode` and `partner_obs_mode` |
| `rl/models/overcooked_cnn.py` | `OvercookedCNN` feature extractor for SB3 |
| `rl/callbacks.py` | `LinearRewardShapingCallback` for reward shaping anneal |
| `rl/train_ppo_cnn_bcpartner.py` | Training script |

### Outputs

Saved under `rl/trained_models/<run-name>/`:
- `final_model.zip` -- the trained PPO policy
- `train_config.json` -- all hyperparameters

TensorBoard logs go to `./logs/ppo_cnn_bc/`.

### Observation modes in the wrapper

The `OvercookedRLWrapper` now supports:

| `obs_mode` | Shape | Description |
|---|---|---|
| `"featurized"` (default) | `(96,)` | Hand-crafted feature vector, same as BC training |
| `"lossless"` | `(H, W, C)` | Full spatial grid encoding for CNN |

`partner_obs_mode` is always `"featurized"` in practice since BC models expect the vector. Both modes can be set independently:

```python
env = OvercookedRLWrapper(
    "cramped_room",
    gym_partner=bc_partner,
    obs_mode="lossless",           # PPO learner sees spatial tensor
    partner_obs_mode="featurized", # BC partner sees 96-D vector
)
```

### Reward shaping anneal

The `LinearRewardShapingCallback` linearly interpolates `reward_shaping_coef` from `start_coef` to `end_coef` over training. This lets PPO bootstrap off dense potential-based shaping early, then shift to the true sparse objective:

```python
callback = LinearRewardShapingCallback(
    total_timesteps=200_000,
    start_coef=1.0,   # full shaping at start
    end_coef=0.0,     # sparse-only by end
)
```

Track `reward_shaping/coef` in TensorBoard to see the anneal.

### Recommended order

1. Train BC (MLP or LSTM) on `cramped_room` first
2. Try **CNN PPO + BC partner** (this section) -- simplest RL setup
3. If that works, try adding the **curriculum** (self-play then anneal) on top

## Live rollout viewer (with optional MP4 save)

You can run a local live rollout viewer while the game is being simulated, and optionally save video. The viewer auto-detects:
- **MLP vs LSTM** BC partner from `config.json` alongside the partner checkpoint
- **Observation mode** (`featurized` vs `lossless`) from `train_config.json` alongside the RL checkpoint

### CNN PPO + BC partner (lossless learner obs)

```bash
python -m rl.live_rollout \
  --checkpoint rl/trained_models/cnn_bc_v1/final_model.zip \
  --algo ppo \
  --layout cramped_room \
  --partner-checkpoint trained_models/bc/bc_lstm_cramped_v1/best.pt \
  --fps 10
```

The `--obs-mode auto` default reads `train_config.json` and detects `lossless`. You can override with `--obs-mode featurized` or `--obs-mode lossless`.

### Curriculum PPO + BC partner (featurized learner obs)

```bash
python -m rl.live_rollout \
  --checkpoint trained_models/rl/curriculum_v1/final_model.zip \
  --algo ppo \
  --layout cramped_room \
  --partner-checkpoint trained_models/bc/bc_cramped_v1/best.pt \
  --sampling-mode sample \
  --fps 10
```

### Recurrent PPO + LSTM BC partner (legacy)

```bash
python -m rl.live_rollout \
  --checkpoint trained_models/rl/rl_lstm_bc_v1/best_model.zip \
  --algo recurrent_ppo \
  --layout cramped_room \
  --partner-checkpoint trained_models/bc/lstm_20260224_153831/best.pt \
  --sampling-mode sample \
  --sampling-temperature 0.8 \
  --fps 10
```

### Save video without display

```bash
python -m rl.live_rollout \
  --checkpoint rl/trained_models/cnn_bc_v1/final_model.zip \
  --algo ppo \
  --sampling-mode argmax \
  --save-video outputs/rollout.mp4 \
  --no-display
```

