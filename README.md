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
this should open this webapp under http://localhost, by creating a docker build. **I cannot seem to install it anyother way easily, cause the dependences are a mess!! like really bad took 1 day to fix it.**

Note: I tested this on WSL so Mac guys you might need to test this thing out yourselves.

To spin it down, use
```bash
./down.sh
```

## Collecting Human Data for Imitation Learning

Currently the webapp is configured for Amazon Mechanical Turk (but I modified it for localhost), and to test it out open http://localhost/psiturk. It will automatically start the game and collect trajectory data under `webapp/server/data/trajectories/`.

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

You can download the original data collected in the overcooked AI repo [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57) and then palce it under `data/` in the root folder of this repo.

Right now BC training is setup from csv first (`data/2019_hh_trials.csv`) and supports both models:
- LSTM (default)
- MLP

To train:
```bash
python train_bc.py --model lstm --run-name bc_lstm_v1
```
or
```bash
python train_bc.py --model mlp --run-name bc_mlp_v1
```

Default training uses both player perspectives in one shared policy (`--player-mode both`).

To train single-player BC for ablations:
```bash
python train_bc.py --model lstm --player-mode single --player-idx 0 --run-name bc_lstm_p0
```

### What data is actually used as input?

For each timestep row in csv:
- `state` is parsed into OvercookedState
- then we run Overcooked featurizer (`mdp.featurize_state`)
- in default mode (`--player-mode both`), both player 0 and player 1 views are added as supervised samples
- this gives feature vectors (typically 96 dim)

Label/target is from `joint_action`:
- default: both players are used (shared policy training)
- optional ablation: only one selected player with `--player-mode single --player-idx {0|1}`

Action space is 6 classes:
- `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `INTERACT`

### Are we training all layouts in csv?

Yes. Current pipeline uses all layouts available in the csv and then does split by `trial_id` (not by layout holdout).  
So if your csv has multiple layouts (like `cramped_room`, `coordination_ring`, `asymmetric_advantages`, `random0`, `random3`), all of them are used.

### Training outputs

Each run saves files like:
- `best.pt`
- `last.pt`
- `metrics.json`
- `config.json`
- `split_summary.json`
- `preprocessing_report.json`

Default output dir right now is:
```bash
trained_models/bc/
```

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

### Example manifest (MLP)

```json
{
  "type": "bc_torch",
  "model_type": "mlp",
  "checkpoint": "best.pt",
  "sampling_mode": "sample",
  "sampling_temperature": 1.0,
  "supported_layouts": ["cramped_room", "coordination_ring", "asymmetric_advantages", "random0", "random3"],
  "input_dim": 96,
  "num_actions": 6,
  "mlp_hidden": [256, 128],
  "dropout": 0.1,
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

### Example manifest (LSTM)

```json
{
  "type": "bc_torch",
  "model_type": "lstm",
  "checkpoint": "best.pt",
  "sampling_mode": "sample",
  "sampling_temperature": 1.0,
  "supported_layouts": ["cramped_room", "coordination_ring", "asymmetric_advantages", "random0", "random3"],
  "input_dim": 96,
  "num_actions": 6,
  "seq_len": 20,
  "hidden_dim": 128,
  "num_layers": 1,
  "dropout": 0.1,
  "planner_cache_dir": ".cache/overcooked_planners"
}
```

Note:
- If selected layout is not in `supported_layouts`, BC agent safely returns `STAY`.
- `player_idx` in manifest is optional. If omitted, the agent uses its runtime slot (player 0 or player 1).
- Default inference now matches Overcooked style (`sampling_mode="sample"`), which helps avoid BC-vs-BC symmetry lock at spawn.
- For deterministic behavior, set `sampling_mode` to `"argmax"` (and optionally `deadlock_break_after` > 0).
- Since server requirements now include `torch`, do a docker rebuild if needed.
