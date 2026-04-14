# Ravioli

A high-fidelity Overcooked-style simulator with three main entrypoints:

- `simulator.py`: run the local simulator, either interactively or headless.
- `ravioli/trainer.py`: train an imitation-learning policy from exported trajectory JSONL files.
- `live_bridge.py`: receive live Overcooked state over HTTP and optionally drive agents from that stream.

## Setup

Commands below assume your working directory is `simulator/`.

```powershell
python -m pip install -r requirements.txt
```

Download training data to `simulator/exports` from [Google drive](https://drive.google.com/file/d/1Oj_uUPIg2dHeH4TyiZtCeDAaOnjqyq99/view?usp=drive_link).

## `simulator.py`

`simulator.py` is the main game runner. It can open the normal game window, run headless for fast simulation, and optionally write trajectory exports to `ravioli/exports/*.jsonl`.

Common agent ids:

- `human`
- `random`
- `auto`
- `auto_onion`
- `auto_soup`
- `imitation`

Examples:

```powershell
python simulator.py
```

Opens the simulator window and shows the in-game menu. Use this when you want to choose agents and level interactively.

```powershell
python simulator.py --player-1 human --player-2 auto --level level_1_1
```

Starts immediately on `level_1_1` with a human on player 1 and the scripted `auto` agent on player 2.

```powershell
python simulator.py --player-1 auto --player-2 auto --level level_1_2 --headless
```

Runs a fully automated headless simulation with no window. This is useful for generating trajectories quickly.

Use `--no-export-trajectories` to skip export.

## `ravioli/trainer.py`

`ravioli/trainer.py` trains a behavior-cloning policy from exported trajectories. By default it reads JSONL files from `ravioli/exports/`, trains either an MLP or LSTM policy, and writes artifacts under `ravioli/training_runs/`.

Typical outputs:

- `best.pt`: best checkpoint for the run
- `last.pt`: last checkpoint from training
- `metrics.json`: train, validation, and test metrics
- `preprocessing_report.json`: dataset and featurization summary
- `feature_schema.json`: feature layout used by the model

Examples:

```powershell
python ravioli/trainer.py --model mlp
```

Trains an MLP on all JSONL files in `ravioli/exports/` using the default settings.

```powershell
python ravioli/trainer.py --model lstm --seq-len 32 --hidden-dim 256 --num-layers 2
```

Trains an LSTM with a longer temporal window and custom hidden size.

```powershell
python ravioli/trainer.py --model lstm --player-mode single --player-slot 1 --run-name player1_lstm
```

Trains only on player 1 behavior and saves the run under a custom name. This is useful when the two player slots have different roles.

```powershell
python ravioli/trainer.py --data-path ravioli/exports/2026-04-12_15-11-12.jsonl --model mlp --epochs 50
```

Trains from a single export file instead of the whole directory.

```powershell
python ravioli/trainer.py --model lstm --player-slot 2
```

Filters the dataset to data from player 2.

## `live_bridge.py`

`live_bridge.py` starts a small HTTP server that accepts streamed Overcooked state and level payloads. It can store the latest payloads on disk, print periodic status summaries, and optionally run non-human agents live from the incoming state stream.

Endpoints:

- `POST /state`
- `POST /level`
- `GET /health`
- `GET /snapshot`

By default it writes the latest payloads to `ingest/` and live agent traces to `ingest/live_agent_trace.jsonl`.

Examples:

```powershell
python live_bridge.py
```

Starts the HTTP bridge on `127.0.0.1:8765` and writes the latest state and level payloads to `ingest/`.

```powershell
python live_bridge.py --player-1 auto --player-2 auto
```

Starts the bridge and runs scripted agents for both players from the streamed state. `live_bridge.py` only supports non-human agents on the CLI.

After starting the live bridge, immediately launch Overcooked! and set it as the focused window. The live bridge will navigate through the menu and select the current level on the map.

## Quick Workflow

Generate trajectories:

```powershell
python simulator.py --player-1 auto --player-2 auto --headless
```

Train a policy:

```powershell
python ravioli/trainer.py --model lstm
```

Run the trained policy in the local simulator:

```powershell
python simulator.py --player-1 imitation --player-2 auto
```
