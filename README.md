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

Files are saved with the naming convention: `game_{game_id}_{game_type}_{timestamp}.json`

Example filenames:
- `game_0_human-human_20260208_175230.json`
- `game_1_human-ai_20260208_175315.json`

Each JSON file contains:
```json
{
    "uid": "unique_identifier_timestamp",
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