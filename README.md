# Cooking Up Collaboration: Training Agents for Human-AI Cooperation

Our goal for this project is to explore state-of-the-art approaches to agent systems in cooperative games and apply them to Overcooked AI to improve generalized cooperative adaptability across changing teammates. The main focus is on developing effective agents that enhance human-AI collaboration.


## About the repo

This repo currently only contains webapp which is modifed UI for visulaizing the agents trained for overcooked ai based on this [repo](https://github.com/HumanCompatibleAI/overcooked-demo).

In order to build and run the development webapp, which includes a deterministic scheduler and helpful debugging logs, run
```bash
cd webapp
./up.sh
```

this should open this webapp under http://localhost, by creating a docker build. **I cannot seem to install it anyother way easily, cause the dependences are a mess!! like really bad took 1 day to fix it.**

Note: I tested this on WSL so Mac guys you might need to test this thing out yourselves.

To spin it down, use
```bash
./down.sh
```