# WasteFlow — Intelligent Waste Operations & Fleet Optimization

Reinforcement-learning dispatch for a simulated waste-collection operation: 15 collection points, 3 vehicles,
one disposal facility and an 8-hour shift. The simulation includes dynamic demand, traffic, vehicle breakdowns
and urgent requests. The agent is trained with four TorchRL algorithms (DQN, REINFORCE, A2C, PPO) and compared
against non-RL baselines.

> Status: project skeleton. Environment, training, evaluation and the three.js visualization are in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv downloads the correct Python (3.12) automatically.

```bash
uv sync
uv run main.py
```

Run the tests:

```bash
uv run pytest
```

## Repository layout

```
project_root/
├── pyproject.toml, uv.lock  dependencies (pinned)
├── README.md
├── main.py                  CLI (train, sweep, evaluate, random-demo, replay) + baselines, evaluation harness, sweep
├── play.py                  run the best agent with GUI + verbose terminal output
├── environment/
│   ├── __init__.py
│   ├── custom_env.py        event-driven simulation core, Gymnasium environment, episode rollouts
│   └── rendering.py         JSON state API + three.js viewer server
├── training/
│   ├── __init__.py          shared RL infrastructure (configs, networks, logging, checkpoints)
│   ├── dqn_training.py      DQN
│   └── pg_training.py       REINFORCE, A2C, PPO
├── models/dqn, models/pg    saved models
├── logs/                    training logs, sweep and evaluation results
├── assets/                  three.js viewer, figures, screenshots
└── tests/                   pytest suite
```
