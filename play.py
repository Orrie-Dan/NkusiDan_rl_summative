"""Run the best trained WasteFlow agents with the three.js GUI and verbose terminal output (Trello card 21).

Each algorithm's agent is its best Phase 2 checkpoint, picked by mean return on the 50 model-selection seeds
(``training.best_checkpoint``; the checkpoints are committed in ``models/``, so no training is needed). The episode
runs on a held-out test seed,
every decision is printed, the viewer shows it live, and a KPI summary is printed at the end. Each episode is also
saved as a ``wasteflow.episode`` JSON for ``uv run main.py replay <file>``.

Usage:
    uv run play.py                          # the best DQN agent
    uv run play.py --algo all               # DQN, REINFORCE, A2C and PPO one after another, same seed
    uv run play.py --algo ppo --scenario high_demand --seed 10003 --speed 20
    uv run play.py --algo all --speed 0 --no-browser --hold 0      # headless: terminal output + KPI table only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ALGOS = ("dqn", "reinforce", "a2c", "ppo")
ALGO_LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
# Held-out test seed where the default agent (DQN, dqn_c08_p2_s4) scores closest to its own 50-seed test mean
# (52.6 vs 53.0), so the demo shows a typical shift rather than a cherry-picked one. Rule fixed before choosing.
DEMO_SEED = 10046
SUMMARY_COLUMNS = (
    ("return", "return", "{:+.1f}"),
    ("tonnes_collected", "collected t", "{:.1f}"),
    ("travel_minutes", "travel min", "{:.0f}"),
    ("overflow_bin_minutes", "overflow bin-min", "{:.0f}"),
    ("urgent_service_rate", "urgent served", "{:.0%}"),
    ("tonnes_left_on_trucks", "left on trucks t", "{:.1f}"),
    ("vehicles_not_returned", "not returned", "{:d}"),
)


def parse_args(argv=None) -> argparse.Namespace:
    from environment.custom_env import SCENARIOS

    p = argparse.ArgumentParser(description="Play the best trained WasteFlow agent(s) in the three.js viewer")
    p.add_argument("--algo", choices=(*ALGOS, "all"), default="dqn", help="which agent to play (default: dqn)")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="base")
    p.add_argument("--seed", type=int, default=DEMO_SEED, help=f"episode seed (default: typical test seed {DEMO_SEED})")
    p.add_argument("--speed", type=float, default=10.0, help="simulated minutes per real second (0: unpaced)")
    p.add_argument("--port", type=int, default=8765, help="viewer port (0: any free port)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser (open the printed URL yourself)")
    p.add_argument("--wait", type=float, default=30.0, help="seconds to wait for the browser to connect")
    p.add_argument("--gap", type=float, default=5.0, help="seconds between agents with --algo all")
    p.add_argument("--hold", type=float, default=30.0, help="seconds to keep the viewer up after the last shift")
    p.add_argument("--record-dir", type=Path, default=Path("logs") / "demo",
                   help="where to save each episode JSON (<algo>_episode.json)")
    p.add_argument("--root", type=Path, default=Path("."), help="project root holding models/ and logs/")
    p.add_argument("--quiet", action="store_true", help="no per-decision lines, only the summary")
    return p.parse_args(argv)


def format_summary(rows: list[dict]) -> str:
    header = ["agent", *[label for _, label, _ in SUMMARY_COLUMNS]]
    table = [[r["policy"], *[fmt.format(r[k]) for k, _, fmt in SUMMARY_COLUMNS]] for r in rows]
    widths = [max(len(str(c)) for c in col) for col in zip(header, *table)]
    line = lambda cells: "  ".join(str(c).rjust(w) if i else str(c).ljust(w)  # noqa: E731
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([line(header), line(["-" * w for w in widths]), *map(line, table)])


def play(args: argparse.Namespace) -> list[dict]:
    """Play the requested agent(s); returns one KPI row per agent (see ``environment.custom_env.run_episode``)."""
    from environment.custom_env import run_episode, WasteFlowEnv
    from training import Agent, best_checkpoint

    algos = ALGOS if args.algo == "all" else (args.algo,)
    agents = []
    for algo in algos:  # load everything first, so a missing checkpoint fails before the viewer opens
        path = best_checkpoint(algo, args.root)
        agents.append(Agent.load(path, name=f"{ALGO_LABEL[algo]} ({path.parent.name})"))

    env = WasteFlowEnv(args.scenario, verbose=not args.quiet, render_mode="human",
                       render_options={"port": args.port, "speed": args.speed, "open_browser": not args.no_browser,
                                       "wait_for_client_s": args.wait})
    rows = []
    try:
        for i, agent in enumerate(agents):
            if i and args.gap > 0:
                time.sleep(args.gap)
            print(f"\n=== {agent.name}: {agent.mode} policy, {agent.frames:,} training frames | "
                  f"scenario {args.scenario}, seed {args.seed}", flush=True)
            rows.append(run_episode(env, agent, args.seed))
            if args.record_dir:
                print(f"episode saved to {env.save_episode(args.record_dir / f'{agent.algo}_episode.json')}")
        print(f"\nKPI summary ({args.scenario}, seed {args.seed}):\n{format_summary(rows)}", flush=True)
        if args.hold > 0:
            print(f"keeping the viewer up for {args.hold:.0f} s (Ctrl+C to quit)", flush=True)
            time.sleep(args.hold)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return rows


def main(argv=None) -> int:
    play(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
