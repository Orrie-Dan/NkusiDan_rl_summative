"""WasteFlow entry point.

Usage:
    uv run main.py                          # status (the best-agent demo comes in card 21)
    uv run main.py random-demo              # random agent in the three.js viewer (live, in the browser)
    uv run main.py replay logs/demo/random_episode.json   # play back a recorded episode
    uv run main.py train dqn --total-frames 100000 --name my_run   # train (flags: training/dqn_training.py)
    uv run main.py train reinforce --total-frames 100000           # (flags: training/pg_training.py)
    uv run main.py --help                   # list commands
"""

import argparse
import sys

COMMANDS = {
    "train": "Train one algorithm (card 10-13)",
    "sweep": "Run the 40-config hyperparameter sweep (card 15)",
    "evaluate": "Evaluate agents and baselines on fixed seeds (card 16)",
    "plots": "Regenerate report figures from logs/ (card 20)",
    "random-demo": "Run a random agent in the three.js viewer (card 05)",
    "replay": "Play back a recorded episode JSON in the three.js viewer (card 05)",
}


def print_status() -> None:
    import gymnasium
    import tensordict
    import torch
    import torchrl

    print("WasteFlow - Intelligent Waste Operations & Fleet Optimization")
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchrl {torchrl.__version__} "
          f"| tensordict {tensordict.__version__} | gymnasium {gymnasium.__version__}")
    print("Available now: random-demo, replay. The trained-agent demo is not implemented yet (card 21).")


def random_demo(args: argparse.Namespace) -> int:
    from baselines.heuristics import RandomPolicy
    from environment.custom_env import WasteFlowEnv

    env = WasteFlowEnv(
        args.scenario,
        verbose=True,
        render_mode="human",
        render_options={"port": args.port, "speed": args.speed, "open_browser": not args.no_browser},
    )
    policy = RandomPolicy()
    policy.reset(args.seed)
    try:
        obs, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(policy.act(obs, env))
        if args.record:
            print(f"episode saved to {env.save_episode(args.record)}")
        if args.hold > 0:
            import time

            print(f"keeping the viewer up for {args.hold:.0f} s (Ctrl+C to quit)")
            time.sleep(args.hold)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


def replay(args: argparse.Namespace) -> int:
    import time
    import webbrowser

    from environment.rendering import ViewerServer

    server = ViewerServer(port=args.port, replay_path=args.path)
    url = server.replay_url
    print(f"WasteFlow replay of {args.path}: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="WasteFlow RL summative")
    sub = parser.add_subparsers(dest="command")
    parsers = {name: sub.add_parser(name, help=help_text) for name, help_text in COMMANDS.items()}

    demo = parsers["random-demo"]
    demo.add_argument("--scenario", default="base")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--speed", type=float, default=10.0, help="simulated minutes per real second")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--record", help="save the episode JSON here (for replay mode)")
    demo.add_argument("--hold", type=float, default=30.0, help="seconds to keep the viewer up after the shift")
    demo.add_argument("--no-browser", action="store_true", help="don't open a browser (open the URL yourself)")

    rep = parsers["replay"]
    rep.add_argument("path", help="episode JSON written by random-demo --record")
    rep.add_argument("--port", type=int, default=8766)
    rep.add_argument("--no-browser", action="store_true")

    tr = parsers["train"]
    tr.add_argument("algo", choices=["dqn", "reinforce"], help="algorithm (A2C/PPO arrive with cards 12-13)")
    tr.add_argument("flags", nargs=argparse.REMAINDER, help="trainer flags, e.g. --total-frames 100000 --seed 1")

    args = parser.parse_args()
    if args.command is None:
        print_status()
        return 0
    if args.command == "random-demo":
        return random_demo(args)
    if args.command == "replay":
        return replay(args)
    if args.command == "train":
        if args.algo == "dqn":
            from training import dqn_training

            return dqn_training.main(args.flags)
        from training import pg_training

        return pg_training.main([args.algo, *args.flags])

    print(f"'{args.command}' is not implemented yet: {COMMANDS[args.command]}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
