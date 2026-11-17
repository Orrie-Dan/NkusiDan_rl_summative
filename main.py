"""WasteFlow entry point.

Usage:
    uv run main.py              # demo of the best trained agent (card 21; currently prints status)
    uv run main.py --help       # list commands
"""

import argparse
import sys

COMMANDS = {
    "train": "Train one algorithm (card 10-13)",
    "sweep": "Run the 40-config hyperparameter sweep (card 15)",
    "evaluate": "Evaluate agents and baselines on fixed seeds (card 16)",
    "plots": "Regenerate report figures from logs/ (card 20)",
    "random-demo": "Run a random agent in the three.js viewer (card 05)",
}


def print_status() -> None:
    import gymnasium
    import tensordict
    import torch
    import torchrl

    print("WasteFlow - Intelligent Waste Operations & Fleet Optimization")
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchrl {torchrl.__version__} "
          f"| tensordict {tensordict.__version__} | gymnasium {gymnasium.__version__}")
    print("Project skeleton is ready. The demo is not implemented yet (card 21).")


def main() -> int:
    parser = argparse.ArgumentParser(description="WasteFlow RL summative")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args()

    if args.command is None:
        print_status()
        return 0

    print(f"'{args.command}' is not implemented yet: {COMMANDS[args.command]}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
