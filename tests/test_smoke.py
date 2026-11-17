"""Smoke tests: the pinned stack imports and the required project layout exists."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_dependencies_import():
    import gymnasium  # noqa: F401
    import tensordict  # noqa: F401
    import torch
    import torchrl  # noqa: F401

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0


def test_required_layout_exists():
    required = [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "main.py",
        "play.py",
        "environment/__init__.py",
        "environment/custom_env.py",
        "environment/rendering.py",
        "training/__init__.py",
        "training/dqn_training.py",
        "training/pg_training.py",
        "models/dqn",
        "models/pg",
        "logs",
        "assets",
        "tests",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_project_packages_import():
    import environment.custom_env  # noqa: F401
    import environment.rendering  # noqa: F401
    import training.dqn_training  # noqa: F401
    import training.pg_training  # noqa: F401
