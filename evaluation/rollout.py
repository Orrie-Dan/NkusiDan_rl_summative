"""Episode rollouts that turn a policy into per-episode KPI rows (card 07; card 16 builds the full harness on this).

A KPI row is a flat dict (one per episode), so a list of rows goes straight into ``pandas.DataFrame``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from environment.custom_env import REWARD_COMPONENTS, RewardConfig, WasteFlowEnv

# Fixed evaluation seeds shared by every baseline and agent (common random numbers). They are kept apart from
# training seeds, which start at 0. Card 16 owns the final evaluation protocol.
EVAL_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_050))


class Policy(Protocol):
    """Common interface for baselines and trained agents.

    ``act`` receives the env observation (``{"observation", "action_mask"}``) and the env itself, so rule-based
    policies can read the simulation state directly. It must return an action allowed by ``obs["action_mask"]``.
    """

    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: dict, env: WasteFlowEnv) -> int: ...


def run_episode(env: WasteFlowEnv, policy: Policy, seed: int) -> dict:
    """Play one full shift and return its KPI row."""
    policy.reset(seed)
    obs, _ = env.reset(seed=seed)
    comps = dict.fromkeys(REWARD_COMPONENTS, 0.0)
    done, info = False, {}
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, env))
        for k, v in info["reward_components"].items():
            comps[k] += v

    s = info["episode_stats"]
    row = {
        "policy": policy.name,
        "scenario": env.sim.config.name,
        "seed": seed,
        "return": info["episode_return"],
        "tonnes_collected": s["kg_collected"] / 1000.0,
        "tonnes_disposed": s["kg_disposed"] / 1000.0,
        "travel_minutes": s["travel_minutes"],
        "overflow_bin_minutes": s["overflow_bin_minutes"],
        "urgent_arrived": s["urgent_arrived"],
        "urgent_served": s["urgent_served"],
        "urgent_missed": s["urgent_missed"],
        "urgent_service_rate": s["urgent_served"] / s["urgent_arrived"] if s["urgent_arrived"] else 1.0,
        "breakdowns": s["breakdowns"],
        "decisions": s["decisions"],
        "bin_visits": s["bin_visits"],
        "disposal_trips": s["disposal_trips"],
        "waits": s["waits"],
        "tonnes_left_on_trucks": s["load_remaining_kg"] / 1000.0,
        "vehicles_not_returned": s["vehicles_not_returned"],
    }
    row.update({f"r_{k}": v for k, v in comps.items()})
    return row


def run_policy(
    policy: Policy,
    scenario: str = "base",
    seeds: Iterable[int] = EVAL_SEEDS,
    reward_config: RewardConfig | None = None,
) -> list[dict]:
    """KPI rows for ``policy`` on every seed in ``seeds``."""
    env = WasteFlowEnv(scenario, reward_config=reward_config)
    return [run_episode(env, policy, seed) for seed in seeds]
