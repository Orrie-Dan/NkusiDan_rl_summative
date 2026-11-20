"""WasteFlow Gymnasium environment (Trello card 04), built on the simulation core in ``simulation.py``.

Design summary (see the "Approved design decisions" and "Spec: reward function & dynamics" cards):
    - 15 collection locations, 3 vehicles, 1 disposal facility (also the depot), 8-hour shift.
    - Event-driven: one idle vehicle decides per step; the simulation then runs until the next vehicle is free.
    - Discrete(17) actions: 0-14 go to location i, 15 go to disposal, 16 wait 10-15 min.
    - Dict observation: {"observation": Box[0, 1] (OBS_DIM,), "action_mask": MultiBinary(17)}, from the
      deciding vehicle's point of view. Every feature is defined once in ``OBS_SPEC``.
    - ``terminated=True`` at shift end (time left is in the observation); never truncated.

Reward (per step, summed over the simulated time the step covers)::

    r = + alpha   * tonnes collected
        - beta    * travel minutes dispatched
        - gamma_o * overflowing bin-minutes
        - delta   * urgent deadlines missed
        - eps_load * tonnes still on trucks at shift end            (terminal)
        - eps_return * vehicles not back at the facility at shift end (terminal)

Each component is reported separately in ``info["reward_components"]``; they sum to the reward exactly.
There is no invalid-action penalty because invalid actions are masked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.simulation import (
    ACTION_DISPOSAL,
    ACTION_WAIT,
    FACILITY,
    N_ACTIONS,
    N_LOCATIONS,
    N_VEHICLES,
    SCENARIOS,
    ScenarioConfig,
    StepOutcome,
    VehicleStatus,
    WasteFlowSim,
)

ENV_ID = "WasteFlow-v0"


# ---------------------------------------------------------------------------------------------------- reward


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Scaled so a typical episode return lands roughly in [-100, 100] (card 14 calibrates).

    Reference magnitudes from the scripted policy in the base scenario: about 18 t collected and 740 travel
    minutes per shift. ``eps_load_per_tonne`` equals ``alpha_per_tonne``, so waste collected but never taken
    to disposal nets zero reward.
    """

    alpha_per_tonne: float = 4.0
    beta_per_travel_min: float = 0.02
    gamma_per_overflow_bin_min: float = 0.02
    delta_per_urgent_missed: float = 2.0
    eps_load_per_tonne: float = 4.0
    eps_per_vehicle_not_returned: float = 2.0


REWARD_COMPONENTS = ("collected", "travel", "overflow", "urgent_missed", "terminal_load", "terminal_not_returned")


def reward_components(out: StepOutcome, rc: RewardConfig) -> dict[str, float]:
    """Signed reward contribution of each term for one step. ``sum(values)`` is the step reward."""
    comps = {
        "collected": rc.alpha_per_tonne * out.kg_collected / 1000.0,
        "travel": -rc.beta_per_travel_min * out.travel_minutes,
        "overflow": -rc.gamma_per_overflow_bin_min * out.overflow_bin_minutes,
        "urgent_missed": -rc.delta_per_urgent_missed * out.urgent_missed,
        "terminal_load": -rc.eps_load_per_tonne * out.load_remaining_kg / 1000.0,
        "terminal_not_returned": -rc.eps_per_vehicle_not_returned * out.vehicles_not_returned,
    }
    return {k: float(v) for k, v in comps.items()}  # plain floats (sim values can be numpy scalars)


# ---------------------------------------------------------------------------------------------- observation

# Fixed normalisation constants. They don't depend on the scenario, so a policy trained on one scenario
# sees the same encoding when it is evaluated on another (generalisation runs).
MAX_FILL_RATE_PER_HOUR = 0.6  # clip for the expected fill rate; high_demand peaks at 0.22 * 1.875 * 1.45 = 0.598
N_ZONES = ScenarioConfig().n_zones  # zone count is part of the fixed layout


@dataclass(frozen=True)
class ObsFeature:
    """One block of the observation vector. ``fn(sim, v)`` returns ``size`` values in [0, 1]."""

    name: str
    size: int
    meaning: str
    encoding: str
    source: str
    cadence: str
    fn: Callable[[WasteFlowSim, int], np.ndarray]


def _others(sim: WasteFlowSim, v: int) -> list[int]:
    """The other vehicles in a fixed order relative to the deciding one: v+1, v+2 (mod 3)."""
    return [(v + k) % N_VEHICLES for k in range(1, N_VEHICLES)]


def _vehicle_pos(sim: WasteFlowSim, u: int) -> int:
    """Where vehicle u is or is heading: its target while busy, else its current node."""
    return int(sim.vehicle_target[u]) if sim.vehicle_target[u] >= 0 else int(sim.vehicle_node[u])


def _max_dist(sim: WasteFlowSim) -> float:
    return float(sim.layout.dist_km.max())


def _taken_by_others(sim: WasteFlowSim, v: int) -> np.ndarray:
    out = np.zeros(N_LOCATIONS)
    for u in _others(sim, v):
        t = sim.vehicle_target[u]
        if 0 <= t < N_LOCATIONS:
            out[t] = 1.0
    return out


def _deadline_left(sim: WasteFlowSim, v: int) -> np.ndarray:
    left = (sim.urgent_deadline - sim.t) / sim.config.urgent_deadline_range_min[1]
    return np.where(sim.urgent_pending, left, 0.0)


def _expected_fill_rate(sim: WasteFlowSim, v: int) -> np.ndarray:
    cfg = sim.config
    rate = sim.layout.base_fill_rate * cfg.demand_multiplier * sim.demand_curve(sim.t)  # per hour, noise-free
    return rate / MAX_FILL_RATE_PER_HOUR


def _other_vehicles(sim: WasteFlowSim, v: int) -> np.ndarray:
    cap = sim.config.vehicle_capacity_kg
    here = int(sim.vehicle_node[v])
    rows = []
    for u in _others(sim, v):
        status = sim.vehicle_status[u]
        rows.append(
            [
                sim.vehicle_load_kg[u] / cap,
                float(status == VehicleStatus.TRAVELLING),
                float(status in (VehicleStatus.SERVICING, VehicleStatus.DISPOSING, VehicleStatus.WAITING)),
                float(status == VehicleStatus.BROKEN_DOWN or sim.vehicle_out_of_service[u]),
                sim.layout.dist_km[here, _vehicle_pos(sim, u)] / _max_dist(sim),
            ]
        )
    return np.asarray(rows).ravel()


OTHER_VEHICLE_FIELDS = ("load_frac", "travelling", "busy_on_site", "broken_down", "dist_from_me")

OBS_SPEC: tuple[ObsFeature, ...] = (
    ObsFeature(
        "time_elapsed", 1, "Fraction of the 8-hour shift already elapsed", "t / shift_minutes, [0, 1]",
        "simulation clock", "every decision", lambda s, v: np.array([s.t / s.config.shift_minutes]),
    ),
    ObsFeature(
        "demand_level", 1, "Current time-of-day demand multiplier", "curve(t) / 1.45 (curve peak), [0, 1]",
        "simulation rule: Gaussian morning peak at 10:30", "every decision",
        lambda s, v: np.array([s.demand_curve(s.t) / 1.45]),
    ),
    ObsFeature(
        "zone_congested", N_ZONES, "Whether each traffic zone is congested", "binary {0, 1}",
        "per-zone two-state Markov chain", "changes every 15-min tick",
        lambda s, v: s.zone_congested.astype(float),
    ),
    ObsFeature(
        "bin_fill", N_LOCATIONS, "Fill level of each location's bin", "fraction of bin capacity, [0, 1]",
        "fill model: base rate x demand x curve x noise, capped at 1", "continuous; read at decision time",
        lambda s, v: s.fill.copy(),
    ),
    ObsFeature(
        "urgent_pending", N_LOCATIONS, "Whether a location has an open urgent request", "binary {0, 1}",
        "Poisson arrivals at random locations", "event-driven (arrival / service / miss)",
        lambda s, v: s.urgent_pending.astype(float),
    ),
    ObsFeature(
        "urgent_deadline_left", N_LOCATIONS, "Time left before each urgent request is missed",
        "minutes left / 90 (max deadline), [0, 1]; 0 if no request", "deadline drawn U(45, 90) min at arrival",
        "every decision", _deadline_left,
    ),
    ObsFeature(
        "dist_to_bin", N_LOCATIONS, "Road distance from the deciding vehicle to each location",
        "km / max map distance, [0, 1]", "fixed layout distance matrix", "every decision (vehicle moves)",
        lambda s, v: s.layout.dist_km[s.vehicle_node[v], :N_LOCATIONS] / _max_dist(s),
    ),
    ObsFeature(
        "expected_fill_rate", N_LOCATIONS, "Expected fill rate of each bin right now (noise-free)",
        f"fraction per hour / {MAX_FILL_RATE_PER_HOUR}, clipped to [0, 1]",
        "base rate x scenario demand multiplier x time-of-day curve", "every decision", _expected_fill_rate,
    ),
    ObsFeature(
        "bin_taken_by_other", N_LOCATIONS, "Whether another vehicle is already heading to each location",
        "binary {0, 1}", "vehicle targets", "event-driven (dispatch / arrival)", _taken_by_others,
    ),
    ObsFeature(
        "my_load", 1, "Load on the deciding vehicle", "kg / vehicle capacity, [0, 1]", "vehicle state",
        "changes on collection / disposal", lambda s, v: np.array([s.vehicle_load_kg[v] / s.config.vehicle_capacity_kg]),
    ),
    ObsFeature(
        "my_at_facility", 1, "Whether the deciding vehicle is at the disposal facility / depot", "binary {0, 1}",
        "vehicle position", "changes on arrival", lambda s, v: np.array([float(s.vehicle_node[v] == FACILITY)]),
    ),
    ObsFeature(
        "my_dist_to_facility", 1, "Road distance from the deciding vehicle to the facility",
        "km / max map distance, [0, 1]", "fixed layout distance matrix", "every decision",
        lambda s, v: np.array([s.layout.dist_km[s.vehicle_node[v], FACILITY] / _max_dist(s)]),
    ),
    ObsFeature(
        "my_id", N_VEHICLES, "Which vehicle is deciding", "one-hot", "event queue (first free vehicle)",
        "every decision", lambda s, v: np.eye(N_VEHICLES)[v],
    ),
    ObsFeature(
        "other_vehicles", 5 * (N_VEHICLES - 1),
        "For each other vehicle (order v+1, v+2): " + ", ".join(OTHER_VEHICLE_FIELDS),
        "load / capacity; binary flags; km / max map distance; all [0, 1]", "vehicle state and targets",
        "every decision", _other_vehicles,
    ),
)

OBS_DIM = sum(f.size for f in OBS_SPEC)


def obs_index() -> dict[str, slice]:
    """Agent-input slice of each feature in the observation vector."""
    out, i = {}, 0
    for f in OBS_SPEC:
        out[f.name] = slice(i, i + f.size)
        i += f.size
    return out


def obs_spec_table() -> str:
    """Markdown provenance table for the report: meaning, encoding/range, source, cadence, input indices."""
    rows = ["| Feature | Dims | Indices | Meaning | Encoding / range | Source / rule | Update cadence |",
            "|---|---|---|---|---|---|---|"]
    for f, sl in zip(OBS_SPEC, obs_index().values()):
        idx = f"{sl.start}" if f.size == 1 else f"{sl.start}-{sl.stop - 1}"
        rows.append(f"| `{f.name}` | {f.size} | {idx} | {f.meaning} | {f.encoding} | {f.source} | {f.cadence} |")
    return "\n".join(rows)


def build_observation(sim: WasteFlowSim, v: int) -> np.ndarray:
    parts = [np.asarray(f.fn(sim, v), dtype=np.float64).reshape(f.size) for f in OBS_SPEC]
    return np.clip(np.concatenate(parts), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- env


def action_name(action: int) -> str:
    if action == ACTION_DISPOSAL:
        return "Disposal"
    if action == ACTION_WAIT:
        return "Wait"
    return f"Bin {action}"


class WasteFlowEnv(gym.Env):
    """Gymnasium wrapper around :class:`WasteFlowSim`.

    Args:
        scenario: scenario name from ``SCENARIOS`` or a ``ScenarioConfig``.
        reward_config: reward weights (defaults to ``RewardConfig()``).
        verbose: print one line per decision (for the demo video's terminal output).
        render_mode: ``"human"`` (three.js viewer in the browser, see ``rendering.py``) or ``None``.
        render_options: keyword arguments for ``rendering.HumanRenderer`` (port, speed, open_browser,
            wait_for_client_s).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        scenario: str | ScenarioConfig = "base",
        reward_config: RewardConfig | None = None,
        verbose: bool = False,
        render_mode: str | None = None,
        render_options: dict | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        config = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.sim = WasteFlowSim(config)
        self.reward_config = reward_config or RewardConfig()
        self.verbose = verbose
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._episode_return = 0.0
        self._last_line = ""
        self._last_decision: dict | None = None
        self._render_options = render_options or {}
        self._renderer = None  # created on the first human render

    @property
    def episode_return(self) -> float:
        return self._episode_return

    @property
    def last_decision(self) -> dict | None:
        """The decision that led to the current state (None right after reset); part of the render snapshot."""
        return self._last_decision

    # -------------------------------------------------------------------------------------------------- api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # The simulation gets its own generator, seeded from the env's np_random, so reset(seed=s) is
        # deterministic and reset() without a seed continues the env's random stream.
        self.sim.reset(seed=int(self.np_random.integers(2**63 - 1)))
        self._episode_return = 0.0
        self._last_line = ""
        self._last_decision = None
        obs = self._obs()
        info = {"vehicle": self.sim.deciding_vehicle, "clock": self.sim.clock_str(), "scenario": self.sim.config.name}
        if self.render_mode == "human":
            self._get_renderer().reset(self)
        return obs, info

    def step(self, action):
        sim = self.sim
        if sim.done:
            raise RuntimeError("step() called after the episode terminated; call reset()")
        v = sim.deciding_vehicle
        action = int(action)
        pre = self._decision_context(v, action)

        out = sim.apply_action(action)  # raises ValueError on a masked action
        comps = reward_components(out, self.reward_config)
        reward = float(sum(comps.values()))
        self._episode_return += reward
        terminated = out.done

        info = {
            "vehicle": v,
            "action": action,
            "clock": sim.clock_str(out.start_time),
            "step_minutes": out.end_time - out.start_time,
            "reward_components": comps,
            "kg_collected": out.kg_collected,
            "travel_minutes": out.travel_minutes,
            "overflow_bin_minutes": out.overflow_bin_minutes,
            "urgent_served": out.urgent_served,
            "urgent_missed": out.urgent_missed,
            "breakdowns": out.breakdowns,
        }
        if terminated:
            info["episode_stats"] = asdict(sim.stats)
            info["episode_return"] = self._episode_return

        self._last_line = self._format_line(pre, reward)
        self._last_decision = {
            "vehicle": v,
            "action": action,
            "label": action_name(action),
            "clock": pre["clock"],
            "t": float(out.start_time),
            "reward": round(reward, 4),
            "text": self._last_line,
        }
        if self.verbose:
            print(self._last_line, flush=True)
            if terminated:
                print(self._format_summary(), flush=True)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, info

    def render(self):
        """Push the current state to the three.js viewer (``render_mode="human"``); paces the episode."""
        if self.render_mode == "human":
            self._get_renderer().step(self)

    def save_episode(self, path):
        """Save the episode recorded by the human renderer as a ``wasteflow.episode`` JSON (for replay mode)."""
        if self._renderer is None:
            raise RuntimeError("save_episode needs render_mode='human' and a reset() first")
        return self._renderer.recorder.save(path)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def _get_renderer(self):
        if self._renderer is None:
            from environment.rendering import HumanRenderer  # lazy: training never needs the HTTP server

            self._renderer = HumanRenderer(**self._render_options)
        return self._renderer

    def action_masks(self) -> np.ndarray:
        """Bool mask for the current decision (convenience for baselines and evaluation)."""
        return self._mask().astype(bool)

    # ----------------------------------------------------------------------------------------------- helpers

    def _mask(self) -> np.ndarray:
        if self.sim.done:
            mask = np.zeros(N_ACTIONS, dtype=np.int8)
            mask[ACTION_WAIT] = 1  # keep one valid action so masked max/softmax never sees all -inf
            return mask
        return self.sim.action_mask().astype(np.int8)

    def _obs(self) -> dict[str, np.ndarray]:
        v = self.sim.deciding_vehicle
        # After the shift ends there is no deciding vehicle; the terminal observation uses vehicle 0's view.
        return {"observation": build_observation(self.sim, 0 if v is None else v), "action_mask": self._mask()}

    def _decision_context(self, v: int, action: int) -> dict:
        sim = self.sim
        ctx = {"clock": sim.clock_str(), "v": v, "action": action}
        if action < N_LOCATIONS:
            ctx["fill"] = float(sim.fill[action])
            ctx["urgent"] = bool(sim.urgent_pending[action])
        return ctx

    def _format_line(self, ctx: dict, reward: float) -> str:
        load = self.sim.vehicle_load_kg[ctx["v"]] / self.sim.config.vehicle_capacity_kg
        target = action_name(ctx["action"])
        if "fill" in ctx:
            target += f" (fill {ctx['fill']:.2f}{', urgent' if ctx['urgent'] else ''})"
        # ASCII arrow: the Windows console (cp1252) can't encode a Unicode one.
        return f"t={ctx['clock']} V{ctx['v']} -> {target} | r={reward:+.2f} | load {load:.2f}"

    def _format_summary(self) -> str:
        s = self.sim.stats
        return (
            f"== shift end | return {self._episode_return:+.1f} | collected {s.kg_collected / 1000:.1f} t | "
            f"travel {s.travel_minutes:.0f} min | overflow {s.overflow_bin_minutes:.0f} bin-min | "
            f"urgent {s.urgent_served}/{s.urgent_arrived} served, {s.urgent_missed} missed | "
            f"breakdowns {s.breakdowns} | left on trucks {s.load_remaining_kg / 1000:.1f} t | "
            f"not returned {s.vehicles_not_returned}"
        )


def register_env() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="environment.custom_env:WasteFlowEnv")


register_env()
