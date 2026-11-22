"""WasteFlow environment (Trello cards 03, 04, 07): the event-driven simulation core, the Gymnasium environment
built on it, and the episode rollouts that turn a policy into KPI rows. Sections are marked by banner comments.

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

import heapq
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ====================================================================================================================
# Simulation core (card 03)  (formerly environment/simulation.py)
# ====================================================================================================================
# WasteFlow simulation core (Trello card 03): an event-driven waste-collection simulation, independent of Gym.
#
# The Gymnasium wrapper (card 04, ``custom_env.py``) builds observations and rewards on top of this module.
#
# Model
# -----
# - Nodes 0..14 are collection locations; node 15 is the disposal facility, which also serves as the depot.
#   Node ids equal action ids, so action ``i`` means "drive to node i".
#   Depot and facility share a node because the action set (Discrete(17)) has no separate "return to depot"
#   action. "Returned at shift end" therefore means "at the facility".
# - Event-driven: the simulation advances until a vehicle becomes free, and that vehicle then decides.
#   Several vehicles that free up at the same moment decide one after another, with no time passing between them.
# - Fill levels grow continuously: base rate x demand multiplier x time-of-day curve x a noise factor that is
#   redrawn every tick. Bins are capped at 100 %, and overflow minutes accrue while a bin is full.
# - Urgent requests arrive as a Poisson process at random locations, each with a 45-90 min deadline. A request is
#   served when a vehicle arrives at its location before the deadline; otherwise it counts as missed once.
# - Traffic: each zone runs a two-state (normal / congested) Markov chain stepped every tick, and travel times
#   get lognormal noise.
# - Breakdowns: a small chance per trip; the vehicle is down for 30-90 min, or for the rest of the shift, and keeps
#   its load.
#
# Randomness
# ----------
# All episode randomness comes from one ``np.random.Generator`` seeded in ``reset(seed)``. The map layout (location
# coordinates, zones, bin sizes, base fill rates) is fixed scenario data. It is generated once from
# ``ScenarioConfig.layout_seed`` so that every episode runs on the same 15 locations.
#
# Accounting conventions (relevant for the reward in card 04)
# -----------------------------------------------------------
# - ``travel_minutes`` is charged in the step that dispatches the trip (driving time only, excluding breakdown
#   downtime). This makes the cost of a decision visible in that decision's step.
# - ``kg_collected`` is credited when the vehicle arrives at the bin, which can be in a later step.
# - ``overflow_bin_minutes`` and ``urgent_missed`` accrue over the simulated time each step covers.
# - Nothing that happens after the shift ends is counted.

N_LOCATIONS = 15
N_VEHICLES = 3
FACILITY = 15  # node id of the disposal facility / depot
ACTION_DISPOSAL = 15
ACTION_WAIT = 16
N_ACTIONS = 17


class VehicleStatus(IntEnum):
    IDLE = 0
    TRAVELLING = 1
    SERVICING = 2
    DISPOSING = 3
    BROKEN_DOWN = 4
    WAITING = 5


class _Ev(IntEnum):
    """Event kinds. The int value breaks ties between events at the same time (lower runs first)."""

    TICK = 0
    URGENT_DEADLINE = 1
    URGENT_ARRIVAL = 2
    BREAKDOWN_START = 3
    BREAKDOWN_PERMANENT = 4
    BREAKDOWN_END = 5
    ARRIVE = 6
    FREE = 7


@dataclass(frozen=True)
class ScenarioConfig:
    """All simulation parameters. The defaults are the base scenario; card 14 calibrates them before the freeze."""

    name: str = "base"
    n_locations: int = N_LOCATIONS
    n_vehicles: int = N_VEHICLES
    shift_minutes: float = 480.0
    shift_start_hour: float = 6.0
    tick_minutes: float = 15.0

    # Layout (fixed across episodes)
    layout_seed: int = 2026
    map_km: float = 12.0
    n_zones: int = 3
    road_factor: float = 1.35  # road distance / straight-line distance
    bin_capacity_kg_range: tuple[float, float] = (400.0, 1200.0)
    # Fraction of the bin per hour, before the demand curve and demand_multiplier.
    fill_rate_range: tuple[float, float] = (0.10, 0.22)

    # Demand. Card 14 calibration (env v1.0, 2026-11-20): at 1.0 urgency-first left almost no overflow, so
    # there was no room for RL on that KPI; at 1.25 random is clearly worst, the two heuristics differ on
    # return, and both still overflow; at 1.5 the two heuristics tied.
    demand_multiplier: float = 1.25
    initial_fill_range: tuple[float, float] = (0.10, 0.55)
    fill_noise_sigma: float = 0.25  # lognormal sigma of the per-tick rate multiplier (mean 1)
    urgent_rate_per_hour: float = 1.0
    urgent_deadline_range_min: tuple[float, float] = (45.0, 90.0)

    # Fleet and operations
    vehicle_capacity_kg: float = 3000.0
    speed_kmh: float = 28.0
    service_base_min: float = 4.0
    service_min_per_tonne: float = 4.0
    disposal_min: float = 20.0
    wait_range_min: tuple[float, float] = (10.0, 15.0)

    # Traffic: per-zone two-state Markov chain, stepped every tick
    traffic_p_congest: float = 0.08
    traffic_p_clear: float = 0.30
    congestion_multiplier: float = 1.6
    travel_noise_sigma: float = 0.15

    # Breakdowns
    breakdown_prob_per_trip: float = 0.02
    breakdown_downtime_range_min: tuple[float, float] = (30.0, 90.0)
    breakdown_rest_of_shift_prob: float = 0.2

    def __post_init__(self) -> None:
        # The action space is fixed at Discrete(17), so the fleet and location counts are fixed too.
        if self.n_locations != N_LOCATIONS or self.n_vehicles != N_VEHICLES:
            raise ValueError(f"WasteFlow v1 is fixed at {N_LOCATIONS} locations and {N_VEHICLES} vehicles")


SCENARIOS: dict[str, ScenarioConfig] = {
    "base": ScenarioConfig(),
    "high_demand": ScenarioConfig(name="high_demand", demand_multiplier=1.875, urgent_rate_per_hour=1.8),  # 1.5x base
    "high_breakdown": ScenarioConfig(name="high_breakdown", breakdown_prob_per_trip=0.08),
    "heavy_traffic": ScenarioConfig(
        name="heavy_traffic", traffic_p_congest=0.25, traffic_p_clear=0.15, congestion_multiplier=2.0
    ),
}


@dataclass(frozen=True)
class Layout:
    """Fixed map data. Arrays indexed by node have length 16 (15 locations + facility)."""

    node_xy: np.ndarray  # (16, 2) km
    node_zone: np.ndarray  # (16,) int
    dist_km: np.ndarray  # (16, 16) road distance
    bin_capacity_kg: np.ndarray  # (15,)
    base_fill_rate: np.ndarray  # (15,) fraction per hour

    @staticmethod
    def generate(cfg: ScenarioConfig) -> Layout:
        rng = np.random.default_rng(cfg.layout_seed)
        loc_xy = rng.uniform(0.0, cfg.map_km, size=(cfg.n_locations, 2))
        facility_xy = np.array([[cfg.map_km / 2, cfg.map_km / 2]])
        node_xy = np.vstack([loc_xy, facility_xy])
        node_zone = np.minimum((node_xy[:, 0] / cfg.map_km * cfg.n_zones).astype(int), cfg.n_zones - 1)
        diff = node_xy[:, None, :] - node_xy[None, :, :]
        dist_km = np.sqrt((diff**2).sum(-1)) * cfg.road_factor
        bin_capacity_kg = rng.uniform(*cfg.bin_capacity_kg_range, size=cfg.n_locations)
        base_fill_rate = rng.uniform(*cfg.fill_rate_range, size=cfg.n_locations)
        for arr in (node_xy, node_zone, dist_km, bin_capacity_kg, base_fill_rate):
            arr.setflags(write=False)
        return Layout(node_xy, node_zone, dist_km, bin_capacity_kg, base_fill_rate)


@dataclass
class StepOutcome:
    """What happened during one decision step: from one decision until the next decision or the shift end."""

    vehicle: int
    action: int
    start_time: float
    end_time: float
    kg_collected: float = 0.0
    kg_disposed: float = 0.0
    travel_minutes: float = 0.0
    overflow_bin_minutes: float = 0.0
    urgent_served: int = 0
    urgent_missed: int = 0
    breakdowns: int = 0
    done: bool = False
    # Filled in only on the final step (done=True):
    load_remaining_kg: float = 0.0
    vehicles_not_returned: int = 0


@dataclass
class EpisodeStats:
    kg_collected: float = 0.0
    kg_disposed: float = 0.0
    travel_minutes: float = 0.0
    overflow_bin_minutes: float = 0.0
    urgent_arrived: int = 0
    urgent_served: int = 0
    urgent_missed: int = 0
    breakdowns: int = 0
    decisions: int = 0
    bin_visits: int = 0
    disposal_trips: int = 0
    waits: int = 0
    load_remaining_kg: float = 0.0
    vehicles_not_returned: int = 0
    extra: dict = field(default_factory=dict)


class WasteFlowSim:
    """Event-driven simulation. Typical loop::

        sim = WasteFlowSim(SCENARIOS["base"])
        sim.reset(seed=0)
        while not sim.done:
            action = policy(sim)                 # for vehicle sim.deciding_vehicle; respect sim.action_mask()
            outcome = sim.apply_action(action)
    """

    def __init__(self, config: ScenarioConfig | None = None) -> None:
        self.config = config or SCENARIOS["base"]
        self.layout = Layout.generate(self.config)
        self._rng: np.random.Generator | None = None
        self._step: StepOutcome | None = None

    # ------------------------------------------------------------------ public state (read-only by convention)

    @property
    def deciding_vehicle(self) -> int | None:
        """Vehicle that must act next, or None once the shift has ended."""
        return None if self.done or not self._idle else self._idle[0]

    @property
    def done(self) -> bool:
        return self.t >= self.config.shift_minutes

    @property
    def urgent_pending(self) -> np.ndarray:
        return np.isfinite(self.urgent_deadline)

    @property
    def zone_congested(self) -> np.ndarray:
        return self._congested.copy()

    def clock_str(self, t: float | None = None) -> str:
        minutes = self.config.shift_start_hour * 60 + (self.t if t is None else t)
        return f"{int(minutes // 60):02d}:{int(minutes % 60):02d}"

    # ------------------------------------------------------------------ episode control

    def reset(self, seed: int | None = None) -> None:
        cfg = self.config
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        n, m = cfg.n_locations, cfg.n_vehicles

        self.t = 0.0
        self.fill = rng.uniform(*cfg.initial_fill_range, size=n)
        self.urgent_deadline = np.full(n, np.inf)
        self._fill_noise = self._draw_fill_noise()
        self._congested = np.zeros(cfg.n_zones, dtype=bool)

        self.vehicle_node = np.full(m, FACILITY, dtype=int)  # current node, or origin while travelling
        self.vehicle_target = np.full(m, -1, dtype=int)  # destination node while busy, else -1
        self.vehicle_load_kg = np.zeros(m)
        self.vehicle_status = np.full(m, VehicleStatus.IDLE, dtype=int)
        self.vehicle_free_at = np.zeros(m)  # planned time the vehicle frees up (inf if out of service)
        self.vehicle_out_of_service = np.zeros(m, dtype=bool)
        # Last trip of each vehicle, for rendering only (no effect on dynamics). The vehicle leaves
        # ``route_from`` at ``route_depart_at`` and has covered ``route_end_frac`` of the way to ``route_to`` at
        # ``route_arrive_at``: 1.0 on arrival (time includes breakdown downtime), less if it broke down for good.
        self.route_from = np.full(m, FACILITY, dtype=int)
        self.route_to = np.full(m, FACILITY, dtype=int)
        self.route_depart_at = np.zeros(m)
        self.route_arrive_at = np.zeros(m)
        self.route_end_frac = np.ones(m)

        self.stats = EpisodeStats()
        self._events: list[tuple[float, int, int, int]] = []
        self._seq = 0
        self._idle: list[int] = list(range(m))
        self._push(cfg.tick_minutes, _Ev.TICK, -1)
        self._schedule_next_urgent()

    def action_mask(self, vehicle: int | None = None) -> np.ndarray:
        """Valid actions for ``vehicle`` (default: the deciding vehicle), as a bool array of length 17.

        Blocks locations already targeted by another vehicle, locations whose current contents exceed the
        vehicle's remaining capacity, waiting when the vehicle is full and disposal when it is empty.
        At least one action is always valid: an empty vehicle can wait, and a non-empty one can dispose.
        """
        v = self.deciding_vehicle if vehicle is None else vehicle
        if v is None:
            raise RuntimeError("no deciding vehicle: the shift has ended")
        cfg = self.config
        mask = np.zeros(N_ACTIONS, dtype=bool)
        remaining = cfg.vehicle_capacity_kg - self.vehicle_load_kg[v]
        others = np.arange(cfg.n_vehicles) != v
        taken = self.vehicle_target[others]
        bin_kg = self.fill * self.layout.bin_capacity_kg
        mask[: cfg.n_locations] = bin_kg <= remaining + 1e-9
        mask[taken[(taken >= 0) & (taken < cfg.n_locations)]] = False
        mask[ACTION_DISPOSAL] = self.vehicle_load_kg[v] > 1e-9
        mask[ACTION_WAIT] = remaining > 1e-9
        return mask

    def apply_action(self, action: int) -> StepOutcome:
        """Apply ``action`` for the deciding vehicle, then advance to the next decision point or the shift end."""
        v = self.deciding_vehicle
        if v is None:
            raise RuntimeError("apply_action called after the shift ended; call reset()")
        action = int(action)
        if not 0 <= action < N_ACTIONS or not self.action_mask(v)[action]:
            raise ValueError(f"invalid action {action} for vehicle {v} at t={self.t:.1f}")

        self._idle.pop(0)
        self._step = StepOutcome(vehicle=v, action=action, start_time=self.t, end_time=self.t)
        self.stats.decisions += 1
        self._dispatch(v, action)
        self._advance()

        out = self._step
        out.end_time = self.t
        out.done = self.done
        if out.done:
            self._finalize(out)
        self._step = None
        return out

    # ------------------------------------------------------------------ dispatch and travel

    def travel_time(self, a: int, b: int) -> float:
        """Sample a travel time in minutes from node a to node b under the current traffic."""
        cfg = self.config
        if a == b:
            return 0.0
        zones = self.layout.node_zone
        mult = np.where(self._congested[[zones[a], zones[b]]], cfg.congestion_multiplier, 1.0).mean()
        noise = self._rng.lognormal(-0.5 * cfg.travel_noise_sigma**2, cfg.travel_noise_sigma)
        return float(self.layout.dist_km[a, b] / cfg.speed_kmh * 60.0 * mult * noise)

    def _dispatch(self, v: int, action: int) -> None:
        cfg, rng, t = self.config, self._rng, self.t
        if action == ACTION_WAIT:
            self.stats.waits += 1
            self.vehicle_status[v] = VehicleStatus.WAITING
            self.vehicle_target[v] = -1
            self.vehicle_free_at[v] = t + rng.uniform(*cfg.wait_range_min)
            self._push(self.vehicle_free_at[v], _Ev.FREE, v)
            return

        dest = action  # 0..14 location, 15 facility
        travel = self.travel_time(self.vehicle_node[v], dest)
        self.vehicle_target[v] = dest
        self.vehicle_status[v] = VehicleStatus.TRAVELLING
        self.route_from[v], self.route_to[v], self.route_depart_at[v] = self.vehicle_node[v], dest, t

        if travel > 0 and rng.random() < cfg.breakdown_prob_per_trip:
            at = t + rng.uniform(0.0, 1.0) * travel
            if rng.random() < cfg.breakdown_rest_of_shift_prob:
                self._step.travel_minutes += at - t
                self.stats.travel_minutes += at - t
                self.vehicle_free_at[v] = np.inf
                self.route_arrive_at[v], self.route_end_frac[v] = at, (at - t) / travel
                self._push(at, _Ev.BREAKDOWN_PERMANENT, v)
                return
            downtime = rng.uniform(*cfg.breakdown_downtime_range_min)
            self._push(at, _Ev.BREAKDOWN_START, v)
            self._push(at + downtime, _Ev.BREAKDOWN_END, v)
        else:
            downtime = 0.0

        self._step.travel_minutes += travel
        self.stats.travel_minutes += travel
        self.vehicle_free_at[v] = np.inf  # unknown until arrival (service time depends on the load picked up)
        self.route_arrive_at[v], self.route_end_frac[v] = t + travel + downtime, 1.0
        self._push(t + travel + downtime, _Ev.ARRIVE, v)

    # ------------------------------------------------------------------ event loop

    def _push(self, time: float, kind: _Ev, arg: int) -> None:
        heapq.heappush(self._events, (float(time), int(kind), self._seq, arg))
        self._seq += 1

    def _advance(self) -> None:
        """Process events until some vehicle is idle (a decision is due) or the shift ends."""
        shift = self.config.shift_minutes
        while not self._idle and self.t < shift:
            next_t = min(self._events[0][0], shift)  # the TICK chain keeps the heap non-empty
            self._evolve_fill(next_t - self.t)
            self.t = next_t
            while self._events and self._events[0][0] <= self.t and self.t < shift:
                _, kind, _, arg = heapq.heappop(self._events)
                self._handle(_Ev(kind), arg)

    def _handle(self, kind: _Ev, arg: int) -> None:
        cfg, rng, t = self.config, self._rng, self.t
        if kind is _Ev.TICK:
            self._fill_noise = self._draw_fill_noise()
            u = rng.random(cfg.n_zones)
            self._congested = np.where(self._congested, u >= cfg.traffic_p_clear, u < cfg.traffic_p_congest)
            self._push(t + cfg.tick_minutes, _Ev.TICK, -1)

        elif kind is _Ev.URGENT_ARRIVAL:
            loc = int(rng.integers(cfg.n_locations))
            if not np.isfinite(self.urgent_deadline[loc]):  # one pending request per location
                self.urgent_deadline[loc] = t + rng.uniform(*cfg.urgent_deadline_range_min)
                self.stats.urgent_arrived += 1
                self._push(self.urgent_deadline[loc], _Ev.URGENT_DEADLINE, loc)
            self._schedule_next_urgent()

        elif kind is _Ev.URGENT_DEADLINE:
            if self.urgent_deadline[arg] <= t:  # still pending (serving resets it to inf)
                self.urgent_deadline[arg] = np.inf
                self._step.urgent_missed += 1
                self.stats.urgent_missed += 1

        elif kind is _Ev.BREAKDOWN_START or kind is _Ev.BREAKDOWN_PERMANENT:
            self.vehicle_status[arg] = VehicleStatus.BROKEN_DOWN
            self._step.breakdowns += 1
            self.stats.breakdowns += 1
            if kind is _Ev.BREAKDOWN_PERMANENT:
                self.vehicle_out_of_service[arg] = True
                self.vehicle_target[arg] = -1  # releases the bin for the other vehicles

        elif kind is _Ev.BREAKDOWN_END:
            self.vehicle_status[arg] = VehicleStatus.TRAVELLING

        elif kind is _Ev.ARRIVE:
            self._arrive(arg)

        elif kind is _Ev.FREE:
            if self.vehicle_status[arg] == VehicleStatus.DISPOSING:
                self._step.kg_disposed += self.vehicle_load_kg[arg]
                self.stats.kg_disposed += self.vehicle_load_kg[arg]
                self.vehicle_load_kg[arg] = 0.0
            self.vehicle_status[arg] = VehicleStatus.IDLE
            self.vehicle_target[arg] = -1
            self._idle.append(arg)

    def _arrive(self, v: int) -> None:
        cfg, t = self.config, self.t
        node = int(self.vehicle_target[v])
        self.vehicle_node[v] = node
        if node == FACILITY:
            self.stats.disposal_trips += 1
            self.vehicle_status[v] = VehicleStatus.DISPOSING
            self.vehicle_free_at[v] = t + cfg.disposal_min
        else:
            self.stats.bin_visits += 1
            cap = self.layout.bin_capacity_kg[node]
            take = min(self.fill[node] * cap, cfg.vehicle_capacity_kg - self.vehicle_load_kg[v])
            take = max(take, 0.0)
            self.fill[node] = max(self.fill[node] - take / cap, 0.0)
            self.vehicle_load_kg[v] += take
            self._step.kg_collected += take
            self.stats.kg_collected += take
            if np.isfinite(self.urgent_deadline[node]):
                self.urgent_deadline[node] = np.inf
                self._step.urgent_served += 1
                self.stats.urgent_served += 1
            self.vehicle_status[v] = VehicleStatus.SERVICING
            self.vehicle_free_at[v] = t + cfg.service_base_min + cfg.service_min_per_tonne * take / 1000.0
        self._push(self.vehicle_free_at[v], _Ev.FREE, v)

    # ------------------------------------------------------------------ continuous dynamics

    def demand_curve(self, t: float) -> float:
        """Time-of-day multiplier on fill rates (morning peak around 10:30); averages about 1 over the shift."""
        hour = self.config.shift_start_hour + t / 60.0
        return 0.55 + 0.9 * np.exp(-0.5 * ((hour - 10.5) / 1.8) ** 2)

    def fill_rate_per_min(self, t: float) -> np.ndarray:
        cfg = self.config
        return self.layout.base_fill_rate * cfg.demand_multiplier * self.demand_curve(t) * self._fill_noise / 60.0

    def _evolve_fill(self, dt: float) -> None:
        if dt <= 0:
            return
        rate = self.fill_rate_per_min(self.t + dt / 2)  # midpoint rule; the curve is smooth over one tick
        new = self.fill + rate * dt
        already_full = self.fill >= 1.0
        crosses = ~already_full & (new > 1.0)
        overflow = np.where(already_full, dt, 0.0)
        overflow[crosses] = dt - (1.0 - self.fill[crosses]) / rate[crosses]
        self.fill = np.minimum(new, 1.0)
        total = float(overflow.sum())
        if self._step is not None:
            self._step.overflow_bin_minutes += total
        self.stats.overflow_bin_minutes += total

    def _draw_fill_noise(self) -> np.ndarray:
        s = self.config.fill_noise_sigma
        return self._rng.lognormal(-0.5 * s**2, s, size=self.config.n_locations)

    def _schedule_next_urgent(self) -> None:
        rate = self.config.urgent_rate_per_hour
        if rate > 0:
            self._push(self.t + self._rng.exponential(60.0 / rate), _Ev.URGENT_ARRIVAL, -1)

    def _finalize(self, out: StepOutcome) -> None:
        returned = (
            (self.vehicle_node == FACILITY)
            & ~self.vehicle_out_of_service
            & (self.vehicle_status != VehicleStatus.TRAVELLING)
            & (self.vehicle_status != VehicleStatus.BROKEN_DOWN)
        )
        out.load_remaining_kg = float(self.vehicle_load_kg.sum())
        out.vehicles_not_returned = int((~returned).sum())
        self.stats.load_remaining_kg = out.load_remaining_kg
        self.stats.vehicles_not_returned = out.vehicles_not_returned


def make_sim(scenario: str = "base", **overrides) -> WasteFlowSim:
    """Build a simulation for a named scenario, optionally overriding config fields (e.g. for tests)."""
    return WasteFlowSim(replace(SCENARIOS[scenario], **overrides))


# ====================================================================================================================
# Gymnasium environment (card 04)  (formerly environment/custom_env.py)
# ====================================================================================================================

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


# ====================================================================================================================
# Episode rollouts (card 07)  (formerly evaluation/rollout.py)
# ====================================================================================================================
# Episode rollouts that turn a policy into per-episode KPI rows (card 07; card 16 builds the full harness on this).
#
# A KPI row is a flat dict (one per episode), so a list of rows goes straight into ``pandas.DataFrame``.

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
