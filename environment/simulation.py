"""WasteFlow simulation core (Trello card 03): an event-driven waste-collection simulation, independent of Gym.

The Gymnasium wrapper (card 04, ``custom_env.py``) builds observations and rewards on top of this module.

Model
-----
- Nodes 0..14 are collection locations; node 15 is the disposal facility, which also serves as the depot.
  Node ids equal action ids, so action ``i`` means "drive to node i".
  Depot and facility share a node because the action set (Discrete(17)) has no separate "return to depot"
  action. "Returned at shift end" therefore means "at the facility".
- Event-driven: the simulation advances until a vehicle becomes free, and that vehicle then decides.
  Several vehicles that free up at the same moment decide one after another, with no time passing between them.
- Fill levels grow continuously: base rate x demand multiplier x time-of-day curve x a noise factor that is
  redrawn every tick. Bins are capped at 100 %, and overflow minutes accrue while a bin is full.
- Urgent requests arrive as a Poisson process at random locations, each with a 45-90 min deadline. A request is
  served when a vehicle arrives at its location before the deadline; otherwise it counts as missed once.
- Traffic: each zone runs a two-state (normal / congested) Markov chain stepped every tick, and travel times
  get lognormal noise.
- Breakdowns: a small chance per trip; the vehicle is down for 30-90 min, or for the rest of the shift, and keeps
  its load.

Randomness
----------
All episode randomness comes from one ``np.random.Generator`` seeded in ``reset(seed)``. The map layout (location
coordinates, zones, bin sizes, base fill rates) is fixed scenario data. It is generated once from
``ScenarioConfig.layout_seed`` so that every episode runs on the same 15 locations.

Accounting conventions (relevant for the reward in card 04)
-----------------------------------------------------------
- ``travel_minutes`` is charged in the step that dispatches the trip (driving time only, excluding breakdown
  downtime). This makes the cost of a decision visible in that decision's step.
- ``kg_collected`` is credited when the vehicle arrives at the bin, which can be in a later step.
- ``overflow_bin_minutes`` and ``urgent_missed`` accrue over the simulated time each step covers.
- Nothing that happens after the shift ends is counted.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np

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
    # Fraction of the bin per hour, before the demand curve. (0.10, 0.22) gives a greedy dispatcher a little
    # overflow in the base scenario and real pressure in high_demand (provisional; card 14 calibrates).
    fill_rate_range: tuple[float, float] = (0.10, 0.22)

    # Demand
    demand_multiplier: float = 1.0
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
    "high_demand": ScenarioConfig(name="high_demand", demand_multiplier=1.5, urgent_rate_per_hour=1.8),
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
