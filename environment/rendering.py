"""three.js rendering bridge for WasteFlow (Trello card 05).

The environment state is serialised to versioned JSON messages. This schema is also the integration API:
anything that can read JSON (the bundled viewer, a dashboard, a test) can follow an episode.

Messages (all carry ``schema`` and ``version``):

- ``wasteflow.layout``: the fixed map (nodes, zones, fleet, shift) plus the playback speed. Sent once per episode.
- ``wasteflow.snapshot``: the dynamic state at a decision point: clock, traffic, bins, vehicles (with their last
  route, so the viewer can animate trucks between decision points), KPIs and the decision that led here.
- ``wasteflow.episode``: a saved episode, ``{layout, frames: [snapshot, ...]}``, for replay mode.

Live mode: :class:`ViewerServer` is a stdlib HTTP server on 127.0.0.1 that serves ``assets/viewer/`` and streams
the messages over Server-Sent Events (``/events``). ``WasteFlowEnv(render_mode="human")`` drives it through
:class:`HumanRenderer`, which opens the browser and paces the episode so the viewer shows it in real time.

Replay mode: ``/replay.json`` serves one saved episode file chosen by the Python side (never a client-supplied
path); the viewer plays it back with a time slider (``index.html?replay``).
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from environment.custom_env import FACILITY, N_LOCATIONS, VehicleStatus, WasteFlowSim

if TYPE_CHECKING:
    from environment.custom_env import WasteFlowEnv

SCHEMA_VERSION = "1.0"
VIEWER_DIR = Path(__file__).resolve().parent.parent / "assets" / "viewer"
DEFAULT_SPEED = 10.0  # simulated minutes per real second in live mode (a shift plays in about 48 s)


# ------------------------------------------------------------------------------------------------ serialisation


def _num(x: float) -> float | None:
    """JSON-safe float: rounded, with inf/nan mapped to null."""
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def layout_message(sim: WasteFlowSim, playback_speed: float | None = None,
                   policy: str | None = None) -> dict[str, Any]:
    cfg, lay = sim.config, sim.layout
    nodes = []
    for i in range(N_LOCATIONS + 1):
        node = {
            "id": i,
            "kind": "facility" if i == FACILITY else "bin",
            "x_km": _num(lay.node_xy[i, 0]),
            "y_km": _num(lay.node_xy[i, 1]),
            "zone": int(lay.node_zone[i]),
        }
        if i < N_LOCATIONS:
            node["capacity_kg"] = _num(lay.bin_capacity_kg[i])
            node["base_fill_rate_per_h"] = _num(lay.base_fill_rate[i])
        nodes.append(node)
    return {
        "schema": "wasteflow.layout",
        "version": SCHEMA_VERSION,
        "scenario": cfg.name,
        "map_km": cfg.map_km,
        "n_zones": cfg.n_zones,
        "shift_minutes": cfg.shift_minutes,
        "shift_start_hour": cfg.shift_start_hour,
        "nodes": nodes,
        "vehicles": [{"id": v, "capacity_kg": cfg.vehicle_capacity_kg} for v in range(cfg.n_vehicles)],
        # None when unpaced (inf) or unknown; the viewer then keeps its own speed.
        "playback_speed": _num(playback_speed) if playback_speed is not None else None,
        # v1.0 additive: who is driving the fleet this episode (None when not given).
        "policy": policy,
    }


def snapshot(env: WasteFlowEnv) -> dict[str, Any]:
    """Dynamic state of ``env`` right now (after ``reset`` or ``step``)."""
    sim = env.sim
    cfg, lay = sim.config, sim.layout
    cap = cfg.vehicle_capacity_kg
    assigned = {int(sim.vehicle_target[u]): u for u in range(cfg.n_vehicles) if 0 <= sim.vehicle_target[u] < N_LOCATIONS}

    bins = []
    for b in range(N_LOCATIONS):
        pending = bool(sim.urgent_pending[b])
        bins.append(
            {
                "id": b,
                "fill": _num(sim.fill[b]),
                "kg": _num(sim.fill[b] * lay.bin_capacity_kg[b]),
                "overflowing": bool(sim.fill[b] >= 1.0),
                "urgent": pending,
                "deadline_left_min": _num(sim.urgent_deadline[b] - sim.t) if pending else None,
                "assigned_vehicle": assigned.get(b),
            }
        )

    vehicles = []
    for v in range(cfg.n_vehicles):
        dispatched = sim.route_arrive_at[v] > 0  # every trip ends after t=0; waiting doesn't create a route
        vehicles.append(
            {
                "id": v,
                "status": VehicleStatus(int(sim.vehicle_status[v])).name.lower(),
                "node": int(sim.vehicle_node[v]),
                "target": int(sim.vehicle_target[v]) if sim.vehicle_target[v] >= 0 else None,
                "load_kg": _num(sim.vehicle_load_kg[v]),
                "load_frac": _num(sim.vehicle_load_kg[v] / cap),
                "out_of_service": bool(sim.vehicle_out_of_service[v]),
                "route": {
                    "from": int(sim.route_from[v]),
                    "to": int(sim.route_to[v]),
                    "depart_t": _num(sim.route_depart_at[v]),
                    "arrive_t": _num(sim.route_arrive_at[v]),
                    "end_frac": _num(sim.route_end_frac[v]),
                }
                if dispatched
                else None,
            }
        )

    s = sim.stats
    return {
        "schema": "wasteflow.snapshot",
        "version": SCHEMA_VERSION,
        "t": _num(min(sim.t, cfg.shift_minutes)),
        "clock": sim.clock_str(min(sim.t, cfg.shift_minutes)),
        "done": bool(sim.done),
        "deciding_vehicle": sim.deciding_vehicle,
        "demand_level": _num(sim.demand_curve(sim.t)),
        "zone_congested": [bool(c) for c in sim.zone_congested],
        "bins": bins,
        "vehicles": vehicles,
        "kpis": {
            "tonnes_collected": _num(s.kg_collected / 1000.0),
            "tonnes_disposed": _num(s.kg_disposed / 1000.0),
            "travel_minutes": _num(s.travel_minutes),
            "overflow_bin_minutes": _num(s.overflow_bin_minutes),
            "urgent_arrived": s.urgent_arrived,
            "urgent_served": s.urgent_served,
            "urgent_missed": s.urgent_missed,
            "breakdowns": s.breakdowns,
            "decisions": s.decisions,
            "episode_return": _num(env.episode_return),
        },
        "last_decision": env.last_decision,
    }


class EpisodeRecorder:
    """Collects the layout and every snapshot of one episode; ``save`` writes a ``wasteflow.episode`` file."""

    def __init__(self) -> None:
        self.layout: dict | None = None
        self.frames: list[dict] = []

    def start(self, env: WasteFlowEnv) -> None:
        self.layout = layout_message(env.sim, policy=getattr(env, "policy_label", None))
        self.frames = [snapshot(env)]

    def record(self, env: WasteFlowEnv) -> None:
        self.frames.append(snapshot(env))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wasteflow.episode", "version": SCHEMA_VERSION, "layout": self.layout, "frames": self.frames}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        return path


def record_episode(env: WasteFlowEnv, policy, seed: int) -> EpisodeRecorder:
    """Play one episode with ``policy`` (``environment.custom_env.Policy``) and return its recording."""
    rec = EpisodeRecorder()
    policy.reset(seed)
    obs, _ = env.reset(seed=seed)
    rec.start(env)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(policy.act(obs, env))
        rec.record(env)
    return rec


# ------------------------------------------------------------------------------------------------------ server


class _Handler(SimpleHTTPRequestHandler):
    """Serves the viewer directory, ``/events`` (SSE) and ``/replay.json``. Bound to 127.0.0.1 only."""

    # Windows can map .js to text/plain in the registry; ES modules need a JavaScript MIME type.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".css": "text/css",
    }

    server: _Server

    def log_message(self, format, *args):  # noqa: A002 - signature from the base class
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/events":
            self._stream_events()
        elif route == "/replay.json":
            self._send_replay()
        else:
            super().do_GET()

    def _send_replay(self):
        path = self.server.replay_path
        if path is None or not path.is_file():
            self.send_error(404, "no replay loaded")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.server.subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=10.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if msg is None:  # server shutting down
                    break
                self.wfile.write(b"data: " + msg + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.server.unsubscribe(q)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, addr, handler):
        super().__init__(addr, handler)
        self.replay_path: Path | None = None
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self._history: list[bytes] = []  # current episode's messages, replayed to clients that join late

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for msg in self._history:
                q.put(msg)
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def broadcast(self, msg: dict, new_episode: bool = False) -> None:
        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        with self._lock:
            if new_episode:
                self._history = []
            self._history.append(data)
            for q in self._clients:
                q.put(data)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def close_clients(self) -> None:
        with self._lock:
            for q in self._clients:
                q.put(None)


class ViewerServer:
    """Local HTTP + SSE server for the three.js viewer.

    Args:
        port: TCP port on 127.0.0.1 (0 picks a free one).
        replay_path: optional ``wasteflow.episode`` file served at ``/replay.json``.
    """

    def __init__(self, port: int = 0, replay_path: str | Path | None = None, viewer_dir: Path = VIEWER_DIR) -> None:
        if not (viewer_dir / "index.html").is_file():
            raise FileNotFoundError(f"viewer not found at {viewer_dir}")
        self._httpd = _Server(("127.0.0.1", port), partial(_Handler, directory=str(viewer_dir)))
        self._httpd.replay_path = Path(replay_path).resolve() if replay_path else None
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="wasteflow-viewer", daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    @property
    def replay_url(self) -> str:
        return self.url + "?replay"

    def publish(self, msg: dict, new_episode: bool = False) -> None:
        self._httpd.broadcast(msg, new_episode=new_episode)

    def client_count(self) -> int:
        return self._httpd.client_count()

    def wait_for_client(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client_count() > 0:
                return True
            time.sleep(0.05)
        return self.client_count() > 0

    def close(self) -> None:
        self._httpd.close_clients()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


class HumanRenderer:
    """``render_mode="human"`` backend: streams snapshots to the browser viewer and paces the episode.

    Options (``WasteFlowEnv(render_options=...)``):
        port (0), speed (DEFAULT_SPEED simulated minutes per second), open_browser (True),
        wait_for_client_s (30: how long the first render waits for the page to connect).
    """

    def __init__(self, port: int = 0, speed: float = DEFAULT_SPEED, open_browser: bool = True,
                 wait_for_client_s: float = 30.0) -> None:
        self.speed = float(speed)
        self.open_browser = open_browser
        self.wait_for_client_s = wait_for_client_s
        self.server = ViewerServer(port=port)
        self.recorder = EpisodeRecorder()
        self._started = False
        self._last_t = 0.0

    def _start(self) -> None:
        self._started = True
        print(f"WasteFlow viewer: {self.server.url}", flush=True)
        if self.open_browser:
            webbrowser.open(self.server.url)
        if self.wait_for_client_s > 0 and not self.server.wait_for_client(self.wait_for_client_s):
            print("WasteFlow viewer: no browser connected; continuing without waiting", flush=True)

    def reset(self, env: WasteFlowEnv) -> None:
        if not self._started:
            self._start()
        self.recorder.start(env)
        self.server.publish(layout_message(env.sim, self.speed, getattr(env, "policy_label", None)), new_episode=True)
        self.server.publish(self.recorder.frames[-1])
        self._last_t = env.sim.t

    def step(self, env: WasteFlowEnv) -> None:
        self.recorder.record(env)
        frame = self.recorder.frames[-1]
        self.server.publish(frame)
        # The viewer animates from the previous decision point to this one while we sleep, so it stays in sync.
        dt = max(float(frame["t"]) - self._last_t, 0.0)
        self._last_t = float(frame["t"])
        if self.speed > 0 and np.isfinite(self.speed):
            time.sleep(dt / self.speed)

    def close(self) -> None:
        self.server.close()
