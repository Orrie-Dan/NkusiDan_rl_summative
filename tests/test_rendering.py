"""Card 05: JSON state API, episode recording, viewer server and render_mode="human".

The browser side (three.js drawing) is checked manually / by screenshot; these tests cover everything the
viewer depends on.
"""

import http.client
import json

import numpy as np
import pytest

from baselines.heuristics import RandomPolicy, UrgencyFirstPolicy
from environment.custom_env import WasteFlowEnv
from environment.rendering import (
    SCHEMA_VERSION,
    VIEWER_DIR,
    EpisodeRecorder,
    ViewerServer,
    layout_message,
    record_episode,
    snapshot,
)
from environment.simulation import FACILITY, N_LOCATIONS, N_VEHICLES

SNAPSHOT_KEYS = {"schema", "version", "t", "clock", "done", "zone_congested", "bins", "vehicles", "kpis", "last_decision"}


@pytest.fixture(scope="module")
def recorded():
    env = WasteFlowEnv("high_breakdown")  # breakdowns exercise the route end_frac < 1 path
    return [record_episode(env, policy, seed) for policy in (RandomPolicy(), UrgencyFirstPolicy()) for seed in (0, 1, 2)]


def test_layout_message_describes_the_fixed_map():
    msg = layout_message(WasteFlowEnv().sim, playback_speed=10.0)
    assert msg["schema"] == "wasteflow.layout" and msg["version"] == SCHEMA_VERSION
    assert len(msg["nodes"]) == N_LOCATIONS + 1 and len(msg["vehicles"]) == N_VEHICLES
    assert [n["kind"] for n in msg["nodes"]].count("facility") == 1 and msg["nodes"][FACILITY]["kind"] == "facility"
    assert all(0 <= n["x_km"] <= msg["map_km"] and 0 <= n["zone"] < msg["n_zones"] for n in msg["nodes"])
    json.dumps(msg, allow_nan=False)


def test_snapshots_are_valid_json_with_the_documented_schema(recorded):
    for rec in recorded:
        frames = rec.frames
        assert frames[0]["last_decision"] is None and frames[0]["t"] == 0
        assert frames[-1]["done"] and not any(f["done"] for f in frames[:-1])
        assert all(f["last_decision"] is not None for f in frames[1:])
        ts = [f["t"] for f in frames]
        assert ts == sorted(ts)
        for f in frames:
            json.dumps(f, allow_nan=False)  # strict JSON: no inf/nan leaks
            assert SNAPSHOT_KEYS <= set(f) and f["schema"] == "wasteflow.snapshot"
            assert len(f["bins"]) == N_LOCATIONS and len(f["vehicles"]) == N_VEHICLES
            for b in f["bins"]:
                assert 0 <= b["fill"] <= 1
                assert (b["deadline_left_min"] is not None) == b["urgent"]
            assigned = [b["assigned_vehicle"] for b in f["bins"] if b["assigned_vehicle"] is not None]
            assert len(assigned) == len(set(assigned))
            for v in f["vehicles"]:
                assert 0 <= v["load_frac"] <= 1


def test_vehicle_routes_are_consistent_with_vehicle_state(recorded):
    """The viewer animates trucks from the route; it must agree with where the simulation says they are."""
    saw_partial_route = False
    for rec in recorded:
        for f in rec.frames:
            for v in f["vehicles"]:
                r = v["route"]
                if r is None:
                    assert v["node"] == FACILITY  # never dispatched yet
                    continue
                assert r["arrive_t"] >= r["depart_t"] and 0 < r["end_frac"] <= 1
                if v["status"] in ("travelling", "broken_down") and v["target"] is not None:
                    assert r["to"] == v["target"] and r["depart_t"] <= f["t"] <= r["arrive_t"] + 1e-6
                elif not v["out_of_service"]:
                    assert r["end_frac"] == 1 and r["to"] == v["node"] and r["arrive_t"] <= f["t"] + 1e-6
                if v["out_of_service"]:
                    saw_partial_route |= r["end_frac"] < 1
    assert saw_partial_route, "expected at least one permanent breakdown mid-route in the sample"


def test_last_decision_matches_the_step(recorded):
    f = recorded[0].frames[1]
    d = f["last_decision"]
    assert {"vehicle", "action", "label", "clock", "t", "reward", "text"} <= set(d)
    assert d["t"] == 0 and d["clock"] == "06:00"


def test_recorder_save_roundtrip(tmp_path, recorded):
    rec = recorded[0]
    path = rec.save(tmp_path / "ep.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "wasteflow.episode" and loaded["version"] == SCHEMA_VERSION
    assert loaded["layout"]["schema"] == "wasteflow.layout"
    assert len(loaded["frames"]) == len(rec.frames) == rec.frames[-1]["kpis"]["decisions"] + 1


def test_rendering_arrays_do_not_change_the_episode():
    """Recording reads the route arrays only; the trajectory matches a plain run with the same seed."""
    a, b = WasteFlowEnv(), WasteFlowEnv()
    rec = record_episode(a, UrgencyFirstPolicy(), 4)
    pol = UrgencyFirstPolicy()
    obs, _ = b.reset(seed=4)
    done = False
    while not done:
        obs, _, done, _, info = b.step(pol.act(obs, b))
    assert rec.frames[-1]["kpis"]["episode_return"] == pytest.approx(info["episode_return"], abs=1e-3)


# ------------------------------------------------------------------------------------------------- server


def _get(server, path):
    host, port = server.url.split("//")[1].rstrip("/").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    conn.request("GET", path)  # sent verbatim (no client-side normalisation of "..")
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type"), body


def _read_events(server, n):
    host, port = server.url.split("//")[1].rstrip("/").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.getheader("Content-Type") == "text/event-stream"
    out = []
    while len(out) < n:
        line = resp.fp.readline().decode()
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    conn.close()
    return out


def test_server_serves_viewer_with_module_mime_types_and_blocks_traversal():
    server = ViewerServer(port=0)
    try:
        assert server.url.startswith("http://127.0.0.1:")
        status, ctype, body = _get(server, "/")
        assert status == 200 and b"viewer.js" in body
        assert _get(server, "/viewer.js")[1].startswith("text/javascript")
        status, ctype, _ = _get(server, "/vendor/three/three.module.min.js")
        assert status == 200 and ctype.startswith("text/javascript")
        assert (VIEWER_DIR / "vendor" / "three" / "LICENSE").is_file()
        for path in ("/../pyproject.toml", "/../../pyproject.toml", "/..%2f..%2fpyproject.toml"):
            status, _, body = _get(server, path)
            assert status == 404 and b"wasteflow" not in body
        assert _get(server, "/replay.json")[0] == 404  # nothing loaded
    finally:
        server.close()


def test_server_serves_only_the_chosen_replay_file(tmp_path, recorded):
    path = recorded[0].save(tmp_path / "ep.json")
    server = ViewerServer(port=0, replay_path=path)
    try:
        status, ctype, body = _get(server, "/replay.json?x=../../pyproject.toml")
        assert status == 200 and ctype == "application/json"
        assert json.loads(body)["schema"] == "wasteflow.episode"
        assert server.replay_url.endswith("/?replay")
    finally:
        server.close()


def test_sse_replays_history_to_late_clients_and_streams_new_messages():
    server = ViewerServer(port=0)
    try:
        server.publish({"schema": "wasteflow.layout", "n": 0}, new_episode=True)
        server.publish({"schema": "wasteflow.snapshot", "n": 1})
        assert [m["n"] for m in _read_events(server, 2)] == [0, 1]
        server.publish({"schema": "wasteflow.layout", "n": 2}, new_episode=True)  # new episode clears history
        assert [m["n"] for m in _read_events(server, 1)] == [2]
    finally:
        server.close()


def test_human_render_mode_streams_and_records_an_episode(tmp_path):
    env = WasteFlowEnv(
        render_mode="human",
        render_options={"port": 0, "speed": float("inf"), "open_browser": False, "wait_for_client_s": 0},
    )
    try:
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=3)
        server = env._renderer.server
        done, steps = False, 0
        while not done:
            obs, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
            steps += 1
        msgs = _read_events(server, steps + 2)  # layout + initial snapshot + one per step
        assert msgs[0]["schema"] == "wasteflow.layout" and msgs[0]["playback_speed"] is None  # inf -> null
        assert msgs[-1]["done"] and msgs[-1]["kpis"]["decisions"] == steps
        saved = json.loads(env.save_episode(tmp_path / "ep.json").read_text(encoding="utf-8"))
        assert len(saved["frames"]) == steps + 1
    finally:
        env.close()
    assert env._renderer is None


def test_save_episode_requires_human_mode():
    with pytest.raises(RuntimeError):
        WasteFlowEnv().save_episode("x.json")
