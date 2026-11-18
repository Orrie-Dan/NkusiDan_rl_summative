// WasteFlow three.js viewer (card 05).
//
// Live mode (default): listens to /events (Server-Sent Events) for wasteflow.layout / wasteflow.snapshot messages.
// Replay mode (?replay): loads /replay.json (a wasteflow.episode) and plays it with a time slider.
// Optional URL params: t=<sim minutes> to start at a time, paused to start paused, speed=<sim min per second>.
//
// Snapshots arrive at decision points. Between two snapshots k-1 and k the viewer animates the clock from t[k-1]
// to t[k]: trucks follow the routes in snapshot k (they include the trip dispatched at t[k-1]), while bins and
// KPIs show snapshot k-1 until the clock reaches t[k]. The Python side sleeps (t[k] - t[k-1]) / speed after
// publishing snapshot k, so live playback stays in sync.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const params = new URLSearchParams(location.search);
const REPLAY = params.has('replay');
const S = 10; // world units per km
const BIN_H = 7; // height of a full bin
const VEHICLE_COLORS = [0x2f7de1, 0xf28c28, 0x8e5bd6];
const BROKEN_COLOR = 0xe53935;
const ZONE_NAMES = ['West', 'Central', 'East'];

const state = {
  layout: null,
  frames: [],
  clock: 0,
  playing: !params.has('paused'),
  speed: Number(params.get('speed')) || 10,
  lastRenderedIndex: -1,
};
window.__wasteflow = state; // handy for debugging and automated screenshots

// ------------------------------------------------------------------------------------------------- three setup

const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe9f3);
scene.fog = new THREE.Fog(0xdfe9f3, 220, 420);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.5, 1000);
camera.position.set(0, 96, 104);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 6);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 30;
controls.maxDistance = 320;

scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9a7b, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-60, 120, 50);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, near: 10, far: 300 });
scene.add(sun);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --------------------------------------------------------------------------------------------------- helpers

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function textSprite(text, { color = '#1d2430', bg = 'rgba(255,255,255,0.88)', scale = 1 } = {}) {
  const font = 'bold 54px system-ui, Segoe UI, sans-serif';
  const probe = document.createElement('canvas').getContext('2d');
  probe.font = font;
  const w = Math.ceil(probe.measureText(text).width + 40);
  const c = document.createElement('canvas');
  c.width = w + 8; c.height = 96;
  const g = c.getContext('2d');
  g.font = font;
  g.fillStyle = bg;
  g.beginPath();
  g.roundRect(4, 14, w, 68, 18);
  g.fill();
  g.fillStyle = color;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, c.width / 2, 50);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false }));
  const h = 3.0 * scale; // world height of the label
  sprite.scale.set((h * c.width) / c.height, h, 1);
  return sprite;
}

function fillColor(f) {
  // green (0) -> yellow (0.5) -> red (1)
  return new THREE.Color().setHSL(0.33 * (1 - Math.min(Math.max(f, 0), 1)), 0.68, 0.5);
}

const hex = (n) => '#' + n.toString(16).padStart(6, '0');

// ------------------------------------------------------------------------------------------------ world build

let world = null; // { nodes, bins, trucks, zones, routeLines }

function nodeXZ(id) {
  const n = state.layout.nodes[id];
  const half = state.layout.map_km / 2;
  return new THREE.Vector2((n.x_km - half) * S, (n.y_km - half) * S);
}

// Point at fraction f along the street path a -> b: first along x, then along z (streets run through every node).
function lPath(a, b, f) {
  const pa = nodeXZ(a), pb = nodeXZ(b);
  const dx = pb.x - pa.x, dz = pb.y - pa.y;
  const len = Math.abs(dx) + Math.abs(dz);
  if (len < 1e-6) return { p: pa, dir: new THREE.Vector2(1, 0) };
  const d = Math.min(Math.max(f, 0), 1) * len;
  if (d <= Math.abs(dx)) return { p: new THREE.Vector2(pa.x + Math.sign(dx) * d, pa.y), dir: new THREE.Vector2(Math.sign(dx) || 1, 0) };
  return { p: new THREE.Vector2(pb.x, pa.y + Math.sign(dz) * (d - Math.abs(dx))), dir: new THREE.Vector2(0, Math.sign(dz) || 1) };
}

function buildWorld(layout) {
  if (world) scene.remove(world.group);
  const group = new THREE.Group();
  const size = layout.map_km * S;
  const half = size / 2;

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(size + 60, size + 60),
    new THREE.MeshStandardMaterial({ color: 0xc9d3bf, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  group.add(ground);

  // Traffic zones: vertical strips by x; tinted red while congested.
  const zones = [];
  const zw = size / layout.n_zones;
  for (let z = 0; z < layout.n_zones; z++) {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(zw, size),
      new THREE.MeshBasicMaterial({ color: 0xe53935, transparent: true, opacity: 0, depthWrite: false }),
    );
    m.rotation.x = -Math.PI / 2;
    m.position.set(-half + zw * (z + 0.5), 0.05, 0);
    group.add(m);
    const label = textSprite(`${ZONE_NAMES[z] ?? 'Zone ' + z} zone`, { color: '#5f6b7a', bg: 'rgba(255,255,255,0.7)', scale: 1.2 });
    label.position.set(-half + zw * (z + 0.5), 2, half + 6);
    group.add(label);
    zones.push(m);
  }

  // Streets through every node in both directions, so truck paths follow them.
  const roadMat = new THREE.MeshStandardMaterial({ color: 0x6b7280, roughness: 0.9 });
  const xs = new Set(), zs = new Set();
  layout.nodes.forEach((n) => { const p = nodeXZ(n.id); xs.add(p.x); zs.add(p.y); });
  for (const x of xs) {
    const r = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.12, size), roadMat);
    r.position.set(x, 0.08, 0); r.receiveShadow = true; group.add(r);
  }
  for (const z of zs) {
    const r = new THREE.Mesh(new THREE.BoxGeometry(size, 0.12, 1.1), roadMat);
    r.position.set(0, 0.08, z); r.receiveShadow = true; group.add(r);
  }

  // Buildings in the blocks between streets (decorative, deterministic), kept clear of nodes and streets.
  const sx = [-half, ...[...xs].sort((a, b) => a - b), half];
  const sz = [-half, ...[...zs].sort((a, b) => a - b), half];
  const rand = mulberry32(7);
  const boxes = [];
  for (let i = 0; i + 1 < sx.length; i++) {
    for (let j = 0; j + 1 < sz.length; j++) {
      const w = sx[i + 1] - sx[i] - 2.4, d = sz[j + 1] - sz[j] - 2.4;
      if (w < 2 || d < 2 || rand() < 0.35) continue;
      const cx = (sx[i] + sx[i + 1]) / 2, cz = (sz[j] + sz[j + 1]) / 2;
      const near = layout.nodes.some((n) => nodeXZ(n.id).distanceTo(new THREE.Vector2(cx, cz)) < 4.5);
      if (near) continue;
      // Kept low (and short of the blocks' edges) so they never hide bins or trucks.
      boxes.push({ cx, cz, w: Math.min(w, 8), d: Math.min(d, 8), h: 0.8 + rand() * rand() * 3.2, tone: rand() });
    }
  }
  if (boxes.length) {
    const inst = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 0.85 }), boxes.length);
    const m4 = new THREE.Matrix4();
    boxes.forEach((b, i) => {
      m4.compose(new THREE.Vector3(b.cx, b.h / 2, b.cz), new THREE.Quaternion(), new THREE.Vector3(b.w, b.h, b.d));
      inst.setMatrixAt(i, m4);
      inst.setColorAt(i, new THREE.Color().setHSL(0.58, 0.08, 0.72 + b.tone * 0.16));
    });
    inst.castShadow = inst.receiveShadow = true;
    group.add(inst);
  }

  // Bins
  const bins = [];
  const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 28);
  unitCyl.translate(0, 0.5, 0);
  for (const n of layout.nodes.filter((n) => n.kind === 'bin')) {
    const p = nodeXZ(n.id);
    const g = new THREE.Group();
    g.position.set(p.x, 0, p.y);
    const radius = 1.0 + 0.6 * (n.capacity_kg - 400) / 800; // bigger bins look bigger
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(radius + 1.2, radius + 1.2, 0.3, 32), new THREE.MeshStandardMaterial({ color: 0xf2f4f7 }));
    pad.position.y = 0.15; pad.receiveShadow = true; g.add(pad);
    const shell = new THREE.Mesh(unitCyl, new THREE.MeshStandardMaterial({ color: 0x9aa5b1, transparent: true, opacity: 0.22, depthWrite: false }));
    shell.scale.set(radius + 0.15, BIN_H, radius + 0.15); shell.position.y = 0.3; g.add(shell);
    const fill = new THREE.Mesh(unitCyl, new THREE.MeshStandardMaterial({ color: 0x3cb371, roughness: 0.6 }));
    fill.scale.set(radius, 0.1, radius); fill.position.y = 0.3; fill.castShadow = true; g.add(fill);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius + 2.0, 0.32, 10, 48), new THREE.MeshBasicMaterial({ color: 0xff2d55 }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.5; ring.visible = false; g.add(ring);
    const label = textSprite(`B${n.id}`, { scale: 0.8 });
    label.position.y = BIN_H + 2.4; g.add(label);
    group.add(g);
    bins.push({ g, fill, ring, label, radius });
  }

  // Disposal facility / depot
  const fp = nodeXZ(layout.nodes.find((n) => n.kind === 'facility').id);
  const fac = new THREE.Group();
  fac.position.set(fp.x, 0, fp.y);
  const hall = new THREE.Mesh(new THREE.BoxGeometry(9, 4.5, 7), new THREE.MeshStandardMaterial({ color: 0x51677d }));
  hall.position.y = 2.25; hall.castShadow = hall.receiveShadow = true; fac.add(hall);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(9.6, 0.5, 7.6), new THREE.MeshStandardMaterial({ color: 0x334455 }));
  roof.position.y = 4.75; fac.add(roof);
  const stack = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.9, 9, 16), new THREE.MeshStandardMaterial({ color: 0x8a96a3 }));
  stack.position.set(3, 4.5, -2); stack.castShadow = true; fac.add(stack);
  const fl = textSprite('Disposal / Depot', { color: '#ffffff', bg: 'rgba(51,68,85,0.9)', scale: 1.15 });
  fl.position.y = 12; fac.add(fl);
  group.add(fac);

  // Trucks
  const trucks = layout.vehicles.map((veh) => {
    const color = VEHICLE_COLORS[veh.id % VEHICLE_COLORS.length];
    const g = new THREE.Group();
    g.scale.setScalar(1.6); // trucks are drawn larger than life so they read at map scale
    const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.45 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(3.4, 2.0, 2.0), bodyMat);
    body.position.set(-0.4, 1.4, 0); body.castShadow = true; g.add(body);
    const cab = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1.6, 1.9), new THREE.MeshStandardMaterial({ color: 0xeef2f6 }));
    cab.position.set(1.95, 1.2, 0); cab.castShadow = true; g.add(cab);
    const wheelGeo = new THREE.CylinderGeometry(0.45, 0.45, 2.2, 14);
    for (const x of [-1.4, 1.5]) {
      const w = new THREE.Mesh(wheelGeo, new THREE.MeshStandardMaterial({ color: 0x222831 }));
      w.rotation.x = Math.PI / 2; w.position.set(x, 0.45, 0); g.add(w);
    }
    // Load bar (stays upright and world-aligned; added to the scene, not the rotating truck group)
    const bar = new THREE.Group();
    const barBg = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.45, 0.45), new THREE.MeshBasicMaterial({ color: 0x2b313b }));
    bar.add(barBg);
    const barFill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.5, 0.5), new THREE.MeshBasicMaterial({ color: 0x3cb371 }));
    barFill.geometry.translate(0.5, 0, 0);
    barFill.position.x = -1.8; bar.add(barFill);
    const label = textSprite(`V${veh.id}`, { color: '#ffffff', bg: hex(color), scale: 0.7 });
    label.position.y = 1.3; bar.add(label);
    const warn = textSprite('BREAKDOWN', { color: '#ffffff', bg: '#e53935', scale: 0.8 });
    warn.position.y = 3.7; warn.visible = false; bar.add(warn);
    group.add(g, bar);
    const routeLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineDashedMaterial({ color, dashSize: 1.4, gapSize: 0.9 }),
    );
    group.add(routeLine);
    return { g, bodyMat, color, bar, barFill, warn, routeLine, offset: new THREE.Vector2((veh.id - 1) * 1.6, (veh.id - 1) * -1.6) };
  });

  scene.add(group);
  world = { group, zones, bins, trucks };
  buildFleetLegend(layout);
}

function buildFleetLegend(layout) {
  document.getElementById('fleet-legend').innerHTML = layout.vehicles
    .map((v) => `<div class="legend-row"><span class="swatch" style="background:${hex(VEHICLE_COLORS[v.id % 3])}"></span>Truck V${v.id} (bar = load)</div>`)
    .join('');
}

// ------------------------------------------------------------------------------------------------- per frame

function frameIndexFor(c) {
  // First frame with t > c (the frame we are animating towards), or the last frame.
  const f = state.frames;
  let lo = 0, hi = f.length - 1;
  if (f[hi].t <= c) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (f[mid].t > c) hi = mid; else lo = mid + 1;
  }
  return lo;
}

function vehiclePose(v, c) {
  const r = v.route;
  if (!r) return { p: nodeXZ(v.node), dir: null };
  if (r.arrive_t === null || c <= r.depart_t) return { ...lPath(r.from, r.to, 0), moving: false };
  const span = Math.max(r.arrive_t - r.depart_t, 1e-6);
  const f = c >= r.arrive_t ? r.end_frac : ((c - r.depart_t) / span) * r.end_frac;
  return { ...lPath(r.from, r.to, f), moving: c < r.arrive_t, frac: f };
}

function updateScene(c, now) {
  const k = frameIndexFor(c);
  const routeFrame = state.frames[k];
  const stateFrame = routeFrame.t > c && k > 0 ? state.frames[k - 1] : routeFrame;
  const pulse = 0.5 + 0.5 * Math.sin(now / 160);

  stateFrame.zone_congested.forEach((cong, z) => {
    world.zones[z].material.opacity = cong ? 0.14 + 0.06 * pulse : 0;
  });

  stateFrame.bins.forEach((b, i) => {
    const w = world.bins[i];
    w.fill.scale.y = Math.max(b.fill, 0.015) * BIN_H;
    w.fill.material.color.copy(fillColor(b.fill));
    w.fill.material.emissive.setHex(b.overflowing ? 0x991111 : 0x000000);
    w.fill.material.emissiveIntensity = b.overflowing ? 0.4 + 0.6 * pulse : 0;
    w.ring.visible = b.urgent;
    if (b.urgent) {
      const s = 1 + 0.18 * pulse;
      w.ring.scale.set(s, s, s);
    }
  });

  routeFrame.vehicles.forEach((v, i) => {
    const t = world.trucks[i];
    const sv = stateFrame.vehicles[i];
    const pose = vehiclePose(v, c);
    const pos = pose.p.clone().add(t.offset);
    t.g.position.set(pos.x, 0, pos.y);
    if (pose.dir) t.g.rotation.y = Math.atan2(-pose.dir.y, pose.dir.x);
    t.bar.position.set(pos.x, 5.8, pos.y);
    t.bar.quaternion.copy(camera.quaternion);

    const broken = sv.status === 'broken_down' || sv.out_of_service;
    t.bodyMat.color.setHex(broken ? BROKEN_COLOR : t.color);
    t.warn.visible = broken && (sv.out_of_service || pulse > 0.3);
    const load = Math.min(Math.max(sv.load_frac, 0), 1);
    t.barFill.scale.x = Math.max(load * 3.6, 0.001);
    t.barFill.material.color.copy(fillColor(load));

    // Dashed line from the truck to where it is heading (while the trip is in progress).
    const r = v.route;
    const showRoute = r && pose.moving && !v.out_of_service;
    t.routeLine.visible = !!showRoute;
    if (showRoute) {
      const end = nodeXZ(r.to);
      const corner = lPath(r.from, r.to, 0).p; // corner of the L is at (to.x, from.z)
      const pts = pose.dir && pose.dir.x !== 0
        ? [new THREE.Vector3(pos.x, 0.6, pos.y), new THREE.Vector3(end.x, 0.6, corner.y), new THREE.Vector3(end.x, 0.6, end.y)]
        : [new THREE.Vector3(pos.x, 0.6, pos.y), new THREE.Vector3(end.x, 0.6, end.y), new THREE.Vector3(end.x, 0.6, end.y)];
      t.routeLine.geometry.setFromPoints(pts);
      t.routeLine.computeLineDistances();
    }
  });

  if (k !== state.lastRenderedIndex || stateFrame !== state.lastStateFrame) {
    state.lastRenderedIndex = k;
    state.lastStateFrame = stateFrame;
    updateHud(stateFrame, k);
  }
}

// ------------------------------------------------------------------------------------------------------- HUD

function fmt(x, d = 1) { return x === null || x === undefined ? '–' : Number(x).toFixed(d); }

function clockStr(t) {
  const m = state.layout.shift_start_hour * 60 + t;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(Math.floor(m % 60)).padStart(2, '0')}`;
}

function updateHud(sf, k) {
  const L = state.layout;
  document.getElementById('scenario').textContent = `scenario: ${L.scenario}`;
  document.getElementById('traffic').innerHTML = sf.zone_congested
    .map((c, z) => `<span class="chip ${c ? 'congested' : ''}">${ZONE_NAMES[z] ?? z}: ${c ? 'congested' : 'clear'}</span>`)
    .join('');
  const kp = sf.kpis;
  const rows = [
    ['Collected', `${fmt(kp.tonnes_collected)} t`],
    ['Disposed', `${fmt(kp.tonnes_disposed)} t`],
    ['Travel', `${fmt(kp.travel_minutes, 0)} min`],
    ['Overflow', `${fmt(kp.overflow_bin_minutes, 0)} bin-min`],
    ['Urgent served', `${kp.urgent_served} / ${kp.urgent_arrived}`],
    ['Urgent missed', `${kp.urgent_missed}`],
    ['Breakdowns', `${kp.breakdowns}`],
    ['Decisions', `${kp.decisions}`],
    ['Episode return', `${kp.episode_return >= 0 ? '+' : ''}${fmt(kp.episode_return, 2)}`],
  ];
  document.getElementById('kpis').innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('');

  // Decisions up to (and including) the one that starts the trip being animated.
  const decisions = [];
  for (let i = Math.min(k, state.frames.length - 1); i >= 1 && decisions.length < 9; i--) {
    if (state.frames[i].last_decision) decisions.push(state.frames[i].last_decision);
  }
  const last = decisions[0];
  document.getElementById('last-decision').innerHTML = last
    ? `<span class="vdot" style="background:${hex(VEHICLE_COLORS[last.vehicle % 3])}"></span>${last.clock} V${last.vehicle} → ${last.label}`
    : 'waiting for the first decision…';
  document.getElementById('log').innerHTML = decisions
    .map((d) => `<li><span><span class="vdot" style="background:${hex(VEHICLE_COLORS[d.vehicle % 3])}"></span>${d.clock} V${d.vehicle} → ${d.label}</span>`
      + `<span class="${d.reward >= 0 ? 'r-pos' : 'r-neg'}">r ${d.reward >= 0 ? '+' : ''}${fmt(d.reward, 2)}</span></li>`)
    .join('');
}

function setMode(text, cls) {
  const el = document.getElementById('mode');
  el.textContent = text;
  el.className = `badge ${cls}`;
}

// ------------------------------------------------------------------------------------------------- data input

function onLayout(layout) {
  state.layout = layout;
  state.frames = [];
  state.clock = 0;
  state.lastRenderedIndex = -1;
  if (!REPLAY && layout.playback_speed) state.speed = layout.playback_speed;
  document.getElementById('scrub').max = layout.shift_minutes;
  buildWorld(layout);
}

function onSnapshot(snap) {
  state.frames.push(snap);
  // A client that joins mid-episode receives the history in a burst; jump close to the newest frame.
  const n = state.frames.length;
  if (!REPLAY && n >= 2 && snap.t - state.clock > 30) state.clock = state.frames[n - 2].t;
}

function startLive() {
  document.body.classList.add('live');
  const es = new EventSource('/events');
  es.onopen = () => setMode('live', 'live');
  es.onerror = () => setMode('reconnecting…', 'error');
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.schema === 'wasteflow.layout') onLayout(msg);
    else if (msg.schema === 'wasteflow.snapshot' && state.layout) onSnapshot(msg);
  };
}

async function startReplay() {
  setMode('replay', 'replay');
  const res = await fetch('replay.json');
  if (!res.ok) { setMode('no replay loaded', 'error'); return; }
  const ep = await res.json();
  onLayout(ep.layout);
  ep.frames.forEach(onSnapshot);
  state.clock = Math.min(Number(params.get('t')) || 0, ep.layout.shift_minutes);
  document.getElementById('speed').value = String(state.speed);
}

// --------------------------------------------------------------------------------------------------- controls

const playBtn = document.getElementById('play');
const scrub = document.getElementById('scrub');
const syncPlay = () => { playBtn.textContent = state.playing ? '❚❚' : '▶'; };
syncPlay();
playBtn.onclick = () => {
  if (state.layout && state.clock >= state.layout.shift_minutes) state.clock = 0;
  state.playing = !state.playing;
  syncPlay();
};
scrub.oninput = () => { state.clock = Number(scrub.value); };
document.getElementById('speed').onchange = (e) => { state.speed = Number(e.target.value); };
document.getElementById('shot').onclick = () => {
  renderer.render(scene, camera);
  renderer.domElement.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `wasteflow_${state.layout ? clockStr(state.clock).replace(':', '') : 'view'}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
};

// ------------------------------------------------------------------------------------------------- main loop

let prev = performance.now();
function loop(now) {
  const dt = Math.min((now - prev) / 1000, 0.25);
  prev = now;
  if (state.layout && state.frames.length) {
    const end = state.layout.shift_minutes;
    const cap = REPLAY ? end : state.frames[state.frames.length - 1].t;
    if (state.playing || !REPLAY) state.clock = Math.min(state.clock + dt * state.speed, cap);
    if (REPLAY && state.clock >= end && state.playing) { state.playing = false; syncPlay(); }
    updateScene(state.clock, now);
    document.getElementById('clock').textContent = clockStr(state.clock);
    document.getElementById('shift-progress').style.width = `${(100 * state.clock) / end}%`;
    if (REPLAY && document.activeElement !== scrub) scrub.value = state.clock;
    const last = state.frames[state.frames.length - 1];
    if (!REPLAY && last.done && state.clock >= last.t) setMode('shift complete', 'live');
    document.body.dataset.ready = '1';
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

if (REPLAY) startReplay(); else startLive();
