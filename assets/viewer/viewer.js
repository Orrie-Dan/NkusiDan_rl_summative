// WasteFlow three.js viewer (cards 05 and 05b).
//
// Live mode (default): listens to /events (Server-Sent Events) for wasteflow.layout / wasteflow.snapshot messages.
// Replay mode (?replay): loads /replay.json (a wasteflow.episode) and plays it with a time slider.
// Optional URL params: t=<sim minutes> to start at a time, paused to start paused, speed=<sim min per second>,
// cam=<orbit|overview|top|street|flyover|follow0|follow1|follow2> to pick the camera, hud=0 to hide the panels.
// Keys: O orbit, V overview, T top-down, S street level, F fly-over, 1-3 follow a truck, H toggle the panels.
//
// Snapshots arrive at decision points. Between two snapshots k-1 and k the viewer animates the clock from t[k-1]
// to t[k]: trucks follow the routes in snapshot k (they include the trip dispatched at t[k-1]), while bins and
// KPIs show snapshot k-1 until the clock reaches t[k]. The Python side sleeps (t[k] - t[k-1]) / speed after
// publishing snapshot k, so live playback stays in sync.
//
// Everything drawn here is a pure function of (snapshots, clock), so scrubbing a replay is exact. The viewer only
// reads schema v1.0 fields; the city decoration (blocks, buildings, trees) is deterministic from the layout.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const params = new URLSearchParams(location.search);
const REPLAY = params.has('replay');
const S = 10; // world units per km
const BIN_H = 6; // height of a full bin
const ROAD_W = 2.8; // street width (world units)
const ROAD_HALF = ROAD_W / 2;
const LANE = 0.7; // trucks drive this far right of the street centre line
const TURN_R = 2.5; // corner radius of truck paths
const TRUCK_SCALE = 1.3; // trucks are drawn larger than life so they read at map scale
const ANIM_REAL_S = 2.2; // a collect / dump animation lasts this many real seconds (or the service time if shorter)
const VEHICLE_COLORS = [0x2f7de1, 0xf28c28, 0x8e5bd6];
const BROKEN_COLOR = 0xe53935;
const ZONE_NAMES = ['West', 'Central', 'East'];
const STATUS_TEXT = {
  idle: 'idle', travelling: 'driving', servicing: 'collecting', disposing: 'dumping',
  broken_down: 'broken down', waiting: 'waiting',
};

const state = {
  layout: null,
  frames: [],
  clock: 0,
  playing: !params.has('paused'),
  speed: Number(params.get('speed')) || 10,
  lastRenderedIndex: -1,
  lastStateFrame: null,
  tripEnd: new Map(), // "<vehicle>:<depart_t>" -> depart_t of that vehicle's next trip (= when its service ended)
  lastTripKey: [],
};
window.__wasteflow = state; // handy for debugging and automated screenshots

const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
const smooth = (x) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };
const smoothstep = (a, b, x) => smooth((x - a) / (b - a));
const hex = (n) => '#' + n.toString(16).padStart(6, '0');

// ------------------------------------------------------------------------------------------------- three setup

const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0xdfe9f3, 240, 520);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.5, 1400);
camera.position.set(0, 96, 104);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 6);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 8;
controls.maxDistance = 320;

// Sky dome: a vertical gradient whose colours follow the time of day.
const skyUniforms = { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } };
const sky = new THREE.Mesh(
  new THREE.SphereGeometry(650, 32, 16),
  new THREE.ShaderMaterial({
    uniforms: skyUniforms,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 horizon; varying vec3 vP;'
      + 'void main(){ float h = clamp(normalize(vP).y, 0.0, 1.0); gl_FragColor = vec4(mix(horizon, top, pow(h, 0.55)), 1.0); }',
  }),
);
scene.add(sky);

const hemi = new THREE.HemisphereLight(0xffffff, 0x7d8a6e, 1.0);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
Object.assign(sun.shadow.camera, { left: -95, right: 95, top: 95, bottom: -95, near: 10, far: 420 });
scene.add(sun, sun.target);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  drawChart(true);
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

// A billboard label drawn on a canvas. `set(lines)` redraws it only when the text changes. Each line is
// {text, color, bg}. Labels ignore depth so data stays readable over buildings. Close to the camera they shrink
// (see fitLabels), so a follow camera isn't filled by the nearest label.
const labels = [];
class Label {
  constructor(height = 3, maxLines = 1) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 512; this.canvas.height = 96 * maxLines;
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.anisotropy = 4;
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.tex, depthTest: false, depthWrite: false }));
    this.sprite.renderOrder = 10;
    this.sprite.scale.set((height * 512) / 96, height * maxLines, 1);
    this.sprite.center.set(0.5, 0);
    this.base = this.sprite.scale.clone();
    this.key = '';
    labels.push(this);
  }

  set(lines) {
    const key = JSON.stringify(lines);
    if (key === this.key) return;
    this.key = key;
    const g = this.canvas.getContext('2d');
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    g.font = 'bold 54px system-ui, Segoe UI, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    const n = this.canvas.height / 96;
    lines.forEach((ln, i) => {
      const y = (n - lines.length + i) * 96; // bottom-aligned, so the first line sits highest
      const w = Math.min(g.measureText(ln.text).width + 40, 504);
      g.fillStyle = ln.bg ?? 'rgba(255,255,255,0.9)';
      g.beginPath();
      g.roundRect((512 - w) / 2, y + 14, w, 68, 18);
      g.fill();
      g.fillStyle = ln.color ?? '#1d2430';
      g.fillText(ln.text, 256, y + 50);
    });
    this.tex.needsUpdate = true;
  }
}

const _lp = new THREE.Vector3();
function fitLabels() {
  for (const l of labels) {
    if (!l.sprite.parent) continue;
    l.sprite.getWorldPosition(_lp);
    const k = clamp(_lp.distanceTo(camera.position) / 55, 0.3, 1);
    l.sprite.scale.set(l.base.x * k, l.base.y * k, 1);
  }
}

function fillColor(f) {
  // green (0) -> yellow (0.5) -> red (1)
  return new THREE.Color().setHSL(0.33 * (1 - clamp(f, 0, 1)), 0.68, 0.5);
}

// Window facade texture shared by all buildings; its emissive copy lights the windows at dawn.
function facadeTextures() {
  const mk = (lit) => {
    const c = document.createElement('canvas');
    c.width = 128; c.height = 128;
    const g = c.getContext('2d');
    g.fillStyle = lit ? '#000' : '#ffffff';
    g.fillRect(0, 0, 128, 128);
    const rand = mulberry32(11);
    for (let r = 0; r < 6; r++) {
      for (let k = 0; k < 4; k++) {
        const on = rand() < 0.55;
        g.fillStyle = lit ? (on ? '#ffd27a' : '#000') : '#7d8fa3';
        g.fillRect(10 + k * 29, 8 + r * 20, 18, 11);
      }
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return { map: mk(false), emissive: mk(true) };
}

// ------------------------------------------------------------------------------------------------ geometry

function nodeXZ(id) {
  const n = state.layout.nodes[id];
  const half = state.layout.map_km / 2;
  return new THREE.Vector2((n.x_km - half) * S, (n.y_km - half) * S);
}

// Street path a -> b: first along x, then along z (streets run through every node), with a rounded corner.
// Returns { len, at(d) -> { p, dir } } where d is the distance along the path.
const pathCache = new Map();
function streetPath(a, b) {
  const key = `${a}>${b}`;
  if (pathCache.has(key)) return pathCache.get(key);
  const A = nodeXZ(a), B = nodeXZ(b);
  const dx = B.x - A.x, dz = B.y - A.y;
  const L1 = Math.abs(dx), L2 = Math.abs(dz);
  const ux = new THREE.Vector2(Math.sign(dx) || 1, 0), uz = new THREE.Vector2(0, Math.sign(dz) || 1);
  let path;
  if (L1 < 1e-6 || L2 < 1e-6) {
    const len = L1 + L2;
    const dir = L1 >= L2 ? ux : uz;
    path = { len, at: (d) => ({ p: A.clone().addScaledVector(dir, clamp(d, 0, len)), dir: dir.clone() }) };
  } else {
    const r = Math.min(TURN_R, L1, L2);
    const C = new THREE.Vector2(B.x, A.y);
    const s1 = L1 - r, arc = (r * Math.PI) / 2, s2 = L2 - r;
    const O = C.clone().addScaledVector(ux, -r).addScaledVector(uz, r);
    const P2 = C.clone().addScaledVector(uz, r);
    path = {
      len: s1 + arc + s2,
      at(d) {
        d = clamp(d, 0, this.len);
        if (d <= s1) return { p: A.clone().addScaledVector(ux, d), dir: ux.clone() };
        if (d <= s1 + arc) {
          const th = ((d - s1) / arc) * (Math.PI / 2);
          return {
            p: O.clone().addScaledVector(uz, -r * Math.cos(th)).addScaledVector(ux, r * Math.sin(th)),
            dir: ux.clone().multiplyScalar(Math.cos(th)).addScaledVector(uz, Math.sin(th)),
          };
        }
        return { p: P2.clone().addScaledVector(uz, d - s1 - arc), dir: uz.clone() };
      },
    };
  }
  pathCache.set(key, path);
  return path;
}

// Right-hand lane offset for a heading in the x-z plane (y up): right = (-dir.z, dir.x).
const laneOffset = (p, dir) => new THREE.Vector2(p.x - dir.y * LANE, p.y + dir.x * LANE);

// ------------------------------------------------------------------------------------------------ world build

let world = null;

function buildWorld(layout) {
  if (world) scene.remove(world.group);
  pathCache.clear();
  labels.length = 0;
  const group = new THREE.Group();
  const size = layout.map_km * S;
  const half = size / 2;
  const rand = mulberry32(7);

  // Ground: grass outside the city, asphalt inside; blocks between streets are raised pavement.
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(size + 260, size + 260), new THREE.MeshStandardMaterial({ color: 0x8fa876, roughness: 1 }));
  grass.rotation.x = -Math.PI / 2; grass.position.y = -0.02; grass.receiveShadow = true;
  const asphalt = new THREE.Mesh(new THREE.PlaneGeometry(size + 2 * ROAD_W, size + 2 * ROAD_W), new THREE.MeshStandardMaterial({ color: 0x5d636c, roughness: 0.95 }));
  asphalt.rotation.x = -Math.PI / 2; asphalt.receiveShadow = true;
  group.add(grass, asphalt);

  const xs = [...new Set(layout.nodes.map((n) => nodeXZ(n.id).x))].sort((a, b) => a - b);
  const zs = [...new Set(layout.nodes.map((n) => nodeXZ(n.id).y))].sort((a, b) => a - b);

  // Where bins and the depot stand: on the pavement at the roomiest corner of their intersection.
  // Free space from coordinate v to the next street in direction `sign` (or to the edge of the city, `lim`).
  const clearance = (v, lines, sign, lim) => {
    const next = lines.filter((u) => (u - v) * sign > 1e-6).map((u) => Math.abs(u - v));
    return next.length ? Math.min(...next) : Math.abs(sign * lim - v);
  };
  const spots = layout.nodes.map((n) => {
    const p = nodeXZ(n.id);
    let best = null;
    for (const qx of [1, -1]) {
      for (const qz of [1, -1]) {
        const c = Math.min(clearance(p.x, xs, qx, half + 6), clearance(p.y, zs, qz, half + 6));
        if (!best || c > best.c) best = { qx, qz, c };
      }
    }
    return { p, ...best };
  });

  // Pavement blocks with buildings, parks, trees and lamps.
  const bx = [-half - ROAD_HALF, ...xs, half + ROAD_HALF], bz = [-half - ROAD_HALF, ...zs, half + ROAD_HALF];
  const pads = [], buildings = [], trees = [], lamps = [], parks = [];
  const keepClear = (x, z) => spots.some((s, i) => {
    const r = layout.nodes[i].kind === 'facility' ? 13 : 5.5;
    return Math.hypot(s.p.x + s.qx * 3 - x, s.p.y + s.qz * 3 - z) < r;
  });
  for (let i = 0; i + 1 < bx.length; i++) {
    for (let j = 0; j + 1 < bz.length; j++) {
      const x0 = bx[i] + ROAD_HALF, x1 = bx[i + 1] - ROAD_HALF, z0 = bz[j] + ROAD_HALF, z1 = bz[j + 1] - ROAD_HALF;
      const w = x1 - x0, d = z1 - z0;
      if (w < 0.8 || d < 0.8) continue;
      const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
      pads.push({ cx, cz, w, d });
      if (w > 3 && d > 3) lamps.push({ x: x1 - 0.4, z: z1 - 0.4 }, { x: x0 + 0.4, z: z0 + 0.4 });
      if (w < 3 || d < 3) continue;
      const park = rand() < 0.2;
      if (park) {
        parks.push({ cx, cz, w: w - 1.2, d: d - 1.2 });
        const nT = Math.min(14, Math.floor((w * d) / 14));
        for (let k = 0; k < nT; k++) trees.push({ x: x0 + 0.9 + rand() * (w - 1.8), z: z0 + 0.9 + rand() * (d - 1.8), s: 0.7 + rand() * 0.6 });
        continue;
      }
      // Lot: split into cells of about 6 units, one building each (kept low so bins and trucks stay visible).
      const nx = Math.max(1, Math.round(w / 6.5)), nz = Math.max(1, Math.round(d / 6.5));
      const cw = w / nx, cd = d / nz;
      for (let a = 0; a < nx; a++) {
        for (let b = 0; b < nz; b++) {
          const ccx = x0 + cw * (a + 0.5), ccz = z0 + cd * (b + 0.5);
          if (keepClear(ccx, ccz)) {
            if (rand() < 0.5) trees.push({ x: ccx, z: ccz, s: 0.6 + rand() * 0.4 });
            continue;
          }
          if (rand() < 0.15) { trees.push({ x: ccx, z: ccz, s: 0.8 + rand() * 0.5 }); continue; }
          const h = 1.6 + rand() * rand() * 5.5;
          buildings.push({ cx: ccx, cz: ccz, w: Math.max(cw - 1.0, 0.8), d: Math.max(cd - 1.0, 0.8), h, tone: rand(), hue: rand() });
        }
      }
    }
  }

  const m4 = new THREE.Matrix4(), q0 = new THREE.Quaternion(), v3 = new THREE.Vector3(), s3 = new THREE.Vector3();
  const instanced = (geo, mat, items, place, color) => {
    if (!items.length) return null;
    const inst = new THREE.InstancedMesh(geo, mat, items.length);
    items.forEach((it, k) => {
      place(it, v3, s3);
      m4.compose(v3, q0, s3);
      inst.setMatrixAt(k, m4);
      if (color) inst.setColorAt(k, color(it));
    });
    group.add(inst);
    return inst;
  };
  const box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 0.5, 0);

  const padMesh = instanced(box, new THREE.MeshStandardMaterial({ color: 0xc9cbc6, roughness: 0.9 }), pads,
    (p, v, s) => { v.set(p.cx, 0, p.cz); s.set(p.w, 0.22, p.d); });
  if (padMesh) padMesh.receiveShadow = true;
  const parkMesh = instanced(box, new THREE.MeshStandardMaterial({ color: 0x7fae5c, roughness: 1 }), parks,
    (p, v, s) => { v.set(p.cx, 0.22, p.cz); s.set(p.w, 0.04, p.d); });
  if (parkMesh) parkMesh.receiveShadow = true;

  // Lane markings: dashed centre lines, broken at intersections.
  const dashes = [];
  const nearCross = (v, lines) => lines.some((u) => Math.abs(u - v) < ROAD_HALF + 0.5);
  for (const x of xs) for (let z = -half; z < half; z += 4.4) if (!nearCross(z, zs) && !nearCross(z + 1.4, zs)) dashes.push({ x, z: z + 0.7, rot: 0 });
  for (const z of zs) for (let x = -half; x < half; x += 4.4) if (!nearCross(x, xs) && !nearCross(x + 1.4, xs)) dashes.push({ x: x + 0.7, z, rot: 1 });
  instanced(box, new THREE.MeshBasicMaterial({ color: 0xd9d4c0, transparent: true, opacity: 0.55 }), dashes,
    (p, v, s) => { v.set(p.x, 0.01, p.z); if (p.rot) s.set(1.4, 0.02, 0.12); else s.set(0.12, 0.02, 1.4); });

  // Buildings: facade texture on the sides, plain roof on top; windows glow while it is dark.
  const fac = facadeTextures();
  const sideMat = new THREE.MeshStandardMaterial({ map: fac.map, emissiveMap: fac.emissive, emissive: 0xffffff, emissiveIntensity: 0, roughness: 0.8 });
  const roofMat = new THREE.MeshStandardMaterial({ color: 0x8b939c, roughness: 0.9 });
  const bMesh = instanced(box, [sideMat, sideMat, roofMat, roofMat, sideMat, sideMat], buildings,
    (b, v, s) => { v.set(b.cx, 0.22, b.cz); s.set(b.w, b.h, b.d); },
    (b) => new THREE.Color().setHSL(b.hue < 0.5 ? 0.08 : 0.58, 0.18, 0.66 + b.tone * 0.2));
  if (bMesh) bMesh.castShadow = bMesh.receiveShadow = true;

  // Trees: trunk + two-tone crown.
  const trunkGeo = new THREE.CylinderGeometry(0.16, 0.22, 1, 6); trunkGeo.translate(0, 0.5, 0);
  instanced(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x6b4f3a }), trees,
    (t, v, s) => { v.set(t.x, 0.22, t.z); s.set(t.s, 1.1 * t.s, t.s); });
  const crown = instanced(new THREE.IcosahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, flatShading: true }), trees,
    (t, v, s) => { v.set(t.x, 0.22 + 1.9 * t.s, t.z); s.set(1.1 * t.s, 1.35 * t.s, 1.1 * t.s); },
    (t) => new THREE.Color().setHSL(0.27 + 0.06 * (t.s - 1), 0.45, 0.33 + 0.1 * (t.x % 1 + 1) / 2));
  if (crown) crown.castShadow = true;

  // Street lamps (switched on while it is dark).
  const poleGeo = new THREE.CylinderGeometry(0.07, 0.09, 1, 6); poleGeo.translate(0, 0.5, 0);
  instanced(poleGeo, new THREE.MeshStandardMaterial({ color: 0x3d434b }), lamps, (l, v, s) => { v.set(l.x, 0.22, l.z); s.set(1, 3.2, 1); });
  const lampMat = new THREE.MeshStandardMaterial({ color: 0xfff1c9, emissive: 0xffc35a, emissiveIntensity: 0 });
  instanced(new THREE.SphereGeometry(0.28, 10, 8), lampMat, lamps, (l, v, s) => { v.set(l.x, 3.45, l.z); s.set(1, 1, 1); });

  // Traffic zones: vertical strips by x; tinted red while congested.
  const zones = [];
  const zw = size / layout.n_zones;
  for (let z = 0; z < layout.n_zones; z++) {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(zw, size),
      new THREE.MeshBasicMaterial({ color: 0xe53935, transparent: true, opacity: 0, depthWrite: false }),
    );
    m.rotation.x = -Math.PI / 2;
    m.position.set(-half + zw * (z + 0.5), 0.4, 0);
    m.renderOrder = 2;
    group.add(m);
    const label = new Label(3.2);
    label.set([{ text: `${ZONE_NAMES[z] ?? 'Zone ' + z} zone`, color: '#5f6b7a', bg: 'rgba(255,255,255,0.75)' }]);
    label.sprite.position.set(-half + zw * (z + 0.5), 0.5, half + 7);
    group.add(label.sprite);
    zones.push({ m, label });
  }

  // Bins: a container on the pavement with a see-through shell, a coloured fill level, a hinged lid and a label.
  const pickables = [];
  const bins = [];
  const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 28); unitCyl.translate(0, 0.5, 0);
  for (const n of layout.nodes.filter((nn) => nn.kind === 'bin')) {
    const sp = spots[n.id];
    const radius = 0.75 + (0.5 * (n.capacity_kg - 400)) / 800; // bigger bins look bigger
    const off = Math.min(ROAD_HALF + radius + 0.35, Math.max(sp.c / 2, ROAD_HALF + 0.4));
    const pos = new THREE.Vector2(sp.p.x + sp.qx * off, sp.p.y + sp.qz * off);
    const g = new THREE.Group();
    g.position.set(pos.x, 0.22, pos.y);
    g.userData = { kind: 'bin', id: n.id };
    const base = new THREE.Mesh(new THREE.CylinderGeometry(radius + 0.35, radius + 0.45, 0.25, 28), new THREE.MeshStandardMaterial({ color: 0x39424d, roughness: 0.6 }));
    base.position.y = 0.12; base.castShadow = true; g.add(base);
    const shell = new THREE.Mesh(unitCyl, new THREE.MeshStandardMaterial({ color: 0x9fb3c4, transparent: true, opacity: 0.25, depthWrite: false, roughness: 0.2 }));
    shell.scale.set(radius + 0.12, BIN_H, radius + 0.12); shell.position.y = 0.25; g.add(shell);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(radius + 0.14, 0.09, 8, 32), new THREE.MeshStandardMaterial({ color: 0x39424d }));
    rim.rotation.x = Math.PI / 2; rim.position.y = BIN_H + 0.25; g.add(rim);
    const fill = new THREE.Mesh(unitCyl, new THREE.MeshStandardMaterial({ color: 0x3cb371, roughness: 0.6 }));
    fill.scale.set(radius, 0.1, radius); fill.position.y = 0.25; fill.castShadow = true; g.add(fill);
    // Lid hinged at the back edge (away from the street corner).
    const hinge = new THREE.Group();
    hinge.position.set(-sp.qx * radius * 0.7, BIN_H + 0.3, -sp.qz * radius * 0.7);
    hinge.rotation.y = Math.atan2(sp.qx, sp.qz);
    const lid = new THREE.Mesh(new THREE.CylinderGeometry(radius + 0.2, radius + 0.2, 0.18, 28), new THREE.MeshStandardMaterial({ color: 0x2f6b4f, roughness: 0.5 }));
    lid.position.set(0, 0, radius * 0.7 + 0.05);
    lid.castShadow = true;
    hinge.add(lid); g.add(hinge);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius + 1.4, 0.26, 10, 48), new THREE.MeshBasicMaterial({ color: 0xff2d55, transparent: true, opacity: 0.9 }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.35; ring.visible = false; g.add(ring);
    const label = new Label(3.4, 2);
    label.sprite.position.y = BIN_H + 1.0; g.add(label.sprite);
    group.add(g);
    for (const m of [shell, fill, base, lid]) { m.userData = g.userData; pickables.push(m); }
    bins.push({ g, fill, ring, label, radius, hinge, top: new THREE.Vector3(pos.x, BIN_H + 0.6, pos.y), corner: sp.p });
  }

  // Disposal facility / depot: a hall with a tipping bay, chimney and fenced yard.
  const fn = layout.nodes.find((nn) => nn.kind === 'facility');
  const fsp = spots[fn.id];
  const facility = new THREE.Group();
  facility.position.set(fsp.p.x + fsp.qx * 7, 0.22, fsp.p.y + fsp.qz * 7);
  facility.userData = { kind: 'facility', id: fn.id };
  const yard = new THREE.Mesh(new THREE.BoxGeometry(13, 0.06, 11), new THREE.MeshStandardMaterial({ color: 0x9a9f94 }));
  yard.position.y = 0.03; yard.receiveShadow = true; facility.add(yard);
  const hall = new THREE.Mesh(new THREE.BoxGeometry(9, 4.5, 6.5), new THREE.MeshStandardMaterial({ color: 0x51677d, roughness: 0.7 }));
  hall.position.set(0, 2.25, 0); hall.castShadow = hall.receiveShadow = true; facility.add(hall);
  const roof = new THREE.Mesh(new THREE.CylinderGeometry(3.4, 3.4, 9.4, 24, 1, false, 0, Math.PI), new THREE.MeshStandardMaterial({ color: 0x334455, side: THREE.DoubleSide }));
  roof.rotation.z = Math.PI / 2; roof.scale.set(0.45, 1, 1); roof.position.y = 4.5; roof.castShadow = true; facility.add(roof);
  const bay = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 0.2), new THREE.MeshStandardMaterial({ color: 0x1f2a35 }));
  bay.position.set(0, 1.5, (-fsp.qz * 6.5) / 2 - 0.1 * fsp.qz); facility.add(bay);
  const stack = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.85, 10, 16), new THREE.MeshStandardMaterial({ color: 0x8a96a3 }));
  stack.position.set(3.2, 5, 2); stack.castShadow = true; facility.add(stack);
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.62, 0.6, 16), new THREE.MeshStandardMaterial({ color: 0xd64545 }));
  band.position.set(3.2, 9.2, 2); facility.add(band);
  const facLabel = new Label(2.6);
  facLabel.set([{ text: 'Disposal / Depot', color: '#ffffff', bg: 'rgba(51,68,85,0.92)' }]);
  facLabel.sprite.position.y = 11; facility.add(facLabel.sprite);
  for (const m of [hall, roof, stack]) { m.userData = facility.userData; pickables.push(m); }
  group.add(facility);

  // Smoke puffs from the chimney (pure function of wall time; purely decorative).
  const smoke = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.6, 1), new THREE.MeshStandardMaterial({ color: 0xdfe3e8, transparent: true, opacity: 0.5, depthWrite: false }), 10);
  smoke.frustumCulled = false;
  group.add(smoke);

  // Trucks
  const trucks = layout.vehicles.map((veh) => buildTruck(veh, group, pickables));

  // Particles for collection / dumping (one pool, positions recomputed every frame).
  const PARTICLES_PER_TRUCK = 18;
  const particles = new THREE.InstancedMesh(new THREE.BoxGeometry(0.42, 0.42, 0.42), new THREE.MeshStandardMaterial({ roughness: 0.8 }), PARTICLES_PER_TRUCK * trucks.length);
  const prand = mulberry32(3);
  for (let k = 0; k < particles.count; k++) {
    particles.setColorAt(k, new THREE.Color().setHSL(prand() < 0.5 ? 0.1 : 0.25, 0.35 + prand() * 0.3, 0.3 + prand() * 0.25));
    particles.setMatrixAt(k, new THREE.Matrix4().makeScale(0, 0, 0));
  }
  particles.castShadow = true;
  particles.frustumCulled = false; // instances move every frame; the initial bounding sphere is meaningless
  group.add(particles);

  scene.add(group);
  world = { group, zones, bins, trucks, facility, smoke, particles, PARTICLES_PER_TRUCK, pickables, sideMat, lampMat, stackTop: new THREE.Vector3().copy(stack.position).setY(10).add(facility.position) };
  buildFleetPanel(layout);
}

function buildTruck(veh, group, pickables) {
  const color = VEHICLE_COLORS[veh.id % VEHICLE_COLORS.length];
  const g = new THREE.Group(); // origin at the truck's centre on the road, forward = +x
  g.scale.setScalar(TRUCK_SCALE);
  g.userData = { kind: 'truck', id: veh.id };
  const dark = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.7 });
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.35, 1.7), dark);
  chassis.position.set(0, 0.62, 0); g.add(chassis);

  // Compactor body on a pivot at its rear-bottom edge so it can tip at the depot.
  const tip = new THREE.Group();
  tip.position.set(-2.1, 0.8, 0);
  g.add(tip);
  const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.1 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.9, 1.9, 1.95), bodyMat);
  body.position.set(1.55, 0.95, 0); body.castShadow = true; tip.add(body);
  const bodyTop = new THREE.Mesh(new THREE.BoxGeometry(2.7, 0.25, 1.7), bodyMat); // stepped roof edge
  bodyTop.position.set(1.5, 2.0, 0); bodyTop.castShadow = true; tip.add(bodyTop);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(2.92, 0.22, 1.97), new THREE.MeshStandardMaterial({ color: 0xf5f7fa }));
  stripe.position.set(1.55, 0.55, 0); tip.add(stripe);
  const hopper = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1.6, 1.85), new THREE.MeshStandardMaterial({ color: 0x5b6470, roughness: 0.6 }));
  hopper.position.set(-0.25, 0.85, 0); hopper.castShadow = true; tip.add(hopper);

  // Cab with windscreen and headlights.
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.35, 1.6, 1.9), new THREE.MeshStandardMaterial({ color: 0xeef2f6, roughness: 0.5 }));
  cab.position.set(1.65, 1.6, 0); cab.castShadow = true; g.add(cab);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.6, 1.6), new THREE.MeshStandardMaterial({ color: 0x1c2a3a, roughness: 0.1, metalness: 0.5 }));
  glass.position.set(2.33, 1.85, 0); g.add(glass);
  const lightMat = new THREE.MeshStandardMaterial({ color: 0xfff8e0, emissive: 0xfff2c0, emissiveIntensity: 0.2 });
  for (const z of [-0.65, 0.65]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.2, 0.32), lightMat);
    hl.position.set(2.34, 1.0, z); g.add(hl);
  }
  // Amber beacon on the cab roof: flashes while collecting or dumping.
  const beaconMat = new THREE.MeshStandardMaterial({ color: 0xffb020, emissive: 0xff9900, emissiveIntensity: 0 });
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.2, 10), beaconMat);
  beacon.position.set(1.65, 2.5, 0); g.add(beacon);

  // Wheels on pivots, spun by the distance driven.
  const wheels = [];
  const wheelGeo = new THREE.CylinderGeometry(0.45, 0.45, 0.35, 16);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1a1d22, roughness: 0.9 });
  const hubMat = new THREE.MeshStandardMaterial({ color: 0xb8c0c8 });
  for (const x of [-1.5, -0.6, 1.55]) {
    for (const z of [-0.88, 0.88]) {
      const pivot = new THREE.Group();
      pivot.position.set(x, 0.45, z);
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.x = Math.PI / 2; pivot.add(w);
      const hub = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.12, 0.37), hubMat); // makes the spin visible
      pivot.add(hub);
      g.add(pivot); wheels.push(pivot);
    }
  }
  for (const m of [body, cab, hopper, chassis]) { m.userData = g.userData; pickables.push(m); }

  // Load bar + name tag (upright, world-aligned: added to the scene, not the rotating truck group).
  const bar = new THREE.Group();
  const barBg = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.45, 0.45), new THREE.MeshBasicMaterial({ color: 0x2b313b, depthTest: false }));
  barBg.renderOrder = 9; bar.add(barBg);
  const barFill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.5, 0.5), new THREE.MeshBasicMaterial({ color: 0x3cb371, depthTest: false }));
  barFill.renderOrder = 10;
  barFill.geometry.translate(0.5, 0, 0);
  barFill.position.x = -1.8; bar.add(barFill);
  const label = new Label(1.9, 2);
  label.sprite.position.y = 0.5; bar.add(label.sprite);

  // Route ribbon from the truck to its target, and a beacon ring at the target.
  const SEG = 48;
  const rgeo = new THREE.BufferGeometry();
  rgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SEG * 2 * 3), 3));
  const idx = [];
  for (let k = 0; k + 1 < SEG; k++) idx.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
  rgeo.setIndex(idx);
  const ribbon = new THREE.Mesh(rgeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }));
  ribbon.frustumCulled = false; ribbon.renderOrder = 3;
  const target = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.18, 8, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }));
  target.rotation.x = -Math.PI / 2;
  group.add(g, bar, ribbon, target);
  return { id: veh.id, g, tip, bodyMat, beaconMat, lightMat, wheels, color, bar, barFill, label, ribbon, SEG, target, lastDir: new THREE.Vector2(1, 0) };
}

function buildFleetPanel(layout) {
  document.getElementById('fleet').innerHTML = layout.vehicles
    .map((v) => `<div class="fleet-row" id="fleet-${v.id}"><span class="swatch" style="background:${hex(VEHICLE_COLORS[v.id % 3])}"></span>`
      + `<b>V${v.id}</b><span class="fleet-status">–</span><span class="mini-bar"><span></span></span></div>`)
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

// Where vehicle v is at clock c, plus what it is doing there. Service phase u in [0, 1] runs over the first
// ANIM_REAL_S seconds after arrival (or the whole service, if shorter).
function vehiclePose(id, v, c) {
  const r = v.route;
  if (!r) return { p: nodeXZ(v.node), dir: null, d: 0, moving: false };
  const path = streetPath(r.from, r.to);
  const dEnd = (r.end_frac ?? 1) * path.len;
  if (r.arrive_t === null || c <= r.depart_t) return { ...path.at(0), d: 0, moving: false, path, dEnd };
  const span = Math.max(r.arrive_t - r.depart_t, 1e-6);
  const d = c >= r.arrive_t ? dEnd : ((c - r.depart_t) / span) * dEnd;
  const pose = { ...path.at(d), d, moving: c < r.arrive_t, path, dEnd };
  if (c >= r.arrive_t && (r.end_frac ?? 1) > 0.999) {
    const end = state.tripEnd.get(`${id}:${r.depart_t}`) ?? Infinity;
    const dur = Math.max(Math.min(end - r.arrive_t, state.speed * ANIM_REAL_S), 1e-6);
    const u = (c - r.arrive_t) / dur;
    if (u < 1) {
      pose.service = state.layout.nodes[r.to].kind === 'facility' ? 'dump' : 'collect';
      pose.u = u;
      pose.bin = r.to;
    }
  }
  return pose;
}

const _m4 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _s = new THREE.Vector3(), _e = new THREE.Euler();

function updateScene(c, now) {
  const k = frameIndexFor(c);
  const routeFrame = state.frames[k];
  const stateFrame = routeFrame.t > c && k > 0 ? state.frames[k - 1] : routeFrame;
  const pulse = 0.5 + 0.5 * Math.sin(now / 160);
  const night = updateSky(c);

  world.sideMat.emissiveIntensity = 0.9 * night;
  world.lampMat.emissiveIntensity = 2.2 * night;

  stateFrame.zone_congested.forEach((cong, z) => {
    world.zones[z].m.material.opacity = cong ? 0.13 + 0.06 * pulse : 0;
  });

  // Work out first which bins are being emptied, so their fill can drain during the animation.
  const poses = routeFrame.vehicles.map((v, i) => vehiclePose(i, v, c));
  const draining = new Map();
  poses.forEach((p) => { if (p.service === 'collect') draining.set(p.bin, p.u); });

  stateFrame.bins.forEach((b, i) => {
    const w = world.bins[i];
    let fill = b.fill;
    const u = draining.get(i);
    if (u !== undefined) {
      const after = routeFrame.bins[i].fill;
      if (after < fill) fill += (after - fill) * smoothstep(0.15, 0.85, u);
    }
    w.fill.scale.y = Math.max(fill, 0.015) * BIN_H;
    w.fill.material.color.copy(fillColor(fill));
    w.fill.material.emissive.setHex(b.overflowing ? 0x991111 : 0x000000);
    w.fill.material.emissiveIntensity = b.overflowing ? 0.4 + 0.6 * pulse : 0;
    w.hinge.rotation.x = u !== undefined ? -1.9 * Math.sin(Math.PI * clamp(u * 1.15, 0, 1)) : 0;
    w.ring.visible = b.urgent;
    if (b.urgent) {
      const s = 1 + 0.18 * pulse;
      w.ring.scale.set(s, s, s);
    }
    const lines = [{ text: `B${i}  ${Math.round(fill * 100)}%`, color: b.overflowing ? '#ffffff' : '#1d2430', bg: b.overflowing ? 'rgba(214,69,69,0.95)' : 'rgba(255,255,255,0.9)' }];
    if (b.urgent && b.deadline_left_min !== null) {
      // Count down with the clock between snapshots (the snapshot value is at stateFrame.t).
      const left = Math.max(b.deadline_left_min - (c - stateFrame.t), 0);
      lines.unshift({ text: `⚠ ${Math.ceil(left)} min${b.assigned_vehicle !== null ? ' · V' + b.assigned_vehicle : ''}`, color: '#ffffff', bg: left < 15 ? 'rgba(214,69,69,0.95)' : 'rgba(255,45,85,0.85)' });
    }
    w.label.set(lines);
  });

  let pi = 0;
  const parts = world.particles;
  routeFrame.vehicles.forEach((v, i) => {
    const t = world.trucks[i];
    const sv = stateFrame.vehicles[i];
    const pose = poses[i];
    if (pose.dir) t.lastDir.copy(pose.dir);
    const dir = t.lastDir;
    const pos = laneOffset(pose.p, dir);
    t.g.position.set(pos.x, 0, pos.y);
    t.g.rotation.y = Math.atan2(-dir.y, dir.x);
    for (const w of t.wheels) w.rotation.z = -pose.d / (0.45 * TRUCK_SCALE);

    const broken = sv.status === 'broken_down' || sv.out_of_service;
    t.bodyMat.color.setHex(broken ? BROKEN_COLOR : t.color);
    t.lightMat.emissiveIntensity = 0.2 + 2.5 * night;

    // Load: drains while dumping, fills while collecting.
    let load = clamp(sv.load_frac, 0, 1);
    if (pose.service) {
      const after = clamp(routeFrame.vehicles[i].load_frac, 0, 1);
      if (after !== load) load += (after - load) * smoothstep(0.2, 0.9, pose.u);
    }
    t.barFill.scale.x = Math.max(load * 3.6, 0.001);
    t.barFill.material.color.copy(fillColor(load));
    t.bar.position.set(pos.x, 6.4, pos.y);
    t.bar.quaternion.copy(camera.quaternion);
    const tag = [{ text: `V${i}`, color: '#ffffff', bg: hex(t.color) }];
    if (broken) tag.unshift({ text: sv.out_of_service ? 'OUT OF SERVICE' : 'BREAKDOWN', color: '#ffffff', bg: pulse > 0.3 || sv.out_of_service ? '#e53935' : '#a52a2a' });
    else if (pose.service) tag.unshift({ text: pose.service === 'collect' ? 'collecting' : 'dumping', color: '#5a3b00', bg: '#ffd166' });
    t.label.set(tag);

    // Tipper body + beacon.
    const tipAmt = pose.service === 'dump' ? Math.sin(Math.PI * clamp(pose.u * 1.1, 0, 1)) : 0;
    t.tip.rotation.z = -0.65 * tipAmt;
    t.beaconMat.emissiveIntensity = pose.service || (broken && !sv.out_of_service) ? (Math.sin(now / 90) > 0 ? 3 : 0.2) : 0;
    if (broken) t.beaconMat.emissive.setHex(0xff2020); else t.beaconMat.emissive.setHex(0xff9900);

    // Particles: waste flying from the bin into the hopper, or falling out of the tipped body.
    const rear = new THREE.Vector3(pos.x - dir.x * 2.9 * TRUCK_SCALE, 2.2, pos.y - dir.y * 2.9 * TRUCK_SCALE);
    for (let j = 0; j < world.PARTICLES_PER_TRUCK; j++, pi++) {
      let shown = false;
      if (pose.service) {
        const start = 0.12 + (0.62 * j) / world.PARTICLES_PER_TRUCK;
        const s = (pose.u - start) / 0.22;
        if (s >= 0 && s <= 1) {
          shown = true;
          if (pose.service === 'collect') {
            const from = world.bins[pose.bin].top;
            _v.lerpVectors(from, rear, s);
            _v.y += 4.5 * s * (1 - s);
          } else {
            const jit = ((j * 37) % 11) / 11 - 0.5;
            _v.set(rear.x - dir.x * 1.2 * s - dir.y * jit, 2.2 + 1.2 * tipAmt - 3.0 * s * s, rear.z - dir.y * 1.2 * s + dir.x * jit);
          }
          _e.set(s * 6 + j, s * 4, j);
          _q.setFromEuler(_e);
          _s.setScalar(1);
          _m4.compose(_v, _q, _s);
        }
      }
      if (!shown) _m4.makeScale(0, 0, 0);
      parts.setMatrixAt(pi, _m4);
    }

    // Route ribbon along the rest of the trip, in the truck's lane, plus a ring at the destination.
    const showRoute = pose.path && pose.moving && !v.out_of_service;
    t.ribbon.visible = t.target.visible = !!showRoute;
    if (showRoute) {
      const arr = t.ribbon.geometry.attributes.position.array;
      for (let s = 0; s < t.SEG; s++) {
        const d = pose.d + ((pose.dEnd - pose.d) * s) / (t.SEG - 1);
        const at = pose.path.at(d);
        const cpt = laneOffset(at.p, at.dir);
        const nx = -at.dir.y * 0.45, nz = at.dir.x * 0.45;
        arr.set([cpt.x - nx, 0.3, cpt.y - nz, cpt.x + nx, 0.3, cpt.y + nz], s * 6);
      }
      t.ribbon.geometry.attributes.position.needsUpdate = true;
      t.ribbon.material.opacity = 0.45 + 0.2 * pulse;
      const end = pose.path.at(pose.dEnd);
      const ep = laneOffset(end.p, end.dir);
      t.target.position.set(ep.x, 0.35, ep.y);
      const ts = 0.85 + 0.15 * pulse;
      t.target.scale.set(ts, ts, ts);
    }
  });
  parts.instanceMatrix.needsUpdate = true;

  // Fleet rows follow the animation (the snapshot status lags it: a truck that has arrived is collecting).
  stateFrame.vehicles.forEach((sv, i) => {
    const row = document.getElementById(`fleet-${i}`);
    if (!row) return;
    const pose = poses[i];
    const r = routeFrame.vehicles[i].route;
    const name = (n) => (state.layout.nodes[n].kind === 'facility' ? 'depot' : `B${n}`);
    const broken = sv.status === 'broken_down' || sv.out_of_service;
    let text;
    if (sv.out_of_service) text = 'out of service';
    else if (broken) text = 'broken down';
    else if (pose.service) text = pose.service === 'collect' ? `collecting at ${name(pose.bin)}` : 'dumping at depot';
    else if (pose.moving && r) text = `driving → ${name(r.to)}`;
    else text = STATUS_TEXT[sv.status] === 'driving' ? 'arrived' : STATUS_TEXT[sv.status] ?? sv.status;
    const el = row.querySelector('.fleet-status');
    if (el.textContent !== text) el.textContent = text;
    row.classList.toggle('broken', broken);
  });

  // Chimney smoke.
  for (let j = 0; j < world.smoke.count; j++) {
    const ph = (now / 4000 + j / world.smoke.count) % 1;
    _v.copy(world.stackTop).add(new THREE.Vector3(ph * 3, ph * 7, -ph * 1.5));
    _s.setScalar(0.6 + ph * 2.2);
    _m4.compose(_v, _q.identity(), _s);
    world.smoke.setMatrixAt(j, _m4);
  }
  world.smoke.instanceMatrix.needsUpdate = true;

  if (k !== state.lastRenderedIndex || stateFrame !== state.lastStateFrame) {
    state.lastRenderedIndex = k;
    state.lastStateFrame = stateFrame;
    updateHud(stateFrame, k);
  }
  return poses;
}

// Day-night: the sun follows the clock (sunrise 06:00, noon high, sunset 18:00). Returns darkness in [0, 1].
const SKY = {
  dawnTop: new THREE.Color(0x2b3f6e), dayTop: new THREE.Color(0x4f8fd6),
  dawnHorizon: new THREE.Color(0xf2a66e), dayHorizon: new THREE.Color(0xdde9f5),
  dawnSun: new THREE.Color(0xffa766), daySun: new THREE.Color(0xfff6e8),
  dawnSky: new THREE.Color(0x8e9cc4), daySky: new THREE.Color(0xffffff),
};
function updateSky(c) {
  const hour = state.layout.shift_start_hour + c / 60;
  const elev = Math.sin((Math.PI * (hour - 6)) / 12); // 0 at 06:00, 1 at noon
  const az = (Math.PI * (hour - 6)) / 12; // 0 = east, pi = west
  const day = smoothstep(-0.05, 0.45, elev);
  const night = 1 - smoothstep(0.0, 0.22, elev);
  const el = Math.max(Math.asin(clamp(elev, -1, 1)), 0.12);
  sun.position.set(Math.cos(az) * Math.cos(el) * 200, Math.sin(el) * 200, -Math.sin(az) * Math.cos(el) * 90 + 40);
  sun.intensity = 0.35 + 1.75 * day;
  sun.color.copy(SKY.dawnSun).lerp(SKY.daySun, day);
  hemi.intensity = 0.35 + 0.75 * day;
  hemi.color.copy(SKY.dawnSky).lerp(SKY.daySky, day);
  skyUniforms.top.value.copy(SKY.dawnTop).lerp(SKY.dayTop, day);
  skyUniforms.horizon.value.copy(SKY.dawnHorizon).lerp(SKY.dayHorizon, day);
  scene.fog.color.copy(skyUniforms.horizon.value);
  return night;
}

// ------------------------------------------------------------------------------------------------------- HUD

function fmt(x, d = 1) { return x === null || x === undefined ? '–' : Number(x).toFixed(d); }

function clockStr(t) {
  const m = state.layout.shift_start_hour * 60 + t;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(Math.floor(m % 60)).padStart(2, '0')}`;
}

function updateHud(sf, k) {
  const L = state.layout;
  document.getElementById('scenario').textContent = `scenario: ${L.scenario}` + (L.policy ? ` · agent: ${L.policy}` : '');
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
  ];
  document.getElementById('kpis').innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('');
  const ret = kp.episode_return;
  const retEl = document.getElementById('return-value');
  retEl.textContent = `${ret >= 0 ? '+' : ''}${fmt(ret, 2)}`;
  retEl.className = ret >= 0 ? 'r-pos' : 'r-neg';

  // Fleet status
  sf.vehicles.forEach((v) => {
    const row = document.getElementById(`fleet-${v.id}`);
    if (!row) return;
    const bar = row.querySelector('.mini-bar span');
    bar.style.width = `${Math.round(clamp(v.load_frac, 0, 1) * 100)}%`;
    bar.style.background = `#${fillColor(v.load_frac).getHexString()}`;
  });

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
  drawChart();
}

// Live chart of the episode return (cumulative reward) over the shift, up to the frame on screen.
const chart = document.getElementById('chart');
let chartKey = '';
function drawChart(force = false) {
  if (!state.layout || !state.frames.length || !state.lastStateFrame) return;
  const upto = state.frames.indexOf(state.lastStateFrame);
  const key = `${upto}:${state.frames.length}:${chart.clientWidth}`;
  if (!force && key === chartKey) return;
  chartKey = key;
  const dpr = Math.min(window.devicePixelRatio, 2);
  const W = chart.clientWidth, H = chart.clientHeight;
  chart.width = W * dpr; chart.height = H * dpr;
  const g = chart.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const end = state.layout.shift_minutes;
  const pts = state.frames.slice(0, upto + 1).map((f) => [f.t, f.kpis.episode_return ?? 0]);
  // Y range covers the whole episode seen so far (replay: whole shift), so the axis doesn't jump around.
  const all = (REPLAY ? state.frames : state.frames.slice(0, upto + 1)).map((f) => f.kpis.episode_return ?? 0);
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  if (hi - lo < 1) hi = lo + 1;
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const X = (t) => 4 + ((W - 8) * t) / end, Y = (r) => H - 14 - ((H - 20) * (r - lo)) / (hi - lo);
  g.strokeStyle = '#d8dee6'; g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(4, Y(0)); g.lineTo(W - 4, Y(0)); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#8a95a3'; g.font = '10px system-ui, Segoe UI, sans-serif';
  g.textAlign = 'left'; g.fillText(clockStr(0), 4, H - 2);
  g.textAlign = 'right'; g.fillText(clockStr(end), W - 4, H - 2);
  g.textAlign = 'left'; g.fillText(`${fmt(hi - pad, 0)}`, 6, 10);
  if (pts.length < 1) return;
  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(47,125,225,0.28)'); grad.addColorStop(1, 'rgba(47,125,225,0.02)');
  g.beginPath(); g.moveTo(X(pts[0][0]), Y(0));
  pts.forEach(([t, r], i) => { if (i) g.lineTo(X(t), Y(pts[i - 1][1])); g.lineTo(X(t), Y(r)); });
  g.lineTo(X(pts[pts.length - 1][0]), Y(0)); g.closePath(); g.fillStyle = grad; g.fill();
  g.beginPath();
  pts.forEach(([t, r], i) => { if (i === 0) g.moveTo(X(t), Y(r)); else { g.lineTo(X(t), Y(pts[i - 1][1])); g.lineTo(X(t), Y(r)); } });
  g.strokeStyle = '#2f7de1'; g.lineWidth = 1.6; g.stroke();
  const [lt, lr] = pts[pts.length - 1];
  g.beginPath(); g.arc(X(lt), Y(lr), 3, 0, Math.PI * 2); g.fillStyle = lr >= 0 ? '#2e9e5b' : '#d64545'; g.fill();
}

function setMode(text, cls) {
  const el = document.getElementById('mode');
  el.textContent = text;
  el.className = `badge ${cls}`;
}

// --------------------------------------------------------------------------------------------------- tooltips

const tooltip = document.getElementById('tooltip');
const raycaster = new THREE.Raycaster();
const mouse = { x: 0, y: 0, ndc: new THREE.Vector2(), inside: false };
renderer.domElement.addEventListener('pointermove', (e) => {
  mouse.x = e.clientX; mouse.y = e.clientY; mouse.inside = true;
  mouse.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
});
renderer.domElement.addEventListener('pointerleave', () => { mouse.inside = false; });

function updateTooltip(c) {
  if (!mouse.inside || !world || !state.lastStateFrame) { tooltip.hidden = true; return; }
  raycaster.setFromCamera(mouse.ndc, camera);
  const hit = raycaster.intersectObjects(world.pickables, false)[0];
  if (!hit) { tooltip.hidden = true; renderer.domElement.style.cursor = ''; return; }
  renderer.domElement.style.cursor = 'help';
  const { kind, id } = hit.object.userData;
  const sf = state.lastStateFrame;
  const L = state.layout;
  let html = '';
  if (kind === 'bin') {
    const b = sf.bins[id], n = L.nodes[id];
    html = `<b>Bin B${id}</b> · ${ZONE_NAMES[n.zone] ?? n.zone} zone<br>`
      + `Fill <b>${Math.round(b.fill * 100)}%</b> (${fmt(b.kg, 0)} / ${fmt(n.capacity_kg, 0)} kg)<br>`
      + `Base fill rate ${fmt(n.base_fill_rate_per_h * 100, 1)} %/h`
      + (b.overflowing ? '<br><span class="r-neg">Overflowing</span>' : '')
      + (b.urgent ? `<br><span class="r-neg">Urgent: ${fmt(Math.max(b.deadline_left_min - (c - sf.t), 0), 0)} min left</span>` : '')
      + (b.assigned_vehicle !== null ? `<br>Assigned to V${b.assigned_vehicle}` : '');
  } else if (kind === 'truck') {
    const v = sf.vehicles[id];
    const tgt = v.target === null ? '–' : L.nodes[v.target].kind === 'facility' ? 'depot' : `B${v.target}`;
    html = `<b>Truck V${id}</b><br>Status: ${v.out_of_service ? 'out of service' : STATUS_TEXT[v.status] ?? v.status}<br>`
      + `Load <b>${fmt(v.load_kg, 0)}</b> / ${fmt(L.vehicles[id].capacity_kg, 0)} kg (${Math.round(v.load_frac * 100)}%)<br>Heading to ${tgt}`;
  } else if (kind === 'facility') {
    html = `<b>Disposal facility / depot</b><br>Disposed so far: ${fmt(sf.kpis.tonnes_disposed)} t<br>Trucks start and end the shift here`;
  }
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const x = Math.min(mouse.x + 14, window.innerWidth - tooltip.offsetWidth - 8);
  const y = Math.min(mouse.y + 14, window.innerHeight - tooltip.offsetHeight - 8);
  tooltip.style.transform = `translate(${x}px, ${y}px)`;
}

// ------------------------------------------------------------------------------------------------ camera director

const PRESETS = {
  overview: { pos: [0, 96, 104], target: [0, 0, 6] },
  top: { pos: [0, 178, 0.01], target: [0, 0, 0] },
  street: { pos: [-62, 16, 58], target: [0, 2, 0] },
};
const cam = { mode: 'orbit', tween: null, flyStart: 0, followPos: new THREE.Vector3(), followLook: new THREE.Vector3(), followInit: false };
const camSelect = document.getElementById('cam');

function setCamera(mode) {
  if (PRESETS[mode]) {
    cam.tween = {
      t0: performance.now(), dur: 1300,
      fromPos: camera.position.clone(), fromTarget: controls.target.clone(),
      toPos: new THREE.Vector3(...PRESETS[mode].pos), toTarget: new THREE.Vector3(...PRESETS[mode].target),
    };
    cam.mode = 'orbit';
  } else {
    cam.tween = null;
    cam.mode = mode;
    cam.flyStart = performance.now();
    cam.followInit = false;
  }
  controls.enabled = cam.mode === 'orbit';
  camSelect.value = PRESETS[mode] ? mode : cam.mode;
}

function updateCamera(now, dt, poses) {
  if (cam.tween) {
    const tw = cam.tween;
    const u = smooth((now - tw.t0) / tw.dur);
    camera.position.lerpVectors(tw.fromPos, tw.toPos, u);
    controls.target.lerpVectors(tw.fromTarget, tw.toTarget, u);
    if (u >= 1) cam.tween = null;
  }
  if (cam.mode === 'flyover') {
    const s = (now - cam.flyStart) / 1000;
    const a = 0.6 + s * 0.11;
    const r = 92 + 18 * Math.sin(s * 0.23);
    camera.position.set(Math.cos(a) * r, 46 + 16 * Math.sin(s * 0.17), Math.sin(a) * r);
    controls.target.set(8 * Math.sin(s * 0.13), 0, 8 * Math.cos(s * 0.11));
    camera.lookAt(controls.target);
    return;
  }
  const m = /^follow(\d)$/.exec(cam.mode);
  if (m && world && world.trucks[+m[1]]) {
    const t = world.trucks[+m[1]];
    const p = t.g.position, d = t.lastDir;
    const wantPos = new THREE.Vector3(p.x - d.x * 24, 14, p.z - d.y * 24);
    const wantLook = new THREE.Vector3(p.x + d.x * 8, 2, p.z + d.y * 8);
    if (!cam.followInit) { cam.followPos.copy(wantPos); cam.followLook.copy(wantLook); cam.followInit = true; }
    const k = 1 - Math.exp(-dt * 2.5);
    cam.followPos.lerp(wantPos, k);
    cam.followLook.lerp(wantLook, 1 - Math.exp(-dt * 5));
    camera.position.copy(cam.followPos);
    controls.target.copy(cam.followLook);
    camera.lookAt(cam.followLook);
    return;
  }
  controls.update();
}

// Grabbing the scene with the mouse hands the camera back to the user.
renderer.domElement.addEventListener('pointerdown', () => { if (cam.mode !== 'orbit' || cam.tween) { cam.tween = null; setCamera('orbit'); } });
camSelect.onchange = (e) => setCamera(e.target.value);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const map = { o: 'orbit', v: 'overview', t: 'top', s: 'street', f: 'flyover', 1: 'follow0', 2: 'follow1', 3: 'follow2' };
  const key = e.key.toLowerCase();
  if (map[key]) setCamera(map[key]);
  else if (key === 'h') document.body.classList.toggle('no-hud');
  else if (key === ' ' && REPLAY) { e.preventDefault(); playBtn.click(); }
});
if (params.get('hud') === '0') document.body.classList.add('no-hud');

// ------------------------------------------------------------------------------------------------- data input

function onLayout(layout) {
  state.layout = layout;
  state.frames = [];
  state.clock = 0;
  state.lastRenderedIndex = -1;
  state.lastStateFrame = null;
  state.tripEnd = new Map();
  state.lastTripKey = [];
  if (!REPLAY && layout.playback_speed) state.speed = layout.playback_speed;
  document.getElementById('scrub').max = layout.shift_minutes;
  buildWorld(layout);
}

function onSnapshot(snap) {
  state.frames.push(snap);
  // Remember when each trip's service ended: a vehicle only decides once it is done (so a snapshot where it is the
  // deciding vehicle ends its current trip), and failing that, its next trip departs then.
  const dv = snap.deciding_vehicle;
  const dr = dv !== null && dv !== undefined ? snap.vehicles[dv]?.route : null;
  if (dr && dr.arrive_t !== null && dr.arrive_t <= snap.t + 1e-6) {
    const key = `${dv}:${dr.depart_t}`;
    if (!state.tripEnd.has(key)) state.tripEnd.set(key, snap.t);
  }
  snap.vehicles.forEach((v) => {
    if (!v.route) return;
    const key = `${v.id}:${v.route.depart_t}`;
    const prev = state.lastTripKey[v.id];
    if (prev !== key) {
      if (prev && !state.tripEnd.has(prev)) state.tripEnd.set(prev, v.route.depart_t);
      state.lastTripKey[v.id] = key;
    }
  });
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
  let poses = null;
  if (state.layout && state.frames.length) {
    const end = state.layout.shift_minutes;
    const cap = REPLAY ? end : state.frames[state.frames.length - 1].t;
    if (state.playing || !REPLAY) state.clock = Math.min(state.clock + dt * state.speed, cap);
    if (REPLAY && state.clock >= end && state.playing) { state.playing = false; syncPlay(); }
    poses = updateScene(state.clock, now);
    document.getElementById('clock').textContent = clockStr(state.clock);
    document.getElementById('shift-progress').style.width = `${(100 * state.clock) / end}%`;
    if (REPLAY && document.activeElement !== scrub) scrub.value = state.clock;
    const last = state.frames[state.frames.length - 1];
    if (!REPLAY && last.done && state.clock >= last.t) setMode('shift complete', 'live');
    updateTooltip(state.clock);
    document.body.dataset.ready = '1';
  }
  updateCamera(now, dt, poses);
  fitLabels();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

setCamera(params.get('cam') || 'orbit');
if (REPLAY) startReplay(); else startLive();
