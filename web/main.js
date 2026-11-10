// The demo: one brush, two worlds. The real one runs sabbia's rules (engine.wasm, built from the same
// revision as the training data); the dream is the network in dreamer.js, which only ever sees its own
// previous frames. Each tick, both get the same brush strokes, then both advance by one frame.

import { Dreamer } from "./dreamer.js";

const MATERIALS = [
  { id: 2, name: "Sand", color: "#e2bf6e" },
  { id: 3, name: "Water", color: "#2f7de1" },
  { id: 4, name: "Fire", color: "#ff8a1c" },
  { id: 7, name: "Wood", color: "#7a4e2a" },
  { id: 8, name: "Oil", color: "#4a3a2a" },
  { id: 9, name: "Lava", color: "#f04a14" },
  { id: 10, name: "Acid", color: "#7ce83a" },
  { id: 11, name: "Ice", color: "#a8e4f5" },
  { id: 12, name: "Plant", color: "#2fa34a" },
  { id: 14, name: "Stone", color: "#8a8680" },
  { id: 1, name: "Wall", color: "#6b6f7a" },
  { id: 0, name: "Eraser", color: "#10121a" },
];
// sabbia's element colours, by id (sogno/palette.py).
const PALETTE = [
  0x10121a, 0x6b6f7a, 0xe2bf6e, 0x2f7de1, 0xff8a1c, 0x555860, 0xb8c4d6, 0x7a4e2a,
  0x4a3a2a, 0xf04a14, 0x7ce83a, 0xa8e4f5, 0x2fa34a, 0xc9e8ef, 0x8a8680, 0xc83a10,
];
const EMPTY = 0, WALL = 1;

const $ = (id) => document.getElementById(id);
const canvases = [$("real"), $("dream")];
const stats = $("stats");

let material = MATERIALS[0];
let brush = 3;
let paused = false;
let temperature = 0.5;

function fail(text) {
  const message = $("message");
  message.innerHTML = text;
  message.hidden = false;
}

// ---------- loading ----------

async function loadWasm(url) {
  const response = fetch(url);
  try {
    return (await WebAssembly.instantiateStreaming(response, {})).instance.exports;
  } catch {
    return (await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), {})).instance.exports;
  }
}

async function gpu() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  return adapter ? adapter.requestDevice() : null;
}

const [engine, device] = await Promise.all([loadWasm("engine.wasm"), gpu()]);
if (!device) {
  fail(`This demo needs WebGPU, which this browser does not offer yet.<br>
    Try a recent Chrome or Edge, or Safari 26. Meanwhile, here is
    <a href="https://github.com/AndreaAcanfora/sogno#readme">a recording</a>.`);
  throw new Error("no WebGPU");
}
device.lost.then((info) => fail(`The GPU went away (${info.message}). Reload the page to dream again.`));
const dreamer = await Dreamer.load("model/", device);
const [H, W] = dreamer.frame;
const CELLS = H * W;
const STRIDE = dreamer.meta.stride ?? 2;
const HISTORY = dreamer.config.history;
for (const canvas of canvases) Object.assign(canvas, { width: W, height: H });
$("dream-note").textContent = `· ${(dreamer.meta.trained?.steps ?? 0).toLocaleString("en")} training steps`;

// ---------- the two worlds ----------

const elements = () => new Uint8Array(engine.memory.buffer, engine.engine_elements(), CELLS).slice();
let real = null; // Uint8Array: the real world now
let dream = []; // the dreamer's last HISTORY frames, oldest first

function newScene(seed = (Math.random() * 2 ** 32) >>> 0) {
  engine.engine_init(W, H, seed);
  const frames = [elements()];
  while (frames.length < HISTORY) {
    engine.engine_step(STRIDE);
    frames.push(elements());
  }
  real = frames.at(-1);
  dream = frames.map((f) => f.slice());
}

function wake() {
  dream = Array.from({ length: HISTORY }, () => real.slice());
}

// Paints with sabbia's brush into the real world, then copies exactly the cells that changed into the
// dream, following the brush's rule there: the eraser and walls overwrite, other materials only fill
// empty cells. This is what the training data did (sogno/data.py, apply_paint).
function paint(x, y) {
  const before = real;
  engine.engine_paint(x, y, brush, material.id);
  const after = elements();
  const now = dream.at(-1);
  for (let i = 0; i < CELLS; i++) {
    if (after[i] === before[i]) continue;
    const e = after[i];
    if (e === EMPTY || e === WALL || now[i] === EMPTY) now[i] = e;
  }
  real = after;
}

// ---------- drawing ----------

// A fixed, faint per-cell grain (the same on both sides) so materials read as texture, not flat colour.
const grain = Float32Array.from({ length: CELLS }, (_, i) => {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  return 0.92 + ((Math.imul(h, 0xc2b2ae35) >>> 0) / 2 ** 32) * 0.16;
});

const images = canvases.map((canvas) => {
  const ctx = canvas.getContext("2d");
  return { ctx, image: ctx.createImageData(W, H) };
});

function draw(index, frame) {
  const { ctx, image } = images[index];
  const px = image.data;
  for (let i = 0; i < CELLS; i++) {
    const c = PALETTE[frame[i]];
    const g = frame[i] === EMPTY ? 1 : grain[i];
    px[i * 4] = ((c >> 16) & 255) * g;
    px[i * 4 + 1] = ((c >> 8) & 255) * g;
    px[i * 4 + 2] = (c & 255) * g;
    px[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

// ---------- the loop ----------

let ticks = 0, dreamMs = 0, clock = performance.now(), counted = 0;

async function tick() {
  if (!paused) {
    paintStroke();
    const past = dream.map((f) => f.slice());
    engine.engine_step(STRIDE);
    real = elements();
    const t0 = performance.now();
    const next = await dreamer.sample(past, { temperature });
    dreamMs = dreamMs ? dreamMs * 0.9 + (performance.now() - t0) * 0.1 : performance.now() - t0;
    dream = [...dream.slice(1), next];
    ticks++;
    counted++;
  } else {
    paintStroke();
  }
  draw(0, real);
  draw(1, dream.at(-1));
  const now = performance.now();
  if (now - clock > 500) {
    let same = 0;
    const d = dream.at(-1);
    for (let i = 0; i < CELLS; i++) same += d[i] === real[i];
    const fps = (counted * 1000) / (now - clock);
    stats.textContent = `${W}×${H} · ${fps.toFixed(0)} frames/s · dream ${dreamMs.toFixed(0)} ms · ` +
      `${((100 * same) / CELLS).toFixed(0)}% cells match`;
    clock = now;
    counted = 0;
  }
  requestAnimationFrame(tick);
}

// ---------- input ----------

let pointer = null; // { x, y, lastX, lastY } in cells while drawing

for (const canvas of canvases) {
  const toCell = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.floor(((event.clientX - rect.left) / rect.width) * W),
      y: Math.floor(((event.clientY - rect.top) / rect.height) * H),
    };
  };
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    const cell = toCell(event);
    pointer = { ...cell, lastX: cell.x, lastY: cell.y };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (pointer) Object.assign(pointer, toCell(event));
  });
  for (const type of ["pointerup", "pointercancel"]) canvas.addEventListener(type, () => (pointer = null));
}

// Paints along the segment since the last tick, so fast strokes leave no gaps. Holding still keeps
// pouring, like a tap.
function paintStroke() {
  if (!pointer) return;
  const dx = pointer.x - pointer.lastX;
  const dy = pointer.y - pointer.lastY;
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, brush / 2)));
  for (let i = 1; i <= steps; i++) {
    paint(Math.round(pointer.lastX + (dx * i) / steps), Math.round(pointer.lastY + (dy * i) / steps));
  }
  pointer.lastX = pointer.x;
  pointer.lastY = pointer.y;
}

// ---------- toolbar ----------

const palette = $("palette");
MATERIALS.forEach((m, index) => {
  const button = document.createElement("button");
  button.className = "material";
  button.setAttribute("role", "radio");
  button.innerHTML = `<span class="swatch" style="background:${m.color}"></span>${m.name}` +
    (index < 10 ? ` <kbd>${(index + 1) % 10}</kbd>` : "");
  button.addEventListener("click", () => select(m));
  m.button = button;
  palette.append(button);
});

function select(m) {
  material = m;
  for (const other of MATERIALS) {
    other.button.classList.toggle("active", other === m);
    other.button.setAttribute("aria-checked", String(other === m));
  }
}

function setBrush(size) {
  brush = size;
  for (const b of document.querySelectorAll(".size")) b.classList.toggle("active", Number(b.dataset.size) === size);
}
for (const b of document.querySelectorAll(".size")) b.addEventListener("click", () => setBrush(Number(b.dataset.size)));

const pauseButton = $("pause");
function togglePause() {
  paused = !paused;
  pauseButton.textContent = paused ? "▶" : "❚❚";
  pauseButton.classList.toggle("active", paused);
}
pauseButton.addEventListener("click", togglePause);
$("scene").addEventListener("click", () => newScene());
$("wake").addEventListener("click", wake);
$("temperature").addEventListener("input", (event) => (temperature = Number(event.target.value)));

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.metaKey || event.ctrlKey) return;
  const digit = Number(event.key);
  if (event.key >= "0" && event.key <= "9") select(MATERIALS[(digit + 9) % 10]);
  else if (event.key === "e") select(MATERIALS.find((m) => m.id === EMPTY));
  else if (event.key === "n") newScene();
  else if (event.key === "w") wake();
  else if (event.key === " ") { event.preventDefault(); togglePause(); }
  else if (event.key === "[") setBrush(brush === 6 ? 3 : 1);
  else if (event.key === "]") setBrush(brush === 1 ? 3 : 6);
});

// ---------- start ----------

newScene(Number(new URLSearchParams(location.search).get("seed")) || undefined);
select(material);
window.sogno = { dreamer, get real() { return real; }, get dream() { return dream.at(-1); } };
requestAnimationFrame(tick);
