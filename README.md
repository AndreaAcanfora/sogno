# sogno

**A neural network that dreams a game, one frame at a time.**

sogno (Italian for “dream”) learns to *be* a game by watching it. There is no game engine at play
time: you act, and the network draws what happens next. It is trained on an Apple Silicon Mac with
[MLX](https://github.com/ml-explore/mlx), and the goal is to run the dream in the browser.

The first game is [sabbia](https://github.com/AndreaAcanfora/sabbia), a falling-sand toy: sand piles
up, water levels out, fire spreads along wood, acid eats walls. Its frames are grids of 16 materials.
The next one is the Game Boy, through the [mattone](https://github.com/AndreaAcanfora/mattone)
emulator (160×144 cells, 4 shades).

> Status: v0.2. A trained model (in `web/model/`) and a browser demo where you draw into the real
> game and the dream side by side. The dream holds its shape for a few seconds, then fire runs away:
> see [What it gets wrong](#what-it-gets-wrong).

![Real game on the left, dream on the right: 150 frames of a held-out episode](docs/dream.gif)

## How it works

1. **Data.** `sogno-gen` (Rust) runs the real sabbia engine while a scripted painter plays: it pours
   sand from the top, draws walls, sets wood on fire. Every frame gives one training pair: the world
   after the painter acted, and the world two physics steps later. It records several thousand
   frames a second on one core, so data is never the bottleneck.
2. **Model.** A small U-Net (1.2M parameters by default, small enough to run in a browser later) sees
   the last two frames and a partly hidden next frame, and predicts a material for every hidden cell.
   Frames are discrete, so instead of denoising pixels it fills in a masked frame,
   [MaskGIT](https://arxiv.org/abs/2202.04200) style.
3. **Dreaming.** The next frame starts fully hidden and is revealed in 4 passes, the most confident
   cells first. Each pass sees what earlier passes chose, which keeps two grains of sand from both
   falling into the same hole. The dreamed frame becomes the input for the next one: from then on the
   network only ever sees its own output.

Because sabbia is random (fire flickers, water picks a side), the dream is not supposed to match the
real game cell by cell. It is supposed to *behave* like it: sand should still pile up, and the amount
of water should not drift.

## Train it on your Mac

You need an Apple Silicon Mac, [Rust](https://rustup.rs) and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/AndreaAcanfora/sogno && cd sogno
uv sync

# 1. Record 2,000 episodes of 300 frames (96x64 cells): ~3.7 GB on disk, a couple of minutes.
cargo run --release --manifest-path gen/Cargo.toml -- --out data/sabbia --episodes 2000

# 2. Train. MLX uses the Mac's GPU on its own; progress and an ETA are printed as it goes.
uv run sogno-train --data data/sabbia --out runs/sabbia
#    Then teach it to recover from its own mistakes (see below), from where it stopped.
uv run sogno-train --data data/sabbia --out runs/unroll --resume runs/sabbia \
  --steps 6000 --unroll 3 --lr 1e-4 --warmup 200

# 3. Watch it dream an episode it never saw, next to the real game.
uv run sogno-dream --model runs/unroll --data data/sabbia --temperature 0.5 --out dream.gif
uv run sogno-dream --model runs/unroll --data data/sabbia --temperature 0.5 --score 16   # drift, below
```

On an 8 GB M1, start with a short run to measure speed, then size the real one from the ETA:

```sh
uv run sogno-train --data data/sabbia --out runs/try --steps 500 --eval-every 250
```

If memory is tight, lower `--batch` (default 16). The episodes are read from disk as needed, so
their size does not count against memory. The checkpoint is saved at every evaluation, so a run can
be stopped any time and continued with `--resume runs/sabbia` (the learning-rate schedule restarts).

Useful knobs:

| flag | default | what it does |
|---|---|---|
| `--steps` | 20000 | training steps |
| `--batch` | 16 | frames per step |
| `--width` | 32 | channels at full resolution (model size grows with its square: 64 gives 4.8M) |
| `--blocks` | 1 | residual blocks per level (2 makes the model 1.7M) |
| `--history` | 2 | past frames the model sees: fire's remaining life is only visible over time |
| `--changed-weight` | 10 | extra weight on cells that change: most of a frame sits still |
| `--unroll` | 0 | frames the model dreams itself before each target (about 3x slower per step) |

`sogno-dream` takes `--passes` (model calls per frame) and `--temperature` (0 = always the most likely
material).

## Results

On an M1 Pro (16 GB), the 20,000 base steps take 40 minutes at 8.5 steps/s and the 6,000 unrolled ones
33 minutes at 3 steps/s. Predicting one frame from real frames is easy: 98% of cells right, 90-93% of
the cells that move. Dreaming is the hard part, because from the second frame on the network only sees
its own output, and its small mistakes become its next input.

`sogno-dream --score 16` measures that on 16 held-out episodes: how far the dream's mix of materials
has drifted from the real game's (0 = the same amount of every material, 1 = nothing in common).

| model | temperature | after 10 frames | 30 | 75 | 150 |
|---|---|---|---|---|---|
| base, 20k steps | 0.5 | 0.056 | 0.148 | 0.394 | 0.614 |
| base, 20k steps | 0 | 0.058 | 0.125 | 0.157 | 0.234 |
| + 6k unrolled steps | 0.5 | **0.032** | **0.072** | **0.119** | **0.167** |
| + 6k unrolled steps | 0 | 0.034 | 0.067 | 0.084 | 0.114 |

**Unrolling** (`--unroll N`, the pushforward trick from neural PDE solvers) is what helps most. Before
each training step, the model dreams up to N frames from a real starting point, with the painter's
strokes applied, and is then asked for the *real* next frame. It learns to steer back towards the game
instead of building on its own errors. Unrolling further (6 frames, 5k more steps) did not help.

Temperature 0 drifts least, but it freezes the world: a material that changes only with low
probability per frame never changes. The demo defaults to 0.5.

### What it gets wrong

Fire, smoke and steam in sabbia die on a hidden countdown (fire lives 20-40 physics steps, smoke
40-90). The frame shows the element, not its age, so with two frames of history the network cannot
tell a fresh flame from a dying one. It guesses an average chance of dying, and at a temperature
below 1 that small chance gets smaller still. So flames outlive the real ones, spread, and turn stone
into lava. In one run of 150 frames at temperature 0.5, the 16 held-out dreams held 2,964 fire cells against
106 in the game, and 5,698 smoke against 280, while walls, sand, wood and glass stay within about a
third of the real amounts.

The fix is to show the age. sabbia's own renderer already shades fire by heat and embers by glow, so
splitting those materials into a few age bands puts on the frame what a player sees. Smoke and steam
would need sabbia to expose their life.

## In the browser

The demo runs both worlds side by side: sabbia's real rules on the left (compiled to WebAssembly from
the same revision the training data came from), the dream on the right. Each brush stroke goes into
both. The dreamer runs on your GPU through WebGPU with no ML library: the U-Net is five small compute
shaders in [`web/dreamer.js`](web/dreamer.js), and its logits match MLX's to 1e-5.

```sh
scripts/build-web.sh runs/unroll     # the game as WebAssembly + this checkpoint's weights (2.4 MB, float16)
python3 -m http.server -d web        # then open http://localhost:8000
```

`node web/test/check.mjs` (after `npm ci` in `web/` and `sogno-export ... --golden web/test/golden.bin`)
checks the WebGPU engine against MLX on the same input and drives the demo in headless Chromium.

## Layout

- `gen/`: the data generator, in Rust, using sabbia as a library (the scripted painter is in `lib.rs`).
- `sogno/data.py`: reads the episodes; `model.py`: the dreamer, its loss and sampler;
  `train.py`, `dream.py` and `export.py`: the commands.
- `web/`: the browser demo. `engine/` is the real game as WebAssembly, `dreamer.js` the network in
  WebGPU, `model/` the trained weights.
- `tests/`: fast checks that run on any CPU (`uv run pytest`).

## License

MIT
