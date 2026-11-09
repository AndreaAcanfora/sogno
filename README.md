# sogno

**A neural network that dreams a game, one frame at a time.**

sogno (Italian for “dream”) learns to *be* a game by watching it. There is no game engine at play
time: you act, and the network draws what happens next. It is trained on an Apple Silicon Mac with
[MLX](https://github.com/ml-explore/mlx), and the goal is to run the dream in the browser.

The first game is [sabbia](https://github.com/AndreaAcanfora/sabbia), a falling-sand toy: sand piles
up, water levels out, fire spreads along wood, acid eats walls. Its frames are grids of 16 materials.
The next one is the Game Boy, through the [mattone](https://github.com/AndreaAcanfora/mattone)
emulator (160×144 cells, 4 shades).

> Status: v0.1, the training pipeline. A trained model and the browser demo come next.

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

# 3. Watch it dream an episode it never saw, next to the real game.
uv run sogno-dream --model runs/sabbia --data data/sabbia --out dream.gif
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

`sogno-dream` takes `--passes` (model calls per frame) and `--temperature` (0 = always the most likely
material).

## In the browser

The demo runs both worlds side by side: sabbia's real rules on the left (compiled to WebAssembly from
the same revision the training data came from), the dream on the right. Each brush stroke goes into
both. The dreamer runs on your GPU through WebGPU with no ML library: the U-Net is five small compute
shaders in [`web/dreamer.js`](web/dreamer.js), and its logits match MLX's to 1e-5.

```sh
scripts/build-web.sh runs/sabbia     # the game as WebAssembly + this checkpoint's weights (2.4 MB, float16)
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
