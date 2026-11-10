"""Fast checks that run on any CPU: the data format, the model's shapes, and that it learns."""

import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx import nn

from sogno import checkpoint
from sogno.data import Episodes, apply_paint
from sogno.model import Config, Dreamer, loss, sample

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "gen" / "target" / "release" / "sogno-gen"


@pytest.fixture(scope="session")
def data(tmp_path_factory) -> Episodes:
    if not GEN.exists():
        subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=ROOT / "gen", check=True)
    out = tmp_path_factory.mktemp("data")
    subprocess.run(
        [
            str(GEN),
            "--out",
            str(out),
            "--episodes",
            "6",
            "--frames",
            "40",
            "--width",
            "32",
            "--height",
            "24",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
    )
    return Episodes(out)


def test_recorded_episodes_follow_the_format(data):
    assert data.states.shape == (6, 41, 24, 32)
    assert data.states.max() < data.classes == len(data.elements) == len(data.lives) == 93
    # Classes past sabbia's 16 elements are older age bands of fire, smoke, steam and embers.
    assert set(data.elements[16:]) == {4, 5, 6, 15}
    assert data.elements[data.states].max() < 16
    assert len(data._keys) > 0, "the painter should have painted something"
    # Painting changes the input frame, and only where the painter drew.
    for e in range(data.episodes):
        for t in range(data.frames):
            cells, elements = data.paints(e, t)
            if len(cells):
                painted = data.painted(e, t)
                assert (painted.reshape(-1)[cells] == elements).all()
                untouched = np.ones(painted.size, bool)
                untouched[cells] = False
                assert (painted.reshape(-1)[untouched] == data.states[e, t].reshape(-1)[untouched]).all()
                return
    pytest.fail("no painted frame found")


def test_the_generator_is_deterministic(tmp_path, data):
    out = tmp_path / "again"
    subprocess.run(
        [
            str(GEN),
            "--out",
            str(out),
            "--episodes",
            "6",
            "--frames",
            "40",
            "--width",
            "32",
            "--height",
            "24",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
    )
    assert np.array_equal(np.load(out / "states.npy"), np.asarray(data.states))


def test_brush_rule_on_dreams():
    state = np.array([[0, 2, 1, 0]], np.uint8)  # empty, sand, wall, empty
    cells, water = np.arange(4), np.full(4, 3, np.uint8)
    apply_paint(state, cells, water, strict=False)
    assert state.tolist() == [[3, 2, 1, 3]], "water only fills empty cells"
    apply_paint(state, cells, np.zeros(4, np.uint8), strict=False)
    assert state.tolist() == [[0, 0, 0, 0]], "the eraser clears everything"


def test_batches_pair_inputs_with_the_next_frame(data):
    batch = data.batch(np.random.default_rng(0), 5, 2, range(data.episodes), flip=False)
    assert batch.past.shape == (5, 2, 24, 32) and batch.target.shape == (5, 24, 32)


def test_model_shapes_and_sampling():
    model = Dreamer(Config(classes=4, history=3, width=8, levels=2, blocks=1))
    past = mx.zeros((2, 3, 8, 12), dtype=mx.uint8)
    logits = model(past, mx.full((2, 8, 12), 4, dtype=mx.uint8))
    assert logits.shape == (2, 8, 12, 4)
    frame = sample(model, past, passes=3)
    assert frame.shape == (2, 8, 12) and frame.max().item() < 4, "every cell gets revealed"


def test_it_learns_to_copy_a_still_world(data):
    """A world where nothing moves: the model should learn to copy the last frame quickly."""
    mx.random.seed(0)
    model = Dreamer(Config(classes=16, history=2, width=8, levels=2, blocks=1))
    optimizer = optim.Adam(learning_rate=3e-3)
    grad = nn.value_and_grad(model, lambda p, t: loss(model, p, t)[0])
    frame = np.asarray(data.states[0, 0])
    past = mx.array(np.stack([frame, frame])[None].repeat(4, 0))
    target = mx.array(frame[None].repeat(4, 0))
    first = None
    for _ in range(60):
        value, grads = grad(past, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        first = first if first is not None else value.item()
    assert value.item() < first / 4
    model.eval()
    assert (sample(model, past[:1], passes=2, temperature=0) == target[:1]).mean().item() > 0.95


def test_checkpoints_round_trip(tmp_path):
    model = Dreamer(Config(classes=4, history=2, width=8, levels=2, blocks=1))
    checkpoint.save(model, tmp_path, {"note": "test"})
    loaded, info = checkpoint.load(tmp_path)
    past, partial = mx.zeros((1, 2, 4, 4), dtype=mx.uint8), mx.full((1, 4, 4), 4, dtype=mx.uint8)
    assert info["note"] == "test"
    assert mx.allclose(model(past, partial), loaded(past, partial)).item()


def test_export_for_the_browser(tmp_path):
    """The browser reads weights.bin through model.json's table: every tensor, float16, back to back."""
    import json

    from mlx.utils import tree_flatten

    from sogno.export import export, golden

    model = Dreamer(Config(width=8, levels=2))
    export(model, {"data": {"height": 16, "width": 24, "stride": 2}}, tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    weights = np.frombuffer((tmp_path / "weights.bin").read_bytes(), np.float16)
    assert meta["frame"] == [16, 24] and meta["config"]["width"] == 8
    end = 0
    for name, value in sorted(tree_flatten(model.parameters())):
        entry = meta["tensors"][name]
        assert entry["shape"] == list(value.shape) and entry["offset"] == end
        stored = weights[end : end + value.size].reshape(value.shape)
        assert np.allclose(stored, np.array(value), atol=1e-3)
        end += value.size
    assert end == weights.size

    golden(model, tmp_path / "golden.bin", (16, 24))
    cells, k, c = 16 * 24, model.config.history, model.config.classes
    assert (tmp_path / "golden.bin").stat().st_size == (k + 1) * cells + cells * c * 4
