"""Saving and loading a trained dreamer: `config.json` + `weights.safetensors` in one folder."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .model import Config, Dreamer


def save(model: Dreamer, folder: str | Path, extra: dict | None = None) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    info = {"config": model.config.to_dict(), **(extra or {})}
    (folder / "config.json").write_text(json.dumps(info, indent=2) + "\n")


def load(folder: str | Path) -> tuple[Dreamer, dict]:
    folder = Path(folder)
    info = json.loads((folder / "config.json").read_text())
    model = Dreamer(Config(**info["config"]))
    model.load_weights(str(folder / "weights.safetensors"))
    model.eval()
    return model, info
