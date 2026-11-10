#!/usr/bin/env bash
# Builds the browser demo into web/: the real game as WebAssembly, and (if a checkpoint is given) the
# dreamer's weights. Then serve it: python3 -m http.server -d web
#
#   scripts/build-web.sh                 # engine only, keeps web/model as it is
#   scripts/build-web.sh runs/sabbia     # engine + export this checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown --manifest-path web/engine/Cargo.toml
cp web/engine/target/wasm32-unknown-unknown/release/sogno_engine.wasm web/engine.wasm
echo "web/engine.wasm: $(wc -c < web/engine.wasm) bytes"
if [ $# -gt 0 ]; then
  uv run sogno-export --model "$1" --out web/model
fi
