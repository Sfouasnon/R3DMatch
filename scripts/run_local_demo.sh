#!/usr/bin/env bash
set -euo pipefail

python -m r3dsplat ingest synthetic.R3D --out /tmp/r3dsplat_demo_dataset --use-synthetic-decoder
python -m r3dsplat train-4d /tmp/r3dsplat_demo_dataset --out /tmp/r3dsplat_demo_run --epochs 1 --window-size 3 --num-gaussians 64
python -m r3dsplat eval-4d /tmp/r3dsplat_demo_run/train-4d.pt
