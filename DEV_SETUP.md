# R3DSplat Development Setup

## Host Requirements

- Linux only for the production target.
- NVIDIA GPU with a working CUDA driver for real gsplat training.
- Python 3.11 or newer.
- CMake 3.20 or newer.
- A local RED SDK install if you want native `.R3D` ingest.

## Python Environment

Create a local virtual environment with Python 3.11+:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install numpy pillow torch torchvision typer pyyaml pytest
```

Install `gsplat` into the same environment.

Option 1: use the vendored source in this workspace:

```bash
python -m pip install ./GSplat/gsplat-main
```

Option 2: install your preferred gsplat build that matches your CUDA and PyTorch versions.

## CUDA / PyTorch Notes

- Use a PyTorch build that matches your CUDA toolkit and NVIDIA driver.
- `gsplat` must be built against that same PyTorch and CUDA stack.
- For CPU-only validation of the synthetic vertical slice, plain `torch` is enough. The renderer will use the internal torch fallback only when `gsplat` is not importable.
- If `gsplat` imports successfully but rasterization fails, R3DSplat will now raise instead of silently falling back.

## RED SDK Integration

R3DSplat does not vendor RED SDK code. The integration boundary is isolated under [cpp/r3d_sdk_wrapper](/Users/sfouasnon/Desktop/R3DSplat/cpp/r3d_sdk_wrapper).

Build the native module with your local SDK paths:

```bash
cmake -S cpp/r3d_sdk_wrapper -B build/r3d_sdk_wrapper \
  -DR3DSPLAT_ENABLE_RED_SDK=ON \
  -DRED_SDK_ROOT=/path/to/R3DSDKv9_x_x \
  -DRED_SDK_INCLUDE_DIR=/path/to/R3DSDKv9_x_x/Include \
  -DRED_SDK_LIBRARY_DIR=/path/to/R3DSDKv9_x_x/Lib/linux64 \
  -DRED_SDK_REDISTRIBUTABLE_DIR=/path/to/R3DSDKv9_x_x/Redistributable/linux

cmake --build build/r3d_sdk_wrapper -j
```

The compiled module is expected to provide:

- `RedSdkConfig`
- `RedDecoderBackend.is_available()`
- `RedDecoderBackend.sdk_diagnostics()`
- `RedDecoderBackend.inspect_clip(path)`
- `RedDecoderBackend.list_frames(path)`
- `RedDecoderBackend.decode_frame(path, frame_index)`

The Python-side architecture already consumes this boundary through `RedSdkIngestBackend`, so a local developer only needs to fill in the actual SDK-backed implementation without changing the rest of the pipeline.

## Validation Commands

Synthetic ingest and 4D training:

```bash
PYTHONPATH=python python -m r3dsplat ingest synthetic.R3D --out /tmp/r3dsplat_dataset --backend mock
PYTHONPATH=python python -m r3dsplat train-4d /tmp/r3dsplat_dataset --out /tmp/r3dsplat_run --epochs 6 --window-size 3 --num-gaussians 64
PYTHONPATH=python python -m r3dsplat eval-4d /tmp/r3dsplat_run/train-4d.pt
PYTHONPATH=python python -m r3dsplat render-sequence /tmp/r3dsplat_run/train-4d.pt
```

Fiducial and COLMAP geometry steps:

```bash
PYTHONPATH=python python -m r3dsplat solve-fiducials /tmp/r3dsplat_dataset --backend mock
PYTHONPATH=python python -m r3dsplat solve-colmap /tmp/r3dsplat_dataset --mode standard
PYTHONPATH=python python -m r3dsplat align-world /tmp/r3dsplat_dataset
```

Run tests:

```bash
PYTHONPATH=python pytest -q
```

Real-backend gated tests:

```bash
export R3DSPLAT_ENABLE_REAL_BACKEND_TESTS=1
export R3DSPLAT_REAL_R3D=/path/to/sample.R3D
export R3DSPLAT_COLMAP_BIN=/usr/local/bin/colmap
export R3DSPLAT_ENABLE_REAL_GSPLAT_TESTS=1
PYTHONPATH=python pytest -q -m real_backend
```
