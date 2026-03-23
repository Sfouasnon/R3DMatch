# R3DSplat

R3DSplat is an internal Linux-first pipeline for native RED `.R3D` ingest into a 4D Gaussian Splatting workflow built around temporal sequences, fiducial-anchored world alignment, COLMAP geometry, and `gsplat` rendering/training.

## Current Status

- Preserves the validated prototype 4D training slice.
- Adds explicit ingest, fiducial, COLMAP, geometry, and masking modules.
- Keeps all backend decisions explicit in manifests and CLI output.
- Uses mock/synthetic validation in this workspace.
- Prepares a real RED SDK native boundary without vendoring RED proprietary code.

## Main Commands

- `r3dsplat inspect <clip.R3D>`
- `r3dsplat ingest <clip.R3D> --out <dataset_dir> [--preset NAME] [--start-frame N] [--max-frames N] [--frame-step N] [--decode-mode MODE] [--resize-scale S | --max-width W --max-height H] [--dry-run]`
- `r3dsplat solve-fiducials <dataset_dir>`
- `r3dsplat solve-colmap <dataset_dir>`
- `r3dsplat align-world <dataset_dir>`
- `r3dsplat train-4d <dataset_dir> --out <run_dir>`
- `r3dsplat train-static <dataset_dir> --out <run_dir>`
- `r3dsplat eval-4d <checkpoint> <dataset_dir>`
- `r3dsplat render-sequence <checkpoint> --out <tensor.pt>`

## Lightweight Real-Data Validation

For quick remote-workstation validation, prefer a small user-owned dataset path such as `~/Desktop/r3d_real_test` instead of `/tmp`.

Example ingest:

```bash
PYTHONPATH=python python -m r3dsplat ingest /path/to/clip.R3D \
  --out ~/Desktop/r3d_real_test \
  --backend red-sdk \
  --preset quick-test
```

Equivalent explicit quick-test style ingest:

```bash
PYTHONPATH=python python -m r3dsplat ingest /path/to/clip.R3D \
  --out ~/Desktop/r3d_real_test \
  --backend red-sdk \
  --max-frames 24 \
  --frame-step 2 \
  --decode-mode half-good \
  --max-width 960
```

Example lightweight training:

```bash
PYTHONPATH=python python -m r3dsplat train-4d ~/Desktop/r3d_real_test \
  --out ~/Desktop/r3d_real_run \
  --epochs 1 \
  --window-size 2 \
  --num-gaussians 64
```

Dry-run estimate:

```bash
PYTHONPATH=python python -m r3dsplat ingest /path/to/clip.R3D \
  --out ~/Desktop/r3d_real_test \
  --backend red-sdk \
  --preset desktop-review \
  --dry-run
```

See [DEV_SETUP.md](/Users/sfouasnon/Desktop/R3DSplat/DEV_SETUP.md) and [LOCAL_RED_SDK_SETUP.md](/Users/sfouasnon/Desktop/R3DSplat/LOCAL_RED_SDK_SETUP.md) for environment details.
