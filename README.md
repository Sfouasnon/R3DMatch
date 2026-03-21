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
- `r3dsplat ingest <clip.R3D> --out <dataset_dir>`
- `r3dsplat solve-fiducials <dataset_dir>`
- `r3dsplat solve-colmap <dataset_dir>`
- `r3dsplat align-world <dataset_dir>`
- `r3dsplat train-4d <dataset_dir> --out <run_dir>`
- `r3dsplat train-static <dataset_dir> --out <run_dir>`
- `r3dsplat eval-4d <checkpoint> <dataset_dir>`
- `r3dsplat render-sequence <checkpoint> --out <tensor.pt>`

See [DEV_SETUP.md](/Users/sfouasnon/Desktop/R3DSplat/DEV_SETUP.md) and [LOCAL_RED_SDK_SETUP.md](/Users/sfouasnon/Desktop/R3DSplat/LOCAL_RED_SDK_SETUP.md) for environment details.
