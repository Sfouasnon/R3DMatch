# Local RED SDK Setup

## Requirements

- A local RED SDK installation that matches the target Linux environment.
- RED SDK headers under an `Include/` directory.
- RED SDK static libraries under `Lib/linux64`.
- RED SDK redistributable shared libraries under `Redistributable/linux`.

## Environment Variables

R3DSplat expects these values to be explicit when building or running against the RED SDK:

- `RED_SDK_ROOT`
- `RED_SDK_INCLUDE_DIR`
- `RED_SDK_LIBRARY_DIR`
- `RED_SDK_REDISTRIBUTABLE_DIR`

## Build Example

```bash
cmake -S cpp/r3d_sdk_wrapper -B build/r3d_sdk_wrapper \
  -DR3DSPLAT_ENABLE_RED_SDK=ON \
  -DRED_SDK_ROOT=$RED_SDK_ROOT \
  -DRED_SDK_INCLUDE_DIR=$RED_SDK_INCLUDE_DIR \
  -DRED_SDK_LIBRARY_DIR=$RED_SDK_LIBRARY_DIR \
  -DRED_SDK_REDISTRIBUTABLE_DIR=$RED_SDK_REDISTRIBUTABLE_DIR

cmake --build build/r3d_sdk_wrapper -j
```

## Native Contract

The native module is expected to provide:

- `RedSdkConfig`
- `RedDecoderBackend.is_available()`
- `RedDecoderBackend.sdk_diagnostics()`
- `RedDecoderBackend.inspect_clip(path)`
- `RedDecoderBackend.list_frames(path)`
- `RedDecoderBackend.decode_frame(path, frame_index)`

Python-side ingest normalizes those outputs into `ClipRecord`, `FrameRecord`, and `CameraRecord`.

## Notes

- The RED SDK remains a local optional dependency and is not included in this repo.
- R3DSplat does not reverse engineer `.R3D`.
- Backend auto-selection should only be used when its behavior is inspectable from diagnostics and manifests.
