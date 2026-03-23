from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RedSdkConfig:
    sdk_root: Optional[str] = None
    libraries_path: Optional[str] = None
    use_gpu_decoder: bool = False
    decode_mode: str = "full-premium"


class RedDecoderBackend:
    def __init__(self, config: Optional[RedSdkConfig] = None) -> None:
        self.config = config or RedSdkConfig()

    def is_available(self) -> bool:
        return False

    def sdk_diagnostics(self) -> Dict[str, Any]:
        return {
            "backend": "red-sdk",
            "available": False,
            "sdk_root": self.config.sdk_root,
            "libraries_path": self.config.libraries_path,
            "message": (
                "RED SDK native backend is not built. Build the optional pybind11 module under "
                "cpp/r3d_sdk_wrapper and point it at a local RED SDK install."
            ),
        }

    def inspect_clip(self, source_path: str):
        raise RuntimeError(self.sdk_diagnostics()["message"])

    def list_frames(self, source_path: str):
        raise RuntimeError(self.sdk_diagnostics()["message"])

    def decode_frame(self, source_path: str, frame_index: int):
        raise RuntimeError(self.sdk_diagnostics()["message"])
