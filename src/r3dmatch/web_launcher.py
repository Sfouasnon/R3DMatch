from __future__ import annotations

import argparse
import importlib
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _bootstrap_module_imports():
    candidates = [
        Path(__file__).resolve().parents[1],
        Path(sys.executable).resolve().parents[1] / "Resources",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.insert(0, Path(str(meipass)).resolve())
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
    desktop_module = importlib.import_module("r3dmatch.desktop_app")
    runtime_module = importlib.import_module("r3dmatch.runtime_env")
    web_module = importlib.import_module("r3dmatch.web_app")
    return (
        desktop_module.launch_desktop_ui,
        runtime_module.runtime_health_payload,
        web_module.DEFAULT_HOST,
        web_module.DEFAULT_PORT,
        web_module.create_app,
    )


launch_desktop_ui, runtime_health_payload, DEFAULT_HOST, DEFAULT_PORT, create_app = _bootstrap_module_imports()


def _find_open_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Unable to find an open port near {preferred_port} on {host}.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from r3dmatch.cli import app as cli_app

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        cli_app()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--desktop-smoke", action="store_true", help="Launch the desktop UI, print smoke info, and exit automatically")
    parser.add_argument("--desktop-smoke-ms", type=int, default=1500, help="Auto-exit delay for --desktop-smoke")
    parser.add_argument("--web", action="store_true", help="Launch the legacy Flask/web shell instead of the desktop UI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--check", action="store_true", help="Validate packaged launcher imports and runtime health")
    args = parser.parse_args()

    health = runtime_health_payload()
    if args.check:
        print("R3DMatch packaged launcher OK")
        print(f"interpreter={health.get('interpreter')}")
        print(f"html_pdf_ready={health.get('html_pdf_ready')}")
        print(f"red_sdk_runtime_ready={bool((health.get('red_sdk_runtime') or {}).get('ready'))}")
        print(f"redline_ready={bool((health.get('redline_tool') or {}).get('ready'))}")
        print(f"red_backend_ready={health.get('red_backend_ready')}")
        return

    if not args.web:
        repo_root = str(Path(__file__).resolve().parents[2])
        launch_desktop_ui(repo_root, smoke_exit_ms=int(args.desktop_smoke_ms) if args.desktop_smoke else 0)
        return

    host = str(args.host or DEFAULT_HOST)
    port = _find_open_port(host, int(args.port or DEFAULT_PORT))
    url = f"http://{host}:{port}"
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)), daemon=True).start()
    print(f"R3DMatch desktop launcher: {url}")
    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
