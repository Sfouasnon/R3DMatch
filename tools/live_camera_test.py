#!/usr/bin/env python3
"""
live_camera_test.py — exercise Detect-on-Network + RCP2 against a real camera.

Runs on the MACHINE THAT IS ON THE CAMERA NETWORK (not in a sandbox). It:
  1. Scans a /24 for RCP2 cameras on port 9998 (the same approach the Cameras
     tab "Detect on Network" button uses).
  2. Connects to the camera via the real r3dmatch3.rcp2.RCP2Session.
  3. Reads camera info + current look state.
  4. Sets exposureAdjust -> 0.0 using the hardened set_and_verify path
     (camera min/max enforcement + independent rcp_get readback).
  5. Reads state back and prints a PASS/FAIL summary.

Usage:
    python3 tools/live_camera_test.py [CAMERA_IP] [SCAN_CIDR]

Defaults target today's setup:
    CAMERA_IP = 169.254.5.136
    SCAN_CIDR = 169.254.5.0/24
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import ipaddress
import socket
import sys
import time
from pathlib import Path

# Make the in-repo package importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from r3dmatch3.rcp2 import RCP2Session, VerifyStatus  # noqa: E402

CAMERA_IP = sys.argv[1] if len(sys.argv) > 1 else "169.254.5.136"
SCAN_CIDR = sys.argv[2] if len(sys.argv) > 2 else "169.254.5.0/24"
PORT = 9998
NUDGE_VALUE = 0.50      # stops — applied, verified, then reset
RESET_VALUE = 0.0       # final resting value


def hr(title: str) -> None:
    print("\n" + "─" * 64 + f"\n{title}\n" + "─" * 64)


def scan(cidr: str):
    """Phase 1 — concurrent TCP probe of port 9998 across the /24."""
    hosts = [str(h) for h in ipaddress.IPv4Network(cidr, strict=False).hosts()]
    print(f"Scanning {cidr}  ({len(hosts)} hosts) on port {PORT} …")

    def probe(ip):
        try:
            socket.create_connection((ip, PORT), timeout=0.3).close()
            return ip
        except Exception:
            return None

    found, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        for fut in concurrent.futures.as_completed({ex.submit(probe, h): h for h in hosts}):
            ip = fut.result()
            if ip:
                found.append(ip)
    found.sort(key=lambda x: tuple(int(p) for p in x.split(".")))
    print(f"  → {len(found)} open in {time.time()-t0:.1f}s: {found or '(none)'}")
    return found


def _ok(fr) -> bool:
    return fr.verify_status in (VerifyStatus.EXACT_MATCH, VerifyStatus.WITHIN_TOLERANCE)


async def rcp2_test(ip: str) -> bool:
    s = RCP2Session(ip, ip, dry_run=False, connect_timeout=6.0, handshake_timeout=6.0)
    ok = True
    try:
        await s.connect()
        info = await s.get_camera_info()
        cam = (info.get("camera_type") or {}).get("str", "?")
        sn = info.get("serial_number", "?")
        fw = (info.get("version") or {}).get("str", "?")
        print(f"  connected: {cam}  SN {sn}  FW {fw}")

        before = await s.read_state()
        print(f"  BEFORE  kelvin={before.kelvin}  tint={before.tint}  "
              f"exposureAdjust={before.exposure_adjust}")

        # Read-only divider/limit report for every parameter (confirms the camera
        # reports authoritative dividers — esp. tint, which was unverified before).
        for f in ("kelvin", "tint", "exposureAdjust"):
            raw, d, lo_, hi_ = await s._get_param_raw(f)
            print(f"  PARAM {f:14} divider={d:<5} range raw=[{lo_}, {hi_}]"
                  f"  current raw={raw}")
            if f == "tint":
                if not d or d <= 0:
                    ok = False
                    print("  !! tint divider not reported by camera")
                else:
                    print(f"  tint divider OK: 1 tint unit = {d} raw counts "
                          f"(range ±{(hi_ / d):.1f})" if hi_ is not None
                          else f"  tint divider OK: 1 tint unit = {d} raw counts")

        # Read divider + camera-reported limits once; reuse for both exposure writes.
        _, div, lo, hi = await s._get_param_raw("exposureAdjust")
        print(f"  exposureAdjust divider={div}  camera range raw=[{lo}, {hi}]")

        # ── Step 1: NUDGE to a non-zero value and verify it actually applied ────
        fr = await s.set_and_verify("exposureAdjust", NUDGE_VALUE, div,
                                    min_raw=lo, max_raw=hi)
        print(f"  NUDGE exposureAdjust={NUDGE_VALUE:+.2f} → status={fr.verify_status.value}  "
              f"readback={fr.readback_value}  Δ={fr.delta}")
        nudged = await s.read_state()
        print(f"  CHECK  exposureAdjust={nudged.exposure_adjust} (expect {NUDGE_VALUE:+.2f})")
        if not _ok(fr) or nudged.exposure_adjust is None \
                or abs(nudged.exposure_adjust - NUDGE_VALUE) > 0.001:
            ok = False
            print(f"  !! nudge did not apply/verify: {fr.error or fr.verify_status.value}")

        # ── Step 2: RESET to 0.0 and verify the change was applied ─────────────
        fr = await s.set_and_verify("exposureAdjust", RESET_VALUE, div,
                                    min_raw=lo, max_raw=hi)
        print(f"  RESET exposureAdjust={RESET_VALUE:+.2f} → status={fr.verify_status.value}  "
              f"readback={fr.readback_value}  Δ={fr.delta}")
        if not _ok(fr):
            ok = False
            print(f"  !! reset did not verify: {fr.error or fr.verify_status.value}")

        after = await s.read_state()
        print(f"  AFTER   kelvin={after.kelvin}  tint={after.tint}  "
              f"exposureAdjust={after.exposure_adjust}")
        if after.exposure_adjust is None or abs(after.exposure_adjust) > 0.001:
            ok = False
            print(f"  !! exposureAdjust did not return to 0 (={after.exposure_adjust})")
    except Exception as exc:
        ok = False
        print(f"  !! RCP2 error: {type(exc).__name__}: {exc}")
    finally:
        await s.close()
    return ok


def main() -> int:
    hr("PHASE 1 — Detect on Network")
    found = scan(SCAN_CIDR)
    detect_ok = CAMERA_IP in found
    print(f"  detect {CAMERA_IP}: {'FOUND ✓' if detect_ok else 'NOT found ✗'}")

    hr(f"PHASE 2 — RCP2 live test (nudge exposureAdjust → {NUDGE_VALUE:+.2f}, then reset → 0)")
    rcp_ok = asyncio.run(rcp2_test(CAMERA_IP))

    hr("SUMMARY")
    print(f"  Detect on Network : {'PASS' if detect_ok else 'FAIL'}")
    print(f"  RCP2 + reset      : {'PASS' if rcp_ok else 'FAIL'}")
    return 0 if (detect_ok and rcp_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
