#!/usr/bin/env python3
"""
test_capture.py — hardware-free unit tests for the Capture module logic.

Covers the parts that must be right *before* you ever point it at a camera:
WebSocket framing, timecode math (including 24h wrap and drop-frame), the
sync-verification verdict, FTP clip selection, and the credential guard.

Run:  python3 -m unittest tools.test_capture   (from the repo root)
  or: python3 tools/test_capture.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from r3dmatch3 import capture as C          # noqa: E402
from r3dmatch3 import capture_ftp as F      # noqa: E402


class TestWebSocketFraming(unittest.TestCase):
    def test_roundtrip_unmasked(self):
        buf = bytearray(C.encode_frame(0x1, b'{"a":1}', mask=False))
        fin, op, payload = C.parse_frame(buf)
        self.assertEqual((fin, op, payload), (0x80, 0x1, b'{"a":1}'))
        self.assertEqual(len(buf), 0)   # frame consumed

    def test_roundtrip_masked(self):
        # A client frame is masked; decode must recover the plaintext.
        raw = C.encode_frame(0x1, b'hello world', mask=True)
        fin, op, payload = C.parse_frame(bytearray(raw))
        self.assertEqual(payload, b'hello world')

    def test_partial_frame_returns_none(self):
        raw = C.encode_frame(0x1, b'xxxxxxxx', mask=False)
        self.assertIsNone(C.parse_frame(bytearray(raw[:3])))

    def test_extended_length(self):
        payload = b'z' * 1000
        fin, op, got = C.parse_frame(bytearray(C.encode_frame(0x2, payload, mask=False)))
        self.assertEqual(got, payload)


class TestTimecode(unittest.TestCase):
    def test_parse_seconds_resolution(self):
        self.assertEqual(C.tc_to_seconds("14:28:34"), 14 * 3600 + 28 * 60 + 34)

    def test_parse_with_frames(self):
        self.assertEqual(C.parse_tc("01:02:03:12"), (1, 2, 3, 12, False))

    def test_parse_drop_frame(self):
        self.assertEqual(C.parse_tc("01:02:03;12"), (1, 2, 3, 12, True))

    def test_parse_invalid(self):
        self.assertIsNone(C.tc_to_seconds("not a timecode"))
        self.assertIsNone(C.parse_tc(""))

    def test_target_basic_lead(self):
        self.assertEqual(C.compute_target_tc("14:28:34", 3), "14:28:37:00")

    def test_target_min_lead_enforced(self):
        # A 1s lead is bumped to the 2s minimum (send latency eats 1s).
        self.assertEqual(C.compute_target_tc("00:00:00", 1), "00:00:02:00")

    def test_target_wraps_midnight(self):
        self.assertEqual(C.compute_target_tc("23:59:59", 3), "00:00:02:00")

    def test_target_preserves_dropframe_delim(self):
        self.assertEqual(C.compute_target_tc("01:02:03;10", 2), "01:02:05;00")

    def test_target_unparseable(self):
        self.assertIsNone(C.compute_target_tc("garbage", 3))


class _FakeLink:
    """Stand-in for CameraLink exposing only .snapshot()."""
    def __init__(self, ip, tc, sync="locked", connected=True, stale=False, rec=0):
        self._s = C.LinkState(ip=ip, connected=connected, stale=stale, timecode=tc,
                              sync_state=sync, record_state=rec,
                              frame_limit_enable=True, frame_limit_frames=1)

    def snapshot(self):
        return C.LinkState(**vars(self._s))


class TestSyncVerify(unittest.TestCase):
    def test_all_in_sync(self):
        links = {"AA": _FakeLink("1.1.1.1", "10:00:05"),
                 "AB": _FakeLink("1.1.1.2", "10:00:05")}
        rep = C.verify_sync(links)
        self.assertTrue(rep.in_sync)
        self.assertEqual(rep.spread_seconds, 0)
        self.assertEqual(rep.reference_tc, "10:00:05")

    def test_timecode_spread_fails(self):
        links = {"AA": _FakeLink("1.1.1.1", "10:00:05"),
                 "AB": _FakeLink("1.1.1.2", "10:00:09")}  # 4s apart
        rep = C.verify_sync(links)
        self.assertFalse(rep.in_sync)
        self.assertEqual(rep.spread_seconds, 4)

    def test_one_second_tolerance(self):
        # A 1s spread is tolerated (absorbs the 1/s TC push boundary).
        links = {"AA": _FakeLink("1.1.1.1", "10:00:05"),
                 "AB": _FakeLink("1.1.1.2", "10:00:06")}
        self.assertTrue(C.verify_sync(links).in_sync)

    def test_free_run_flagged(self):
        links = {"AA": _FakeLink("1.1.1.1", "10:00:05"),
                 "AB": _FakeLink("1.1.1.2", "10:00:05", sync="free-run")}
        rep = C.verify_sync(links)
        self.assertTrue(any("sync_state" in p for p in rep.problems))

    def test_stale_link_not_in_sync(self):
        links = {"AA": _FakeLink("1.1.1.1", "10:00:05"),
                 "AB": _FakeLink("1.1.1.2", "10:00:05", stale=True)}
        self.assertFalse(C.verify_sync(links).in_sync)


class TestFtpLogic(unittest.TestCase):
    def test_clip_sort_key_orders_by_number(self):
        self.assertLess(F._clip_sort_key("A001_C007.RDC"),
                        F._clip_sort_key("A001_C063.RDC"))

    def test_credentials_required(self):
        with self.assertRaises(F.FTPConfigError):
            F.ftp_connect("10.0.0.1", "", "")
        with self.assertRaises(F.FTPConfigError):
            F.pull_array({"AA": "10.0.0.1"}, "/tmp/x", "", "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
