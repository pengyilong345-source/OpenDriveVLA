"""IPC unit tests: synthetic six-camera arrays through the full shm + socket path."""
from __future__ import annotations
import json
import os
import socket
import tempfile
import time
import unittest

from carla_vla.online import ipc_protocol as P
from carla_vla.online import shared_frame_buffer as S
from carla_vla.online import latency_profiler as L
from carla_vla.online import process_health as H


def _make_cam_arrays(seed: int):
    import numpy as np
    return [np.full((S.CAM_H, S.CAM_W, 3), (seed + i) % 256, dtype=np.uint8)
            for i in range(S.N_CAMS)]


class FrameBufferTests(unittest.TestCase):
    def test_publish_read_roundtrip(self):
        path = tempfile.mkstemp(prefix="odvla_ipc_", dir="/dev/shm")[1]
        os.remove(path)  # FrameWriter creates it
        try:
            w = S.FrameWriter(path)
            cams = _make_cam_arrays(7)
            packed = S.pack_cameras(cams)
            seq = w.publish(frame_id=3, sensor_timestamp_ns=123, cam_bytes=packed,
                             episode_id="ep1")
            self.assertEqual(seq, 1)
            r = S.FrameReader(path)
            hdr, cam, rseq = r.read_latest()
            self.assertEqual(hdr["frame_id"], 3)
            self.assertEqual(hdr["write_seq"], 1)
            self.assertEqual(hdr["sha256"], P.sha256_hex(packed))
            self.assertEqual(rseq, 1)
            unpacked = S.unpack_cameras(cam)
            for a, b in zip(cams, unpacked):
                self.assertTrue((a == b).all())
            w.close(); r.close()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_torn_read_retries(self):
        path = tempfile.mkstemp(prefix="odvla_ipc_", dir="/dev/shm")[1]
        os.remove(path)
        try:
            w = S.FrameWriter(path)
            cams = _make_cam_arrays(1)
            packed = S.pack_cameras(cams)
            w.publish(frame_id=1, sensor_timestamp_ns=1, cam_bytes=packed, episode_id="ep")
            r = S.FrameReader(path)
            hdr, cam, seq = r.read_latest(max_retries=2)
            self.assertIsNotNone(hdr)
            self.assertEqual(seq, 1)
            w.close(); r.close()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_pack_unpack_shape_check(self):
        import numpy as np
        bad = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(6)]
        with self.assertRaises(ValueError):
            S.pack_cameras(bad)


class SocketEnvelopeTests(unittest.TestCase):
    def test_send_recv_envelope(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path = tempfile.mkstemp(prefix="odvla_sock_", dir="/tmp")[1]
        os.remove(path)
        srv.bind(path)
        srv.listen(1)
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect(path)
        acc, _ = srv.accept()
        P.send_envelope(cli, {"kind": "request", "frame_id": 42, "x": 1.5})
        got = P.recv_envelope(acc, timeout_s=2.0)
        self.assertEqual(got["frame_id"], 42)
        # second envelope on the same socket
        P.send_envelope(cli, {"kind": "response", "frame_id": 43})
        got2 = P.recv_envelope(acc, timeout_s=2.0)
        self.assertEqual(got2["frame_id"], 43)
        cli.close(); acc.close(); srv.close()
        os.remove(path)

    def test_stale_detection(self):
        self.assertTrue(P.is_stale_response(5, 4))
        self.assertFalse(P.is_stale_response(5, 5))


class LatencyTests(unittest.TestCase):
    def test_record_and_deltas(self):
        r = L.LatencyRecord(episode_id="e", frame_id=1, request_id="r", model_group="G1")
        base = 1_000_000_000
        for i, s in enumerate(L.STAGES):
            r.set(s, base + i * 1_000_000)  # 1 ms per stage
        d = r.deltas_ms()
        for k, _, _ in L.DELTAS:
            self.assertAlmostEqual(d[k], 1.0, places=3, msg=k)
        self.assertAlmostEqual(d["total_decision_latency_ms"], 10.0, places=3)

    def test_aggregate_and_deadline(self):
        recs = []
        for i in range(5):
            r = L.LatencyRecord(episode_id="e", frame_id=i, request_id="r", model_group="G1")
            base = 1_000_000_000
            r.set("T0", base)
            r.set("T10", base + (50 + i * 50) * 1_000_000)  # 50,100,150,200,250 ms
            recs.append(r)
        agg = L.aggregate(recs, deadline_ms=150.0)
        self.assertEqual(agg["n_valid"], 5)
        self.assertEqual(agg["deadline_miss_count"], 2)
        self.assertFalse(agg["strict_verdict_pass"])
        # sorted = [50,100,150,200,250]; p90 by linear interp = 200 + 0.6*(250-200) = 230
        self.assertAlmostEqual(agg["totals"]["p90"], 230.0, places=1)
        self.assertAlmostEqual(agg["totals"]["max"], 250.0, places=1)

    def test_aggregate_empty(self):
        agg = L.aggregate([], deadline_ms=150.0)
        self.assertEqual(agg["n_valid"], 0)
        self.assertEqual(agg["totals"]["count"], 0)


class HealthTests(unittest.TestCase):
    def test_heartbeat_alive_and_restart(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "hb.jsonl")
            h = H.HeartbeatLogger(log, role="gateway", period_s=0.0)
            for _ in range(3):
                h.beat(); time.sleep(0.01)
            snap = H.diagnose(log, timeout_s=5.0)
            self.assertTrue(snap.alive)
            self.assertEqual(snap.n_restarts, 0)
            # simulate a restart (new boot_id)
            h2 = H.HeartbeatLogger(log, role="gateway", period_s=0.0)
            h2.beat()
            snap2 = H.diagnose(log, timeout_s=5.0)
            self.assertEqual(snap2.n_restarts, 1)

    def test_diagnose_missing_log(self):
        snap = H.diagnose("/nonexistent/path.jsonl", timeout_s=1.0)
        self.assertFalse(snap.alive)


if __name__ == "__main__":
    unittest.main()
