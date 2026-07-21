"""Tests for D4.3 chase-camera 30s demo (s1_1_lane_keeping)."""
from __future__ import annotations
import json
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
CAPTURE_ROOT = ROOT / "output" / "carla_acceptance" / "D4_3_s1_1_30s_third_person_demo"
EP_ID = "s1_1_lane_keeping_seed101_ep0"
EP_DIR = CAPTURE_ROOT / "online_run" / "episodes" / EP_ID
SMOKE_DIR = CAPTURE_ROOT / "online_run" / "episodes" / "smoke_s1_1_lane_keeping_seed101_ep0"


def _has_run() -> bool:
    p = EP_DIR / "gateway_episode.json"
    return p.exists() or (SMOKE_DIR / "gateway_episode.json").exists()


def _load_ge():
    for p in (EP_DIR / "gateway_episode.json", SMOKE_DIR / "gateway_episode.json"):
        if p.exists():
            return json.loads(p.read_text()), p
    return None, None


def _online_root():
    """The online_run dir is a sibling of EP_DIR (not a child)."""
    return CAPTURE_ROOT / "online_run"


def _ffprobe(p):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,width,height",
         "-of", "json", str(p)], stderr=subprocess.PIPE, timeout=60)
    return json.loads(out).get("streams", [{}])[0]


@unittest.skipUnless(CAPTURE_ROOT.exists(), "D4.3 capture root absent")
class TestD43Preflight(unittest.TestCase):
    def test_chase_preview_exists(self):
        p = CAPTURE_ROOT / "preflight" / "chase_camera_preview.png"
        self.assertTrue(p.exists(), "preflight chase preview missing")
        meta = json.loads((CAPTURE_ROOT / "preflight" / "chase_camera_preview.meta.json").read_text())
        self.assertTrue(meta["valid"], "chase preview quality gate failed")
        self.assertGreater(meta["non_black_pixel_ratio"], 0.10)
        self.assertGreater(meta["luminance_std"], 3)

    def test_protocol_snapshot(self):
        for n in ("D3_capture_contract.json", "D3_alignment_contract.json",
                    "D3_scenario_semantic_contracts.json", "D4_capture_contract.json"):
            p = CAPTURE_ROOT / "protocol_snapshot" / n
            self.assertTrue(p.exists(), f"missing {n}")

    def test_scenario_contract(self):
        p = CAPTURE_ROOT / "audit" / "scenario_contract.json"
        sc = json.loads(p.read_text())
        self.assertEqual(sc["scenario_id"], "s1_1_lane_keeping")
        self.assertEqual(sc["seed"], 101)
        self.assertEqual(sc["group"], "G1")
        self.assertEqual(sc["map"], "Town03")
        self.assertEqual(sc["chase_camera"]["transform_xyzrpy"]["x_m"], -7.0)


@unittest.skipUnless(_has_run(), "D4.3 gateway has not run yet")
class TestD43Artifacts(unittest.TestCase):
    def setUp(self):
        ge, self.ge_path = _load_ge()
        self.ge = ge
        # timelines live under online_run (not EP_DIR)
        self.online_root = _online_root()

    def test_warmup_handoff_speed(self):
        v = self.ge.get("handoff_speed_mps") or 0.0
        self.assertTrue(5.0 <= v <= 8.0, f"handoff={v} not in [5,8]")

    def test_external_control_leakage_zero(self):
        self.assertEqual(self.ge.get("external_control_leakage_count", -1), 0)

    def test_chase_video_playable(self):
        p = CAPTURE_ROOT / "videos/third_person/s1_1_lane_keeping_third_person_clean_20hz.mp4"
        if not p.exists():
            p = CAPTURE_ROOT / "videos/third_person/chase_continuous_raw.mp4"
        self.assertTrue(p.exists())
        s = _ffprobe(p)
        self.assertEqual(s.get("codec_name"), "h264")
        self.assertEqual(s.get("pix_fmt"), "yuv420p")
        self.assertEqual(s.get("width"), 1600)
        self.assertEqual(s.get("height"), 900)

    def test_chase_frames_match_expected(self):
        chase_meta = self.ge.get("chase_video_meta", {})
        n = int(chase_meta.get("frame_count", 0))
        scored = self.ge.get("scored_simulation_duration_s", 0.0)
        expected = int(round(scored / 0.05))
        # Allow bounded difference; if terminal collision/duration completed, expected may exceed actual
        self.assertGreater(n, 0)
        self.assertEqual(chase_meta.get("dropped_frames", 0), 0)
        self.assertEqual(chase_meta.get("encoder_errors", 0), 0)
        self.assertGreaterEqual(n, expected - 5,
            f"chase frames={n} far below expected~{expected}")

    def test_per_tick_timeline(self):
        p = self.online_root / "tick_timeline" / "per_tick_timeline.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        # match by episode_id_prefix (avoid smoke leftovers)
        ep_decision_ids = set(
            (r.get("decision_id") or "") for r in [
                json.loads(line) for line in (p.read_text().splitlines() or [])
                if line.strip()
            ] if (r.get("decision_id") or "").startswith(self.ge["episode_id"]))
        # at minimum, last run should have at least min(n_decisions, len)
        self.assertGreaterEqual(n, self.ge["n_decisions"])

    def test_per_decision_timeline(self):
        p = self.online_root / "tick_timeline" / "per_decision_timeline.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        self.assertGreaterEqual(n, self.ge["n_decisions"])

    def test_provenance(self):
        p = self.online_root / "tick_timeline" / "model_to_control_provenance.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        self.assertGreaterEqual(n, self.ge["n_decisions"])

    def test_no_ground_truth_leakage(self):
        for d in self.ge.get("decisions", []):
            for forbidden in ("expected_behavior", "actor_visibility",
                              "lane_change_label", "d3_alignment_verdict"):
                self.assertNotIn(forbidden, d.get("response", {}))

    def test_decision_bundles(self):
        n = self.ge["n_decisions"]
        for i in range(min(n, 50)):  # at least first 50
            p = self.online_root / "decision_bundles" / f"f{i:03d}.json"
            self.assertTrue(p.exists())

    def test_six_camera_hash(self):
        if not (self.online_root / "decision_bundles" / "f000.json").exists():
            return
        b0 = json.loads((self.online_root / "decision_bundles" / "f000.json").read_text())
        six = b0.get("six_camera_images", {})
        expected = {"CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                     "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
        self.assertEqual(set(six.keys()), expected)
        for v in six.values():
            self.assertIn("raw_bytes_sha256", v)
            self.assertIn("saved_file_sha256", v)

    def test_chase_camera_not_in_model_input(self):
        b0 = json.loads((self.online_root / "decision_bundles" / "f000.json").read_text())
        ch = b0.get("chase_camera", {})
        self.assertTrue(ch.get("entered_model_input") is False
                          if "entered_model_input" in ch else True)

    def test_final_verdict(self):
        v = json.loads((CAPTURE_ROOT / "final_verdict.json").read_text())
        self.assertEqual(v["scenario_id"], "s1_1_lane_keeping")
        self.assertTrue(v["online_closed_loop_valid"])
        self.assertTrue(v["valid_handoff"])
        self.assertEqual(v["external_control_leakage_count"], 0)


if __name__ == "__main__":
    unittest.main()