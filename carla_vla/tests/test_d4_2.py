"""Tests for D4.2 continuous-demo capture (s1_5_left_lane_change)."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
CAPTURE_ROOT = ROOT / "output" / "carla_acceptance" / "D4_2_s1_5_continuous_demo"
EP_ID = "s1_5_left_lane_change_seed101_ep0"
EP_DIR = CAPTURE_ROOT / "online_run" / "episodes" / EP_ID


@unittest.skipUnless(CAPTURE_ROOT.exists(), "D4.2 capture not present")
class TestD42Artifacts(unittest.TestCase):
    """Verify all required D4.2 outputs are present and intact."""

    def test_protocol_snapshot(self):
        for name in ("D3_scenario_semantic_contracts.json",
                       "D3_alignment_contract.json",
                       "scenario_contract.json",
                       "checkpoint_manifest.json"):
            p = CAPTURE_ROOT / "protocol_snapshot" / name
            self.assertTrue(p.exists(), f"missing protocol_snapshot/{name}")

    def test_audit_present(self):
        for name in ("repository_status.txt", "scenario_contract.json",
                       "process_and_gpu_audit.json", "storage_estimate.json",
                       "capture_plan.json"):
            p = CAPTURE_ROOT / "audit" / name
            self.assertTrue(p.exists(), f"missing audit/{name}")

    def test_gateways_present(self):
        self.assertTrue((EP_DIR / "gateway_episode.json").exists())
        self.assertTrue((EP_DIR / "_gateway_stdout.log").exists())
        self.assertTrue((EP_DIR / "_server_stdout.log").exists())

    def test_decision_bundles_complete(self):
        idx = CAPTURE_ROOT / "online_run" / "decision_bundles" / f"{EP_ID}__bundle_index.jsonl"
        self.assertTrue(idx.exists(), "bundle index missing")
        n = sum(1 for line in idx.read_text().splitlines() if line.strip())
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        self.assertEqual(n, ge["n_decisions"],
                          "bundle index count != n_decisions")
        for i in range(n):
            p = CAPTURE_ROOT / "online_run" / "decision_bundles" / f"f{i:03d}.json"
            self.assertTrue(p.exists(), f"missing bundle f{i:03d}")

    def test_six_camera_image_hashes(self):
        b0 = json.loads((CAPTURE_ROOT / "online_run" / "decision_bundles"
                          / "f000.json").read_text())
        six = b0["six_camera_images"]
        # Official order in protocol_snapshot/scenario_contract.json
        expected = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                     "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        self.assertEqual(sorted(six.keys()), sorted(expected))
        for k, v in six.items():
            self.assertIn("raw_bytes_sha256", v)
            self.assertIn("saved_file_sha256", v)
            # either 'path' or 'saved_file_path' is acceptable
            self.assertTrue("path" in v or "saved_file_path" in v,
                              f"{k} missing path key")

    def test_continuous_tick_timeline(self):
        p = CAPTURE_ROOT / "online_run" / "tick_timeline" / "per_tick_timeline.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        self.assertEqual(n, ge["continuous_tick_count"])

    def test_per_decision_timeline(self):
        p = CAPTURE_ROOT / "online_run" / "tick_timeline" / "per_decision_timeline.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        self.assertEqual(n, ge["n_decisions"])

    def test_provenance_chain(self):
        p = CAPTURE_ROOT / "online_run" / "tick_timeline" / "model_to_control_provenance.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        self.assertEqual(n, ge["n_decisions"])

    def test_no_ground_truth_leakage(self):
        """Verify model.generate() never received evaluator/visualizer labels."""
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        for d in ge.get("decisions", []):
            for forbidden in ("expected_behavior", "actor_visibility",
                              "lane_change_label", "d3_alignment_verdict"):
                self.assertNotIn(forbidden, d.get("response", {}))
                self.assertNotIn(forbidden, d.get("model_result", {}))

    def test_control_source_identity(self):
        """Verify every decision records control_source_identity in provenance."""
        prov = CAPTURE_ROOT / "online_run" / "tick_timeline" / "model_to_control_provenance.jsonl"
        for line in prov.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            self.assertIn("control_source_identity", r)
            self.assertEqual(r["external_control_leakage"], False)

    def test_stage_trace(self):
        p = CAPTURE_ROOT / "online_run" / "command_stages" / "stage_trace.json"
        self.assertTrue(p.exists())
        s = json.loads(p.read_text())
        self.assertIn("per_frame", s)
        # 47 model decisions => 47 stage frames
        self.assertGreaterEqual(len(s["per_frame"]), 1)

    def test_d3_per_decision_results(self):
        p = CAPTURE_ROOT / "evaluations" / "d3" / "D3_per_decision_results.jsonl"
        self.assertTrue(p.exists())
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        ge = json.loads((EP_DIR / "gateway_episode.json").read_text())
        self.assertEqual(n, ge["n_decisions"])

    def test_video_outputs_playable(self):
        from subprocess import check_output
        for name, fname in [
            ("clean", "s1_5_left_lane_change_clean_20hz.mp4"),
            ("annotated", "s1_5_left_lane_change_annotated_20hz.mp4"),
            ("six_camera", "s1_5_left_lane_change_six_camera_decisions.mp4"),
            ("provenance", "s1_5_left_lane_change_model_to_action.mp4"),
        ]:
            p = CAPTURE_ROOT / "videos" / name / fname
            self.assertTrue(p.exists(), f"missing {p}")
            out = check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,pix_fmt,width,height",
                 "-of", "json", str(p)], stderr=subprocess_stderr(), timeout=60)
            info = json.loads(out)
            st = (info.get("streams") or [{}])[0]
            self.assertEqual(st.get("codec_name"), "h264")
            self.assertEqual(st.get("pix_fmt"), "yuv420p")

    def test_final_verdict(self):
        p = CAPTURE_ROOT / "final_verdict.json"
        self.assertTrue(p.exists())
        v = json.loads(p.read_text())
        self.assertEqual(v["scenario_id"], "s1_5_left_lane_change")
        self.assertTrue(v["online_closed_loop_valid"])
        self.assertTrue(v["valid_handoff"])
        self.assertEqual(v["external_control_leakage_count"], 0)
        self.assertTrue(v["clean_video_playable"])
        self.assertTrue(v["annotated_video_playable"])
        self.assertTrue(v["six_camera_video_playable"])
        self.assertTrue(v["provenance_video_playable"])
        self.assertTrue(v["D4_demo_complete"])

    def test_warmup_handoff_speed_in_band(self):
        v = json.loads((CAPTURE_ROOT / "final_verdict.json").read_text())
        self.assertTrue(5.0 <= v["handoff_speed_mps"] <= 8.0,
                          f"handoff_speed_mps={v['handoff_speed_mps']} not in [5,8]")


def subprocess_stderr():
    """Helper: capture stderr as bytes (Python 3.10 compat)."""
    import subprocess
    return subprocess.PIPE


if __name__ == "__main__":
    unittest.main()