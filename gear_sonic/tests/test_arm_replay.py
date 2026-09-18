"""Offline recording conversion and launcher tests. No SDK imports or robot I/O."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gear_sonic.scripts import launch_arm_replay as replay


class ArmReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "recordings"
        self.source.mkdir()
        self.metadata = {"format_version": 1, "status": "completed", "topic": "rt/lowstate",
                         "units": {"q": "rad"},
                         "joints": [{"name": name, "sdk_index": i + 15}
                                    for i, name in enumerate(replay.JOINTS)]}
        self.write_recording()

    def write_recording(self, stamps=(1000000000, 1009000000, 1021000000, 1045000000), values=None):
        (self.source / "metadata.json").write_text(json.dumps(self.metadata))
        with (self.source / "samples.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            # Shuffled source columns must still produce hardware-ordered targets.
            writer.writerow(["received_monotonic_ns", "mode_pr", *[n + "_q" for n in reversed(replay.JOINTS)],
                             "left_elbow_joint_dq", "left_elbow_joint_tau_est"])
            for i, stamp in enumerate(stamps):
                pose = values or [0.1 * i + j * 0.001 for j in range(14)]
                writer.writerow([stamp, 0, *reversed(pose), 9999, -9999])

    def test_nearest_selection_preserves_duration_and_ignores_feedback_commands(self):
        original = (self.source / "samples.csv").read_bytes()
        output, report = replay.prepare(self.source, self.root / "derived")
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["time_s"]) for row in rows], [0, .02, .04, .045])
        self.assertEqual([float(row["left_shoulder_pitch_joint_q"]) for row in rows], [0, .2, .3, .3])
        self.assertAlmostEqual(float(rows[1]["right_wrist_yaw_joint_q"]), .213)
        self.assertEqual(len(rows[0]), 15)
        self.assertEqual(report["replayed_fields"], ["q"])
        self.assertEqual(report["controller"], "ARM_REPLAY_G1_ENCODER_V1")
        self.assertEqual(report["encoder"]["mode_id"], 0)
        self.assertEqual(report["encoder"]["lookahead_s"], .9)
        self.assertAlmostEqual(report["duration_s"], .045)
        self.assertEqual((self.source / "samples.csv").read_bytes(), original)

    def test_invalid_times_nan_bounds_and_gaps_are_rejected(self):
        for stamps, values in [((1, 1), None), ((2, 1), None), ((1, 200000001), None),
                                ((1, 20000001), [float("nan")] * 14),
                                ((1, 20000001), [4.0] * 14)]:
            with self.subTest(stamps=stamps, values=values):
                self.write_recording(stamps, values)
                with self.assertRaises(ValueError):
                    replay.read_recording(self.source)

    def test_metadata_and_original_directory_are_protected(self):
        with self.assertRaises(ValueError):
            replay.prepare(self.source, self.source)
        self.metadata["joints"][0]["sdk_index"] = 0
        self.write_recording()
        with self.assertRaises(ValueError):
            replay.read_recording(self.source)

    def test_sample_rate_is_limited_to_control_rate(self):
        for hz in (0, 9, 51, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                replay.downsample([0, 1], [(0,) * 14] * 2, hz)

    def test_offline_modes_never_launch_deployment(self):
        for mode in ("--dry-run", "--prepare-only", "--check"):
            with self.subTest(mode=mode), patch.object(replay, "check_binary") as check, \
                    patch.object(replay.subprocess, "run") as run, redirect_stdout(io.StringIO()) as output:
                replay.main(["eno2", "--recording", str(self.source), "--output", str(self.root / "out"), mode])
                run.assert_not_called()
                self.assertEqual(check.call_count, int(mode == "--check"))
                self.assertIn("--input-type arm_replay", output.getvalue())
                self.assertIn("--disable-dex3-hands", output.getvalue())
                self.assertNotIn("--only-arms-output", output.getvalue())


if __name__ == "__main__":
    unittest.main()
