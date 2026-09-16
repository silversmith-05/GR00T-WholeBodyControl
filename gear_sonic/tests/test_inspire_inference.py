"""Deployment contracts and real SDK over fake transport. No robot connection."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import msgpack
import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.launch_inspire_inference import check_output, commands, parser
from gear_sonic.scripts.run_inspire_inference import LiveSensors, Runner
from gear_sonic.tests import test_inspire_hand_controller as legacy
from gear_sonic.utils.inference.inspire_bridge import (
    ActionTimeline,
    PolicyClient,
    Prediction,
    SensorCache,
    _decode,
    _encode,
    build_observation,
    pack_action,
)


def hand(now):
    return dict(
        connected=True,
        angle_valid=True,
        angle_monotonic=now,
        angle=[850, 850, 850, 850, 1000, 3],
        fault_count=0,
        thumb_rotation=3,
        thumb_target_valid=True,
        write_status=2,
    )


def fresh_cache(now=10):
    cache = SensorCache()
    cache.config = dict(
        dex3_hands_enabled=False,
        model_path="policy/release/model_decoder.onnx",
        encoder_file="policy/release/model_encoder.onnx",
        obs_config_path="policy/release/observation_config.yaml",
        control_frequency=50,
    )
    cache.update_body(dict(index=1, body_q=list(range(29)), base_quat=[1, 0, 0, 0], token_state=[0.1] * 64), now)
    cache.update_camera(
        dict(
            images={k: np.zeros((480, 640, 3), np.uint8) for k in ("ego_view", "left_wrist", "right_wrist")},
            timestamps={k: 100.0 for k in ("ego_view", "left_wrist", "right_wrist")},
        ),
        now,
    )
    return cache


def prediction(observed=10, epoch=0, request_id=1):
    return Prediction(
        epoch,
        request_id,
        observed,
        observed + 0.01,
        observed + 0.1,
        dict(
            motion_token=np.full((1, 40, 64), 0.2, np.float32),
            hand=np.tile(np.array([0.49, 0.5], np.float32), (1, 40, 1)),
        ),
    )


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    def test_body_order_matches_collection_and_hand_feedback_is_not_binarized(self):
        cache = fresh_cache()
        hands = [hand(10), hand(10)]
        obs = build_observation(cache.body, cache.camera, hands, self.model, "pick up the ball")
        body = self.model.get_configuration_from_actuated_joints(body_actuated_joint_values=np.arange(29))
        collected = body[self.model.get_body_actuated_joint_indices()]
        reconstructed = np.concatenate(
            [obs["state"][k][0, 0] for k in ("left_leg", "right_leg", "waist", "left_arm", "right_arm")]
        )
        np.testing.assert_array_equal(reconstructed, collected)
        self.assertEqual(sum(v.shape[-1] for v in obs["state"].values()), 44)
        np.testing.assert_array_equal(obs["state"]["left_hand"][0, 0], hands[0]["angle"])
        np.testing.assert_array_equal(obs["state"]["projected_gravity"][0, 0], [0, 0, -1])
        self.assertEqual(set(obs["video"]), {"ego_view", "left_wrist", "right_wrist"})
        self.assertEqual(obs["video"]["right_wrist"].shape, (1, 1, 480, 640, 3))

    def test_missing_camera_and_nonfinite_body_rejected(self):
        cache = fresh_cache()
        del cache.camera["images"]["right_wrist"]
        with self.assertRaises(KeyError):
            build_observation(cache.body, cache.camera, [hand(10)] * 2, self.model, "task")
        cache = fresh_cache()
        cache.body["body_q"][0] = float("nan")
        with self.assertRaises(ValueError):
            build_observation(cache.body, cache.camera, [hand(10)] * 2, self.model, "task")

    def test_repeated_source_markers_never_refresh_freshness(self):
        cache = fresh_cache()
        cache.update_body(cache.body, 10.3)
        cache.update_camera(cache.camera, 10.3)
        with self.assertRaisesRegex(RuntimeError, "Body"):
            cache.snapshot([hand(10.3)] * 2, 10.3)
        cache.update_body(dict(cache.body, index=2), 10.3)
        with self.assertRaisesRegex(RuntimeError, "Camera"):
            cache.snapshot([hand(10.3)] * 2, 10.3)

    def test_single_camera_stall_and_stale_hand(self):
        cache = fresh_cache()
        camera = copy.deepcopy(cache.camera)
        camera["timestamps"]["ego_view"] += 1
        camera["timestamps"]["left_wrist"] += 1
        cache.update_camera(camera, 10.3)
        cache.update_body(dict(cache.body, index=2), 10.3)
        with self.assertRaisesRegex(RuntimeError, "right_wrist"):
            cache.snapshot([hand(10.3)] * 2, 10.3)
        with self.assertRaisesRegex(RuntimeError, "hand"):
            fresh_cache().snapshot([hand(8.9)] * 2, 10)

    def test_hand_timeout_warns_once_recovers_and_stops_after_one_second(self):
        cache = fresh_cache()
        logger = "gear_sonic.utils.inference.inspire_bridge"
        with self.assertLogs(logger, level="WARNING") as logs:
            cache.snapshot([hand(9.4), hand(10)], 10)
            cache.snapshot([hand(9.3), hand(10)], 10)
            cache.snapshot([hand(9.0), hand(10)], 10)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("side=left age_ms=600.000", logs.output[0])
        with self.assertLogs(logger, level="INFO") as logs:
            cache.snapshot([hand(10)] * 2, 10)
        self.assertIn("hand_feedback_recovered", logs.output[0])
        with self.assertLogs(logger, level="WARNING"):
            cache.snapshot([hand(10), hand(9.4)], 10)
        with self.assertRaisesRegex(RuntimeError, "right.*1000 ms.*age_ms=1001"):
            cache.snapshot([hand(10), hand(8.999)], 10)

    def test_observation_checks_time_after_hand_snapshot(self):
        runner = Runner.__new__(Runner)
        runner.hands = Mock()
        runner.hands.snapshot.return_value = {"hands": [hand(10.0001)] * 2}
        runner.sensors = SimpleNamespace(cache=fresh_cache())
        runner.robot_model = self.model
        runner.config = SimpleNamespace(prompt="task")
        with patch("gear_sonic.utils.inference.inspire_bridge.time.monotonic", return_value=10.0002):
            runner.observation(10.0)

    def test_disconnected_invalid_and_future_hand_feedback_still_rejected(self):
        for changes in ({"connected": False}, {"angle_valid": False}, {"angle_monotonic": 11}):
            bad = dict(hand(10), **changes)
            with self.assertRaisesRegex(RuntimeError, "left hand"):
                fresh_cache().snapshot([bad, hand(10)], 10)

    def test_config_and_restart_rejected(self):
        cache = fresh_cache()
        cache.config["dex3_hands_enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "Dex3|dex3"):
            cache.snapshot([hand(10)] * 2, 10)
        cache = fresh_cache()
        cache.config["model_path"] = "policy/sonic_v1_1/model_decoder.onnx"
        with self.assertRaisesRegex(RuntimeError, "SONIC"):
            cache.snapshot([hand(10)] * 2, 10)
        with self.assertRaisesRegex(RuntimeError, "restarted"):
            cache.update_body(dict(cache.body, index=0), 10.1)


class TimelineTests(unittest.TestCase):
    def test_latency_selection_and_binary_threshold(self):
        timeline = ActionTimeline()
        self.assertTrue(timeline.accept(prediction(), 10.101))
        step = timeline.step(10.101)
        self.assertEqual(step["action_index"], 5)
        np.testing.assert_array_equal(step["hand"], [0, 1])
        self.assertIsNone(timeline.step(10.102))
        self.assertAlmostEqual(step["valid_until"], 10.8)

    def test_exhaustion_is_fault_not_last_frame_repeat(self):
        timeline = ActionTimeline()
        timeline.accept(prediction(), 10.1)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            timeline.step(10.81)

    def test_expired_nonfinite_and_wrong_shape_predictions(self):
        timeline = ActionTimeline()
        with self.assertRaisesRegex(RuntimeError, "expired"):
            timeline.accept(prediction(), 10.81)
        bad = prediction()
        bad.actions["hand"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            timeline.accept(bad, 10.1)
        bad = prediction()
        bad.actions["motion_token"] = np.zeros((1, 40, 63))
        with self.assertRaises(ValueError):
            timeline.accept(bad, 10.1)

    def test_pause_prompt_and_initialization_invalidate_late_results(self):
        timeline = ActionTimeline()
        old = prediction()
        for _ in range(3):
            timeline.invalidate()
            self.assertFalse(timeline.accept(old, 10.1))
            self.assertIsNone(timeline.step(10.1))

    def test_protocol_has_no_dex3_fields(self):
        raw = pack_action(np.arange(64, dtype=np.float32), 42)
        self.assertEqual(raw[:4], b"pose")
        header = json.loads(raw[4:1284].rstrip(b"\0"))
        self.assertEqual(header["v"], 4)
        self.assertEqual([f["name"] for f in header["fields"]], ["token_state", "frame_index"])
        np.testing.assert_array_equal(np.frombuffer(raw[1284 : 1284 + 256], "<f4"), np.arange(64))

    def test_official_wire_numpy_and_modality_markers(self):
        value = {"state": np.ones((1, 1, 6), np.float32)}
        decoded = msgpack.unpackb(msgpack.packb(value, default=_encode), raw=False, object_hook=_decode)
        np.testing.assert_array_equal(decoded["state"], value["state"])
        self.assertEqual(
            _decode({"__ModalityConfig__": True, "as_json": {"modality_keys": ["hand"]}}),
            {"modality_keys": ["hand"]},
        )
        with self.assertRaises(ValueError):
            _decode({b"nd": True, b"kind": b"O", b"data": b"do not unpickle"})

    def test_rpc_request_matches_official_keyword_dispatch(self):
        # Exercise serialization without opening sockets or importing GR00T.
        client = PolicyClient.__new__(PolicyClient)
        client.socket = Mock()
        client.socket.recv.return_value = msgpack.packb([{}, {}])
        obs = {"state": {"left_hand": np.zeros((1, 1, 6), np.float32)}}
        client.get_action(obs)
        request = msgpack.unpackb(client.socket.send.call_args.args[0], raw=False, object_hook=_decode)
        self.assertEqual(request["endpoint"], "get_action")
        self.assertEqual(set(request["data"]), {"observation", "options"})

        def handler(observation, options):
            return observation

        np.testing.assert_array_equal(handler(**request["data"])["state"]["left_hand"], obs["state"]["left_hand"])


class PolicyHandTests(unittest.TestCase):
    # Share the real SDK/fake-device fixture, without inheriting legacy tests.
    setUp = legacy.ControllerTests.setUp
    tearDown = legacy.ControllerTests.tearDown
    make_controller = legacy.ControllerTests.make_controller
    wait = legacy.ControllerTests.wait

    def submit(self, targets, sequence=0, deadline=None):
        now = time.monotonic()
        self.hands.update_policy(
            targets, sequence=sequence, sample_monotonic=now, valid_until=deadline or now + 0.8
        )

    def test_first_close_and_first_open_both_execute_without_trigger_release(self):
        self.wait(lambda: all(h["angle_valid"] for h in self.hands.snapshot()["hands"]))
        thumbs = self.hands.begin_policy_session()
        self.assertEqual(thumbs, [339, 339])
        self.assertFalse(self.left.writes or self.right.writes)
        self.submit([1, 0])
        self.wait(lambda: bool(self.left.angles_sent and self.right.angles_sent))
        self.assertEqual(self.left.angles_sent[0], [250, 250, 250, 250, 300, 339])
        self.assertEqual(self.right.angles_sent[0], [850, 850, 850, 850, 1000, 339])
        for seq in range(1, 5):
            self.submit([1, 0], seq)
        time.sleep(0.03)
        self.assertEqual(len(self.left.angles_sent), 1)
        self.assertEqual(len(self.right.angles_sent), 1)

    def test_thumb_seed_is_fresh_measured_value_and_kept(self):
        self.left.angle[5], self.right.angle[5] = 3, 0
        self.wait(lambda: self.hands.snapshot()["hands"][0]["angle"][5] == 3)
        self.assertEqual(self.hands.begin_policy_session(), [3, 0])
        self.submit([1, 1])
        self.wait(lambda: bool(self.left.angles_sent and self.right.angles_sent))
        self.left.angle[5] = 8
        self.wait(lambda: self.hands.snapshot()["hands"][0]["angle"][5] == 8)
        self.submit([0, 0], 1)
        self.wait(lambda: len(self.left.angles_sent) == 2)
        self.assertEqual(self.left.angles_sent[-1][-1], 3)
        self.hands.cancel()
        time.sleep(0.03)
        self.assertEqual(self.hands.begin_policy_session(), [8, 0])

    def test_session_cancel_and_expired_chunk_block_writes(self):
        self.wait(lambda: all(h["angle_valid"] for h in self.hands.snapshot()["hands"]))
        self.hands.begin_policy_session()
        self.hands.cancel()
        with self.assertRaises(RuntimeError):
            self.submit([1, 1])
        self.assertFalse(self.left.writes or self.right.writes)
        self.hands.begin_policy_session()
        with self.assertRaises(RuntimeError):
            self.submit([1, 1], deadline=time.monotonic() - 0.01)
        self.assertFalse(self.left.writes or self.right.writes)

    def test_write_timeout_never_replays_and_invalidates_session(self):
        self.wait(lambda: all(h["angle_valid"] for h in self.hands.snapshot()["hands"]))
        self.hands.begin_policy_session()
        self.left.fail_address = 1486
        self.submit([1, 0])
        self.wait(lambda: self.hands.snapshot()["hands"][0]["fault_count"] > 0)
        sent = len(self.left.angles_sent)
        time.sleep(0.07)
        with self.assertRaises(RuntimeError):
            self.submit([1, 0], 1)
        self.assertEqual(len(self.left.angles_sent), sent)

    def test_new_policy_goal_cancels_partial_write_without_fault_and_executes_latest(self):
        self.wait(lambda: all(h["angle_valid"] for h in self.hands.snapshot()["hands"]))
        self.hands.begin_policy_session()
        self.left.write_gate.clear()
        try:
            self.submit([1, 0])
            self.wait(self.left.write_entered.is_set)
            # Replace the close goal while its force write is in progress.
            self.submit([0, 0], 1)
            with self.assertLogs("gear_sonic.utils.teleop.inspire_hand_controller", level="WARNING") as logs:
                self.left.write_gate.set()
                self.wait(lambda: bool(self.left.angles_sent))
            self.assertTrue(any("superseded=True" in line for line in logs.output))
            self.assertEqual(self.left.angles_sent, [[850, 850, 850, 850, 1000, 339]])
            state = self.hands.snapshot()["hands"][0]
            self.assertEqual(state["fault_count"], 0)
            self.assertEqual(state["write_status"], 2)
            self.assertTrue(state["write_current"])
            # Same session still accepts and executes subsequent commands.
            self.submit([1, 0], 2)
            self.wait(lambda: len(self.left.angles_sent) == 2)
            self.assertEqual(self.left.angles_sent[-1], [250, 250, 250, 250, 300, 339])
        finally:
            self.left.write_gate.set()

    def test_expired_input_is_not_treated_as_normal_policy_replacement(self):
        self.wait(lambda: all(h["angle_valid"] for h in self.hands.snapshot()["hands"]))
        self.hands.begin_policy_session()
        self.left.write_gate.clear()
        try:
            self.submit([1, 0])
            self.wait(self.left.write_entered.is_set)
            self.submit([0, 0], 1)
            time.sleep(0.27)
            self.left.write_gate.set()
            self.wait(lambda: self.hands.snapshot()["hands"][0]["fault_count"] > 0)
            self.assertFalse(self.left.angles_sent)
        finally:
            self.left.write_gate.set()


class RunnerTests(unittest.TestCase):
    def test_deploy_port_overrides_parse_without_starting_control(self):
        script = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/deploy.sh"
        result = subprocess.run(
            ["bash", str(script), "--zmq-port", "5596", "--zmq-out-port", "5597", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for invalid in ("0", "65536", "-1", "abc", ""):
            with self.subTest(port=invalid):
                result = subprocess.run(
                    ["bash", str(script), "--zmq-port", invalid, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("1..65535", result.stderr)

    def test_body_receiver_startup_error_is_reported(self):
        sensors = LiveSensors.__new__(LiveSensors)
        sensors.cache = SensorCache()
        config = SimpleNamespace(state_host="127.0.0.1", state_port=5557)
        with patch(
            "gear_sonic.scripts.run_inspire_inference.ZMQStateSubscriber",
            side_effect=RuntimeError("test startup failure"),
        ):
            sensors._body(config)
        self.assertIn("test startup failure", sensors.cache.error)

    def test_camera_receiver_startup_error_is_reported(self):
        sensors = LiveSensors.__new__(LiveSensors)
        sensors.cache = SensorCache()
        config = SimpleNamespace(camera_host="127.0.0.1", camera_port=5555)
        with patch(
            "gear_sonic.camera.composed_camera.ComposedCameraClientSensor",
            side_effect=RuntimeError("test camera failure"),
        ):
            sensors._camera(config)
        self.assertIn("test camera failure", sensors.cache.error)

    def make_runner(self, mode="real"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        token = Path(temp.name) / "token.json"
        token.write_text(json.dumps(dict(token=[0.2] * 64, source={"episode": 0, "frame": 0})))
        c = SimpleNamespace(
            mode=mode, initial_token=token, prompt="task", execution_horizon=16, initial_pose_blend_duration=1
        )
        hands = Mock()
        hands.snapshot.return_value = {"hands": [hand(10), hand(10)]}
        worker = Mock()
        worker.take.return_value = None
        worker.submit.return_value = True
        runner = Runner(c, hands, SimpleNamespace(cache=fresh_cache()), worker, Mock(), Mock())
        # These state-machine tests use synthetic tick times. Match the fresh
        # validation clock to that time; the snapshot race has its own test.
        observation = runner.observation
        def observe_at_tick(now):
            with patch("gear_sonic.utils.inference.inspire_bridge.time.monotonic", return_value=now):
                return observation(now)
        runner.observation = observe_at_tick
        return runner

    def test_observe_never_outputs_and_ignores_motion_keys(self):
        runner = self.make_runner("observe")
        runner.worker.take.return_value = prediction()
        runner.tick(10.1)
        for key in ("k", "i", "p", "o"):
            runner.key(key, 10.1)
        runner.publisher.send.assert_not_called()
        runner.hands.update_policy.assert_not_called()
        runner.hands.begin_policy_session.assert_not_called()

    def test_fault_sends_stop_and_latches(self):
        runner = self.make_runner()
        runner.cpp_started = True
        runner.phase = "RUNNING"
        runner.tick(10.6)
        self.assertEqual(runner.phase, "FAULT")
        runner.hands.cancel.assert_called()
        self.assertEqual(runner.publisher.send.call_count, 3)
        for key in ("k", "i", "p"):
            runner.key(key, 10.61)
        self.assertEqual(runner.publisher.send.call_count, 3)

    def test_no_fresh_token_refuses_initial_pose_no_snap(self):
        runner = self.make_runner()
        runner.cpp_started = True
        runner.sensors.cache.body["token_state"] = []
        with self.assertRaises(ValueError):
            runner.key("i", 10.1)
        runner.publisher.send.assert_not_called()

    def test_pending_hand_write_is_not_a_fault(self):
        runner = self.make_runner()
        runner.cpp_started = True
        runner.phase = "RUNNING"
        runner.fault_counts = [0, 0]
        runner.hands.snapshot.return_value["hands"][0]["write_status"] = 1
        runner.timeline.accept(prediction(), 10.1)
        runner.tick(10.11)
        self.assertEqual(runner.phase, "RUNNING")
        runner.hands.update_policy.assert_called_once()
        wire = runner.publisher.send.call_args.args[0]
        body_sequence = np.frombuffer(wire[-8:], "<i8")[0]
        self.assertEqual(body_sequence, runner.hands.update_policy.call_args.kwargs["sequence"])

    def test_pause_discards_late_failed_request_without_rearming(self):
        runner = self.make_runner()
        runner.cpp_started, runner.phase = True, "RUNNING"
        old = prediction()
        old.error = "old request timed out"
        runner.key("p", 10.1)
        runner.worker.take.return_value = old
        runner.tick(10.11)
        self.assertEqual(runner.phase, "PAUSED")
        runner.hands.update_policy.assert_not_called()
        runner.publisher.send.assert_not_called()

    def test_action_exhaustion_stops_even_when_all_sensors_are_fresh(self):
        runner = self.make_runner()
        runner.cpp_started, runner.phase, runner.fault_counts = True, "RUNNING", [0, 0]
        runner.timeline.accept(prediction(), 10.1)
        runner.sensors.cache = fresh_cache(10.81)
        runner.hands.snapshot.return_value = {"hands": [hand(10.81)] * 2}
        runner.tick(10.81)
        self.assertEqual(runner.phase, "FAULT")
        self.assertIn("exhausted", runner.stop_reason)
        runner.hands.update_policy.assert_not_called()
        self.assertEqual(runner.publisher.send.call_count, 3)

    def test_initialization_blends_from_fresh_current_token(self):
        runner = self.make_runner()
        runner.cpp_started = True
        with patch("gear_sonic.scripts.run_inspire_inference.time.monotonic", return_value=10.1):
            runner.key("i", 10.1)
        runner.publisher.reset_mock()
        runner.tick(10.1)
        np.testing.assert_allclose(np.frombuffer(runner.publisher.send.call_args.args[0][1284:1540], "<f4"), 0.1)
        runner.sensors.cache = fresh_cache(10.6)
        runner.hands.snapshot.return_value = {"hands": [hand(10.6)] * 2}
        runner.tick(10.6)
        np.testing.assert_allclose(np.frombuffer(runner.publisher.send.call_args.args[0][1284:1540], "<f4"), 0.15)
        runner.hands.update_policy.assert_not_called()

    def test_sustained_slow_inference_stops(self):
        runner = self.make_runner()
        runner.cpp_started, runner.phase, runner.fault_counts = True, "RUNNING", [0, 0]
        for index in range(5):
            now = 11.0 + index
            runner.sensors.cache = fresh_cache(now)
            runner.hands.snapshot.return_value = {"hands": [hand(now)] * 2}
            result = prediction(observed=now - 0.4, request_id=index)
            result.completed = now
            runner.worker.take.return_value = result
            runner.tick(now)
        self.assertEqual(runner.phase, "FAULT")
        self.assertIn("update budget", runner.stop_reason)

    def test_launcher_keeps_dex3_disabled_and_modes_separate(self):
        config = parser().parse_args(["real"])
        cmd = commands(config, Path("/tmp/run with space"))
        self.assertIn("--disable-dex3-hands", cmd["deploy"])
        self.assertIn("new_embodiment", cmd["server"])
        self.assertIn("pick up the ball", cmd["client"])
        self.assertNotIn("Pico", " ".join(cmd["client"]))
        with tempfile.TemporaryDirectory() as name:
            (Path(name) / "existing").write_text("preserve")
            with self.assertRaises(FileExistsError):
                check_output(Path(name))


if __name__ == "__main__":
    unittest.main()
