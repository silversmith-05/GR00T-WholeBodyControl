"""Offline geometry, wire-message and manager checks; no SDK or socket is started."""

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock, patch

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as pico
from gear_sonic.utils.teleop import inspire_hand_controller as inspire
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE, build_planner_message


def body_sample():
    poses = np.zeros((24, 7), dtype=np.float32)
    poses[:, 6] = 1
    poses[12, :3] = [0, 1.4, 0]
    poses[22, :3] = [-0.4, 1.2, 0.2]
    poses[23, :3] = [0.4, 1.2, 0.2]
    return poses


def move_body(poses):
    moved = poses.copy()
    rot = Rotation.from_euler("xyz", [30, 50, -20], degrees=True)
    for i in range(len(poses)):
        moved[i, :3] = rot.apply(poses[i, :3]) + [0.8, -0.3, 0.4]
        moved[i, 3:] = (rot * Rotation.from_quat(poses[i, 3:])).as_quat()
    return moved


def unpack(message, topic):
    data = message[len(topic):]
    header = json.loads(data[:HEADER_SIZE].rstrip(b"\0"))
    payload, result, offset = data[HEADER_SIZE:], {}, 0
    dtypes = {"i32": "<i4", "i64": "<i8", "f32": "<f4", "f64": "<f8", "u8": "u1", "bool": "?"}
    for field in header["fields"]:
        dtype = np.dtype(dtypes[field["dtype"]])
        count = int(np.prod(field["shape"]))
        result[field["name"]] = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += count * dtype.itemsize
    return result


class ArmsDetectTests(unittest.TestCase):
    def test_wrist_coordinates_use_original_pelvis_frame(self):
        poses = body_sample()
        original = poses.copy()
        expected = pico._process_3pt_pose(poses)
        # A common world-space transform cancels in the original pelvis frame.
        actual = pico._process_3pt_pose(move_body(poses))
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_array_equal(poses, original)
        # Moving only the neck no longer changes either raw wrist target.
        moved = poses.copy()
        moved[12] = move_body(poses)[12]
        np.testing.assert_array_equal(pico._process_3pt_pose(moved)[:2], expected[:2])
        # Pelvis motion with fixed world-space wrists does affect their targets.
        moved = poses.copy()
        moved[0] = move_body(poses)[0]
        self.assertFalse(np.allclose(pico._process_3pt_pose(moved)[:2], expected[:2]))

    def test_calibration_aligns_measured_wrists_and_keeps_neck_upright(self):
        model = Mock()
        q = np.linspace(-0.1, 0.2, 29)
        reference = {
            "left_wrist": {"position": np.array([0.2, 0.3, 0.1]),
                           "orientation_wxyz": np.array([1., 0, 0, 0])},
            "right_wrist": {"position": np.array([0.2, -0.3, 0.1]),
                            "orientation_wxyz": np.array([1., 0, 0, 0])},
        }
        tracker = pico.ThreePointPose(robot_model=model, only_arms_detect=True)
        with patch.object(pico, "get_g1_key_frame_poses", return_value=reference), \
                redirect_stdout(io.StringIO()):
            tracker.reset_with_measured_q(q)
            initial = tracker.process_smpl_pose(body_sample())
            np.testing.assert_allclose(initial[0, :3], reference["left_wrist"]["position"], atol=1e-6)
            np.testing.assert_allclose(initial[1, :3], reference["right_wrist"]["position"], atol=1e-6)
            np.testing.assert_array_equal(model.get_configuration_from_actuated_joints.call_args.kwargs[
                "body_actuated_joint_values"], q)
            moved = move_body(body_sample())
            np.testing.assert_allclose(tracker.process_smpl_pose(moved), initial, atol=1e-6)
            moved[22, :3] += [0.1, 0, 0]
            moved[22, 3:] = (Rotation.from_euler("x", 20, degrees=True)
                              * Rotation.from_quat(moved[22, 3:])).as_quat()
            tracked = tracker.process_smpl_pose(moved)
            self.assertFalse(np.allclose(tracked[0, :3], initial[0, :3]))
            self.assertFalse(np.allclose(tracked[0, 3:], initial[0, 3:]))
            np.testing.assert_allclose(tracked[1], initial[1], atol=1e-6)
            np.testing.assert_allclose(tracked[2], [0, 0, 0.4, 1, 0, 0, 0], atol=1e-6)

    def planner(self, only_arms_detect=True):
        reader = Mock()
        reader.get_timestamp_ns.side_effect = [1, 2, 2, 3]
        reader.get_latest.return_value = {"body_poses_np": body_sample()}
        tracker = pico.ThreePointPose(robot_model=Mock(), only_arms_detect=only_arms_detect)
        with patch.object(pico, "FeedbackReader"):
            planner = pico.PlannerStreamer(Mock(), reader, tracker, hand_backend="inspire",
                                            only_arms_detect=only_arms_detect)
        return planner

    def test_detect_preserves_original_wrist_calibration_and_only_fixes_neck(self):
        original = pico.ThreePointPose(robot_model=Mock())
        detect = pico.ThreePointPose(robot_model=Mock(), only_arms_detect=True)
        reference = {
            "left_wrist": {"position": np.array([0.2, 0.3, 0.1]),
                           "orientation_wxyz": np.array([1., 0, 0, 0])},
            "right_wrist": {"position": np.array([0.2, -0.3, 0.1]),
                            "orientation_wxyz": np.array([1., 0, 0, 0])},
        }
        initial = move_body(body_sample())
        moved = initial.copy()
        moved[0, 3:] = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_quat()
        moved[12, 3:] = Rotation.from_euler("xyz", [-25, 10, 40], degrees=True).as_quat()
        moved[22, :3] += [0.1, -0.2, 0.05]
        with patch.object(pico, "get_g1_key_frame_poses", return_value=reference), \
                redirect_stdout(io.StringIO()):
            for tracker in (original, detect):
                tracker.reset_with_measured_q(np.linspace(-0.1, 0.1, 29))
            for sample in (initial, moved):
                raw = original.process_smpl_pose(sample)
                actual = detect.process_smpl_pose(sample)
                np.testing.assert_array_equal(actual[:2], raw[:2])
                np.testing.assert_allclose(actual[2], [0, 0, 0.4, 1, 0, 0, 0], atol=1e-6)
            self.assertFalse(np.allclose(raw[2], actual[2]))

    def test_planner_wire_keeps_idle_and_never_reads_locomotion_inputs(self):
        planner = self.planner()
        planner.mode = pico.LocomotionMode.RUN
        with patch.object(pico, "get_controller_axes", side_effect=AssertionError("joystick read")), \
                patch.object(pico, "get_abxy_buttons", side_effect=AssertionError("locomotion buttons read")), \
                patch.object(pico, "get_controller_inputs", return_value=(False, 0.3, 0.8, 0, 0)), \
                patch.object(pico.time, "sleep"), redirect_stdout(io.StringIO()):
            planner.run_once(pico.StreamMode.PLANNER)
            planner.run_once(pico.StreamMode.PLANNER_VR_3PT)
            planner.run_once(pico.StreamMode.PLANNER_VR_3PT)  # Same timestamp: no duplicate.
        self.assertEqual(planner.socket.send.call_count, 2)
        for call in planner.socket.send.call_args_list:
            msg = unpack(call.args[0], b"planner")
            self.assertEqual(msg["mode"].item(), 0)
            np.testing.assert_array_equal(msg["movement"], [0, 0, 0])
            np.testing.assert_array_equal(msg["facing"], [1, 0, 0])
            self.assertEqual(msg["speed"].item(), -1)
            self.assertEqual(msg["height"].item(), -1)
            self.assertNotIn("upper_body_position", msg)
            self.assertNotIn("arm_position", msg)
            self.assertNotIn("left_hand_joints", msg)  # Inspire remains independent.
        idle = unpack(planner.socket.send.call_args_list[0].args[0], b"planner")
        self.assertNotIn("vr_position", idle)
        tracking = unpack(planner.socket.send.call_args_list[1].args[0], b"planner")
        np.testing.assert_allclose(tracking["vr_position"][-3:], [0, 0, 0.4], atol=1e-6)
        np.testing.assert_array_equal(tracking["vr_orientation"][-4:], [1, 0, 0, 0])

    def test_default_planner_still_accepts_movement_and_turning(self):
        planner = self.planner(only_arms_detect=False)
        planner.mode = pico.LocomotionMode.WALK
        with patch.object(pico, "get_controller_axes", return_value=(0., 1., 0.8, 0.)), \
                patch.object(pico, "get_abxy_buttons", return_value=(False,) * 4), \
                patch.object(pico.time, "sleep"):
            planner.run_once(pico.StreamMode.PLANNER)
        msg = unpack(planner.socket.send.call_args.args[0], b"planner")
        self.assertEqual(msg["mode"].item(), pico.LocomotionMode.WALK)
        self.assertGreater(np.linalg.norm(msg["movement"]), 0)
        self.assertFalse(np.allclose(msg["facing"], [1, 0, 0]))

    def test_tracking_entry_requires_measured_robot_pose(self):
        planner = self.planner()
        planner.three_point = Mock()
        with redirect_stdout(io.StringIO()):
            for invalid in (None, np.zeros(28), np.full(29, np.nan)):
                planner.feedback_reader.full_body_q_measured = invalid
                self.assertFalse(planner.recalibrate_for_vr3pt())
            planner.three_point.reset_with_measured_q.assert_not_called()
            planner.feedback_reader.full_body_q_measured = np.zeros(29)
            self.assertTrue(planner.recalibrate_for_vr3pt())
            planner.three_point.reset_with_measured_q.assert_called_once()

    def pose_streamer(self):
        with redirect_stdout(io.StringIO()):
            return pico.PoseStreamer(
                Mock(), Mock(), pico.ThreePointPose(robot_model=Mock(), only_arms_detect=True),
                num_frames_to_send=2, target_fps=50, use_cuda=False,
                record_dir="", record_format="npz", hand_backend="inspire", only_arms_detect=True)

    def test_smpl_preserves_arm_rotations_and_neutralizes_body_before_fk(self):
        parents = self.pose_streamer().parent_indices
        rng = np.random.default_rng(17)
        sample = body_sample()
        sample[:, 3:] = Rotation.random(24, random_state=rng).as_quat()
        original = pico.compute_from_body_poses(parents, "cpu", sample)
        actual = pico.compute_from_body_poses(parents, "cpu", sample, only_arms_detect=True)
        arms = np.array([13, 14, 16, 17, 18, 19, 20, 21]) - 1  # body_pose excludes root.
        body = [i for i in range(21) if i not in arms]
        original_pose = original["smpl_pose"].numpy().reshape(-1, 3)
        actual_pose = actual["smpl_pose"].numpy().reshape(-1, 3)
        np.testing.assert_array_equal(actual_pose[arms], original_pose[arms])
        np.testing.assert_array_equal(actual_pose[body], 0)
        np.testing.assert_allclose(actual["global_orient_quat"], [[1, 0, 0, 0]], atol=1e-6)

        # Change all body rotations while retaining the same local arm rotations.
        locals_ = Rotation.from_rotvec(original_pose)
        globals_ = [Rotation.from_euler("xyz", [40, -20, 70], degrees=True)]
        for i in range(1, 24):
            local = locals_[i - 1] if i - 1 in arms else Rotation.random(random_state=rng)
            globals_.append(globals_[parents[i]] * local)
        changed = sample.copy()
        for i, rot in enumerate(globals_):
            changed[i, 3:] = (rot * Rotation.from_euler("y", 180, degrees=True).inv()).as_quat()
        changed[:, :3] += [1, 2, 3]
        filtered = pico.compute_from_body_poses(parents, "cpu", changed, only_arms_detect=True)
        for key in ("smpl_pose", "smpl_joints_local", "global_orient_quat", "adjusted_transl"):
            np.testing.assert_allclose(filtered[key], actual[key], atol=1e-6, err_msg=key)

    def test_elbow_motion_survives_smpl_even_with_identical_vr3pt_targets(self):
        parents = self.pose_streamer().parent_indices
        a = body_sample()
        b = a.copy()
        b[18, 3:] = Rotation.from_euler("xyz", [20, 35, -10], degrees=True).as_quat()
        np.testing.assert_array_equal(pico._process_3pt_pose(a), pico._process_3pt_pose(b))
        first = pico.compute_from_body_poses(parents, "cpu", a, only_arms_detect=True)
        second = pico.compute_from_body_poses(parents, "cpu", b, only_arms_detect=True)
        self.assertGreater(float((first["smpl_joints_local"] - second["smpl_joints_local"]).abs().max()), .01)
        self.assertGreater(float((first["smpl_pose"] - second["smpl_pose"]).abs().max()), .1)

    def test_neutral_body_keeps_original_smpl_arm_geometry(self):
        parents = self.pose_streamer().parent_indices
        sample = body_sample()
        root = Rotation.from_euler("y", -90, degrees=True)
        sample[:, 3:] = root.as_quat()
        # Only the arm chains differ from a canonical neutral SMPL body.
        for i in (13,14,16,17,18,19,20,21,22,23):
            parent = Rotation.from_quat(sample[parents[i], 3:])
            sample[i, 3:] = (parent * Rotation.from_euler("xyz", [i, -5, 8], degrees=True)).as_quat()
        original = pico.compute_from_body_poses(parents, "cpu", sample)
        actual = pico.compute_from_body_poses(parents, "cpu", sample, only_arms_detect=True)
        np.testing.assert_array_equal(actual["smpl_pose"][:, :63], original["smpl_pose"][:, :63])
        np.testing.assert_allclose(actual["smpl_joints_local"], original["smpl_joints_local"], atol=1e-6)
        np.testing.assert_allclose(actual["global_orient_quat"], original["global_orient_quat"], atol=1e-6)

    def test_pose_wire_retains_original_smpl_protocol_and_blocks_joystick_heading(self):
        streamer = self.pose_streamer()
        sample = body_sample()
        sample[18, 3:] = Rotation.from_euler("x", 30, degrees=True).as_quat()
        sample[19, 3:] = Rotation.from_euler("x", -30, degrees=True).as_quat()
        streamer.reader.get_latest.side_effect = [
            {"body_poses_np": sample, "timestamp_ns": i * 20_000_000} for i in range(1, 6)]
        with patch.object(pico, "get_controller_axes", side_effect=AssertionError("joystick read")), \
                patch.object(pico, "get_abxy_buttons", return_value=(False,) * 4), \
                patch.object(pico, "get_controller_inputs", return_value=(False, .2, .7, 0., 0.)), \
                patch.object(pico.time, "sleep"), redirect_stdout(io.StringIO()):
            for _ in range(5):
                streamer.run_once()
        self.assertGreater(streamer.socket.send.call_count, 0)
        for call in streamer.socket.send.call_args_list:
            raw = call.args[0]
            self.assertTrue(raw.startswith(b"pose"))
            header = json.loads(raw[4:4 + HEADER_SIZE].rstrip(b"\0"))
            self.assertEqual(header["v"], 3)  # C++ selects SMPL encoder mode 2.
            msg = unpack(raw, b"pose")
            self.assertIn("smpl_joints", msg)
            self.assertIn("joint_pos", msg)  # Original elbow-to-wrist decomposition remains.
            self.assertNotIn("arm_position", msg)
            self.assertNotIn("left_hand_joints", msg)
            self.assertEqual(msg["heading_increment"].item(), 0)
            np.testing.assert_allclose(msg["body_quat_w"].reshape(-1, 4), [[1,0,0,0]] * 2, atol=1e-6)
            pose = msg["smpl_pose"].reshape(-1, 21, 3)
            self.assertGreater(np.max(np.abs(pose[:, 17])), .1)  # Left elbow.
            np.testing.assert_array_equal(pose[:, [0,1,2,3,4,5,6,7,8,9,10,11,14]], 0)

    def test_freeze_sends_only_measured_arms_and_failed_resume_preserves_hold(self):
        planner = self.planner()
        feedback = pico.FeedbackReader.__new__(pico.FeedbackReader)  # No subscriber/socket.
        feedback.poller = Mock()
        feedback.upper_body_joint_indices = feedback._get_upper_body_joint_indices()
        feedback.full_body_q_measured = None
        target_fields = ("upper_body_position_target", "left_hand_position_target", "right_hand_position_target")
        for key in target_fields:
            setattr(feedback, key, None)
        q = np.linspace(-0.2, 0.2, 29)
        feedback.poller.get_data.side_effect = [msgpack.packb({
            "body_q_measured": q.tolist(), "left_hand_q_measured": [0.1] * 7,
            "right_hand_q_measured": [0.2] * 7,
        }), None, msgpack.packb({"body_q_measured": (q + 0.1).tolist()})]
        planner.feedback_reader = feedback
        with redirect_stdout(io.StringIO()), patch.object(pico.time, "sleep"):
            self.assertTrue(planner.save_upper_body_position_target())
            held = {key: getattr(feedback, key) for key in target_fields}
            planner.reader.get_latest.side_effect = AssertionError("frozen mode read live pose")
            planner.run_once(pico.StreamMode.PLANNER_FROZEN_UPPER_BODY)
            self.assertFalse(planner.recalibrate_for_vr3pt())  # Feedback disappears on resume.
            for key in target_fields:
                self.assertEqual(getattr(feedback, key), held[key])
            planner.run_once(pico.StreamMode.PLANNER_FROZEN_UPPER_BODY)
            # The next freeze captures a new measurement, not the old cached target.
            self.assertTrue(planner.save_upper_body_position_target())
        expected = q[[12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]]
        for call in planner.socket.send.call_args_list:
            msg = unpack(call.args[0], b"planner")
            np.testing.assert_allclose(msg["arm_position"], expected[3:], atol=1e-6)
            self.assertNotIn("upper_body_position", msg)
            self.assertNotIn("upper_body_velocity", msg)
            self.assertEqual(msg["mode"].item(), 0)
            np.testing.assert_array_equal(msg["movement"], [0, 0, 0])
            self.assertNotIn("vr_position", msg)
        np.testing.assert_allclose(feedback.upper_body_position_target, expected + 0.1)

    def test_default_freeze_still_sends_waist_and_arms(self):
        planner = self.planner(only_arms_detect=False)
        planner.feedback_reader.upper_body_position_target = np.linspace(-0.2, 0.2, 17)
        with patch.object(pico, "get_controller_axes", return_value=(0., 0., 0., 0.)), \
                patch.object(pico, "get_abxy_buttons", return_value=(False,) * 4), \
                patch.object(pico.time, "sleep"):
            planner.run_once(pico.StreamMode.PLANNER_FROZEN_UPPER_BODY)
        msg = unpack(planner.socket.send.call_args.args[0], b"planner")
        np.testing.assert_allclose(msg["upper_body_position"],
                                   planner.feedback_reader.upper_body_position_target, atol=1e-6)
        self.assertNotIn("arm_position", msg)

    def test_arm_wire_rejects_wrong_length_or_conflicting_waist_targets(self):
        for kwargs in ({"arm_position": [0.] * 17},
                       {"arm_position": [0.] * 14, "upper_body_position": [0.] * 17},
                       {"arm_position": [0.] * 14, "upper_body_velocity": [0.] * 17}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_planner_message(0, [0, 0, 0], [1, 0, 0], **kwargs)

    def test_manager_restores_original_ax_by_smpl_and_hand_lifecycle(self):
        off, start, ax, by = (False,) * 4, (True,) * 4, (True, False, True, False), (False, True, False, True)
        buttons = [off, start, off, ax, off, by, by, off, by, off, off, off, ax, off,
                   off, off, by, off, by, off, off, off, off, off, by, off, start]
        clicks = [(i in (14, 20, 22), False) for i in range(len(buttons))]
        expected_modes = [0, 2, 2, 1, 1, 3, 3, 3, 1, 1, 4, 1, 2, 2,
                          5, 5, 1, 1, 3, 3, 5, 5, 3, 3, 1, 1, 0]
        context = Mock()
        with patch.object(pico, "_init_input_source"), patch.object(pico.zmq, "Context", return_value=context), \
                patch.object(pico, "ThreePointPose") as tracker, patch.object(pico, "PoseStreamer") as pose, \
                patch.object(pico, "PlannerStreamer") as planner, \
                patch.object(inspire, "InspireHandController") as hands, \
                patch.object(inspire, "PicoHandBridge") as bridge, \
                patch.object(pico, "get_abxy_buttons", side_effect=buttons), \
                patch.object(pico, "get_axis_clicks", side_effect=clicks), \
                patch.object(pico, "get_controller_inputs", side_effect=[
                    (i == 10, 0, 0, 0, 0) for i in range(len(buttons))]), \
                patch.object(pico.time, "sleep"), redirect_stdout(io.StringIO()):
            hands.return_value.snapshot.return_value = {}
            planner.return_value.recalibrate_for_vr3pt.return_value = True
            planner.return_value.save_upper_body_position_target.return_value = True
            with self.assertRaises(SystemExit):
                pico.run_pico_manager(hand_backend="inspire", enable_hand_control=True, only_arms_detect=True)
            self.assertTrue(tracker.call_args.kwargs["only_arms_detect"])
            self.assertTrue(planner.call_args.kwargs["only_arms_detect"])
            self.assertTrue(pose.call_args.kwargs["only_arms_detect"])
            self.assertEqual(pose.return_value.run_once.call_count, expected_modes.count(1))
            modes = [c.args[0] for c in planner.return_value.run_once.call_args_list]
            self.assertEqual([mode.value for mode in modes], [m for m in expected_modes if m in (2,3,5)])
            hand_modes = [c.args[3] for c in bridge.return_value.update.call_args_list]
            self.assertEqual(hand_modes, expected_modes)
            self.assertEqual(planner.return_value.save_upper_body_position_target.call_count, 3)
            self.assertEqual(planner.return_value.recalibrate_for_vr3pt.call_count, 2)
            tracker.return_value.calibrate_now.assert_called_once()
            hands.return_value.close.assert_called_once()
        messages = [c.args[0] for c in context.socket.return_value.send.call_args_list
                    if c.args[0].startswith(b"command")]
        expected_planner = [int(mode != 1) for previous, mode in zip(expected_modes, expected_modes[1:])
                            if previous != mode and mode != 4]
        self.assertEqual([unpack(msg, b"command")["planner"].item() for msg in messages], expected_planner)
        self.assertEqual(unpack(messages[-1], b"command")["stop"].item(), 1)

    def test_failed_freeze_keeps_smpl_and_does_not_pause_hands(self):
        off, start, ax, by = (False,) * 4, (True,) * 4, (True, False, True, False), (False, True, False, True)
        buttons = [off, start, off, ax, off, by, by, off, by, off, by, off, start]
        with patch.object(pico, "_init_input_source"), patch.object(pico.zmq, "Context"), \
                patch.object(pico, "ThreePointPose"), patch.object(pico, "PoseStreamer") as pose, \
                patch.object(pico, "PlannerStreamer") as planner, \
                patch.object(inspire, "InspireHandController") as hands, \
                patch.object(inspire, "PicoHandBridge") as bridge, \
                patch.object(pico, "get_abxy_buttons", side_effect=buttons), \
                patch.object(pico, "get_axis_clicks", return_value=(False, False)), \
                patch.object(pico, "get_controller_inputs", return_value=(False, 0, 0, 0, 0)), \
                patch.object(pico.time, "sleep"), redirect_stdout(io.StringIO()):
            hands.return_value.snapshot.return_value = {}
            planner.return_value.save_upper_body_position_target.side_effect = [False, True]
            with self.assertRaises(SystemExit):
                pico.run_pico_manager(hand_backend="inspire", enable_hand_control=True, only_arms_detect=True)
            self.assertEqual([c.args[3] for c in bridge.return_value.update.call_args_list],
                             [0,2,2,1,1,1,1,1,3,3,1,1,0])
            planner.return_value.recalibrate_for_vr3pt.assert_not_called()
            self.assertEqual(pose.return_value.run_once.call_count, 7)


if __name__ == "__main__":
    unittest.main()
