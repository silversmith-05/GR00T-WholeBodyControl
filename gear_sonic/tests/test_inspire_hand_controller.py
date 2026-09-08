"""Offline only: real SDK protocol over fake transports; never opens a socket."""
import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from inspire_rh56e2 import HandClient
from gear_sonic.utils.teleop.inspire_hand_controller import (
    ANGLES, CLOSE_ANGLES, InspireHandController, PicoHandBridge, ThumbButtonGesture, ThumbRotationConfig,
    THUMB_HOLD_DELAY, validate_close_angles,
)
from gear_sonic.utils.data_collection.inspire_hand import hand_features, snapshot_to_frame, validate_snapshot


class FakeDevice:
    def __init__(self):
        self.connected = False
        self.connects = self.closes = 0
        self.writes = []
        self.reads = []
        self.angle = [500] * 6  # Never mirror commanded targets into feedback.
        self.read_entered = threading.Event()
        self.read_gate = threading.Event()
        self.read_gate.set()
        self.write_gate = threading.Event()
        self.write_gate.set()
        self.write_entered = threading.Event()
        self.read_fail = False
        self.fail_address = None
        self.reject_address = None
        self.active = 0
        self.max_active = 0

    def connect(self):
        self.connects += 1
        self.connected = True
        return True

    def close(self):
        self.closes += 1
        self.connected = False

    def _enter(self):
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def read_holding_registers(self, address, *, count, device_id):
        self._enter()
        try:
            assert address == 1546 and count == 6 and device_id == 255
            self.reads.append(address)
            self.read_entered.set()
            if not self.read_gate.wait(1):
                raise TimeoutError('fake read timeout')
            if self.read_fail:
                raise OSError('fake read disconnect')
            return SimpleNamespace(isError=lambda: False, registers=list(self.angle), function_code=3)
        finally:
            self.active -= 1

    def write_registers(self, address, values, *, device_id):
        self._enter()
        try:
            assert device_id == 255
            self.writes.append((address, list(values)))
            self.write_entered.set()
            if not self.write_gate.wait(1):
                raise TimeoutError('fake write timeout')
            if address == self.fail_address:
                raise TimeoutError('fake write result unknown')
            return SimpleNamespace(isError=lambda: address == self.reject_address,
                                   function_code=16, address=address, count=len(values))
        finally:
            self.active -= 1

    @property
    def angles_sent(self):
        return [values for address, values in self.writes if address == 1486]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.devices = {"192.168.123.211": FakeDevice(), "192.168.123.210": FakeDevice()}
        self.left, self.right = self.devices.values()
        self.left.angle[5] = self.right.angle[5] = 339
        self.hands = self.make_controller()
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))

    def make_controller(self, **kwargs):
        return InspireHandController(enabled=kwargs.pop('enabled', True),
            client_factory=lambda c: HandClient(c, transport=self.devices[c.host]),
            lock_dir=self.tmp.name, poll_hz=100, reconnect_delay=0.03, **kwargs)

    def tearDown(self):
        for d in self.devices.values():
            d.read_gate.set()
            d.write_gate.set()
        self.hands.close()
        self.tmp.cleanup()

    def update(self, left=0, right=0, active=True, age=0, buttons=None, left_grip=0):
        self.hands.update(left, right, sample_monotonic=time.monotonic()-age,
                          sample_time=time.time()-age, active=active,
                          buttons=buttons, left_grip=left_grip)

    def wait(self, predicate, timeout=1):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= end:
                self.fail('condition timed out')
            time.sleep(0.003)

    def hold(self, left=0, right=0, duration=.07, **kwargs):
        end = time.monotonic()+duration
        while time.monotonic() < end:
            self.update(left, right, **kwargs)
            time.sleep(.003)

    def test_startup_held_trigger_never_moves_then_requires_new_press(self):
        self.hold(1, 1)
        self.assertFalse(self.left.writes or self.right.writes)
        self.hold(0, 0)
        self.assertFalse(self.left.writes or self.right.writes)
        self.hold(1, 0)
        self.assertEqual(self.left.angles_sent, [list(ANGLES[1])])
        self.assertEqual(self.right.writes, [])

    def test_exact_presets_left_right_and_deduplication(self):
        self.hold()
        self.hold(1, 0)
        self.hold(1, 1)
        self.hold(0, 1)
        self.hold(0, 1)
        self.assertEqual(self.left.angles_sent, [[250,250,250,250,300,339],
                                                [1000,1000,1000,1000,1000,339]])
        self.assertEqual(self.right.angles_sent, [[250,250,250,250,300,339]])
        for d in (self.left, self.right):
            for start in range(0, len(d.writes), 3):
                self.assertEqual(d.writes[start:start+2], [(1498, [200]*6), (1522, [200]*6)])
            self.assertEqual(d.max_active, 1)

    def test_threshold_is_strictly_greater_than_one_half(self):
        self.hold(.5, .5)
        self.assertFalse(self.left.writes or self.right.writes)
        self.hold(.50001, .5)
        self.hold(.9, .5)
        self.assertEqual(self.left.angles_sent, [list(ANGLES[1])])
        self.hold(.5, .5)
        self.assertEqual(self.left.angles_sent[-1], list(ANGLES[0]))

    def test_slow_left_does_not_block_submit_or_right(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        start = time.monotonic()
        self.update(1, 1)
        self.assertLess(time.monotonic()-start, .02)
        self.wait(lambda: bool(self.right.angles_sent))
        self.assertFalse(self.left.writes)
        self.left.read_gate.set()
        self.wait(lambda: bool(self.left.angles_sent))

    def test_pause_clears_queued_target(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.update(1, 0)
        self.update(1, 0, active=False)
        self.left.read_gate.set()
        self.hold(1, 0)
        self.assertFalse(self.left.writes)
        self.hold(0, 0)
        self.hold(1, 0)
        self.assertEqual(len(self.left.angles_sent), 1)

    def test_input_expiry_clears_queued_target_and_does_not_replay(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.update(1, 0)
        time.sleep(.27)  # Simulates stopped Pico/main loop during a slow read.
        self.left.read_gate.set()
        self.hold(1, 0)
        self.assertFalse(self.left.writes)
        self.assertFalse(self.hands.snapshot()['hands'][0]['target_valid'])

    def test_explicit_stale_or_invalid_input(self):
        self.hold()
        self.update(1, 1, age=1)
        self.hold(1, 1)
        self.assertFalse(self.left.writes or self.right.writes)
        self.update(float('nan'), 1)
        json.dumps(self.hands.snapshot(), allow_nan=False)

    def test_disconnect_reconnect_never_replays(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.update(1, 0)
        self.left.read_fail = True
        self.left.read_gate.set()
        self.wait(lambda: self.hands.snapshot()['hands'][0]['fault_count'] > 0)
        self.left.read_fail = False
        self.wait(lambda: self.left.connects > 1 and self.hands.snapshot()['hands'][0]['connected'])
        self.hold(1, 0)
        self.assertFalse(self.left.writes)
        self.hold()
        self.hold(1, 0)
        self.assertEqual(len(self.left.angles_sent), 1)

    def test_write_timeout_unconfirmed_no_retry_even_after_reconnect(self):
        self.hold()
        self.left.fail_address = 1486
        self.hold(1, 0, duration=.14)
        self.assertEqual(len(self.left.angles_sent), 1)
        s = self.hands.snapshot()['hands'][0]
        self.assertEqual(s['write_status'], 3)
        self.assertFalse(s['target_valid'])
        self.assertFalse(s['write_current'])
        self.assertGreater(s['fault_count'], 0)

    def test_rejected_write_records_failure(self):
        self.hold()
        self.left.reject_address = 1498
        self.hold(1, 0)
        s = self.hands.snapshot()['hands'][0]
        self.assertEqual(s['write_status'], 4)
        self.assertEqual(len(self.left.writes), 1)
        self.assertFalse(self.left.angles_sent)

    def test_guard_cancels_remaining_registers_after_pause(self):
        self.hold()
        self.left.write_gate.clear()
        self.update(1, 0)
        self.wait(self.left.write_entered.is_set)
        self.update(1, 0, active=False)
        self.left.write_gate.set()
        self.wait(lambda: self.hands.snapshot()['hands'][0]['write_status'] == 5)
        self.assertEqual(self.left.writes, [(1498, [200]*6)])
        self.assertGreater(self.hands.snapshot()['hands'][0]['fault_count'], 0)

    def test_target_and_measured_feedback_are_distinct_and_frame_schema(self):
        self.hold()
        self.hold(1, 1)
        s = json.loads(json.dumps(self.hands.snapshot()))
        validate_snapshot(s)
        frame = snapshot_to_frame(s, frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertEqual(frame['action.hand'].tolist(), [1, 1])
        self.assertEqual(frame['hand.angle_act'].tolist(), ([500]*5 + [339])*2)
        self.assertFalse(frame['hand.at_target'].any())
        self.assertTrue(frame['hand.at_target_valid'].all())
        self.assertTrue(frame['hand.training_valid'][0])
        for key, spec in hand_features().items():
            self.assertEqual(frame[key].shape, spec['shape'])
            self.assertEqual(str(frame[key].dtype), spec['dtype'])
        future = snapshot_to_frame(s, frame_time=time.time()+1, frame_monotonic=time.monotonic()+1)
        self.assertFalse(future['hand.training_valid'][0])
        self.assertTrue(future['hand.episode_fault'][0])
        missing = snapshot_to_frame(None, frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertEqual(missing['action.hand'].tolist(), [-1, -1])
        self.assertFalse(missing['hand.angle_valid'].any())

    def test_one_owner_and_shutdown_releases_lock(self):
        duplicate = self.make_controller()
        with self.assertRaisesRegex(RuntimeError, 'Another driver'):
            duplicate.start()
        self.hands.close()
        self.assertTrue(all(not w.thread.is_alive() for w in self.hands.workers))
        self.assertTrue(all(not d.connected for d in self.devices.values()))
        replacement = self.make_controller(enabled=False)
        try:
            replacement.start()
            time.sleep(.03)
            replacement.update(0, 0, sample_monotonic=time.monotonic(), sample_time=time.time(), active=True)
            replacement.update(1, 1, sample_monotonic=time.monotonic(), sample_time=time.time(), active=True)
            time.sleep(.03)
            self.assertFalse(self.left.writes or self.right.writes)
        finally:
            replacement.close()

    def test_shutdown_cancels_a_pending_command_before_join(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.update(1, 0)
        closing = threading.Thread(target=self.hands.close)
        closing.start()
        self.wait(lambda: self.hands.workers[0].stop_event.is_set())
        self.left.read_gate.set()
        closing.join(timeout=1)
        self.assertFalse(closing.is_alive())
        self.assertFalse(self.left.writes)
        self.assertFalse(self.left.connected or self.right.connected)

    def test_bridge_uses_device_timestamp_and_mode_transitions(self):
        bridge = PicoHandBridge(self.hands)
        sample = dict(timestamp_ns=10, timestamp_monotonic=time.monotonic(), timestamp_realtime=time.time())
        reader = SimpleNamespace(get_latest=lambda: sample, disconnected=False)
        bridge.update(reader, 0, 0, 1)
        first = bridge.sample_monotonic
        sample['timestamp_monotonic'] += .01
        bridge.update(reader, 1, 0, 4)  # pause cancels pending work
        self.assertEqual(bridge.sample_monotonic, first)
        bridge.update(reader, 1, 0, 5)
        self.hold(1, 0)
        self.assertFalse(self.left.writes)
        sample.update(timestamp_ns=11, timestamp_monotonic=time.monotonic(), controller_data={})
        bridge.update(reader, 0, 0, 5)
        self.assertFalse(self.hands.snapshot()['hands'][0]['input_valid'])

    def click(self, key, **kwargs):
        self.update(buttons=(False,)*4, **kwargs)
        self.update(buttons=tuple(c == key for c in 'abxy'), **kwargs)
        self.update(buttons=(False,)*4, **kwargs)

    def test_thumb_all_four_buttons_are_independent_release_only_and_no_repeat(self):
        self.hold()
        for key, device, expected in [('y',self.left,349), ('x',self.left,339),
                                      ('b',self.right,349), ('a',self.right,339)]:
            count = len(device.angles_sent)
            self.update(buttons=(False,)*4)
            self.hold(buttons=tuple(c == key for c in 'abxy'))
            self.assertEqual(len(device.angles_sent), count)
            self.hold(buttons=(False,)*4)
            self.assertEqual(len(device.angles_sent), count+1)
            # SDK encodes the documented -1 no-motion sentinel as 0xffff.
            self.assertEqual(device.angles_sent[-1], [65535]*5+[expected])
        self.assertEqual(self.left.max_active, 1)
        self.assertEqual(self.right.max_active, 1)

    def test_thumb_steps_limits_and_trigger_keep_rotation(self):
        self.hands.close()
        self.hands = self.make_controller(left_thumb=ThumbRotationConfig(7,330,350),
                                          right_thumb=ThumbRotationConfig(25,300,370))
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))
        self.hold()
        for _ in range(3):
            self.click('y')
            self.hold()
        self.assertEqual([a[5] for a in self.left.angles_sent], [346,350])
        self.click('a')
        self.hold()
        self.assertEqual(self.right.angles_sent[-1][-1], 314)
        self.hold(1,1)
        self.assertEqual(self.left.angles_sent[-1], list(ANGLES[1][:5])+[350])
        self.assertEqual(self.right.angles_sent[-1], list(ANGLES[1][:5])+[314])
        self.click('x', left=1, right=1)
        self.hold(1,1)
        self.assertEqual(self.left.angles_sent[-1], list(ANGLES[1][:5])+[343])
        self.hold(0,0)
        self.assertEqual(self.left.angles_sent[-1], list(ANGLES[0][:5])+[343])
        self.assertEqual(self.right.angles_sent[-1], list(ANGLES[0][:5])+[314])
        frame = snapshot_to_frame(self.hands.snapshot(), frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertEqual(frame['action.hand'].tolist(), [0,0])
        self.assertEqual(frame['action.thumb_rotation'].tolist(), [343,314])
        self.assertEqual(frame['hand.thumb_step'].tolist(), [7,25])
        self.assertTrue(frame['hand.training_valid'][0])

    def test_thumb_baseline_is_actual_and_outside_limit_never_jumps(self):
        self.left.angle[5] = 612
        self.hands.cancel()
        self.hold()
        self.assertFalse(self.left.writes)
        self.click('y')
        self.hold()
        self.assertEqual(self.left.angles_sent[-1][-1], 622)
        self.hands.close()
        self.hands = self.make_controller(left_thumb=ThumbRotationConfig(10,300,400))
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))
        count = len(self.left.writes)
        self.hold()
        self.click('x')
        self.hold(1,0)
        self.assertEqual(len(self.left.writes), count)
        self.assertFalse(self.hands.snapshot()['hands'][0]['armed'])
        self.assertIn('outside configured limits', self.hands.snapshot()['hands'][0]['error'])

    def test_thumb_pending_cancel_during_slow_read_and_other_hand_continues(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        start = time.monotonic()
        self.click('y')
        self.click('b')
        self.assertLess(time.monotonic()-start, .02)
        self.wait(lambda: bool(self.right.angles_sent))
        self.assertFalse(self.left.writes)
        self.update(active=False)
        self.left.read_gate.set()
        self.hold()
        self.assertFalse(self.left.writes)
        self.hold(1,0)
        self.assertEqual(self.left.angles_sent[-1][-1], 339)  # Cancelled 349 not replayed.

    def test_thumb_expired_pending_not_replayed(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.click('y')
        time.sleep(.27)
        self.left.read_gate.set()
        self.hold()
        self.assertFalse(self.left.writes)
        self.hold(1,0)
        self.assertEqual(self.left.angles_sent[-1][-1], 339)

    def test_thumb_disconnect_consumes_click_and_does_not_replay(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.update(buttons=(False,False,False,True))
        self.left.read_fail = True
        self.wait(lambda: self.hands.snapshot()['hands'][0]['fault_count'] > 0)
        self.left.read_fail = False
        self.wait(lambda: self.left.connects > 1 and self.hands.snapshot()['hands'][0]['connected'])
        self.hold(buttons=(False,False,False,True))
        self.hold(buttons=(False,)*4)
        self.assertFalse(self.left.writes)
        self.click('y')
        self.hold()
        self.assertEqual(self.left.angles_sent[-1][-1], 349)

    def test_thumb_timeout_recorded_without_retry_or_old_target_on_new_grasp(self):
        self.hold()
        self.left.fail_address = 1486
        self.click('y')
        self.hold(duration=.14)
        self.assertEqual(len(self.left.angles_sent), 1)
        snap = self.hands.snapshot()
        self.assertEqual(snap['hands'][0]['write_status'], 3)
        self.assertEqual(snap['hands'][0]['write_thumb_rotation'], 349)
        frame = snapshot_to_frame(snap, frame_time=time.time(), frame_monotonic=time.monotonic())
        self.assertTrue(frame['hand.episode_fault'][0])
        self.assertFalse(frame['hand.training_valid'][0])
        self.left.fail_address = None
        self.hold(1,0)
        self.assertEqual(self.left.angles_sent[-1][-1], 339)

    def test_thumb_at_target_compares_dynamic_rotation_and_pending_masks_training(self):
        self.hold()
        self.hold(1,1)
        self.left.angle = list(ANGLES[1][:5])+[339]
        self.hold(1,1)
        self.assertTrue(self.hands.snapshot()['hands'][0]['at_target'])
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.left.write_gate.clear()
        self.left.write_entered.clear()
        for _ in range(3):
            self.click('y',left=1,right=1)
        self.left.read_gate.set()
        self.wait(self.left.write_entered.is_set)
        snap = self.hands.snapshot()
        self.assertEqual(snap['hands'][0]['thumb_rotation'], 369)
        self.assertFalse(snap['hands'][0]['at_target'])
        frame = snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertFalse(frame['hand.training_valid'][0])
        self.assertFalse(frame['hand.episode_fault'][0])
        self.left.write_gate.set()

    def test_bridge_actual_button_path_cancels_on_mode_pause_and_stale_source(self):
        bridge = PicoHandBridge(self.hands)
        sample = {}
        reader = SimpleNamespace(get_latest=lambda: sample, disconnected=False)

        def send(keys='', mode=1, age=0):
            sample.update(timestamp_ns=sample.get('timestamp_ns',0)+1,
                          timestamp_monotonic=time.monotonic()-age, timestamp_realtime=time.time()-age)
            bridge.update(reader,0,0,mode,buttons=tuple(k in keys for k in 'abxy'),left_grip=0)

        def baseline(mode=1):
            end = time.monotonic()+.04
            while time.monotonic() < end:
                send(mode=mode)
                time.sleep(.003)

        baseline()
        send('y')
        send('y',mode=4)
        baseline()
        self.assertFalse(self.left.writes)
        send('y')
        send('y',age=1)
        baseline()
        self.assertFalse(self.left.writes)
        send('y')
        reader.disconnected = True
        send('y')
        reader.disconnected = False
        baseline()
        self.assertFalse(self.left.writes)
        send('y')
        send()
        self.wait(lambda: bool(self.left.angles_sent))
        self.assertEqual(self.left.angles_sent[-1][-1],349)

    def test_bridge_isaac_snapshot_buttons_override_loop_buttons_and_missing_cancels(self):
        bridge = PicoHandBridge(self.hands)
        ctrl = dict(left_trigger_value=0,right_trigger_value=0,left_squeeze_value=0,
                    right_primary_click=0,right_secondary_click=0,left_primary_click=0,left_secondary_click=0)
        sample = dict(controller_data=ctrl)
        reader = SimpleNamespace(get_latest=lambda: sample, disconnected=False)

        def send():
            sample.update(timestamp_ns=sample.get('timestamp_ns',0)+1,
                          timestamp_monotonic=time.monotonic(),timestamp_realtime=time.time())
            bridge.update(reader,1,1,1,buttons=(True,)*4,left_grip=1)

        for _ in range(15):
            send()
            time.sleep(.003)
        ctrl['left_secondary_click'] = 1
        send()
        ctrl['left_secondary_click'] = 0
        send()
        self.wait(lambda: bool(self.left.angles_sent))
        self.assertEqual(self.left.angles_sent[-1], [65535]*5+[349])
        ctrl['right_secondary_click'] = 1
        send()
        del ctrl['left_primary_click']  # Missing is not an observed release.
        send()
        ctrl.update(left_primary_click=0,right_secondary_click=0)
        for _ in range(15):
            send()
            time.sleep(.003)
        self.assertFalse(self.right.writes)

    def test_readonly_thumb_and_shutdown_pending_never_write(self):
        self.hold()
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.click('y')
        closing = threading.Thread(target=self.hands.close)
        closing.start()
        self.wait(lambda: self.hands.workers[0].stop_event.is_set())
        self.left.read_gate.set()
        closing.join(timeout=1)
        self.assertFalse(closing.is_alive())
        self.assertFalse(self.left.writes or self.right.writes)
        self.hands = self.make_controller(enabled=False)
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))
        self.hold()
        self.click('y')
        self.click('b')
        self.hold(1,1)
        self.hold(0,0,buttons=(False,)*4)
        self.hold(buttons=(False,False,False,True),duration=.8)
        self.assertFalse(self.left.writes or self.right.writes)

    def test_custom_five_finger_grasp_and_rotation_are_composed(self):
        self.hands.close()
        custom = (210,220,230,240,290)
        self.hands = self.make_controller(close_angles=custom)
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))
        self.hold()
        self.click('y')
        self.hold()
        self.hold(1,1)
        self.assertEqual(self.left.angles_sent[-1], [210,220,230,240,290,349])
        self.assertEqual(self.right.angles_sent[-1], [210,220,230,240,290,339])
        frame = snapshot_to_frame(self.hands.snapshot(),frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertEqual(frame['hand.close_angles'].tolist(),list(custom)*2)
        self.assertEqual(frame['hand.angle_target'].tolist(),list(custom)+[349]+list(custom)+[339])
        self.assertEqual(frame['hand.angle_act'].tolist(),([500]*5+[339])*2)
        self.assertEqual(frame['action.hand'].tolist(),[1,1])

    def test_hold_delay_slow_small_steps_and_release_has_no_extra_click(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.hold(buttons=(False,False,False,True),duration=.54)
        self.assertFalse(self.left.writes)
        self.hold(buttons=(False,False,False,True),duration=.6)
        self.hold(buttons=(False,)*4,duration=.08)
        targets=[a[-1] for a in self.left.angles_sent]
        self.assertTrue(targets)
        self.assertTrue(all(b-a==5 for a,b in zip([339]+targets, targets)))
        self.assertLessEqual(targets[-1]-339,10)
        self.assertFalse(self.right.writes)
        count=len(targets)
        self.hold(buttons=(False,)*4,duration=.2)
        self.assertEqual(len(self.left.angles_sent),count)
        self.assertFalse(self.hands.snapshot()['hands'][0]['thumb_hold_active'])

    def test_hold_per_hand_rate_and_hardware_settings(self):
        self.hands.close()
        self.hands = self.make_controller(left_thumb=ThumbRotationConfig(50,0,1000,10),
                                          right_thumb=ThumbRotationConfig(50,0,1000,30))
        self.hands.start()
        self.wait(lambda: all(h['connected'] for h in self.hands.snapshot()['hands']))
        self.hold()
        for key, device, delta in [('y',self.left,1),('a',self.right,-3)]:
            self.update(buttons=(False,)*4)
            self.hold(buttons=tuple(k==key for k in 'abxy'),duration=1.02)
            self.hold(buttons=(False,)*4)
            targets=[a[-1] for a in device.angles_sent]
            self.assertGreaterEqual(len(targets),2)
            self.assertTrue(all(b-a==delta for a,b in zip([339]+targets,targets)))
            # Hold rate is independent of the short-click step (50 here).
            self.assertEqual(device.writes[:2],[(1498,[200]*6),(1522,[200]*6)])
            self.assertEqual(sum(address==1498 for address,_ in device.writes),1)
            self.assertEqual(device.max_active,1)

    def test_hold_slow_worker_never_catches_up_or_replays_released_input(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.hold(buttons=(False,False,False,True),duration=.8)
        self.assertFalse(self.left.writes)
        self.assertEqual(self.hands.snapshot()['hands'][0]['thumb_rotation'],339)
        self.left.read_gate.set()
        self.hold(buttons=(False,False,False,True),duration=.15)
        self.hold(buttons=(False,)*4)
        self.assertTrue(self.left.angles_sent)
        self.assertLessEqual(self.left.angles_sent[-1][-1],349)
        count=len(self.left.angles_sent)
        self.hold(buttons=(False,)*4,duration=.2)
        self.assertEqual(len(self.left.angles_sent),count)

    def test_hold_release_during_slow_preparation_sends_no_angle(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.left.write_gate.clear()
        self.left.write_entered.clear()
        self.hold(buttons=(False,False,False,True),duration=.78)
        self.assertTrue(self.left.write_entered.is_set())
        self.update(buttons=(False,)*4)
        self.left.write_gate.set()
        self.hold(buttons=(False,)*4)
        self.assertFalse(self.left.angles_sent)
        snap=self.hands.snapshot()['hands'][0]
        self.assertEqual(snap['thumb_rotation'],339)
        self.assertEqual(snap['write_status'],0)
        self.assertEqual(snap['thumb_hold_cancel_count'],1)
        self.assertEqual(snap['fault_count'],0)

    def test_hold_waits_for_measured_motion_and_does_not_build_large_target_error(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.hold(buttons=(False,False,False,True),duration=1.4)
        self.assertEqual(self.left.angles_sent[-1][-1],349)
        count=len(self.left.angles_sent)
        self.hold(buttons=(False,False,False,True),duration=.2)
        self.assertEqual(len(self.left.angles_sent),count)
        # Explicit simulated physical progress, not automatic target mirroring.
        self.left.angle[5]=345
        self.hold(buttons=(False,False,False,True),duration=.15)
        self.assertGreater(self.left.angles_sent[-1][-1],349)
        self.assertLessEqual(self.left.angles_sent[-1][-1],355)

    def test_hold_timeout_never_retries_held_key_after_reconnect(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.left.fail_address=1486
        self.hold(buttons=(False,False,False,True),duration=1.05)
        self.assertEqual(len(self.left.angles_sent),1)
        snap=self.hands.snapshot()['hands'][0]
        self.assertEqual(snap['write_status'],3)
        self.assertEqual(snap['write_thumb_rotation'],344)
        self.assertFalse(snap['target_valid'])
        self.assertGreater(snap['fault_count'],0)
        self.left.fail_address=None
        self.hold(buttons=(False,False,False,True),duration=.2)
        self.assertEqual(len(self.left.angles_sent),1)

    def test_hold_pause_and_expiry_clear_worker_intent_before_slow_read_returns(self):
        for invalid in ({'active':False}, {'age':1}):
            with self.subTest(invalid=invalid):
                self.hold()
                self.update(buttons=(False,)*4)
                self.left.read_entered.clear()
                self.left.read_gate.clear()
                self.wait(self.left.read_entered.is_set)
                self.hold(buttons=(False,False,False,True),duration=.73)
                self.update(buttons=(False,False,False,True),**invalid)
                self.left.read_gate.set()
                self.hold(buttons=(False,False,False,True),duration=.08)
                self.hold(buttons=(False,)*4)
                self.assertFalse(self.left.writes)

    def test_shutdown_clears_hold_before_worker_is_free(self):
        self.hold()
        self.update(buttons=(False,)*4)
        self.left.read_entered.clear()
        self.left.read_gate.clear()
        self.wait(self.left.read_entered.is_set)
        self.hold(buttons=(False,False,False,True),duration=.73)
        closing=threading.Thread(target=self.hands.close)
        closing.start()
        self.wait(lambda:self.hands.workers[0].stop_event.is_set())
        self.left.read_gate.set()
        closing.join(timeout=1)
        self.assertFalse(closing.is_alive())
        self.assertFalse(self.left.writes)
        self.assertTrue(all(not w.thread.is_alive() for w in self.hands.workers))

    def test_hold_inflight_masks_data_and_late_ack_does_not_replace_new_grasp(self):
        self.hold()
        self.hold(1,1)
        self.update(1,1,buttons=(False,)*4)
        self.left.write_gate.clear()
        self.left.write_entered.clear()
        self.hold(1,1,buttons=(False,False,False,True),duration=.78)
        self.assertTrue(self.left.write_entered.is_set())
        snap=self.hands.snapshot()
        frame=snapshot_to_frame(snap,frame_time=time.time(),frame_monotonic=time.monotonic())
        self.assertEqual(frame['action.thumb_rotation'].tolist(),[339,339])
        self.assertEqual(frame['hand.write_thumb_rotation'].tolist(),[344,339])
        self.assertFalse(frame['hand.training_valid'][0])
        self.assertFalse(frame['hand.episode_fault'][0])
        self.update(0,1,buttons=(False,)*4)
        self.left.write_gate.set()
        self.hold(0,1,buttons=(False,)*4)
        snap=self.hands.snapshot()['hands'][0]
        self.assertEqual(snap['target'],0)
        self.assertEqual(self.left.angles_sent[-1],list(ANGLES[0]))
        self.assertTrue(snap['write_current'])


class GestureTests(unittest.TestCase):
    def sequence(self, gestures, sessions=(1,1)):
        detector = ThumbButtonGesture()
        events = []
        detector.update((False,)*4, 0, sessions)
        for keys, grip in gestures:
            event = detector.update(tuple(k in keys for k in 'abxy'), grip, sessions)
            if event is not None:
                events.append(event)
        return events

    def test_original_chords_and_staggered_release_never_emit_thumb(self):
        for chord in ('ax','by','ab','xy','abxy','ay','bx'):
            with self.subTest(chord=chord):
                seq = [(chord[:i],0) for i in range(1,len(chord)+1)]
                seq += [(chord[i:],0) for i in range(1,len(chord)+1)]
                self.assertEqual(self.sequence(seq), [])
        # Overlapping buttons, even if no two were down on the same sampled frame.
        self.assertEqual(self.sequence([('a',0),('x',0),('',0)]), [])

    def test_grip_record_discard_combos_in_either_order_are_consumed(self):
        for key in 'abxy':
            for seq in ([('',1),(key,1),(key,0),('',0)],
                        [(key,0),(key,1),('',1),('',0)],
                        [(key,0),(key,1),(key,0),('',0)]):
                self.assertEqual(self.sequence(seq), [])

    def test_startup_held_and_reset_require_new_click(self):
        detector = ThumbButtonGesture()
        for _ in range(2):
            self.assertIsNone(detector.update((False,False,False,True),0,(1,1)))
            self.assertIsNone(detector.update((False,)*4,0,(1,1)))
            detector.reset()
        self.assertEqual(self.sequence([('x',0),('x',0),('',0),('',0)]), [(0,-1,1)])

    def test_missing_input_invalid_grip_and_session_change_cancel_click(self):
        for middle in [(None,0,(1,1)), ((False,)*4,float('nan'),(1,1)),
                       ((False,False,False,True),0,(None,None))]:
            detector = ThumbButtonGesture()
            detector.update((False,)*4,0,(1,1))
            detector.update((False,False,False,True),0,(1,1))
            self.assertIsNone(detector.update(*middle))
            self.assertIsNone(detector.update((False,)*4,0,(1,1)))
        detector = ThumbButtonGesture()
        detector.update((False,)*4,0,(1,1))
        detector.update((False,False,False,True),0,(1,1))
        self.assertIsNone(detector.update((False,)*4,0,(2,1)))
        detector.update((False,True,False,False),0,(2,1))
        self.assertEqual(detector.update((False,)*4,0,(None,1)), (1,1,1))

    def test_thumb_configuration_rejects_invalid_values(self):
        for args in [(0,0,1000),(-1,0,1000),(1001,0,1000),(10,-1,1000),
                     (10,0,1001),(10,500,500),(10,501,500),(0.5,0,1000)]:
            with self.assertRaises(ValueError):
                ThumbRotationConfig(*args)
        for rate in (0, -1, 51, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ThumbRotationConfig(hold_rate=rate)
        for angles in ((),(1,2,3,4),(1,2,3,4,1000),(-1,2,3,4,5),(1,2,3,4,5.5)):
            with self.assertRaises(ValueError):
                validate_close_angles(angles)

    def test_hold_delay_release_chord_grip_and_reset_with_deterministic_clock(self):
        detector=ThumbButtonGesture()
        detector.update((False,)*4,0,(1,1),now=10)
        detector.update((False,False,False,True),0,(1,1),now=10.1)
        detector.update((False,False,False,True),0,(1,1),now=10.69)
        self.assertIsNone(detector.held)
        detector.update((False,False,False,True),0,(1,1),now=10.71)
        self.assertEqual(detector.held,(0,1,1))
        detector.update((False,True,False,True),0,(1,1),now=10.8)
        self.assertIsNone(detector.held)
        detector.update((False,False,False,True),0,(1,1),now=11.5)
        self.assertIsNone(detector.held)
        self.assertIsNone(detector.update((False,)*4,0,(1,1),now=11.6))
        detector.update((True,False,False,False),0,(1,1),now=12)
        detector.update((True,False,False,False),0,(1,1),now=12.7)
        self.assertEqual(detector.held,(1,-1,1))
        detector.update((True,False,False,False),1,(1,1),now=12.8)
        self.assertIsNone(detector.held)
        self.assertIsNone(detector.update((False,)*4,0,(1,1),now=13))
        detector.reset()
        detector.update((False,False,True,False),0,(2,2),now=20)
        detector.update((False,False,True,False),0,(2,2),now=22)
        self.assertIsNone(detector.held)
        self.assertIsNone(detector.update((False,)*4,0,(2,2),now=23))


if __name__ == '__main__':
    unittest.main()
