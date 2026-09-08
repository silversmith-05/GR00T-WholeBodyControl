"""Standalone hand UI over a fake transport; no hardware or user input needed."""
from contextlib import redirect_stderr
import io
import tempfile
import unittest

from inspire_rh56e2 import HandClient
from gear_sonic.scripts.test_inspire_hand import main, run_session
from gear_sonic.tests.test_inspire_hand_controller import FakeDevice
from gear_sonic.utils.teleop.inspire_hand_controller import acquire_hand_lock


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.device = FakeDevice()
        self.configs = []
        self.output = []

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, config):
        self.configs.append(config)
        return HandClient(config, transport=self.device)

    def run_commands(self, commands=(), *, side="right", enabled=True, **kwargs):
        commands = iter(commands)

        def read(_):
            # Even with writes enabled on the command line, startup must be read-only.
            if not self.output:
                raise AssertionError("Missing startup feedback")
            return next(commands)

        return run_session(side, enable_control=enabled, input_fn=read,
                           output_fn=self.output.append, lock_dir=self.tmp.name,
                           client_factory=self.factory, **kwargs)

    def test_default_read_only_never_requests_input_or_writes(self):
        self.assertEqual(self.run_commands(enabled=False), 0)
        self.assertEqual(self.device.writes, [])
        self.assertEqual(self.device.reads, [1546])
        self.assertFalse(self.device.connected)

    def test_quit_and_invalid_input_never_move(self):
        self.assertEqual(self.run_commands(["", "bad", "q"]), 0)
        self.assertFalse(self.device.writes)
        self.assertFalse(self.device.connected)

    def test_open_close_values_dedup_and_actual_feedback(self):
        self.assertEqual(self.run_commands(["0", "0", "s", "1", "q"]), 0)
        self.assertEqual(self.device.angles_sent, [[1000,1000,1000,1000,1000,339],
                                                  [250,250,250,250,300,339]])
        self.assertEqual(self.device.writes[:2], [(1498, [200]*6), (1522, [200]*6)])
        self.assertTrue(any("[500, 500, 500, 500, 500, 500]" in line for line in self.output))
        self.assertTrue(any("不表示实际运动到位" in line for line in self.output))
        self.assertEqual(self.device.max_active, 1)

    def test_custom_five_finger_pose_is_printed_and_sent_consistently(self):
        self.assertEqual(self.run_commands(['1','s','q'],close_angles=(210,220,230,240,290)),0)
        self.assertEqual(self.device.angles_sent,[[210,220,230,240,290,339]])
        self.assertTrue(any('[210, 220, 230, 240, 290, 339]' in line for line in self.output))

    def test_side_and_verified_transport_configuration(self):
        for side, host in (("left", "192.168.123.211"), ("right", "192.168.123.210")):
            self.assertEqual(self.run_commands(side=side, enabled=False), 0)
            config = self.configs[-1]
            self.assertEqual((config.host, config.port, config.unit_id), (host, 6000, 255))
            self.assertEqual((config.model, config.byte_layout, config.tactile_order),
                             ("RH56E2-T1", "packed_little", "big"))

    def test_write_timeout_exits_without_retry_or_reconnect(self):
        self.device.fail_address = 1486
        self.assertEqual(self.run_commands(["1"]), 2)
        self.assertEqual(len(self.device.angles_sent), 1)
        self.assertEqual(self.device.connects, 1)
        self.assertFalse(self.device.connected)
        self.assertTrue(any("写入结果未确认" in line for line in self.output))

    def test_read_failure_never_writes(self):
        self.device.read_fail = True
        self.assertEqual(self.run_commands(["1"]), 1)
        self.assertFalse(self.device.writes)
        self.assertFalse(self.device.connected)

    def test_existing_driver_lock_blocks_connection(self):
        with acquire_hand_lock("192.168.123.210", 6000, self.tmp.name):
            self.assertEqual(self.run_commands(["1"]), 1)
            self.assertFalse(self.configs)
            self.assertEqual(self.device.connects, 0)
        self.assertEqual(self.run_commands(enabled=False), 0)

    def test_both_hands_cannot_be_enabled_by_accident(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--enable-control"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
