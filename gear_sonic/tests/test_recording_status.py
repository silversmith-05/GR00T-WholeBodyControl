"""Recording state checks with fake exporter writes and loopback HTTP, never hardware."""

from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from gear_sonic.camera.web_server import CameraRelay, CameraWebServer
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.recording_status import (
    RecordingStatusPublisher, default_recording_status_path, read_recording_status,
)


class RecordingStatusTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "recording.json"
        self.publisher = RecordingStatusPublisher(self.path, "test-dataset")
        self.addCleanup(self.publisher.close)
        self.wait_status(state="idle")

    def wait_status(self, **expected):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = read_recording_status(self.path)
            if status["available"] and all(status.get(key) == value for key, value in expected.items()):
                return status
            time.sleep(0.01)
        self.fail(f"Expected {expected}, received {status}")

    def collector(self):
        collector = GrootDataCollector.__new__(GrootDataCollector)
        collector._recording_status = self.publisher
        collector._episode_state = EpisodeState()
        collector._keyboard_listener = Mock()
        collector._keyboard_listener.read_msg.return_value = None
        collector._manager_toggle_dc = collector._manager_toggle_da = False
        collector.hand_backend = "dex3"
        collector._hand_episode_fault = False
        collector._hand_fault_counts = None
        collector.latest_hand_msg = None
        collector.frequency = 50
        collector.sonic_timing_monitor = Mock()
        collector._print_and_say = Mock()
        collector.data_exporter = Mock()
        collector.data_exporter.episode_buffer = {"episode_index": 7, "size": 2}

        def save(discarded):
            self.wait_status(state="saving", recording=False, discarded=discarded)
            collector.data_exporter.episode_buffer["episode_index"] += 1
            collector.data_exporter.episode_buffer["size"] = 0

        collector.data_exporter.save_episode.side_effect = lambda: save(False)
        collector.data_exporter.save_episode_as_discarded.side_effect = lambda: save(True)
        return collector

    def toggle(self, collector, discard=False):
        if discard:
            collector._manager_toggle_da = True
        else:
            collector._manager_toggle_dc = True
        collector._check_recording_commands()

    def test_controller_discard_latches_for_last_episode_and_resets_on_next_start(self):
        collector = self.collector()
        self.toggle(collector, discard=True)  # B while idle is ignored by the exporter.
        collector.data_exporter.save_episode_as_discarded.assert_not_called()
        self.toggle(collector)
        self.wait_status(state="recording", recording=True, episode_index=7, discarded=False)
        self.toggle(collector, discard=True)
        self.wait_status(state="idle", recording=False, episode_index=7, discarded=True,
                         discard_reason="operator", result="discarded")
        self.assertEqual(collector.current_episode_index, 8)
        self.assertEqual(collector._episode_state.get_state(), "idle")
        self.toggle(collector)
        self.wait_status(state="recording", episode_index=8, discarded=False, discard_reason=None, result=None)

    def test_normal_stop_save_and_empty_episode(self):
        collector = self.collector()
        self.toggle(collector)
        self.toggle(collector)
        self.wait_status(state="saving", recording=False)
        collector._finalize_frame(time.monotonic())
        self.wait_status(state="idle", episode_index=7, discarded=False, result="saved")
        collector.data_exporter.save_episode.assert_called_once()
        self.toggle(collector)
        self.toggle(collector)
        collector._finalize_frame(time.monotonic())
        self.wait_status(state="idle", episode_index=8, discarded=False, result="empty")
        collector.data_exporter.save_episode.assert_called_once()

    def test_hand_fault_and_failed_save_are_not_reported_as_normal_success(self):
        collector = self.collector()
        collector.hand_backend = "inspire"
        self.toggle(collector)
        collector._mark_hand_episode_fault()
        self.wait_status(state="recording", discarded=True, discard_reason="hand_fault")
        self.toggle(collector)
        collector._finalize_frame(time.monotonic())
        self.wait_status(state="idle", result="discarded", discard_reason="hand_fault")
        self.toggle(collector)
        collector.data_exporter.save_episode_as_discarded.side_effect = RuntimeError("Disk failed")
        with self.assertRaisesRegex(RuntimeError, "Disk failed"):
            self.toggle(collector, discard=True)
        self.wait_status(state="error", recording=False, result="save_failed")

    def test_all_browsers_receive_actual_status_even_without_camera_frames(self):
        relay = CameraRelay("test-camera", 5555)
        server = CameraWebServer(("127.0.0.1", 0), relay, self.path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.publisher.update(state="recording", episode_index=0)
            self.wait_status(recording=True)
            for _ in range(2):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                try:
                    connection.request("GET", "/status")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    data = json.loads(response.read())
                    self.assertFalse(data["online"])
                    self.assertTrue(data["recording"]["recording"])
                    self.assertEqual(data["recording"]["episode_index"], 0)
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            relay.close()

    def test_heartbeat_continues_while_saving_and_stops_on_exit(self):
        self.publisher.update(state="saving", episode_index=3)
        original = self.wait_status(state="saving")["updated_at"]
        deadline = time.monotonic() + 2
        while read_recording_status(self.path).get("updated_at", 0) <= original:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(read_recording_status(self.path, stale_after=0.5)["state"], "saving")
        self.publisher.close()
        self.assertFalse(read_recording_status(self.path)["available"])

    def test_missing_stale_and_malformed_status_is_unknown(self):
        path = self.path.with_name("invalid.json")
        self.assertIsNone(read_recording_status(path)["recording"])
        valid = dict(version=1, state="recording", discarded=False, episode_index=0, updated_at=100)
        with patch("gear_sonic.utils.data_collection.recording_status.time.time", return_value=104):
            for value in (valid, [], None, {}, {**valid, "updated_at": 104, "discarded": "false"},
                          {**valid, "updated_at": float("nan")}, {**valid, "updated_at": 200}):
                path.write_text(json.dumps(value))
                self.assertFalse(read_recording_status(path)["available"])
                self.assertIsNone(read_recording_status(path)["recording"])
        path.write_text('{"state":')
        self.assertFalse(read_recording_status(path)["available"])
        self.assertEqual(default_recording_status_path("camera", 5555), default_recording_status_path(" CAMERA ", 5555))
        self.assertNotEqual(default_recording_status_path("camera", 5555), default_recording_status_path("camera", 5556))


if __name__ == "__main__":
    unittest.main()
