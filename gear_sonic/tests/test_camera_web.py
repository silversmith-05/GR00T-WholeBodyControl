"""Exercise the relay over loopback ZMQ/HTTP; never connect to robot hardware."""

from contextlib import redirect_stdout
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import msgpack
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.camera.web_server import CameraRelay, CameraWebServer
from gear_sonic.scripts import launch_data_collection as launcher


class CameraWebTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.XPUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.camera_port = self.publisher.bind_to_random_port("tcp://127.0.0.1")
        self.relay = CameraRelay("127.0.0.1", self.camera_port, fps=60)
        self.server = CameraWebServer(("127.0.0.1", 0), self.relay)
        self.http_port = self.server.server_port
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.http_thread.start()
        self.relay.start()
        self.connections = []
        self.assertTrue(self.publisher.poll(3000), "ZMQ subscriber did not connect")
        self.assertEqual(self.publisher.recv(), b"\x01")

    def tearDown(self):
        for connection in self.connections:
            connection.close()
        self.relay.close()
        self.server.shutdown()
        self.server.server_close()
        self.http_thread.join(timeout=3)
        self.publisher.close()
        self.context.term()
        self.assertFalse(self.relay.thread.is_alive())
        self.assertFalse(self.http_thread.is_alive())

    def get(self, path):
        connection = HTTPConnection("127.0.0.1", self.http_port, timeout=3)
        self.connections.append(connection)
        connection.request("GET", path)
        return connection.getresponse()

    def publish(self, images):
        payload = ImageMessageSchema({name: time.time() for name in images}, images).serialize()
        previous = self.relay.sequence
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.publisher.send(msgpack.packb(payload, use_bin_type=True))
            with self.relay.condition:
                if self.relay.condition.wait_for(lambda: self.relay.sequence > previous, timeout=0.05):
                    return
        self.fail(f"No frame received: {self.relay.status()}")

    def read_jpeg(self, response):
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
        self.assertEqual(response.readline(), b"--frame\r\n")
        self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
        length = int(response.readline().split(b":", 1)[1])
        self.assertEqual(response.readline(), b"\r\n")
        jpeg = response.read(length)
        self.assertEqual(response.read(2), b"\r\n")
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    def test_page_is_available_before_camera_and_bad_routes(self):
        response = self.get("/")
        self.assertEqual(response.status, 200)
        self.assertIn(b"SONIC Camera Preview", response.read())
        status = json.loads(self.get("/status").read())
        self.assertFalse(status["online"])
        self.assertEqual(status["cameras"], [])
        response = self.get("/stream")
        self.assertEqual(response.status, 503)
        response.read()
        self.assertEqual(self.get("/missing").status, 404)

    def test_wire_formats_colors_mono_and_multiple_browsers(self):
        # Legacy array -> JPEG/base64 contains RGB channel order; bytes are normal JPEG.
        red_rgb = np.full((80, 120, 3), (255, 0, 0), dtype=np.uint8)
        blue_bgr = np.full((80, 120, 3), (255, 0, 0), dtype=np.uint8)
        _, blue_jpeg = cv2.imencode(".jpg", blue_bgr)
        self.publish({"a_ego": red_rgb, "b_wrist": blue_jpeg.tobytes(),
                      "c_mono": np.full((80, 120), 125, dtype=np.uint8)})
        streams = [self.get("/stream"), self.get("/stream?t=browser2")]
        for response in streams:
            image = self.read_jpeg(response)
            self.assertEqual(image.shape, (224, 240, 3))
            np.testing.assert_allclose(image[65, 50], [0, 0, 255], atol=8)
            np.testing.assert_allclose(image[65, 170], [255, 0, 0], atol=8)
            np.testing.assert_allclose(image[180, 50], [125, 125, 125], atol=8)
        # The two clients both receive a newer frame from the same subscriber.
        self.publish({"a_ego": np.full((80, 120, 3), (0, 255, 0), dtype=np.uint8)})
        for response in streams:
            np.testing.assert_allclose(self.read_jpeg(response)[65, 50], [0, 255, 0], atol=8)

    def test_stale_partial_updates_and_recovery_after_bad_message(self):
        self.relay.stale_after = 0.15
        frame = np.full((80, 120, 3), 120, dtype=np.uint8)
        self.publish({"ego": frame, "wrist": frame})
        time.sleep(0.2)
        self.assertFalse(json.loads(self.get("/status").read())["online"])
        self.assertEqual(self.get("/stream").status, 503)
        # Invalid msgpack must not kill the receiver.
        self.publisher.send(b"\xc1")
        with self.relay.condition:
            self.relay.condition.wait_for(lambda: self.relay.error is not None, timeout=0.2)
        self.publish({"ego": frame})
        status = json.loads(self.get("/status").read())
        self.assertTrue(status["online"])
        self.assertIsNone(status["error"])
        self.assertEqual({c["name"]: c["stale"] for c in status["cameras"]},
                         {"ego": False, "wrist": True})
        image = self.read_jpeg(self.get("/stream"))
        self.assertLess(image[65, 170].mean(), 50)

    def unused_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def test_existing_browser_stream_resumes_after_camera_server_restart(self):
        self.relay.stale_after = 0.15
        self.publish({"ego": np.full((80, 120, 3), (255, 0, 0), dtype=np.uint8)})
        response = self.get("/stream")
        self.read_jpeg(response)
        self.publisher.close()
        time.sleep(1.1)
        self.assertFalse(json.loads(self.get("/status").read())["online"])
        self.publisher = self.context.socket(zmq.XPUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.bind(f"tcp://127.0.0.1:{self.camera_port}")
        self.assertTrue(self.publisher.poll(3000), "Relay did not reconnect to restarted publisher")
        self.assertEqual(self.publisher.recv(), b"\x01")
        self.publish({"ego": np.full((80, 120, 3), (0, 255, 0), dtype=np.uint8)})
        # Skip the cached heartbeat, then observe new pixels on the same HTTP connection.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            pixel = self.read_jpeg(response)[65, 50]
            if pixel[1] > 240 and pixel[2] < 10:
                break
        else:
            self.fail("Browser stream did not resume after camera server restart")

    def wait_http(self, port):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            connection = HTTPConnection("127.0.0.1", port, timeout=0.2)
            try:
                connection.request("GET", "/status")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                return json.loads(response.read())
            except OSError:
                time.sleep(0.05)
            finally:
                connection.close()
        self.fail(f"Preview did not start on port {port}")

    def assert_port_released(self, port):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                    return
                except OSError:
                    time.sleep(0.05)
        self.fail(f"Preview still holds port {port}")

    def test_standalone_bootstrap_port_conflict_and_sigterm_cleanup(self):
        root = Path(__file__).resolve().parents[2]
        port = self.unused_port()
        command = [shutil.which("python3"), str(root / "gear_sonic/scripts/run_camera_web.py"),
                   "--camera-host", "127.0.0.1", "--camera-port", str(self.camera_port),
                   "--port", str(port)]
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            self.assertEqual(self.wait_http(port)["source"], f"tcp://127.0.0.1:{self.camera_port}")
            duplicate = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("Choose another --port", duplicate.stderr)
            self.assertIsNone(process.poll())
            process.terminate()
            output, _ = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, output)
            self.assert_port_released(port)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

    @unittest.skipUnless(shutil.which("tmux"), "tmux not installed")
    def test_tmux_session_owns_relay_and_releases_port_on_exit(self):
        root = Path(__file__).resolve().parents[2]
        port = self.unused_port()
        with tempfile.TemporaryDirectory(prefix="sonic-camera-test-") as directory:
            # An isolated socket/config ensures this test never touches the user's sessions.
            tmux = ["tmux", "-S", str(Path(directory) / "tmux.socket")]
            subprocess.run([*tmux, "-f", "/dev/null", "new-session", "-d", "-s", "preview_test",
                            "-n", "base", "sleep 60"], check=True, capture_output=True)
            run = subprocess.run
            try:
                config = launcher.DataCollectionLaunchConfig(
                    camera_host="127.0.0.1", camera_port=self.camera_port, camera_web_port=port)
                with patch.object(launcher, "SESSION_NAME", "preview_test"), \
                     patch.object(launcher.subprocess, "run", side_effect=lambda args, **kw: run(
                         [*tmux, *args[1:]], **kw)):
                    launcher._start_camera_web(config, root)
                self.assertFalse(self.wait_http(port)["online"])
                active = run([*tmux, "display-message", "-p", "-t", "preview_test", "#{window_name}"],
                             check=True, capture_output=True, text=True)
                self.assertEqual(active.stdout.strip(), "base", "Relay must not change the active window")
                run([*tmux, "kill-session", "-t", "preview_test"], check=True, capture_output=True)
                self.assert_port_released(port)
            finally:
                run([*tmux, "kill-server"], capture_output=True)


class CameraWebLaunchTests(unittest.TestCase):
    def run_launcher(self, **options):
        config = launcher.DataCollectionLaunchConfig(**options)
        output = io.StringIO()
        result = subprocess.CompletedProcess([], 0, "0")
        with patch.object(launcher, "_check_prerequisites"), \
             patch.object(launcher, "_get_local_ip", return_value="127.0.0.1"), \
             patch.object(launcher.time, "sleep"), \
             patch.object(launcher.subprocess, "run", return_value=result) as run, \
             redirect_stdout(output):
            launcher.main(config)
        return [call.args[0] for call in run.call_args_list], output.getvalue()

    def test_existing_inspire_command_starts_relay_with_same_camera(self):
        calls, output = self.run_launcher(
            camera_host="192.168.123.164", task_prompt="pick up the cup",
            hand_backend="inspire", enable_hand_control=True,
            inspire_left_ip="192.168.123.211", inspire_right_ip="192.168.123.210",
            inspire_port=6000, camera_port=5560, camera_web_port=8090,
            camera_web_host="0.0.0.0", camera_viewer=False,
        )
        windows = [call for call in calls if call[:2] == ["tmux", "new-window"]]
        self.assertEqual(len(windows), 1)
        self.assertIn("camera_web", windows[0])
        command = shlex.split(windows[0][-1])
        self.assertEqual(command[0], "exec")
        self.assertIn("run_camera_web.py", command[3])
        for flag, value in (("--camera-host", "192.168.123.164"), ("--camera-port", "5560"),
                            ("--host", "0.0.0.0"), ("--port", "8090")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("http://localhost:8090", output)
        pane_commands = [call[-2] for call in calls if call[:2] == ["tmux", "send-keys"]]
        self.assertTrue(any("--disable-dex3-hands" in command for command in pane_commands))
        self.assertTrue(any("--enable-hand-control" in command for command in pane_commands))
        self.assertFalse(any("run_camera_viewer.py" in command for command in pane_commands))

    def test_relay_can_be_disabled_independently_of_viewer(self):
        calls, output = self.run_launcher(camera_web=False)
        self.assertFalse(any(call[:2] == ["tmux", "new-window"] for call in calls))
        self.assertIn("Disabled", output)
        self.assertTrue(any("run_camera_viewer.py" in call[-2]
                            for call in calls if call[:2] == ["tmux", "send-keys"]))

    def test_paths_and_arguments_are_shell_quoted(self):
        root = Path("/tmp/preview project")
        config = launcher.DataCollectionLaunchConfig(camera_host="host; echo example")
        with patch.object(launcher.subprocess, "run") as run:
            launcher._start_camera_web(config, root)
        command = shlex.split(run.call_args.args[0][-1])
        self.assertEqual(command[1], str(root / ".venv_data_collection/bin/python"))
        self.assertEqual(command[command.index("--camera-host") + 1], config.camera_host)


if __name__ == "__main__":
    unittest.main()
