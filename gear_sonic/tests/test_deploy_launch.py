"""Launcher-to-shell regression tests with fake deployment tools; no hardware I/O."""

from contextlib import redirect_stdout
import io
import itertools
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import tyro

from gear_sonic.scripts import launch_data_collection as launcher


class DeployLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sonic-deploy-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        deploy_dir = Path(__file__).resolve().parents[2] / "gear_sonic_deploy"
        shutil.copy2(deploy_dir / "deploy.sh", self.root / "deploy.sh")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/setup_env.sh").write_text(":\n")
        for name in ("policy/release/model_decoder.onnx", "policy/release/model_encoder.onnx",
                     "policy/release/observation_config.yaml", "planner/target_vel/V2/planner_sonic.onnx"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / "reference/example").mkdir(parents=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # Only the test shim can receive `just run`; never launch the real executable.
        self.make_tool("just", 'if [[ "$1" == run ]]; then printf "%s\\0" "$@" > "$DEPLOY_TEST_ARGS"; fi\n')
        for name in ("cmake", "clang", "git"):
            self.make_tool(name, ":\n")
        self.make_tool("ip", "cat <<'EOF'\n1: lo: <LOOPBACK>\n    inet 127.0.0.1/8\n"
                       "2: test0: <BROADCAST>\n    inet 192.168.123.1/24\nEOF\n")

    def make_tool(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nset -e\n" + body)
        path.chmod(0o755)

    def run_deploy(self, args):
        captured = self.root / "arguments.bin"
        captured.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(self.root / "deploy.sh"), *args], input="y\n",
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                 "TensorRT_ROOT": str(self.root), "DEPLOY_TEST_ARGS": str(captured)},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(captured.exists(), result.stdout + result.stderr)
        return captured.read_bytes().decode().split("\0")[:-1]

    def launch_commands(self, config):
        result = subprocess.CompletedProcess([], 0, "0")
        with patch.object(launcher, "_check_prerequisites"), \
             patch.object(launcher, "_get_local_ip", return_value="127.0.0.1"), \
             patch.object(launcher.time, "sleep"), \
             patch.object(launcher.subprocess, "run", return_value=result) as run, \
             redirect_stdout(io.StringIO()):
            launcher.main(config)
        return [shlex.split(call.args[0][-2]) for call in run.call_args_list
                if call.args[0][:2] == ["tmux", "send-keys"]]

    def test_hand_backend_and_motor_gains_reach_deploy_together(self):
        for backend, sim, gains in itertools.product(("dex3", "inspire"), (False, True), (False, True)):
            with self.subTest(backend=backend, sim=sim, gains=gains):
                config = launcher.DataCollectionLaunchConfig(
                    hand_backend=backend, sim=sim, camera_web=False, camera_viewer=False,
                    deploy_motor_kp_scale="4,10=1.5" if gains else "",
                    deploy_motor_kd_scale="4-5=0.8" if gains else "",
                )
                commands = self.launch_commands(config)
                deploy = next(command for command in commands if "./deploy.sh" in command)
                mode = "sim" if sim else "real"
                self.assertEqual(deploy[-1], mode)
                self.assertEqual(deploy.count(mode), 1)
                actual = self.run_deploy(deploy[deploy.index("./deploy.sh") + 1:])
                self.assertEqual(actual[:3], ["run", "g1_deploy_onnx_ref", "lo" if sim else "test0"])
                expected = ["--disable-dex3-hands"] if backend == "inspire" else []
                if sim:
                    expected += ["--disable-crc-check"]
                if gains:
                    expected += ["--motor-kp-scale", "4,10=1.5", "--motor-kd-scale", "4-5=0.8"]
                self.assertEqual(actual[actual.index("--zmq-host") + 2:], expected)
                for script in ("pico_manager_thread_server.py", "run_data_exporter.py"):
                    command = next(command for command in commands
                                   if f"gear_sonic/scripts/{script}" in command)
                    self.assertEqual(command[command.index("--hand-backend") + 1], backend)
                    self.assertNotIn("--motor-kp-scale", command)

    def test_repeated_gain_options_remain_separate_arguments(self):
        actual = self.run_deploy([
            "--motor-kp-scale", "4,10=1.5", "--disable-dex3-hands",
            "--motor-kp-scale", "11-12=1.2", "--motor-kd-scale", "4-5=0.8", "sim",
        ])
        self.assertEqual(actual[actual.index("--zmq-host") + 2:], [
            "--disable-dex3-hands", "--disable-crc-check",
            "--motor-kp-scale", "4,10=1.5", "--motor-kp-scale", "11-12=1.2",
            "--motor-kd-scale", "4-5=0.8",
        ])

    def test_cli_accepts_upstream_gains_with_inspire_and_camera_options(self):
        config = tyro.cli(launcher.DataCollectionLaunchConfig, args=[
            "--deploy-motor-kp-scale", "4,10=1.5", "--deploy-motor-kd-scale", "4-5=0.8",
            "--hand-backend", "inspire", "--enable-hand-control", "--sim",
            "--camera-web-port", "8090", "--inspire-left-thumb-step", "5",
        ])
        self.assertEqual((config.deploy_motor_kp_scale, config.deploy_motor_kd_scale),
                         ("4,10=1.5", "4-5=0.8"))
        self.assertEqual((config.hand_backend, config.inspire_left_thumb_step), ("inspire", 5))
        self.assertTrue(config.enable_hand_control)
        self.assertTrue(config.sim)
        self.assertTrue(config.camera_web)
        self.assertEqual(config.camera_web_port, 8090)


if __name__ == "__main__":
    unittest.main()
