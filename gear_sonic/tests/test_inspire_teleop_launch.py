"""Teleop launcher checks with stand-in processes; never connect to a robot."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gear_sonic.scripts import launch_inspire_teleop as launcher


class InspireTeleopLaunchTests(unittest.TestCase):
    def run_standins(self, args):
        with tempfile.TemporaryDirectory(prefix="inspire teleop '") as directory:
            root = Path(directory)
            scripts = root / "gear_sonic/scripts"
            scripts.mkdir(parents=True)
            probe = scripts / "pico_manager_thread_server.py"
            probe.write_text("import json, sys; print(json.dumps(sys.argv[1:]))\n")
            python = root / ".venv_teleop/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            deploy = root / "gear_sonic_deploy/deploy.sh"
            deploy.parent.mkdir()
            deploy.write_text("#!/bin/bash\nexec " + shlex.join([sys.executable, str(probe)]) + ' "$@"\n')
            results = {}
            for name, command in launcher.build_commands(launcher.parse_args(args), root):
                result = subprocess.run(["bash", "-c", "exec " + shlex.join(command)],
                                        cwd=root, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                results[name] = json.loads(result.stdout)
            return results

    def option(self, argv, name):
        return argv[argv.index(name) + 1]

    def test_defaults_launch_body_and_inspire_without_vla_recording_or_cameras(self):
        commands = self.run_standins([])
        self.assertEqual(list(commands), ["deploy", "teleop"])
        deploy, teleop = commands["deploy"], commands["teleop"]
        self.assertIn("--disable-dex3-hands", deploy)
        self.assertNotIn("--only-arms-output", deploy)
        self.assertEqual(self.option(deploy, "--input-type"), "zmq_manager")
        self.assertEqual(self.option(deploy, "--output-type"), "zmq")
        self.assertEqual(deploy[-1], "real")
        self.assertEqual(self.option(teleop, "--hand-backend"), "inspire")
        self.assertIn("--enable-hand-control", teleop)
        self.assertIn("--manager", teleop)
        for name in ("--model-path", "--record_dir", "--camera-host", "--dataset-name"):
            self.assertNotIn(name, teleop)

    def test_custom_ports_hand_targets_and_quoted_paths_reach_children(self):
        model = "policy/operator's model; $(false) `false`"
        commands = self.run_standins([
            "eth0", "--cp", model, "--zmq-port", "5566", "--zmq-out-port", "5567",
            "--inspire-left-ip", "192.168.123.212", "--inspire-port", "6001",
            "--inspire-close-angles", "100", "200", "300", "400", "500",
            "--inspire-left-thumb-hold-rate", "10", "--motor-kp-scale", "4,10=1.5",
        ])
        deploy, teleop = commands["deploy"], commands["teleop"]
        self.assertEqual(self.option(deploy, "--cp"), model)
        self.assertEqual(deploy[-1], "eth0")
        self.assertEqual(self.option(deploy, "--zmq-port"), self.option(teleop, "--port"))
        self.assertEqual(self.option(deploy, "--zmq-out-port"), self.option(teleop, "--zmq_feedback_port"))
        self.assertEqual(self.option(teleop, "--inspire-left-ip"), "192.168.123.212")
        index = teleop.index("--inspire-close-angles")
        self.assertEqual(teleop[index + 1:index + 6], ["100", "200", "300", "400", "500"])
        self.assertEqual(self.option(teleop, "--inspire-left-thumb-hold-rate"), "10.0")

    def test_readonly_hands_still_disable_legacy_driver(self):
        commands = self.run_standins(["--read-only-hands"])
        self.assertNotIn("--enable-hand-control", commands["teleop"])
        self.assertIn("--disable-dex3-hands", commands["deploy"])

    def test_only_arms_output_reaches_body_deploy_and_preserves_inspire_control(self):
        commands = self.run_standins(["--only-arms-output"])
        self.assertIn("--only-arms-output", commands["deploy"])
        self.assertIn("--disable-dex3-hands", commands["deploy"])
        self.assertIn("--enable-hand-control", commands["teleop"])
        self.assertNotIn("--only-arms-output", commands["teleop"])
        commands = self.run_standins(["--component", "deploy", "--only-arms-output",
                                     "--teleop-host", "192.168.3.2"])
        self.assertIn("--only-arms-output", commands["deploy"])

    def test_only_arms_output_rejects_old_binary_before_starting_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deploy = root / "gear_sonic_deploy"
            deploy.mkdir()
            (deploy / "deploy.sh").touch()
            binary = deploy / "target/release/g1_deploy_onnx_ref"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"--disable-dex3-hands")
            args = launcher.parse_args(["--component", "deploy", "--only-arms-output"])
            with patch.object(launcher.shutil, "which", return_value="tmux"), \
                    patch.object(launcher.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "does not support --only-arms-output"):
                    launcher.check_prerequisites(args, root)
                binary.write_bytes(b"--disable-dex3-hands --only-arms-output")
                launcher.check_prerequisites(args, root)
                run.assert_not_called()

    def test_split_components_use_explicit_remote_addresses(self):
        commands = self.run_standins(["--component", "deploy", "--teleop-host", "192.168.3.2"])
        self.assertEqual(list(commands), ["deploy"])
        self.assertEqual(self.option(commands["deploy"], "--zmq-host"), "192.168.3.2")
        commands = self.run_standins(["--component", "teleop", "--state-host", "192.168.3.164"])
        self.assertEqual(list(commands), ["teleop"])
        self.assertEqual(self.option(commands["teleop"], "--zmq_feedback_host"), "192.168.3.164")

    def test_invalid_configuration_never_starts_processes(self):
        for argv in (["sim"], ["lo"], ["--zmq-port", "0"], ["--zmq-port", "5557"],
                     ["--inspire-left-ip", "192.168.123.210"], ["--session", "bad;name"],
                     ["--inspire-left-thumb-hold-rate", "51"],
                     ["--inspire-close-angles", "1000", "250", "250", "250", "300"],
                     ["--component", "teleop", "--only-arms-output"],
                     ["--teleop-host", "192.168.3.2"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                    patch.object(launcher.subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    launcher.main(argv)
                run.assert_not_called()

    def test_dry_run_works_without_venv_or_site_packages_from_another_directory(self):
        result = subprocess.run([sys.executable, "-S", str(Path(launcher.__file__).resolve()), "--dry-run"],
                                cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--disable-dex3-hands", result.stdout)
        self.assertIn("--enable-hand-control", result.stdout)

    def test_check_and_failed_checks_never_create_a_session(self):
        with redirect_stdout(io.StringIO()), patch.object(launcher, "check_prerequisites"), \
                patch.object(launcher.subprocess, "run") as run:
            launcher.main(["--check"])
            run.assert_not_called()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(launcher, "check_prerequisites", side_effect=RuntimeError("SDK missing")), \
                patch.object(launcher.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                launcher.main([])
            run.assert_not_called()

    def test_existing_session_is_preserved(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(launcher, "check_prerequisites"), \
                patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(launcher, "launch_session") as launch:
            with self.assertRaises(SystemExit):
                launcher.main([])
            launch.assert_not_called()

    def test_tmux_prepares_panes_before_start_and_does_not_send_control_keys(self):
        commands = launcher.build_commands(launcher.parse_args([]))
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            output = "%1\n" if argv[1] == "new-session" else "%2\n"
            return subprocess.CompletedProcess(argv, 0, stdout=output)

        with patch.object(launcher.subprocess, "run", side_effect=fake_run):
            launcher.launch_session("test_inspire", commands)
        operations = [call[1] for call in calls]
        self.assertNotIn("send-keys", operations)
        self.assertLess(operations.index("split-window"), operations.index("respawn-pane"))
        starts = [call for call in calls if call[1] == "respawn-pane"]
        self.assertEqual(len(starts), 2)
        for call, (_, expected) in zip(starts, commands):
            shell = shlex.split(call[-1])
            self.assertEqual(shell[:2], ["bash", "-c"])
            self.assertEqual(shlex.split(shell[2]), ["exec", *expected])
        self.assertEqual(calls[-1], ["tmux", "select-pane", "-t", "%1"])


if __name__ == "__main__":
    unittest.main()
