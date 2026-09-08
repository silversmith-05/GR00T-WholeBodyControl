"""Wireless launch regression tests; never start robot, camera or teleop clients."""

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

from gear_sonic.scripts import launch_data_collection_wireless as launcher


class WirelessLaunchTests(unittest.TestCase):
    def test_script_runs_without_site_packages_from_another_directory(self):
        for role in ("--nx", "--workstation"):
            result = subprocess.run(
                [sys.executable, "-S", str(Path(launcher.__file__).resolve()), role, "--dry-run"],
                cwd="/tmp", capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("192.168.3.164:5557", result.stdout)
            self.assertIn("192.168.3.2:5556", result.stdout)

    def test_roles_and_invalid_options_fail_before_starting_any_process(self):
        for args in ([], ["--nx", "--workstation"], ["--nx", "--deploy-output-type", "ros2"],
                     ["--workstation", "--camera-port", "0"],
                     ["--workstation", "--data-exporter-frequency", "0"],
                     ["--nx", "--hand-backend", "inspire", "--no-pico-manager"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), \
                 patch.object(launcher.subprocess, "run") as run:
                with self.assertRaises(SystemExit) as exc:
                    launcher.parse_args(args)
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()

    def run_commands_with_fake_programs(self, config):
        """Execute the actual shell commands against stand-ins, including quoting/venv activation."""
        with tempfile.TemporaryDirectory(prefix="sonic wireless '") as temporary:
            root = Path(temporary)
            scripts = root / "gear_sonic/scripts"
            scripts.mkdir(parents=True)
            probe = "import json, sys; print(json.dumps(sys.argv[1:]))\n"
            for name in ("pico_manager_thread_server.py", "run_data_exporter.py",
                         "run_camera_viewer.py", "run_camera_web.py"):
                (scripts / name).write_text(probe)
            for venv in (".venv_teleop", ".venv_data_collection"):
                bin_dir = root / venv / "bin"
                bin_dir.mkdir(parents=True)
                (bin_dir / "python").symlink_to(sys.executable)
                (bin_dir / "activate").write_text(f"export PATH={shlex.quote(str(bin_dir))}:\"$PATH\"\n")
            deploy_dir = root / "gear_sonic_deploy"
            deploy_dir.mkdir()
            deploy = deploy_dir / "deploy.sh"
            deploy.write_text(f"#!{sys.executable}\n" + probe)
            deploy.chmod(0o755)
            results = {}
            for name, command in launcher.build_commands(config, root):
                result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                results[name] = json.loads(result.stdout)
            return results

    def assert_option(self, argv, flag, expected):
        self.assertEqual(argv[argv.index(flag) + 1], expected)

    def test_nx_starts_only_deploy_and_keeps_robot_interface_separate(self):
        config = launcher.parse_args([
            "--nx", "--deploy-interface", "enP8p1s0", "--workstation-host", "192.168.3.20",
            "--deploy-checkpoint", "policy/my model", "--deploy-motor-kp-scale", "4,10=1.5",
            "--hand-backend", "inspire",
        ])
        commands = self.run_commands_with_fake_programs(config)
        self.assertEqual(list(commands), ["deploy"])
        deploy = commands["deploy"]
        self.assert_option(deploy, "--zmq-host", "192.168.3.20")
        self.assert_option(deploy, "--output-type", "zmq")
        self.assert_option(deploy, "--cp", "policy/my model")
        self.assert_option(deploy, "--motor-kp-scale", "4,10=1.5")
        self.assertIn("--disable-dex3-hands", deploy)
        self.assertEqual(deploy[-1], "enP8p1s0")

    def test_workstation_receives_remote_feedback_state_cameras_and_local_pose(self):
        task = "pick up operator's cup; $(false) `false`\nnext instruction"
        config = launcher.parse_args(["--workstation", "--task-prompt", task, "--dataset-name", "cup 'one'"])
        commands = self.run_commands_with_fake_programs(config)
        self.assertEqual(set(commands), {"teleop", "exporter", "camera_viewer", "camera_web"})
        self.assert_option(commands["teleop"], "--zmq_feedback_host", "192.168.3.164")
        self.assert_option(commands["teleop"], "--zmq_feedback_port", "5557")
        self.assert_option(commands["teleop"], "--port", "5556")
        self.assert_option(commands["exporter"], "--state-zmq-host", "192.168.3.164")
        self.assert_option(commands["exporter"], "--sonic-zmq-host", "127.0.0.1")
        self.assert_option(commands["exporter"], "--task-prompt", task)
        self.assert_option(commands["exporter"], "--dataset-name", "cup 'one'")
        for name in ("exporter", "camera_viewer", "camera_web"):
            self.assert_option(commands[name], "--camera-host", "192.168.3.164")

    def test_host_overrides_and_optional_components(self):
        config = launcher.parse_args([
            "--workstation", "--nx-host", "192.168.3.10", "--camera-host", "192.168.3.11",
            "--camera-port", "6005", "--no-camera-viewer", "--no-camera-web", "--no-text-to-speech",
            "--record-wrist-cameras", "--pico-input-source", "isaac-teleop", "--pico-vis-vr3pt",
        ])
        commands = self.run_commands_with_fake_programs(config)
        self.assertEqual(set(commands), {"teleop", "exporter"})
        self.assert_option(commands["teleop"], "--zmq_feedback_host", "192.168.3.10")
        self.assert_option(commands["teleop"], "--input-source", "isaac-teleop")
        self.assertIn("--vis_vr3pt", commands["teleop"])
        self.assert_option(commands["exporter"], "--state-zmq-host", "192.168.3.10")
        self.assert_option(commands["exporter"], "--camera-host", "192.168.3.11")
        self.assert_option(commands["exporter"], "--camera-port", "6005")
        self.assertIn("--no-text-to-speech", commands["exporter"])
        self.assertIn("--record-wrist-cameras", commands["exporter"])

    def test_prerequisites_are_local_to_each_role(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(launcher.shutil, "which", return_value="tmux"):
            root = Path(temporary)
            (root / "gear_sonic_deploy/scripts").mkdir(parents=True)
            for name in ("deploy.sh", "scripts/setup_env.sh"):
                (root / "gear_sonic_deploy" / name).touch()
            launcher.check_prerequisites(launcher.parse_args(["--nx"]), root)
            with self.assertRaisesRegex(ValueError, ".venv_teleop"):
                launcher.check_prerequisites(launcher.parse_args(["--workstation"]), root)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(launcher.shutil, "which", return_value="tmux"):
            root = Path(temporary)
            for venv in (".venv_teleop", ".venv_data_collection"):
                (root / venv / "bin").mkdir(parents=True)
                for name in ("python", "activate"):
                    (root / venv / "bin" / name).touch()
            launcher.check_prerequisites(launcher.parse_args(["--workstation"]), root)

    def test_ethernet_auto_detection_does_not_fall_back_to_wifi(self):
        wifi = {"ifname": "wlan0", "addr_info": [{"local": "192.168.3.164"}]}
        ethernet = {"ifname": "eth0", "addr_info": [{"local": "192.168.123.164"}]}
        for interfaces, expected in (([wifi, ethernet], "eth0"), ([wifi], None),
                                     ([ethernet, {**ethernet, "ifname": "eth1"}], None)):
            result = subprocess.CompletedProcess([], 0, json.dumps(interfaces))
            with patch.object(launcher.subprocess, "run", return_value=result):
                if expected:
                    self.assertEqual(launcher.resolve_deploy_interface("real"), expected)
                else:
                    with self.assertRaisesRegex(ValueError, "--deploy-interface"):
                        launcher.resolve_deploy_interface("real")

    def test_dry_run_and_existing_session_never_start_processes(self):
        for role in ("--nx", "--workstation"):
            with patch.object(launcher.subprocess, "run") as run, redirect_stdout(io.StringIO()):
                launcher.main(launcher.parse_args([role, "--dry-run"]))
                run.assert_not_called()
        with patch.object(launcher, "check_prerequisites"), \
             patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            with self.assertRaisesRegex(ValueError, "already exists"):
                launcher.main(launcher.parse_args(["--nx"]))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][1], "has-session")

    def test_normal_launch_uses_only_role_commands_and_preserves_pane_processes(self):
        for role in ("--nx", "--workstation"):
            panes = iter(("%31", "%42", "%53", "%64"))

            def fake_tmux(argv, **kwargs):
                self.assertEqual(argv[0], "tmux")
                output = next(panes) if argv[1] in {"new-session", "split-window", "new-window"} else ""
                return subprocess.CompletedProcess(argv, 1 if argv[1] == "has-session" else 0, output)

            with patch.object(launcher, "check_prerequisites"), \
                 patch.object(launcher, "resolve_deploy_interface", return_value="eth0"), \
                 patch.object(launcher.subprocess, "run", side_effect=fake_tmux) as run, \
                 redirect_stdout(io.StringIO()):
                launcher.main(launcher.parse_args([role, "--no-attach"]))
            calls = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any(argv[1] in {"attach", "kill-session", "send-keys"} for argv in calls))
            commands = [shlex.split(argv[-1]) for argv in calls if argv[1] in {"respawn-pane", "split-window"}]
            self.assertEqual(len(commands), 1 if role == "--nx" else 4)
            for command in commands:
                self.assertEqual(command[:2], ["bash", "-lc"])
                self.assertIn("&& exec", command[2])
                self.assertEqual("./deploy.sh" in command[2], role == "--nx")
            self.assertTrue(any("remain-on-exit" in argv for argv in calls))
            self.assertEqual(calls[-1], ["tmux", "select-pane", "-t", "%31" if role == "--nx" else "%42"])


if __name__ == "__main__":
    unittest.main()
