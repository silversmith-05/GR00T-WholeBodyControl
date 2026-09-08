r"""Launch SONIC data collection across an NX and a workstation over Wi-Fi.

On NX (192.168.3.164):
    python gear_sonic/scripts/launch_data_collection_wireless.py --nx
On the workstation (192.168.3.2):
    python gear_sonic/scripts/launch_data_collection_wireless.py --workstation \
        --task-prompt "pick up the cup"

The NX runs C++ deploy and uses Ethernet for robot DDS traffic. The workstation
runs teleop, the exporter, and camera previews. Keep the camera server running on
the NX as in the wired setup. Use --dry-run to inspect commands without starting
any processes. The launcher itself only needs Python's standard library.
"""

import argparse
from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

# Support both direct script execution and package imports without an installed
# gear_sonic package or workstation Python environment on the NX.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.scripts.launch_data_collection import (
    DataCollectionLaunchConfig,
    hand_launch_arguments,
)


@dataclass
class WirelessDataCollectionLaunchConfig(DataCollectionLaunchConfig):
    nx: bool = False
    workstation: bool = False
    nx_host: str = "192.168.3.164"
    workstation_host: str = "192.168.3.2"
    deploy_interface: str = "real"
    deploy_zmq_host: str = ""
    deploy_output_type: str = "zmq"
    camera_host: str = ""
    dry_run: bool = False
    no_attach: bool = False


def parse_args(argv=None) -> WirelessDataCollectionLaunchConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    role = parser.add_mutually_exclusive_group(required=True)
    role.add_argument("--nx", action="store_true", help="Run only C++ deploy on the NX.")
    role.add_argument("--workstation", action="store_true", help="Run teleop, exporter and previews.")
    defaults = WirelessDataCollectionLaunchConfig()

    def option(group, name, help_text, **kwargs):
        default = getattr(defaults, name)
        if isinstance(default, bool):
            kwargs.setdefault("action", argparse.BooleanOptionalAction)
        else:
            kwargs.setdefault("type", type(default))
            help_text += f" (default: {default!r})"
        group.add_argument(
            "--" + name.replace("_", "-"), default=default,
            help=help_text, **kwargs,
        )

    network = parser.add_argument_group("Wireless network")
    option(network, "nx_host", "NX Wi-Fi address; robot state and default camera host")
    option(network, "workstation_host", "Workstation Wi-Fi address; teleop publisher and PICO PC IP")
    option(network, "camera_host", "Override camera host; empty uses --nx-host")
    option(network, "camera_port", "Camera server TCP port")
    option(network, "deploy_zmq_host", "Override teleop publisher; empty uses --workstation-host")

    deploy = parser.add_argument_group("NX deployment")
    option(deploy, "deploy_interface", "NX robot Ethernet interface/IP; 'real' detects 192.168.123.x")
    option(deploy, "deploy_input_type", "C++ input handler")
    option(deploy, "deploy_output_type", "State output; must include ZMQ", choices=("zmq", "all"))
    for name, description in (
        ("checkpoint", "Checkpoint prefix"), ("obs_config", "Observation configuration"),
        ("planner", "Planner model"), ("motion_data", "Motion data path"),
        ("motor_kp_scale", "Hardware motor Kp scales"), ("motor_kd_scale", "Hardware motor Kd scales"),
    ):
        option(deploy, "deploy_" + name, description + "; empty uses deploy.sh default")

    workstation = parser.add_argument_group("Workstation collection and previews")
    option(workstation, "task_prompt", "Language task description")
    option(workstation, "dataset_name", "Dataset name; empty generates a timestamp")
    option(workstation, "data_exporter_frequency", "Recording frequency in Hz")
    option(workstation, "record_wrist_cameras", "Record left and right wrist cameras")
    option(workstation, "text_to_speech", "Enable voice feedback")
    option(workstation, "pico_manager", "Enable the PICO mode manager")
    option(workstation, "pico_input_source", "Teleop input source", choices=("xrt", "isaac-teleop"))
    option(workstation, "pico_vis_vr3pt", "Show VR 3-point visualization")
    option(workstation, "pico_vis_smpl", "Show SMPL visualization")
    option(workstation, "pico_waist_tracking", "Enable waist tracking")
    option(workstation, "camera_viewer", "Start OpenCV camera viewer")
    option(workstation, "camera_web", "Start browser camera preview")
    option(workstation, "camera_web_host", "Browser preview HTTP bind address")
    option(workstation, "camera_web_port", "Browser preview HTTP port")

    hands = parser.add_argument_group("Hands (use the same backend on both machines)")
    option(hands, "hand_backend", "Hand backend", choices=("dex3", "inspire"))
    option(hands, "enable_hand_control", "Permit Inspire writes after trigger arming")
    option(hands, "inspire_port", "Inspire TCP port; endpoints must be reachable from workstation")
    for side in ("left", "right"):
        option(hands, f"inspire_{side}_ip", f"{side.title()} hand address")
        for setting in ("step", "min", "max", "hold_rate"):
            option(hands, f"inspire_{side}_thumb_{setting}", f"{side.title()} thumb {setting}")
    option(hands, "inspire_close_angles", "Little, ring, middle, index and thumb bend targets",
           type=int, nargs=5)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without checking dependencies or launching.",
    )
    parser.add_argument("--no-attach", action="store_true", help="Start tmux detached.")
    config = WirelessDataCollectionLaunchConfig(**vars(parser.parse_args(argv)))
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def validate_config(config: WirelessDataCollectionLaunchConfig):
    if config.nx == config.workstation:
        raise ValueError("Choose exactly one of --nx or --workstation")
    if config.sim:
        raise ValueError("For simulation use launch_data_collection.py --sim")
    for name in ("nx_host", "workstation_host"):
        if not getattr(config, name).strip():
            raise ValueError(f"--{name.replace('_', '-')} cannot be empty")
    if config.deploy_output_type not in {"zmq", "all"}:
        raise ValueError("Data collection requires --deploy-output-type zmq or all")
    for name in ("camera_port", "camera_web_port", "inspire_port"):
        if not 1 <= getattr(config, name) <= 65535:
            raise ValueError(f"--{name.replace('_', '-')} must be in 1..65535")
    if config.data_exporter_frequency <= 0:
        raise ValueError("--data-exporter-frequency must be positive")
    if not config.deploy_interface.strip():
        raise ValueError("--deploy-interface cannot be empty")
    if config.hand_backend not in {"dex3", "inspire"}:
        raise ValueError("--hand-backend must be dex3 or inspire")
    if config.enable_hand_control and config.hand_backend != "inspire":
        raise ValueError("--enable-hand-control requires --hand-backend inspire")
    if config.hand_backend == "inspire":
        if not config.pico_manager:
            raise ValueError("Inspire requires --pico-manager")
        for address in (config.inspire_left_ip, config.inspire_right_ip):
            ipaddress.IPv4Address(address)
        if config.inspire_left_ip == config.inspire_right_ip:
            raise ValueError("Inspire left and right IP addresses must differ")
        hand_launch_arguments(config)


def check_prerequisites(config: WirelessDataCollectionLaunchConfig, repo_root: Path):
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux is missing. Install with: sudo apt install tmux")
    if config.nx:
        for name in ("deploy.sh", "scripts/setup_env.sh"):
            if not (repo_root / "gear_sonic_deploy" / name).is_file():
                errors.append(f"Missing gear_sonic_deploy/{name}; set up C++ deployment on NX")
    else:
        for venv, installer in (("teleop", "install_pico.sh"), ("data_collection", "install_data_collection.sh")):
            for name in ("python", "activate"):
                if not (repo_root / f".venv_{venv}/bin" / name).exists():
                    errors.append(f".venv_{venv} is missing/incomplete. Run: bash install_scripts/{installer}")
                    break
        if config.hand_backend == "inspire" and not errors:
            check = subprocess.run(
                [str(repo_root / ".venv_teleop/bin/python"), "-c",
                 "from gear_sonic.utils.teleop.inspire_hand_controller import validate_inspire_sdk; "
                 "validate_inspire_sdk()"], capture_output=True, text=True,
            )
            if check.returncode:
                errors.append("Install inspire-rh56e2==0.3.0 into .venv_teleop: " + check.stderr)
    if errors:
        raise ValueError("Prerequisites not met:\n  - " + "\n  - ".join(errors))


def resolve_deploy_interface(interface: str) -> str:
    """Resolve 'real' without deploy.sh's fallback to a possible Wi-Fi interface."""
    if interface != "real":
        return interface
    result = subprocess.run(["ip", "-j", "-4", "addr", "show"], capture_output=True, text=True, check=True)
    candidates = {
        entry["ifname"] for entry in json.loads(result.stdout)
        for addr in entry.get("addr_info", [])
        if addr.get("local", "").startswith("192.168.123.")
    }
    if len(candidates) != 1:
        raise ValueError(
            "Expected one NX Ethernet interface in 192.168.123.x; "
            "check the robot cable/IP or specify --deploy-interface <Ethernet-interface>. "
            f"Found: {', '.join(sorted(candidates)) or 'none'}"
        )
    return candidates.pop()


def build_commands(config: WirelessDataCollectionLaunchConfig, repo_root: Path) -> list[tuple[str, str]]:
    """Build only local commands. No imports of runtime clients, sockets or hardware."""
    hand_deploy, hand_teleop, hand_exporter = hand_launch_arguments(config)
    if config.nx:
        argv = ["./deploy.sh", "--input-type", config.deploy_input_type,
                "--zmq-host", config.deploy_zmq_host or config.workstation_host,
                "--output-type", config.deploy_output_type]
        for name, flag in (("checkpoint", "--cp"), ("obs_config", "--obs-config"),
                           ("planner", "--planner"), ("motion_data", "--motion-data"),
                           ("motor_kp_scale", "--motor-kp-scale"), ("motor_kd_scale", "--motor-kd-scale")):
            value = getattr(config, "deploy_" + name)
            if value:
                argv += [flag, value]
        argv += hand_deploy + [config.deploy_interface]
        return [("deploy", f"cd {shlex.quote(str(repo_root / 'gear_sonic_deploy'))} && exec {shlex.join(argv)}")]

    def python_command(venv, script, argv):
        return (
            f"cd {shlex.quote(str(repo_root))} && "
            f"source {shlex.quote(str(repo_root / venv / 'bin/activate'))} && "
            "exec " + shlex.join(["python", "-u", "gear_sonic/scripts/" + script, *argv])
        )

    teleop = ["--input-source", config.pico_input_source, "--port", "5556",
              "--zmq_feedback_host", config.nx_host, "--zmq_feedback_port", "5557", *hand_teleop]
    for enabled, flag in ((config.pico_manager, "--manager"), (config.pico_vis_vr3pt, "--vis_vr3pt"),
                          (config.pico_vis_smpl, "--vis_smpl"), (config.pico_waist_tracking, "--waist_tracking")):
        if enabled:
            teleop.append(flag)
    camera = ["--camera-host", config.camera_host or config.nx_host, "--camera-port", str(config.camera_port)]
    exporter = ["--task-prompt", config.task_prompt,
                "--data-collection-frequency", str(config.data_exporter_frequency),
                "--state-zmq-host", config.nx_host, "--state-zmq-port", "5557",
                "--sonic-zmq-host", "127.0.0.1", "--sonic-zmq-port", "5556", *camera, *hand_exporter]
    if config.dataset_name:
        exporter += ["--dataset-name", config.dataset_name]
    if config.record_wrist_cameras:
        exporter.append("--record-wrist-cameras")
    if not config.text_to_speech:
        exporter.append("--no-text-to-speech")
    commands = [
        ("teleop", python_command(".venv_teleop", "pico_manager_thread_server.py", teleop)),
        ("exporter", python_command(".venv_data_collection", "run_data_exporter.py", exporter)),
    ]
    if config.camera_viewer:
        commands.append(("camera_viewer", python_command(".venv_data_collection", "run_camera_viewer.py", camera)))
    if config.camera_web:
        commands.append(("camera_web", python_command(
            ".venv_data_collection", "run_camera_web.py",
            [*camera, "--host", config.camera_web_host, "--port", str(config.camera_web_port)],
        )))
    return commands


def launch_session(session: str, commands: list[tuple[str, str]], repo_root: Path):
    """Launch foreground pane processes; keep failed panes visible for diagnosis."""
    def tmux(*args):
        return subprocess.run(["tmux", *args], capture_output=True, text=True, check=True).stdout.strip()

    first_pane = tmux("new-session", "-d", "-s", session, "-n", "data_collection",
                      "-c", str(repo_root), "-P", "-F", "#{pane_id}", "bash --noprofile --norc")
    tmux("set-option", "-t", session, "mouse", "on")
    tmux("set-option", "-w", "-t", first_pane, "remain-on-exit", "on")
    tmux("set-option", "-w", "-t", first_pane, "pane-border-status", "top")
    tmux("set-option", "-w", "-t", first_pane, "pane-border-format", "#{pane_title}")
    selected = first_pane
    for index, (name, command) in enumerate(commands):
        shell_command = shlex.join(["bash", "-lc", command])
        if index == 0:
            tmux("respawn-pane", "-k", "-t", first_pane, shell_command)
            pane = first_pane
        elif name == "camera_web":
            pane = tmux("new-window", "-d", "-t", session, "-n", name,
                        "-c", str(repo_root), "-P", "-F", "#{pane_id}", "bash --noprofile --norc")
            tmux("set-option", "-w", "-t", pane, "remain-on-exit", "on")
            tmux("respawn-pane", "-k", "-t", pane, shell_command)
        else:
            pane = tmux("split-window", "-d", "-h", "-t", first_pane,
                        "-c", str(repo_root), "-P", "-F", "#{pane_id}", shell_command)
            tmux("select-layout", "-t", first_pane, "tiled")
        tmux("select-pane", "-t", pane, "-T", name)
        if name == "exporter":
            selected = pane
    tmux("select-window", "-t", selected)
    tmux("select-pane", "-t", selected)


def main(config: WirelessDataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parents[2]
    validate_config(config)
    role = "nx" if config.nx else "workstation"
    session = "sonic_data_collection_" + role
    if not config.dry_run:
        check_prerequisites(config, repo_root)
        existing = subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True)
        if existing.returncode == 0:
            raise ValueError(f"Session '{session}' already exists; reattach with: tmux attach -t {session}")
        if config.nx:
            config.deploy_interface = resolve_deploy_interface(config.deploy_interface)
    commands = build_commands(config, repo_root)
    print(f"SONIC wireless data collection — {role}")
    print(f"  Teleop: {config.deploy_zmq_host or config.workstation_host}:5556 -> NX deploy")
    print(f"  Robot state: {config.nx_host}:5557 -> workstation teleop + exporter")
    print(f"  Camera: {config.camera_host or config.nx_host}:{config.camera_port} -> workstation")
    if config.workstation:
        print(f"  PICO PC IP: {config.workstation_host}")
        if config.camera_web:
            host = config.workstation_host if config.camera_web_host == "0.0.0.0" else config.camera_web_host
            print(f"  Browser preview: http://{host}:{config.camera_web_port}")
        if config.hand_backend == "inspire":
            print("  Inspire requires workstation routes to both hand IPs; configure routing separately.")
    for name, command in commands:
        print(f"\n[{name}]\n{command}")
    if config.dry_run:
        return
    launch_session(session, commands, repo_root)
    print(f"\nCreated tmux session: {session}")
    if config.nx:
        print("Complete deploy.sh's confirmation in the NX deploy pane, then wait for 'Init done'.")
        print("Keep the camera server running on NX and launch --workstation on the workstation.")
    print("Ctrl+b, arrows: switch panes; Ctrl+b, n/p: switch windows; Ctrl+b, d: detach.")
    print(f"Reattach: tmux attach -t {session}")
    print(f"Stop this machine's session: tmux kill-session -t {session}")
    print("NX and workstation sessions are independent; stopping one does not stop the other.")
    if not config.no_attach:
        try:
            subprocess.run(["tmux", "attach", "-t", session], check=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main(parse_args())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        sys.exit(1)
