"""Launch SONIC body teleoperation and Inspire hands in tmux, without VLA or recording.

No arguments: run both components locally against the real robot.
Use --component deploy/teleop for an NX/workstation split, --check for offline
dependency checks, or --dry-run to print commands without starting processes.
"""

import argparse
import ipaddress
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

from gear_sonic.scripts.launch_data_collection import (
    DataCollectionLaunchConfig,
    hand_launch_arguments,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("interface", nargs="?", default="real", help="real, robot network interface, or IP")
    parser.add_argument("--component", choices=("both", "deploy", "teleop"), default="both")
    parser.add_argument("--teleop-host", "--zmq-host", default="localhost",
                        help="Pico publisher host for C++ (default: localhost)")
    parser.add_argument("--state-host", default="localhost", help="C++ feedback host for Pico")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-out-port", type=int, default=5557)
    parser.add_argument("--pico-input-source", choices=("xrt", "isaac-teleop"), default="xrt")
    parser.add_argument("--read-only-hands", action="store_true", help="Read hand feedback without hand writes")
    defaults = DataCollectionLaunchConfig()
    for name in ("inspire_left_ip", "inspire_right_ip", "inspire_port",
                 *[f"inspire_{side}_thumb_{setting}" for side in ("left", "right")
                   for setting in ("step", "min", "max", "hold_rate")]):
        value = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value,
                            help=f"Default: {value}")
    parser.add_argument("--inspire-close-angles", type=int, nargs=5, default=defaults.inspire_close_angles)
    parser.add_argument("--cp", "--checkpoint", dest="checkpoint")
    for flag in ("obs-config", "planner", "motion-data"):
        parser.add_argument("--" + flag)
    for flag in ("motor-kp-scale", "motor-kd-scale"):
        parser.add_argument("--" + flag, action="append", default=[])
    parser.add_argument("--session", default="sonic_inspire_teleop")
    parser.add_argument("--no-attach", action="store_true", help="Leave tmux running without attaching")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands only; no dependency checks or device I/O")
    mode.add_argument("--check", action="store_true", help="Check local dependencies only; no device I/O")
    args = parser.parse_args(argv)
    try:
        if args.interface in {"sim", "lo", "lo0"} or args.interface.startswith("127."):
            raise ValueError("This entry controls real Inspire hands; simulation/loopback is not supported")
        if any(not 1 <= port <= 65535 for port in (args.zmq_port, args.zmq_out_port, args.inspire_port)):
            raise ValueError("Ports must be in 1..65535")
        if args.zmq_port == args.zmq_out_port:
            raise ValueError("Action and feedback ports must be different")
        for address in (args.inspire_left_ip, args.inspire_right_ip):
            ipaddress.IPv4Address(address)
        if args.inspire_left_ip == args.inspire_right_ip:
            raise ValueError("Left and right hands must have different IP addresses")
        if not args.session or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                   for c in args.session):
            raise ValueError("Session name must contain only letters, digits, underscores and hyphens")
        if args.component == "both" and any(host not in {"localhost", "127.0.0.1"}
                                            for host in (args.teleop_host, args.state_host)):
            raise ValueError("Use --component deploy/teleop when the other component is on another machine")
        # Reuse the same hand presets, bounds and argument validation as collection.
        hand_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def hand_arguments(args):
    values = {name: value for name, value in vars(args).items() if name.startswith("inspire_")}
    config = DataCollectionLaunchConfig(hand_backend="inspire",
                                       enable_hand_control=not args.read_only_hands, **values)
    deploy, teleop, _ = hand_launch_arguments(config)
    return deploy, teleop


def build_commands(args, root=REPO):
    deploy_hand, teleop_hand = hand_arguments(args)
    deploy = ["bash", str(root / "gear_sonic_deploy/deploy.sh"), "--input-type", "zmq_manager",
              "--output-type", "zmq", "--zmq-host", args.teleop_host,
              "--zmq-port", str(args.zmq_port), "--zmq-out-port", str(args.zmq_out_port), *deploy_hand]
    for name, flag in (("checkpoint", "--cp"), ("obs_config", "--obs-config"),
                       ("planner", "--planner"), ("motion_data", "--motion-data")):
        if getattr(args, name):
            deploy += [flag, getattr(args, name)]
    for name in ("motor_kp_scale", "motor_kd_scale"):
        for value in getattr(args, name):
            deploy += ["--" + name.replace("_", "-"), value]
    deploy.append(args.interface)
    teleop = [str(root / ".venv_teleop/bin/python"),
              str(root / "gear_sonic/scripts/pico_manager_thread_server.py"),
              "--manager", "--input-source", args.pico_input_source,
              "--port", str(args.zmq_port), "--zmq_feedback_host", args.state_host,
              "--zmq_feedback_port", str(args.zmq_out_port), *teleop_hand]
    return [(name, command) for name, command in (("deploy", deploy), ("teleop", teleop))
            if args.component in ("both", name)]


def check_prerequisites(args, root=REPO):
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux is missing (install with sudo apt install tmux)")
    if args.component in ("both", "deploy"):
        for relative in ("gear_sonic_deploy/deploy.sh", "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"):
            if not (root / relative).is_file():
                errors.append(f"Missing {root / relative}")
        binary = root / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        if binary.is_file() and b"--disable-dex3-hands" not in binary.read_bytes():
            errors.append("Rebuild g1_deploy_onnx_ref: this binary does not support --disable-dex3-hands")
    if args.component in ("both", "teleop"):
        python = root / ".venv_teleop/bin/python"
        if not python.is_file():
            errors.append(f"Missing {python}; set up .venv_teleop with install_scripts/install_pico.sh")
        else:
            source_module = "xrobotoolkit_sdk" if args.pico_input_source == "xrt" else "isaacteleop"
            probe = (
                "import importlib.util; "
                "from gear_sonic.utils.teleop.inspire_hand_controller import validate_inspire_sdk; "
                "validate_inspire_sdk(); "
                f"modules = ['torch', 'numpy', 'scipy', 'zmq', 'msgpack', '{source_module}']; "
                "missing = [name for name in modules if importlib.util.find_spec(name) is None]; "
                "assert not missing, 'Missing teleop modules: ' + ', '.join(missing)"
            )
            result = subprocess.run([str(python), "-c", probe], cwd=root, capture_output=True, text=True)
            if result.returncode:
                errors.append("Teleop environment check failed:\n" + result.stderr.strip())
                errors.append("Install inspire-rh56e2==0.3.0 and pymodbus==3.11.4 into .venv_teleop. "
                              "For a local SDK checkout: uv pip install --python .venv_teleop/bin/python /path/to/Inspire_RH56DFTP_sdk")
    if errors:
        raise RuntimeError("\n".join(errors))


def launch_session(session, commands, root=REPO):
    def tmux(*argv):
        return subprocess.run(["tmux", *argv], capture_output=True, text=True, check=True).stdout.strip()

    # Prepare all panes before starting either component; never replace an existing session.
    first = tmux("new-session", "-d", "-s", session, "-n", "teleop", "-c", str(root),
                 "-P", "-F", "#{pane_id}", "bash --noprofile --norc")
    tmux("set-option", "-t", session, "mouse", "on")
    tmux("set-option", "-w", "-t", first, "remain-on-exit", "on")
    tmux("set-option", "-w", "-t", first, "pane-border-status", "top")
    tmux("set-option", "-w", "-t", first, "pane-border-format", "#{pane_title}")
    panes = [first]
    for _ in commands[1:]:
        panes.append(tmux("split-window", "-d", "-h", "-t", first, "-c", str(root),
                          "-P", "-F", "#{pane_id}", "bash --noprofile --norc"))
    for pane, (name, command) in zip(panes, commands):
        tmux("select-pane", "-t", pane, "-T", name)
        tmux("respawn-pane", "-k", "-t", pane, shlex.join(["bash", "-c", "exec " + shlex.join(command)]))
    tmux("select-pane", "-t", first)


def main(argv=None):
    args = parse_args(argv)
    commands = build_commands(args)
    for name, command in commands:
        print(f"[{name}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    try:
        check_prerequisites(args)
        if args.check:
            print("Local dependency checks passed. Hardware connectivity and motion were not tested.")
            return
        exists = subprocess.run(["tmux", "has-session", "-t", "=" + args.session], capture_output=True)
        if exists.returncode == 0:
            raise RuntimeError(f"Session already exists. Attach with: tmux attach -t {args.session}")
        launch_session(args.session, commands)
        print(f"Session: {args.session}. Switch panes with Ctrl+b then an arrow key.")
        if args.component in ("both", "deploy"):
            print("The deploy pane retains deploy.sh's confirmation prompt. No start key is sent automatically.")
        print(f"Detach: Ctrl+b d. Reattach: tmux attach -t {args.session}")
        print("Detaching leaves control processes running. Use the existing robot stop controls before cleanup.")
        if not args.no_attach:
            subprocess.run(["tmux", "attach-session", "-t", args.session], check=True)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        print(f"ERROR: {detail.strip()}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
