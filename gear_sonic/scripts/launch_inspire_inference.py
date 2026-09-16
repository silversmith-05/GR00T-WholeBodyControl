"""Independent wired deployment launcher; check is offline, observe has no robot outputs."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GR00T = Path("/home/pku/workspace/Isaac-GR00T")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "observe", "real", "keyboard"))
    p.add_argument("--gr00t-repo", type=Path, default=DEFAULT_GR00T)
    p.add_argument("--model-path", type=Path, default=DEFAULT_GR00T / "models/inspire/checkpoint-10000")
    p.add_argument("--dataset-root", type=Path, default=REPO / "processed_datasets/pick_up_ball_inspire_v1")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prompt", default="pick up the ball")
    p.add_argument("--cuda-visible-devices", default="auto", help="Default: select local A100 UUID")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=5550)
    p.add_argument("--state-host", default="127.0.0.1")
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--action-host", default="127.0.0.1")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--keyboard-host", default="127.0.0.1")
    p.add_argument("--keyboard-port", type=int, default=5580)
    p.add_argument("--camera-host", default="192.168.123.164")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--inspire-left-ip", default="192.168.123.211")
    p.add_argument("--inspire-right-ip", default="192.168.123.210")
    p.add_argument("--inspire-port", type=int, default=6000)
    p.add_argument("--initial-episode", type=int, default=0)
    p.add_argument("--initial-frame", type=int, default=0)
    p.add_argument("--initial-pose-blend-duration", type=float, default=1.0)
    p.add_argument("--execution-horizon", type=int, default=16)
    p.add_argument("--deploy-interface", default="real")
    p.add_argument("--duration", type=float, default=0, help="Observe mode seconds; 0 runs until interrupted")
    p.add_argument(
        "--gpu-check", action="store_true", help="check mode: also run recorded-observation GPU inference"
    )
    return p


def check_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a fresh output directory: {path}")


def port_free(host, port):
    with socket.socket() as probe:
        probe.bind((host, port))


def choose_gpu(value):
    if value != "auto":
        return value
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        name, uuid = line.split(",", 1)
        if "A100" in name:
            return uuid.strip()
    raise RuntimeError("No A100 found; select the intended GPU with --cuda-visible-devices")


def resolve_interface(value):
    if value != "real":
        return value
    info = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr", "show"], text=True))
    names = [
        r["ifname"]
        for r in info
        if any(a.get("local", "").startswith("192.168.123.") for a in r.get("addr_info", []))
    ]
    if len(names) != 1:
        raise ValueError("Need one wired 192.168.123.x interface, or explicit --deploy-interface")
    return names[0]


def verify_command(c, directory):
    return [
        str(c.gr00t_repo / ".venv/bin/python"),
        str(REPO / "gear_sonic/scripts/verify_inspire_checkpoint.py"),
        "--gr00t-repo",
        str(c.gr00t_repo),
        "--model-path",
        str(c.model_path),
        "--dataset-root",
        str(c.dataset_root),
        "--output-dir",
        str(directory),
        "--initial-episode",
        str(c.initial_episode),
        "--initial-frame",
        str(c.initial_frame),
    ]


def commands(c, output):
    python = str(REPO / ".venv_teleop/bin/python")
    server = [
        str(c.gr00t_repo / ".venv/bin/python"),
        str(c.gr00t_repo / "gr00t/eval/run_gr00t_server.py"),
        "--model-path",
        str(c.model_path),
        "--embodiment-tag",
        "new_embodiment",
        "--device",
        "cuda:0",
        "--host",
        c.policy_host,
        "--port",
        str(c.policy_port),
    ]
    client = [
        python,
        str(REPO / "gear_sonic/scripts/run_inspire_inference.py"),
        "--mode",
        c.mode,
        "--output-dir",
        str(output / "client"),
        "--initial-token",
        str(output / "initial_token.json"),
        "--prompt",
        c.prompt,
    ]
    for name in (
        "policy_host",
        "policy_port",
        "state_host",
        "state_port",
        "action_host",
        "action_port",
        "keyboard_host",
        "keyboard_port",
        "camera_host",
        "camera_port",
        "inspire_left_ip",
        "inspire_right_ip",
        "inspire_port",
        "execution_horizon",
        "initial_pose_blend_duration",
        "duration",
    ):
        client += ["--" + name.replace("_", "-"), str(getattr(c, name))]
    keyboard = [
        python,
        str(Path(__file__).resolve()),
        "keyboard",
        "--keyboard-host",
        c.keyboard_host,
        "--keyboard-port",
        str(c.keyboard_port),
    ]
    deploy = [
        str(REPO / "gear_sonic_deploy/deploy.sh"),
        "--input-type",
        "zmq_manager",
        "--zmq-host",
        c.action_host,
        "--zmq-port",
        str(c.action_port),
        "--zmq-out-port",
        str(c.state_port),
        "--output-type",
        "zmq",
        "--disable-dex3-hands",
        "--cp",
        "policy/release/model",
        "--obs-config",
        "policy/release/observation_config.yaml",
        c.deploy_interface,
    ]
    return dict(server=server, client=client, keyboard=keyboard, deploy=deploy)


def keyboard(c):
    import zmq

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://{c.keyboard_host}:{c.keyboard_port}")
    print(
        "Enter + return: k start/STOP C++; i initialize; p enable/pause; o stop; "
        "t TEXT change task; s/f trial success/failure"
    )
    print("Pause does not guarantee stillness. k/o stop exits C++ and requires restart.")
    try:
        while True:
            key = input().strip()
            pub.send_string("prompt:" + key[2:] if key.startswith("t ") else key)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        pub.close(linger=0)
        context.term()


def start_window(session, name, command, output, *, cwd=REPO, new=False, env=None):
    # shlex.join, not string interpolation, preserves arbitrary prompts/paths.
    argv = (["env", *[f"{k}={v}" for k, v in env.items()]] if env else []) + command
    script = "set -o pipefail\n" + shlex.join(argv) + " 2>&1 | tee " + shlex.quote(str(output / f"{name}.log"))
    script += '\nstatus=${PIPESTATUS[0]}\necho "Process exit: $status"\nexec bash\n'
    target = session if new else session + ":"
    tool = [
        "tmux",
        "new-session" if new else "new-window",
        "-d",
        "-s" if new else "-t",
        target,
        "-n",
        name,
        "-c",
        str(cwd),
        shlex.join(["bash", "-c", script]),
    ]
    subprocess.run(tool, check=True)


def main(args=None):
    c = parser().parse_args(args)
    if c.mode == "keyboard":
        keyboard(c)
        return
    if (
        not c.prompt.strip()
        or not 1 <= c.execution_horizon <= 40
        or c.duration < 0
        or c.initial_episode < 0
        or c.initial_frame < 0
        or c.initial_pose_blend_duration <= 0
        or (c.mode == "real" and c.duration)
    ):
        raise ValueError("Invalid prompt/horizon/initialization/duration")
    for name in ("policy_host", "state_host", "action_host", "keyboard_host"):
        if getattr(c, name) not in ("127.0.0.1", "localhost"):
            raise ValueError(f"Wired all-local topology requires loopback --{name.replace('_', '-')}")
    ports = [c.policy_port, c.action_port, c.state_port, c.keyboard_port]
    if len(set(ports)) != 4 or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("Local process ports must be distinct and within 1..65535")
    output = (
        c.output_dir or REPO / "outputs" / ("inspire_deploy_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    check_output(output)
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.teleop.inspire_hand_controller import validate_inspire_sdk

    validate_inspire_sdk()
    instantiate_g1_robot_model()
    for path in (
        c.gr00t_repo / ".venv/bin/python",
        REPO / "gear_sonic_deploy/deploy.sh",
        *[
            REPO / "gear_sonic_deploy/policy/release" / n
            for n in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")
        ],
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if c.mode != "check":
        if not shutil.which("tmux"):
            raise RuntimeError("tmux is required")
        for host, port in [(c.policy_host, c.policy_port), (c.keyboard_host, c.keyboard_port)]:
            port_free(host, port)
        if c.mode == "real":
            port_free(c.action_host, c.action_port)
            port_free(c.state_host, c.state_port)
            c.deploy_interface = resolve_interface(c.deploy_interface)
    env = dict(os.environ, MPLBACKEND="Agg", NO_ALBUMENTATIONS_UPDATE="1")
    if c.mode != "check" or c.gpu_check:
        env["CUDA_VISIBLE_DEVICES"] = choose_gpu(c.cuda_visible_devices)
    output.mkdir(parents=True)
    verification = verify_command(c, output)
    if c.mode != "check":
        verification.append("--export-observation")
    if c.gpu_check:
        verification.append("--gpu")
    with (output / "check.log").open("x") as log:
        subprocess.run(verification, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO)
    cmd = commands(c, output)
    manifest = dict(
        mode=c.mode,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(c).items()},
        commands=cmd,
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        sonic_sha256={
            name: hashlib.sha256((REPO / "gear_sonic_deploy/policy/release" / name).read_bytes()).hexdigest()
            for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")
        },
    )
    (output / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    saved_config = output / "configs"
    saved_config.mkdir()
    for name in ("config.json", "processor_config.json", "embodiment_id.json", "statistics.json"):
        shutil.copy2(c.model_path / name, saved_config / name)
    shutil.copy2(REPO / "gear_sonic_deploy/policy/release/observation_config.yaml", saved_config)
    # Save the actual unified hand presets used by the driver.
    from gear_sonic.utils.teleop.inspire_hand_controller import ANGLES, HAND_FORCE

    (saved_config / "hand.json").write_text(
        json.dumps(
            dict(
                open=list(ANGLES[0][:5]),
                closed=list(ANGLES[1][:5]),
                speed=200,
                force=HAND_FORCE,
                thumb="Fresh measured angle on each manual enable; then hold",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Checks passed. Artifacts: {output}")
    if c.mode == "check":
        return
    session = "inspire_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    start_window(
        session,
        "server",
        cmd["server"],
        output,
        new=True,
        env={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "MPLBACKEND", "NO_ALBUMENTATIONS_UPDATE")},
    )
    from gear_sonic.utils.inference.inspire_bridge import PolicyClient

    probe = PolicyClient(c.policy_host, c.policy_port, timeout_ms=1000)
    try:
        deadline = time.monotonic() + 180
        while True:
            try:
                probe.call("ping")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Server did not become ready; see {output / 'server.log'}")
                time.sleep(1)
    finally:
        probe.close()
    # Check the same lightweight RPC path used by control, before any hardware pane.
    with (output / "warmup.log").open("x") as log:
        subprocess.run(
            [
                str(REPO / ".venv_teleop/bin/python"),
                str(REPO / "gear_sonic/scripts/check_inspire_rpc.py"),
                "--recorded-dir",
                str(output),
                "--output-dir",
                str(output / "warmup"),
                "--host",
                c.policy_host,
                "--port",
                str(c.policy_port),
                "--budget-ms",
                str(c.execution_horizon * 20),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=REPO,
        )
    performance = json.loads((output / "warmup/rpc_report.json").read_text())
    if c.mode == "real" and not performance["meets_budget"]:
        raise RuntimeError(
            "Inference exceeds the update budget; hardware panes not started. "
            f"See {output / 'warmup/rpc_report.json'}"
        )
    start_window(session, "keyboard", cmd["keyboard"], output)
    start_window(session, "client", cmd["client"], output)
    if c.mode == "real":
        start_window(
            session,
            "deploy",
            cmd["deploy"],
            output,
            cwd=REPO / "gear_sonic_deploy",
            env={"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
        )
    else:
        print("observe does not start or stop SONIC. It requires an existing g1_debug stream with Dex3 disabled.")
    print(f"tmux attach -t {session}\nLogs: {output}\nNo keyboard motion command is sent automatically.")


if __name__ == "__main__":
    main()
