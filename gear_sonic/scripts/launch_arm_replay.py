"""Prepare recorded arms as G1 Encoder references and optionally launch SONIC.

--prepare-only, --check and --dry-run never initialize DDS or touch the robot.
All modes write a derived CSV/report under outputs/arm_replay; source recordings
are not modified. Normal launch retains deploy.sh's confirmation and requires
the operator to press ] after initialization to start policy control/replay.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
JOINTS = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
URDF = REPO / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"


def read_recording(directory):
    directory = Path(directory).resolve()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("format_version") != 1 or metadata.get("status") != "completed"
            or metadata.get("topic") != "rt/lowstate" or metadata.get("units", {}).get("q") != "rad"):
        raise ValueError("Expected a completed format-1 rt/lowstate recording with q in radians")
    mapping = {joint["name"]: joint["sdk_index"] for joint in metadata["joints"]}
    if len(metadata["joints"]) != 14 or mapping != dict(zip(JOINTS, range(15, 29))):
        raise ValueError("Recording must contain exactly the 14 arm joints at hardware indices 15..28")
    limits = {j.attrib["name"]: (float(j.find("limit").attrib["lower"]),
                                  float(j.find("limit").attrib["upper"]))
              for j in ET.parse(URDF).getroot().findall("joint") if j.attrib["name"] in JOINTS}
    times, positions = [], []
    first_ns = previous_ns = None
    max_gap = 0.0
    with (directory / "samples.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["received_monotonic_ns", "mode_pr", *[f"{name}_q" for name in JOINTS]]
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or any(
                name not in reader.fieldnames for name in required):
            raise ValueError("Recording CSV has missing or duplicate columns")
        for line, row in enumerate(reader, 2):
            stamp = int(row["received_monotonic_ns"])
            if stamp < 0 or (previous_ns is not None and stamp <= previous_ns):
                raise ValueError(f"Non-increasing monotonic timestamp at CSV row {line}")
            if int(row["mode_pr"]) != 0:
                raise ValueError(f"Expected PR-mode recording at CSV row {line}")
            if previous_ns is not None:
                max_gap = max(max_gap, (stamp - previous_ns) * 1e-9)
            if first_ns is None:
                first_ns = stamp
            previous_ns = stamp
            q = tuple(float(row[f"{name}_q"]) for name in JOINTS)
            for name, value in zip(JOINTS, q):
                lo, hi = limits[name]
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"Invalid/out-of-range {name} at CSV row {line}: {value}")
            times.append((stamp - first_ns) * 1e-9)
            positions.append(q)
    if len(times) < 2 or times[-1] <= 0:
        raise ValueError("Recording needs at least two distinct samples")
    if max_gap > 0.1:
        raise ValueError(f"Recording has a {max_gap:.3f}s gap; refusing to bridge missing motion")
    written = metadata.get("statistics", {}).get("frames_written")
    if written is not None and written != len(times):
        raise ValueError("CSV sample count differs from metadata frames_written")
    return times, positions, max_gap


def downsample(times, positions, hz):
    if not math.isfinite(hz) or not 10 <= hz <= 50:
        raise ValueError("Replay sample rate must be 10..50 Hz (SONIC control runs at 50 Hz)")
    duration = times[-1]
    targets = [i / hz for i in range(math.floor(duration * hz) + 1)]
    if duration - targets[-1] > 1e-9:
        targets.append(duration)
    else:
        targets[-1] = duration
    selected = []
    for time in targets:
        index = bisect.bisect_left(times, time)
        if index == len(times):
            index -= 1
        elif index and time - times[index - 1] <= times[index] - time:
            index -= 1
        selected.append(index)
    return targets, [positions[i] for i in selected], selected


def prepare(directory, output, hz=50):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents:
        raise ValueError("Derived output must be outside the source recording directory")
    times, q, max_gap = read_recording(directory)
    target_times, target_q, indices = downsample(times, q, hz)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "arms.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", *[f"{name}_q" for name in JOINTS]])
        writer.writerows((format(t, ".12g"), *[format(v, ".12g") for v in pose])
                         for t, pose in zip(target_times, target_q))
    report = {
        "source": str(directory), "source_samples": len(times), "prepared_samples": len(target_times),
        "duration_s": times[-1], "target_hz": hz,
        "selection": "nearest original sample at each target timestamp; final endpoint retained",
        "clock": "received_monotonic_ns relative to first sample; original duration preserved",
        "max_source_gap_s": max_gap,
        "max_selection_error_s": max(abs(t - times[i]) for t, i in zip(target_times, indices)),
        "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "replayed_fields": ["q"], "ignored_feedback_fields": ["dq", "tau_est"],
        "controller": "ARM_REPLAY_G1_ENCODER_V1",
        "reference_interpolation": "shape-preserving cubic Hermite; zero endpoint derivatives",
        "reference_velocity": "analytic derivative of interpolated q and reference blends",
        "encoder": {"mode": "g1", "mode_id": 0, "token_dimension": 64,
                    "future_frames": 10, "frame_spacing_s": 0.1, "lookahead_s": 0.9},
        "motor_commands": "all 29 joints from SONIC; no post-policy arm override",
        "hardware_indices": list(range(15, 29)),
        "settle_s": 5, "blend_in_s": 3, "blend_out_s": 3, "repeat": False,
        "after_completion": "arm reference returns to planner IDLE; SONIC continues until O",
        "joints": [{"name": name, "min_q": min(row[j] for row in q),
                    "max_q": max(row[j] for row in q),
                    "max_selected_step_rad": max(abs(b[j] - a[j]) for a, b in zip(target_q, target_q[1:]))}
                   for j, name in enumerate(JOINTS)],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return destination, report


def build_command(args, prepared):
    command = ["bash", str(REPO / "gear_sonic_deploy/deploy.sh"),
               "--input-type", "arm_replay", "--output-type", "zmq", "--disable-dex3-hands",
               "--arm-replay-file", str(prepared)]
    for name in ("motor_kp_scale", "motor_kd_scale"):
        for value in getattr(args, name):
            command += ["--" + name.replace("_", "-"), value]
    return [*command, args.interface]


def check_binary(prepared):
    binary = REPO / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    if not binary.is_file() or b"ARM_REPLAY_G1_ENCODER_V1" not in binary.read_bytes():
        raise ValueError("Build first: cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2")
    # Match deploy.sh -> scripts/setup_env.sh. ROS-enabled builds link FastRTPS
    # and can reject an inherited Cyclone RMW selection even before main().
    subprocess.run([str(binary), "--check-arm-replay", str(prepared),
                    str(REPO / "gear_sonic_deploy/policy/release/observation_config.yaml")], check=True,
                   env={**os.environ, "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("interface", nargs="?", default="eno2")
    parser.add_argument("--recording", type=Path, default=REPO / "recordings")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/arm_replay")
    parser.add_argument("--hz", type=float, default=50)
    for name in ("motor-kp-scale", "motor-kd-scale"):
        parser.add_argument("--" + name, action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Prepare CSV/report only; no process launch")
    mode.add_argument("--dry-run", action="store_true", help="Prepare CSV/report and print launch command only")
    mode.add_argument("--check", action="store_true", help="Prepare and validate CSV in C++; no device I/O")
    args = parser.parse_args(argv)
    try:
        prepared, report = prepare(args.recording, args.output, args.hz)
        print(f"Arms: {report['source_samples']} -> {report['prepared_samples']} samples at {args.hz:g} Hz; "
              f"duration {report['duration_s']:.6f} s", flush=True)
        print(f"Prepared: {prepared}\nReport: {prepared.parent / 'report.json'}", flush=True)
        print("Mode: G1 Encoder arm reference -> 64D token -> SONIC full-body control", flush=True)
        command = build_command(args, prepared)
        print(shlex.join(command), flush=True)
        if args.prepare_only or args.dry_run:
            return
        check_binary(prepared)
        if args.check:
            print("Offline checks passed. No robot connection or motion was attempted.")
            return
        print("Deployment confirmation comes next. C++ initialization moves to the default pose. "
              "After Init Done press ] for G1 Encoder arm tracking; O stops control. "
              "After replay, SONIC continues standing until stopped.", flush=True)
        subprocess.run(command, cwd=REPO, check=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
