"""Bounded live read-only probe: never starts SONIC, publishes actions, or enables hand writes."""

import json
import time

from gear_sonic.scripts.run_inspire_inference import LiveSensors, parser
from gear_sonic.utils.teleop.inspire_hand_controller import InspireHandController


def main():
    config = parser().parse_args()
    if config.mode != "observe":
        raise ValueError("The sensor probe only supports --mode observe")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "sensor_probe.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    hands = InspireHandController(
        left_host=config.inspire_left_ip,
        right_host=config.inspire_right_ip,
        port=config.inspire_port,
        enabled=False,
    )
    sensors = None
    passed, reason, samples = False, "No samples", 0
    try:
        hands.start()
        sensors = LiveSensors(config)
        deadline = time.monotonic() + (config.duration or 5)
        while time.monotonic() < deadline:
            snapshot = hands.snapshot()["hands"]
            try:
                sensors.cache.snapshot(snapshot, time.monotonic())
                passed, reason = True, "All required streams are fresh"
                samples += 1
            except Exception as exc:
                passed, reason = False, str(exc)
            time.sleep(0.02)
        snapshot = hands.snapshot()["hands"]
        with sensors.cache.lock:
            report = dict(
                passed=passed,
                reason=reason,
                complete_samples=samples,
                hardware_commands_sent=False,
                hand_writes_enabled=False,
                sensor_error=sensors.cache.error,
                camera_endpoint=f"tcp://{config.camera_host}:{config.camera_port}",
                body_received=sensors.cache.body is not None,
                camera_keys=list((sensors.cache.camera or {}).get("images", {})),
                camera_shapes={
                    key: list(value.shape) if value is not None else None
                    for key, value in (sensors.cache.camera or {}).get("images", {}).items()
                },
                camera_age_ms={
                    key: (time.monotonic() - arrived) * 1000 for key, arrived in sensors.cache.image_times.items()
                },
                robot_config=sensors.cache.config,
                hands=[
                    dict(
                        host=h["host"],
                        connected=h["connected"],
                        angle=h["angle"],
                        angle_valid=h["angle_valid"],
                        thumb_measured=h["angle"][5],
                        error=h["error"],
                    )
                    for h in snapshot
                ],
            )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        hands.close()
        if sensors:
            sensors.close()


if __name__ == "__main__":
    main()
