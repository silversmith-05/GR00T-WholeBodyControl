"""Wired G1 Inspire client. observe never creates an action publisher or enables writes."""

import argparse
import json
import logging
from pathlib import Path
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.inspire_bridge import (
    ActionTimeline,
    HAND_FEEDBACK_STOP_SECONDS,
    PolicyClient,
    PolicyWorker,
    SensorCache,
    array,
    build_observation,
    pack_action,
    validate_modality,
    validate_sonic_config,
)
from gear_sonic.utils.teleop.inspire_hand_controller import InspireHandController
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

LOG = logging.getLogger("inspire_inference")


class LiveSensors:
    def __init__(self, config):
        self.cache = SensorCache()
        self.stop = threading.Event()
        self.threads = [
            threading.Thread(target=fn, args=(config,), daemon=True) for fn in (self._body, self._camera)
        ]
        for thread in self.threads:
            thread.start()

    def _body(self, config):
        sub = context = cfg = None
        try:
            sub = ZMQStateSubscriber(host=config.state_host, port=config.state_port)
            context = zmq.Context()
            cfg = context.socket(zmq.SUB)
            cfg.setsockopt_string(zmq.SUBSCRIBE, "robot_config")
            cfg.setsockopt(zmq.CONFLATE, 1)
            cfg.connect(f"tcp://{config.state_host}:{config.state_port}")
            while not self.stop.is_set():
                message = sub.get_msg()
                if message is not None:
                    self.cache.update_body(message, time.monotonic())
                if cfg.poll(0):
                    data = msgpack.unpackb(cfg.recv()[len(b"robot_config") :], raw=False)
                    with self.cache.lock:
                        self.cache.config = data
                self.stop.wait(0.005)
        except Exception as exc:
            self.cache.error = f"Body receiver failed: {exc}"
        finally:
            if sub:
                sub.close()
            if cfg is not None:
                cfg.close(linger=0)
            if context:
                context.term()

    def _camera(self, config):
        # Same decoder as data collection; preserve its RGB/encoding behavior.
        camera = None
        try:
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

            camera = ComposedCameraClientSensor(config.camera_host, config.camera_port)
            while not self.stop.is_set():
                message = camera.read()
                if message is not None:
                    self.cache.update_camera(message, time.monotonic())
                self.stop.wait(0.005)
        except Exception as exc:
            self.cache.error = f"Camera receiver failed: {exc}"
        finally:
            if camera:
                camera.close()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)


class EventLog:
    def __init__(self, directory):
        self.stream = (directory / "events.jsonl").open("x")

    def emit(self, kind, **fields):
        row = dict(event=kind, time=time.time(), monotonic=time.monotonic(), **fields)
        self.stream.write(json.dumps(row, allow_nan=False, default=lambda x: x.tolist()) + "\n")
        self.stream.flush()
        if kind != "action":
            LOG.info("%s %s", kind, fields)

    def close(self):
        self.stream.close()


class Runner:
    """Main-thread control state machine; inference and I/O never block 50 Hz."""

    def __init__(self, config, hands, sensors, worker, publisher, events):
        self.config, self.hands, self.sensors = config, hands, sensors
        self.worker, self.publisher, self.events = worker, publisher, events
        self.robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        self.timeline = ActionTimeline()
        self.phase = "OBSERVE" if config.mode == "observe" else "DISARMED"
        self.cpp_started = False
        self.cpp_start_time = None
        self.fault_counts = None
        self.initialized = False
        self.blend = None
        self.request_id = self.sequence = 0
        self.last_request = -1e9
        self.last_status = -1e9
        self.latencies = []
        self.slow_predictions = 0
        self.loop_intervals = []
        self.late_loop_count = 0
        self.last_loop = None
        self.stop_reason = None
        self.arm_started = None
        self.initial_token = None
        self.control_commands = 0
        self.task_success = None
        if config.mode == "real":
            initial = json.loads(config.initial_token.read_text())
            self.initial_token = array(initial["token"], (64,), "initial token")
            self.events.emit("initial_pose_source", source=initial["source"])

    def invalidate(self):
        self.timeline.invalidate()
        self.hands.cancel()
        self.blend = None
        self.last_request = -1e9
        self.slow_predictions = 0

    def command(self, *, start=False, stop=False, planner=False):
        if self.config.mode != "real":
            raise RuntimeError("Observe mode cannot send C++ commands")
        message = build_command_message(start=start, stop=stop, planner=planner)
        # Small bounded repetition mitigates PUB slow joining. No pose is sent
        # concurrently from this main-thread-owned socket.
        for _ in range(3):
            self.publisher.send(message)
            time.sleep(0.01)
        self.control_commands += 1
        self.events.emit("cpp_command", start=start, stop=stop, planner=planner)

    def fault(self, reason):
        if self.phase == "FAULT":
            return
        self.stop_reason = str(reason)
        self.invalidate()
        self.phase = "FAULT"
        self.events.emit("fault", reason=self.stop_reason, recovery="Restart deployment; no automatic rearm")
        if self.cpp_started:
            self.command(stop=True)
            self.cpp_started = False

    def observation(self, now):
        hands = self.hands.snapshot()["hands"]
        body, camera, observed = self.sensors.cache.snapshot(hands)
        obs = build_observation(body, camera, hands, self.robot_model, self.config.prompt)
        return obs, observed, hands, body

    def key(self, key, now):
        if self.phase == "FAULT":
            return
        if key.startswith("prompt:"):
            prompt = key[len("prompt:") :].strip()
            if prompt:
                self.config.prompt = prompt
                self.invalidate()
                self.phase = "OBSERVE" if self.config.mode == "observe" else "PAUSED"
                self.events.emit("prompt_changed", prompt=prompt, manual_resume=True)
            return
        if self.config.mode == "observe":
            self.events.emit("key_ignored", key=key, reason="observe mode has no hardware outputs")
            return
        if key in ("o", "O", "q") or (key == "k" and self.cpp_started):
            self.fault("Operator stop: C++ exits and enters its damping shutdown")
        elif key in ("s", "f"):
            self.task_success = key == "s"
            self.events.emit("trial_result", success=self.task_success, source="operator")
        elif key == "k":
            # g1_debug is only published by C++ after operator start. Requiring
            # body feedback here would deadlock startup; require its config.
            validate_sonic_config(self.sensors.cache.config)
            self.command(start=True, planner=True)
            self.cpp_started = True
            self.cpp_start_time = now
            self.phase = "PLANNER"
            self.events.emit("planner_started", next="Wait for fresh feedback, then press i")
        elif key == "i":
            if not self.cpp_started:
                raise RuntimeError("Press k to start C++ before initializing")
            _, _, _, body = self.observation(now)
            current = array(body.get("token_state"), (64,), "fresh C++ token_state")
            self.invalidate()
            self.initialized = False
            self.command(start=True, planner=False)
            self.blend = (time.monotonic(), current.copy())
            self.phase = "INITIALIZING"
            self.events.emit("initializing", seconds=self.config.initial_pose_blend_duration)
        elif key == "p":
            if self.phase in ("RUNNING", "ARMING"):
                self.invalidate()
                self.phase = "PAUSED"
                self.events.emit(
                    "paused", warning="C++ remains running; pause does not guarantee stillness. k stops C++."
                )
            else:
                if not self.cpp_started or not self.initialized:
                    raise RuntimeError("Initialize with k then i before enabling policy")
                _, _, hands, _ = self.observation(now)
                self.invalidate()
                thumbs = self.hands.begin_policy_session()
                self.fault_counts = [hand["fault_count"] for hand in hands]
                self.arm_started = now
                self.phase = "ARMING"
                self.events.emit(
                    "armed", thumb_targets=thumbs, thumb_measured=[hand["angle"][5] for hand in hands]
                )

    def tick(self, now):
        if self.last_loop is not None:
            self.loop_intervals.append(now - self.last_loop)
        if self.last_loop is not None and now - self.last_loop > 0.03:
            self.late_loop_count += 1
        self.last_loop = now
        if self.phase == "FAULT":
            return
        status = self.hands.snapshot()["hands"]
        if now - self.last_status >= 1:
            self.last_status = now
            self.events.emit(
                "status",
                phase=self.phase,
                thumbs=[
                    dict(
                        side=side,
                        measured=h["angle"][5],
                        target=h["thumb_rotation"],
                        feedback_valid=h["angle_valid"],
                        target_valid=h["thumb_target_valid"],
                        write_status=h["write_status"],
                        fault_count=h["fault_count"],
                        angle=h["angle"],
                        grasp_target=h.get("target"),
                        policy_sequence=h.get("policy_sequence"),
                        command_id=h.get("command_id"),
                        write_id=h.get("write_id"),
                        write_current=h.get("write_current"),
                        error=h.get("error"),
                    )
                    for side, h in zip(("left", "right"), status)
                ],
                late_loop_count=self.late_loop_count,
            )
        try:
            obs, observed, status, body = self.observation(now)
        except Exception as exc:
            # C++ initialization needs time after k before publishing g1_debug.
            grace = self.phase == "PLANNER" and self.sensors.cache.body is None and now - self.cpp_start_time < 10
            if self.cpp_started and not grace:
                self.fault(exc)
            elif now - self.last_request >= 2:
                self.events.emit("waiting_for_sensors", reason=str(exc))
                self.last_request = now
            return
        if self.phase in ("ARMING", "RUNNING"):
            if [h["fault_count"] for h in status] != self.fault_counts:
                self.events.emit("hand_fault", previous_fault_counts=self.fault_counts, hands=status)
                self.fault("Inspire write/connection fault; no retry")
                return
        result = self.worker.take()
        if result is not None:
            try:
                if self.timeline.accept(result, now):
                    latency = result.completed - result.observed
                    self.latencies.append(latency)
                    self.slow_predictions = (
                        self.slow_predictions + 1 if latency > self.config.execution_horizon / 50 else 0
                    )
                    self.events.emit(
                        "prediction",
                        request_id=result.request_id,
                        latency_ms=1000 * latency,
                        compute_ms=1000 * (result.completed - result.started),
                        state_ranges={k: [float(v.min()), float(v.max())] for k, v in obs["state"].items()},
                        motion_token_shape=[1, 40, 64],
                        hand_shape=[1, 40, 2],
                    )
                    if self.slow_predictions >= 5:
                        self.fault(
                            "Five consecutive predictions exceeded the update budget; "
                            "inspect performance before a new trial"
                        )
                        return
                    if self.phase == "ARMING":
                        self.phase = "RUNNING"
            except Exception as exc:
                self.fault(exc)
                return
        if self.phase == "INITIALIZING":
            started, initial = self.blend
            fraction = min(1.0, (now - started) / self.config.initial_pose_blend_duration)
            token = (1 - fraction) * initial + fraction * self.initial_token
            self.publisher.send(pack_action(token, self.sequence))
            self.sequence += 1
            if fraction >= 1:
                self.blend = None
                self.initialized = True
                self.phase = "PAUSED"
                self.events.emit("initialized", next="Check stance and thumbs, press p to enable policy")
            return
        if self.phase in ("OBSERVE", "ARMING", "RUNNING"):
            if now - self.last_request >= self.config.execution_horizon / 50:
                if self.worker.submit(self.timeline.epoch, self.request_id, obs, observed):
                    self.request_id += 1
                    self.last_request = now
            if self.phase == "ARMING" and now - self.arm_started >= 0.8:
                self.fault("No usable first prediction within 800 ms")
                return
            try:
                step = self.timeline.step(now)
                if step is not None:
                    if self.phase == "RUNNING":
                        self.hands.update_policy(
                            step["hand"],
                            sequence=self.sequence,
                            sample_monotonic=now,
                            valid_until=step["valid_until"],
                        )
                        self.publisher.send(pack_action(step["token"], self.sequence))
                    self.events.emit(
                        "action",
                        sequence=self.sequence,
                        mode=self.phase,
                        request_id=step["request_id"],
                        action_index=step["action_index"],
                        token=step["token"],
                        raw_hand=step["raw_hand"],
                        hand=step["hand"],
                        hardware_outputs=self.phase == "RUNNING",
                    )
                    self.sequence += 1
            except Exception as exc:
                self.fault(exc)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("observe", "real"), default="observe")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=5550)
    p.add_argument("--state-host", default="127.0.0.1")
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--action-host", default="127.0.0.1")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--camera-host", default="192.168.123.164")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--keyboard-host", default="127.0.0.1")
    p.add_argument("--keyboard-port", type=int, default=5580)
    p.add_argument("--inspire-left-ip", default="192.168.123.211")
    p.add_argument("--inspire-right-ip", default="192.168.123.210")
    p.add_argument("--inspire-port", type=int, default=6000)
    p.add_argument("--prompt", default="pick up the ball")
    p.add_argument("--initial-token", type=Path)
    p.add_argument("--initial-pose-blend-duration", type=float, default=1.0)
    p.add_argument("--execution-horizon", type=int, default=16)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--duration", type=float, default=0, help="Bounded read-only session seconds; 0 runs until stopped"
    )
    return p


def main(args=None):
    c = parser().parse_args(args)
    if (
        not 1 <= c.execution_horizon <= 40
        or c.initial_pose_blend_duration <= 0
        or c.duration < 0
        or (c.mode == "real" and (c.initial_token is None or c.duration))
    ):
        raise ValueError("Invalid horizon/blend duration; real requires initial token and no duration limit")
    c.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    events = EventLog(c.output_dir)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    client = PolicyClient(c.policy_host, c.policy_port, timeout_ms=1000)
    try:
        client.call("ping")
        validate_modality(client.call("get_modality_config"))
    finally:
        client.close()
    hands = InspireHandController(
        left_host=c.inspire_left_ip, right_host=c.inspire_right_ip, port=c.inspire_port,
        enabled=c.mode == "real", feedback_timeout=HAND_FEEDBACK_STOP_SECONDS,
    )
    sensors = worker = keyboard = context = runner = None
    started = time.monotonic()
    try:
        hands.start()
        sensors = LiveSensors(c)
        worker = PolicyWorker(c.policy_host, c.policy_port)
        publisher = None
        if c.mode == "real":
            context = zmq.Context()
            publisher = context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.bind(f"tcp://{c.action_host}:{c.action_port}")
        keyboard = ZMQKeyboardSubscriber(host=c.keyboard_host, port=c.keyboard_port)
        runner = Runner(c, hands, sensors, worker, publisher, events)
        events.emit(
            "started",
            mode=c.mode,
            hardware_enabled=False,
            keys="k=start planner/stop C++; i=initialize; p=enable/pause; o=stop; t TEXT=prompt",
        )
        while not stop.is_set():
            now = time.monotonic()
            key = keyboard.read_msg()
            if key:
                try:
                    runner.key(key, now)
                except Exception as exc:
                    # Invalid startup/keyboard transitions do not enable output.
                    events.emit("key_refused", key=key, reason=str(exc))
                    if runner.cpp_started and runner.phase in ("RUNNING", "INITIALIZING"):
                        runner.fault(exc)
            runner.tick(time.monotonic())
            if runner.phase == "FAULT" or (c.duration and now - started >= c.duration):
                break
            stop.wait(max(0, 0.02 - (time.monotonic() - now)))
    finally:
        if runner and runner.cpp_started:
            runner.fault("Client shutdown")
        hands.close()
        if worker:
            worker.close()
        if sensors:
            sensors.close()
        if keyboard:
            keyboard.close()
        if context:
            publisher.close(linger=0)
            context.term()
        if runner:
            latencies = runner.latencies
            summary = dict(
                mode=c.mode,
                predictions=len(latencies),
                latency_p95_ms=float(np.percentile(latencies, 95) * 1000) if latencies else None,
                target_budget_ms=c.execution_horizon * 20,
                late_loop_count=runner.late_loop_count,
                loop_interval_p95_ms=float(np.percentile(runner.loop_intervals, 95) * 1000)
                if runner.loop_intervals
                else None,
                loop_mean_hz=float(1 / np.mean(runner.loop_intervals)) if runner.loop_intervals else None,
                fault=runner.stop_reason,
                sensor_error=runner.sensors.cache.error,
                task_success=runner.task_success,
                hardware_commands_sent=(runner.sequence > 0 or runner.control_commands > 0) and c.mode == "real",
            )
            summary["passed"] = bool(latencies and not runner.stop_reason)
            (c.output_dir / "session_report.json").write_text(json.dumps(summary, indent=2) + "\n")
        events.close()
    if runner and runner.stop_reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
