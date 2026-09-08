"""Nonblocking Pico adapter; SDK owns all Modbus encoding and write safeguards.

One worker/connection per hand. No I/O runs under the mailbox lock. A release
baseline followed by a NEW press is required after every safety reset. Already
transmitted registers cannot be recalled; the SDK guard cancels subsequent writes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import fcntl
import logging
import math
from pathlib import Path
import threading
import time

LOG = logging.getLogger(__name__)
# Fixed five-finger targets. Lower bend values close further on RH56E2.
# Starting values for small-ball calibration, NOT a verified ball diameter/pose.
CLOSE_ANGLES = (250, 250, 250, 250, 300)
ANGLES = ((1000, 1000, 1000, 1000, 1000, 339), (*CLOSE_ANGLES, 339))
PRESETS = ("release", "close")
HAND_SCHEMA_VERSION = 4
PRESET_REVISION = 3
SDK_ANGLES = ((1000, 1000, 1000, 592, 720, 339), (1000, 1000, 1000, 1000, 1000, 339))
WRITE_STATUS = {"none": 0, "pending": 1, "confirmed": 2, "unconfirmed": 3,
                "failed": 4, "cancelled": 5}
ACTIVE_MODES = {1, 5}  # POSE, PLANNER_VR_3PT; frozen upper body keeps hands frozen.
THUMB_HOLD_DELAY = 0.6
THUMB_HOLD_INTERVAL = 0.1
THUMB_HOLD_MAX_LEAD = 10


def validate_close_angles(values):
    values = tuple(values)
    if len(values) != 5 or any(type(v) is not int or not 0 <= v < 1000 for v in values):
        raise ValueError("Five-finger close angles need five integers in 0..999 (little, ring, middle, index, thumb bend)")
    return values


def validate_inspire_sdk():
    """Shared by the teleop controller and the standalone hand tester."""
    from importlib.metadata import version
    from inspire_rh56e2.presets import RELEASE, CLOSE

    if version("inspire-rh56e2") != "0.3.0":
        raise RuntimeError("Inspire backend requires inspire-rh56e2==0.3.0")
    for i, preset in enumerate((RELEASE, CLOSE)):
        if (preset.name != PRESETS[i] or preset.angle != SDK_ANGLES[i]
                or preset.speed != (200,) * 6 or preset.force != (200,) * 6):
            raise RuntimeError("SDK built-in presets differ from the expected 0.3.0 definitions")


def hand_preset(target, thumb_rotation=None, close_angles=CLOSE_ANGLES):
    """Use SDK custom presets; never modify its installed built-in definitions."""
    from inspire_rh56e2.presets import HandPreset

    if target not in (0, 1):
        raise ValueError("Hand target must be 0=release or 1=close")
    bends = validate_close_angles(close_angles) if target else ANGLES[0][:5]
    angles = (*bends, 339 if thumb_rotation is None else thumb_rotation)
    return HandPreset(PRESETS[target], angles, force=200, speed=200)


@dataclass(frozen=True)
class ThumbRotationConfig:
    """Device scale, not degrees. Limits are software bounds, not calibration."""
    step: int = 10
    minimum: int = 0
    maximum: int = 1000
    hold_rate: float = 50.0

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.step, self.minimum, self.maximum))
                or not 1 <= self.step <= 1000 or not 0 <= self.minimum < self.maximum <= 1000):
            raise ValueError("Thumb step must be an integer in 1..1000; limits need 0 <= min < max <= 1000")
        if not math.isfinite(self.hold_rate) or not 1 <= self.hold_rate <= 50:
            raise ValueError("Thumb hold rate must be 1..50 device scale units/second")


class ThumbButtonGesture:
    """Short clicks on release, slow hold after a chord-disambiguation delay.

    A gesture runs from first face-button down until ALL face buttons are up.
    Per-hand session tokens prevent a reconnect from completing an old click.
    """
    MAPPING = {"x": (0, -1), "y": (0, 1), "a": (1, -1), "b": (1, 1)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.ready = False
        self.seen = set()
        self.blocked = False
        self.sessions = (None, None)
        self.started_at = 0.0
        self.long_press = False
        self.held = None

    def update(self, buttons, left_grip, sessions, *, now=None):
        now = time.monotonic() if now is None else now
        self.held = None
        if (buttons is None or len(buttons) != 4 or any(type(b) is not bool for b in buttons)
                or not math.isfinite(left_grip) or not 0 <= left_grip <= 1
                or all(s is None for s in sessions)):
            self.reset()
            return None
        pressed = {key for key, down in zip("abxy", buttons) if down}
        if not self.ready:
            # A key held at startup/reset is not a new click.
            self.ready = not pressed and left_grip <= 0.5
            return None
        if pressed:
            if not self.seen:
                self.sessions = sessions
                self.started_at = now
                self.long_press = False
            self.seen.update(pressed)
        self.blocked |= left_grip > 0.5 or len(self.seen) > 1
        if len(self.seen) == 1 and now - self.started_at >= THUMB_HOLD_DELAY:
            self.long_press = True
        if len(self.seen) == 1 and not self.blocked:
            side, direction = self.MAPPING[next(iter(self.seen))]
            if sessions[side] is None or sessions[side] != self.sessions[side]:
                self.blocked = True
            elif pressed and self.long_press:
                self.held = (side, direction, sessions[side])
        if pressed:
            return None
        event = None
        if len(self.seen) == 1 and not self.blocked and not self.long_press:
            side, direction = self.MAPPING[next(iter(self.seen))]
            if sessions[side] is not None and sessions[side] == self.sessions[side]:
                event = (side, direction, sessions[side])
        self.seen.clear()
        self.blocked = left_grip > 0.5
        return event


@dataclass(frozen=True)
class _HandCommand:
    epoch: int
    command_id: int
    target: int
    thumb_rotation: int


class _HoldReleased(Exception):
    """A hold ended before an angle write; force/speed may have been confirmed."""


def acquire_hand_lock(host, port, lock_dir="/tmp"):
    """Return an owned file handle; close it to release, never unlink the file."""
    path = Path(lock_dir) / f"sonic-inspire-{host}-{port}.lock"
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"Another driver owns {host}:{port}")
    return handle


class PicoHandBridge:
    """Use device sequence advancement, never loop arrival, as input freshness."""
    def __init__(self, controller):
        self.controller = controller
        self.stamp = None
        self.sample_monotonic = self.sample_time = 0.0
        self.mode = None

    def update(self, reader, left, right, mode, *, buttons=None, left_grip=0.0):
        sample = reader.get_latest() or {}
        stamp = sample.get("timestamp_ns", 0)
        if stamp and stamp != self.stamp:
            if self.stamp is not None and stamp < self.stamp:
                self.controller.cancel()  # Input source restarted with a new clock epoch.
            self.stamp = stamp
            self.sample_monotonic = sample.get("timestamp_monotonic", 0.0)
            self.sample_time = sample.get("timestamp_realtime", 0.0)
        if mode != self.mode:
            self.controller.cancel()
            self.mode = mode
        if "controller_data" in sample:
            ctrl = sample["controller_data"] or {}
            left = float(ctrl.get("left_trigger_value", float("nan")))
            right = float(ctrl.get("right_trigger_value", float("nan")))
            keys = ("right_primary_click", "right_secondary_click",
                    "left_primary_click", "left_secondary_click")
            values = [float(ctrl.get(key, float("nan"))) for key in keys]
            buttons = (tuple(v > 0.5 for v in values)
                       if all(math.isfinite(v) and 0 <= v <= 1 for v in values) else None)
            left_grip = float(ctrl.get("left_squeeze_value", float("nan")))
        buttons_valid = (buttons is not None and len(buttons) == 4
                         and all(type(b) is bool for b in buttons)
                         and math.isfinite(left_grip) and 0 <= left_grip <= 1)
        self.controller.update(left, right, sample_monotonic=self.sample_monotonic,
                               sample_time=self.sample_time,
                               active=mode in ACTIVE_MODES and not reader.disconnected and buttons_valid,
                               buttons=buttons, left_grip=left_grip)


class _Cancelled(Exception):
    pass


class _HandWorker:
    def __init__(self, config, enabled, factory, *, input_timeout, feedback_timeout,
                 reconnect_delay, poll_hz, thumb_config, close_angles):
        self.config, self.enabled, self.factory = config, enabled, factory
        self.input_timeout, self.feedback_timeout = input_timeout, feedback_timeout
        self.reconnect_delay, self.period = reconnect_delay, 1 / poll_hz
        self.thumb_config = thumb_config
        self.close_angles = validate_close_angles(close_angles)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"inspire-{config.host}")
        self.pending = None
        self.epoch = 0
        self.armed = False
        self.previous = None
        self.last_update = 0.0
        self.baseline_after = 0.0
        self.next_command_id = 0
        self.settings_ready = False
        self.hold_direction = 0
        self.hold_token = 0
        self.hold_next = self.hold_credit = 0.0
        self.s = dict(host=config.host, port=config.port, unit_id=config.unit_id, connected=False, input_valid=False, armed=False, target=-1,
                      target_valid=False, trigger=-1.0, input_time=0.0,
                      input_monotonic=0.0, angle=[-1] * 6, angle_time=0.0,
                      angle_monotonic=0.0, angle_elapsed_ms=0.0, angle_valid=False,
                      command_id=0, write_id=0, write_target=-1, write_status=0,
                      write_time=0.0, fault_count=0, error="waiting for connection",
                      thumb_rotation=-1, thumb_target_valid=False, write_thumb_rotation=-1,
                      thumb_step=thumb_config.step, thumb_min=thumb_config.minimum,
                      thumb_max=thumb_config.maximum, thumb_hold_rate=thumb_config.hold_rate,
                      thumb_hold_active=False, thumb_hold_cancel_count=0,
                      close_angles=list(self.close_angles))

    def _reset(self):
        # Caller holds lock. Do not overwrite the last transaction result.
        self.epoch += 1
        self.pending = None
        self.armed = False
        self.previous = None
        self.baseline_after = time.monotonic()
        self.hold_direction = 0
        self.hold_token += 1
        self.hold_credit = 0.0
        self.s.update(input_valid=False, target_valid=False, armed=False, thumb_target_valid=False,
                      thumb_hold_active=False)

    def cancel(self):
        with self.lock:
            self._reset()

    def update(self, trigger, sample_monotonic, sample_time, active):
        now = time.monotonic()
        valid = (active and math.isfinite(trigger) and 0 <= trigger <= 1
                 and 0 <= now - sample_monotonic <= self.input_timeout)
        with self.lock:
            # Catch a gap even if the worker was inside a slow read throughout it.
            if now - self.last_update > self.input_timeout:
                self._reset()
            self.last_update = now
            self.s.update(trigger=trigger if math.isfinite(trigger) else -1.0,
                          input_time=sample_time, input_monotonic=sample_monotonic)
            if not valid or not self.enabled or not self.s["connected"]:
                self._reset()
                return
            self.s["input_valid"] = True
            if not self._seed_thumb(now):
                # Wait for a new measured baseline; never jump to a limit or 339.
                self.armed = self.s["armed"] = False
                self.previous = None
                return
            target = int(trigger > 0.5)
            if not self.armed:
                # A held trigger can NEVER cause a startup/reconnect command.
                if target == 0:
                    self.armed = self.s["armed"] = True
                    self.previous = 0
                return
            if target == self.previous:
                return
            self.previous = target
            self.s.update(target=target, target_valid=True)
            self._queue()

    def _seed_thumb(self, now):
        # Caller holds lock. Baseline selection is read-only and never a command.
        if self.s["thumb_target_valid"]:
            return True
        if (not self.s["angle_valid"] or self.s["angle_monotonic"] < self.baseline_after
                or not 0 <= now - self.s["angle_monotonic"] <= self.feedback_timeout):
            return False
        measured = self.s["angle"][5]
        if not self.thumb_config.minimum <= measured <= self.thumb_config.maximum:
            self.s["error"] = "Measured thumb rotation outside configured limits; control not armed"
            return False
        self.s.update(thumb_rotation=measured, thumb_target_valid=True, error="")
        return True

    def _queue(self):
        self.next_command_id += 1
        self.s["command_id"] = self.next_command_id
        target = self.s["target"] if self.s["target_valid"] else -1
        self.pending = _HandCommand(self.epoch, self.s["command_id"], target, self.s["thumb_rotation"])

    def input_session(self):
        with self.lock:
            return self.epoch if self.armed and self._fresh(time.monotonic()) else None

    def adjust_thumb(self, direction, session):
        with self.lock:
            if (session != self.epoch or not self.armed or not self.s["thumb_target_valid"]
                    or not self._fresh(time.monotonic())):
                return
            target = min(self.thumb_config.maximum, max(self.thumb_config.minimum,
                         self.s["thumb_rotation"] + direction * self.thumb_config.step))
            if target == self.s["thumb_rotation"]:
                return  # No register writes at a limit or while holding a button.
            self.s["thumb_rotation"] = target
            self._queue()

    def set_thumb_hold(self, direction, session=None):
        with self.lock:
            if session != self.epoch or not self.armed or not self._fresh(time.monotonic()):
                direction = 0
            if direction != self.hold_direction:
                self.hold_token += 1
                self.hold_direction = direction
                self.hold_next = time.monotonic() + THUMB_HOLD_INTERVAL
                self.hold_credit = 0.0
            self.s["thumb_hold_active"] = bool(direction)

    def _take_hold(self, now):
        # Caller holds lock. Generate at most one small step only when this worker
        # is free. A slow connection never accumulates future rotation targets.
        if (not self.hold_direction or now < self.hold_next or not self._fresh(now)
                or not self.s["thumb_target_valid"] or not self.s["angle_valid"]
                or not 0 <= now - self.s["angle_monotonic"] <= self.feedback_timeout):
            return None
        self.hold_next = now + THUMB_HOLD_INTERVAL
        self.hold_credit += self.thumb_config.hold_rate * THUMB_HOLD_INTERVAL
        step = int(self.hold_credit + 1e-9)
        self.hold_credit -= step
        target = min(self.thumb_config.maximum, max(self.thumb_config.minimum,
                     self.s["thumb_rotation"] + self.hold_direction * step))
        if target == self.s["thumb_rotation"]:
            return None
        # Do not drive a target far ahead of a blocked/lagging real thumb and
        # then let it catch up quickly when contact disappears. Wait for feedback.
        if self.hold_direction * (target - self.s["angle"][5]) > THUMB_HOLD_MAX_LEAD:
            return None
        self.next_command_id += 1
        command = _HandCommand(self.epoch, self.next_command_id,
                               self.s["target"] if self.s["target_valid"] else -1, target)
        return command, self.hold_token, self.s["command_id"]

    def _guard_hold(self, command, token, base_id):
        with self.lock:
            if (command.epoch != self.epoch or token != self.hold_token or not self.hold_direction
                    or base_id != self.s["command_id"] or not self._fresh(time.monotonic())):
                raise _HoldReleased("hold released, chord detected, input expired, or newer grasp")

    def _fresh(self, now):
        return (not self.stop_event.is_set() and self.enabled and self.s["input_valid"]
                and 0 <= now - self.last_update <= self.input_timeout
                and 0 <= now - self.s["input_monotonic"] <= self.input_timeout)

    def _guard(self, command):
        with self.lock:
            if (command.epoch != self.epoch or command.command_id != self.s["command_id"]
                    or not self._fresh(time.monotonic())):
                raise _Cancelled("input expired, mode changed, or command superseded")

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if self.s["input_valid"] and not self._fresh(now):
                self._reset()
            s = copy.deepcopy(self.s)
        s["angle_valid"] = bool(s["connected"] and s["angle_valid"]
                                and 0 <= now - s["angle_monotonic"] <= self.feedback_timeout)
        s["write_current"] = bool(s["target_valid"] and s["thumb_target_valid"]
                                  and s["write_id"] == s["command_id"]
                                  and s["write_status"] == WRITE_STATUS["confirmed"])
        # This is a measured tolerance check, not a physical completion guarantee.
        s["at_target_valid"] = bool(s["target_valid"] and s["thumb_target_valid"] and s["angle_valid"])
        bends = self.close_angles if s["target"] == 1 else ANGLES[0][:5]
        target_angles = (*bends, s["thumb_rotation"])
        s["angle_target"] = list(target_angles) if s["target_valid"] else [-1]*5 + [s["thumb_rotation"]]
        s["at_target"] = bool(s["at_target_valid"] and
                              all(abs(a - b) <= 20 for a, b in zip(s["angle"], target_angles)))
        return s

    def _run(self):
        client = None
        try:
            while not self.stop_event.is_set():
                try:
                    if client is None:
                        client = self.factory(self.config)
                        client.connect()  # read-only; enable_writes does not send registers
                        self.settings_ready = False
                        with self.lock:
                            self._reset()
                            self.s.update(connected=True, error="")
                    with self.lock:
                        if self.s["input_valid"] and not self._fresh(time.monotonic()):
                            self._reset()
                        command, self.pending = self.pending, None
                    if command is not None:
                        self._write(client, command)
                    else:
                        with self.lock:
                            held = self._take_hold(time.monotonic())
                        if held is not None:
                            self._write_hold(client, *held)
                    if self.stop_event.is_set():
                        break
                    reading = client.read_state_field("angle_act")
                    with self.lock:
                        self.s.update(angle=list(reading.values), angle_time=reading.timestamp,
                                      angle_monotonic=time.monotonic(),
                                      angle_elapsed_ms=reading.elapsed_ms, angle_valid=True)
                    self.stop_event.wait(self.period)
                except Exception as exc:
                    with self.lock:
                        self._reset()
                        self.s.update(connected=False, angle_valid=False, error=str(exc))
                        self.s["fault_count"] += 1
                    LOG.warning("Inspire %s: %s", self.config.host, exc)
                    if client is not None:
                        client.close()
                        client = None
                    self.stop_event.wait(self.reconnect_delay)
        finally:
            if client is not None:
                client.close()  # disconnect, NEVER close_hand()
            with self.lock:
                self._reset()
                self.s.update(connected=False, angle_valid=False)

    def _write(self, client, command):
        from inspire_rh56e2 import WriteUnconfirmed

        try:
            self._guard(command)
        except _Cancelled:
            return  # nothing started
        with self.lock:
            self.s.update(write_id=command.command_id, write_target=command.target,
                          write_thumb_rotation=command.thumb_rotation, write_status=1)
        status, error = "confirmed", ""
        try:
            client.enable_writes()
            # SDK order: force=200, speed=200, then angle. Guard runs before EACH write.
            if command.target == -1:
                from inspire_rh56e2.presets import HandPreset
                # RH56E2 manual §2.6.11 / SDK simulator: -1 leaves that DOF still.
                # Before any explicit grasp, rotation must not invent finger targets.
                preset = HandPreset("thumb_rotation", (-1,) * 5 + (command.thumb_rotation,),
                                    force=200, speed=200)
            else:
                preset = hand_preset(command.target, command.thumb_rotation, self.close_angles)
            client.apply_preset(preset, guard=lambda: self._guard(command))
        except _Cancelled as exc:
            status, error = "cancelled", str(exc)
        except WriteUnconfirmed as exc:
            status, error = "unconfirmed", str(exc)
        except Exception as exc:
            status, error = "failed", str(exc)
        finally:
            client.enable_writes(False)
        with self.lock:
            self.s.update(write_status=WRITE_STATUS[status], write_time=time.time(), error=error)
            if status == "confirmed":
                self.settings_ready = True
            if status == "cancelled":
                # A later confirmed write must not hide a partial cancellation
                # between two telemetry publications during recording.
                self.s["fault_count"] += 1
            if status in {"failed", "unconfirmed"}:
                self._reset()
        if status in {"failed", "unconfirmed"}:
            # Reconnect read-only and require a NEW release/press; never replay.
            raise RuntimeError(f"write {status} (no retry): {error}")

    def _write_hold(self, client, command, token, base_id):
        """One rate-limited step, with no mailbox of future held-key commands.

        The target advances only on register acknowledgement. In-flight intent
        is write_thumb_rotation; pending/unconfirmed data remains masked. An
        interrupted preparation cannot masquerade as a completed angle write.
        """
        from inspire_rh56e2 import Command, WriteUnconfirmed

        guard = lambda: self._guard_hold(command, token, base_id)
        try:
            guard()
        except _HoldReleased:
            return
        fields = ("write_id", "write_target", "write_thumb_rotation", "write_status", "write_time", "error")
        with self.lock:
            previous_write = {key: self.s[key] for key in fields}
            self.s.update(write_id=command.command_id, write_target=command.target,
                          write_thumb_rotation=command.thumb_rotation, write_status=1)
        status, error = "confirmed", ""
        try:
            client.enable_writes()
            if not self.settings_ready:
                client.command(Command(force=(200,)*6, speed=(200,)*6), guard=guard)
                self.settings_ready = True
            bends = ((-1,)*5 if command.target == -1 else
                     self.close_angles if command.target == 1 else ANGLES[0][:5])
            # One angle operation: if its guard cancels, NO angle was sent.
            client.command(Command(angle=(*bends, command.thumb_rotation)), guard=guard)
        except _HoldReleased:
            with self.lock:
                self.s.update(previous_write)
                self.s["thumb_hold_cancel_count"] += 1
            return
        except WriteUnconfirmed as exc:
            status, error = "unconfirmed", str(exc)
        except Exception as exc:
            status, error = "failed", str(exc)
        finally:
            client.enable_writes(False)
        with self.lock:
            self.s.update(write_status=WRITE_STATUS[status], write_time=time.time(), error=error)
            if status != "confirmed":
                self._reset()
            elif command.epoch == self.epoch and base_id == self.s["command_id"]:
                # Never replace a newer trigger/click goal with a late hold ACK.
                self.s.update(command_id=command.command_id, thumb_rotation=command.thumb_rotation)
        if status != "confirmed":
            raise RuntimeError(f"hold write {status} (no retry): {error}")


class InspireHandController:
    def __init__(self, left_host="192.168.123.211", right_host="192.168.123.210",
                 port=6000, enabled=False, input_timeout=0.25, feedback_timeout=0.5,
                 timeout=0.2, reconnect_delay=1.0, poll_hz=20, client_factory=None,
                 lock_dir="/tmp", left_thumb=None, right_thumb=None, close_angles=CLOSE_ANGLES):
        from inspire_rh56e2 import HandClient, HandConfig

        validate_inspire_sdk()
        close_angles = validate_close_angles(close_angles)
        if left_host == right_host:
            raise ValueError("Left and right hands must have different IP addresses")
        self.workers = [
            _HandWorker(HandConfig(host=host, port=port, unit_id=255, model="RH56E2-T1",
                                   byte_layout="packed_little", tactile_order="big", timeout=timeout),
                        enabled, client_factory or HandClient, input_timeout=input_timeout,
                        feedback_timeout=feedback_timeout, reconnect_delay=reconnect_delay,
                        poll_hz=poll_hz, thumb_config=thumb or ThumbRotationConfig(), close_angles=close_angles)
            for host, thumb in zip((left_host, right_host), (left_thumb, right_thumb))
        ]
        self.gesture = ThumbButtonGesture()
        self.locks = []
        self.lock_dir = Path(lock_dir)
        self.started = False

    def start(self):
        if self.started:
            raise RuntimeError("Controller already started")
        try:
            # Persistent lock files must not be unlinked (avoids inode races).
            for worker in self.workers:
                self.locks.append(acquire_hand_lock(worker.config.host, worker.config.port, self.lock_dir))
            for worker in self.workers:
                worker.thread.start()
            self.started = True
        except BaseException:
            self.close()
            raise

    def update(self, left_trigger, right_trigger, *, sample_monotonic, sample_time,
               active, buttons=None, left_grip=0.0):
        for worker, trigger in zip(self.workers, (left_trigger, right_trigger)):
            worker.update(trigger, sample_monotonic, sample_time, active)
        event = self.gesture.update(buttons, left_grip,
                                    tuple(worker.input_session() for worker in self.workers))
        held = self.gesture.held
        for side, worker in enumerate(self.workers):
            worker.set_thumb_hold(held[1] if held is not None and side == held[0] else 0,
                                  held[2] if held is not None and side == held[0] else None)
        if event is not None:
            side, direction, session = event
            self.workers[side].adjust_thumb(direction, session)

    def cancel(self):
        self.gesture.reset()
        for worker in self.workers:
            worker.cancel()

    def snapshot(self):
        return dict(schema_version=HAND_SCHEMA_VERSION, backend="inspire", model="RH56E2-T1",
                    published_time=time.time(), published_monotonic=time.monotonic(),
                    hands=[worker.snapshot() for worker in self.workers])

    def close(self):
        # Signal BOTH before joining either. Network calls have finite SDK timeouts.
        for worker in self.workers:
            worker.stop_event.set()
            worker.cancel()
        for worker in self.workers:
            if worker.thread.ident is not None:
                worker.thread.join()
        for handle in self.locks:
            handle.close()
        self.locks.clear()
