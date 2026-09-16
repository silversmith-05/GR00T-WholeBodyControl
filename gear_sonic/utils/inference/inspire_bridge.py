"""Hardware-independent G1 Inspire observation, protocol and scheduling contracts."""

from dataclasses import dataclass
import json
import logging
import math
import queue
import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

STATE_WIDTHS = dict(
    left_leg=6, right_leg=6, waist=3, left_arm=7, right_arm=7, projected_gravity=3, left_hand=6, right_hand=6
)
VIDEO_KEYS = ("ego_view", "left_wrist", "right_wrist")
HORIZON = 40
CONTROL_HZ = 50
HAND_FEEDBACK_WARN_SECONDS = 0.5
HAND_FEEDBACK_STOP_SECONDS = 1.0
LOG = logging.getLogger(__name__)


def array(value, shape, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite {shape}, got {result.shape}")
    return result


def validate_modality(config):
    expected = {
        "video": list(VIDEO_KEYS),
        "state": list(STATE_WIDTHS),
        "action": ["motion_token", "hand"],
        "language": ["annotation.human.task_description"],
    }
    for modality, keys in expected.items():
        entry = config.get(modality, {})
        steps = list(range(HORIZON)) if modality == "action" else [0]
        if entry.get("modality_keys") != keys or entry.get("delta_indices") != steps:
            raise ValueError(f"Incompatible new_embodiment {modality} configuration")
    if len(config["action"].get("action_configs", [])) != 2:
        raise ValueError("Expected exactly two action configurations")
    for action in config["action"]["action_configs"]:
        if tuple(action[key].lower() for key in ("rep", "type", "format")) != ("absolute", "non_eef", "default"):
            raise ValueError("Inspire requires absolute non-EEF actions")


def build_observation(body, camera, hands, robot_model, prompt):
    """Match collection's body conversion, including MuJoCo -> model ordering."""
    q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=array(body["body_q"], (29,), "body_q")
    )
    state = {
        key: np.asarray(q[robot_model.get_joint_group_indices(key)], dtype=np.float32)
        for key in list(STATE_WIDTHS)[:5]
    }
    state["projected_gravity"] = compute_projected_gravity(array(body["base_quat"], (4,), "base_quat"))
    for side, hand in zip(("left", "right"), hands):
        state[f"{side}_hand"] = array(hand["angle"], (6,), f"{side} angle_act")
    for key, width in STATE_WIDTHS.items():
        state[key] = array(state[key], (width,), key)[None, None]
    video = {}
    for key in VIDEO_KEYS:
        image = np.asarray(camera["images"][key])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 RGB (480,640,3), got {image.shape}/{image.dtype}")
        video[key] = image[None, None]
    if not prompt.strip():
        raise ValueError("Task prompt cannot be empty")
    return {"state": state, "video": video, "language": {"annotation.human.task_description": [[prompt]]}}


def validate_sonic_config(config):
    if not config or config.get("dex3_hands_enabled") is not False:
        raise RuntimeError("Awaiting robot_config with dex3_hands_enabled=false")
    for key, suffix in (
        ("model_path", "release/model_decoder.onnx"),
        ("encoder_file", "release/model_encoder.onnx"),
        ("obs_config_path", "release/observation_config.yaml"),
    ):
        if not config.get(key, "").endswith(suffix):
            raise RuntimeError(f"SONIC differs from collection: {key}")
    if config.get("control_frequency") != CONTROL_HZ:
        raise RuntimeError("SONIC control frequency must be 50 Hz")


class SensorCache:
    """Freshness uses arrival monotonic time AND progressing source markers.

    Camera timestamps live on NX; do not compare its monotonic clock to the PC.
    Repeated cached images or body packets never refresh their arrival time.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.body = self.camera = self.config = None
        self.body_index = None
        self.body_time = -math.inf
        self.image_marks, self.image_times = {}, {}
        self.error = None
        self.hand_timeout_warnings = set()

    def update_body(self, body, now):
        marker = int(body["index"])
        with self.lock:
            if self.body_index is not None and marker < self.body_index:
                raise RuntimeError("SONIC state sequence restarted")
            if self.body_index is None or marker > self.body_index:
                self.body, self.body_index, self.body_time = body, marker, now

    def update_camera(self, camera, now):
        with self.lock:
            for key in VIDEO_KEYS:
                marker = float(camera.get("timestamps", {}).get(key, float("nan")))
                if key not in camera.get("images", {}) or not math.isfinite(marker):
                    raise ValueError(f"Missing image/timestamp: {key}")
                previous = self.image_marks.get(key, -math.inf)
                if marker < previous:
                    raise RuntimeError(f"Camera clock/stream restarted: {key}")
                if marker > previous:
                    self.image_marks[key], self.image_times[key] = marker, now
            self.camera = camera

    def snapshot(self, hands, now=None):
        with self.lock:
            # Sample after the hand snapshot and after acquiring the cache lock:
            # feedback threads may have advanced since the control tick began.
            if now is None:
                now = time.monotonic()
            if self.error:
                raise RuntimeError(self.error)
            if self.body is None or now - self.body_time > 0.25:
                raise RuntimeError("Body feedback missing/stale (>250 ms)")
            for key in VIDEO_KEYS:
                if now - self.image_times.get(key, -math.inf) > 0.25:
                    raise RuntimeError(f"Camera missing/stale (>250 ms): {key}")
            validate_sonic_config(self.config)
            if len(hands) != 2:
                raise RuntimeError("Missing hand feedback")
            for side, hand in zip(("left", "right"), hands):
                age = now - hand["angle_monotonic"]
                if math.isfinite(age) and age > HAND_FEEDBACK_WARN_SECONDS:
                    if side not in self.hand_timeout_warnings:
                        LOG.warning(
                            "hand_feedback_timeout side=%s age_ms=%.3f warning_ms=500 stop_ms=1000",
                            side, age * 1000,
                        )
                        self.hand_timeout_warnings.add(side)
                elif hand["connected"] and hand["angle_valid"] and 0 <= age:
                    if side in self.hand_timeout_warnings:
                        LOG.info("hand_feedback_recovered side=%s age_ms=%.3f", side, age * 1000)
                        self.hand_timeout_warnings.remove(side)
                if (
                    not hand["connected"]
                    or not hand["angle_valid"]
                    or not 0 <= age <= HAND_FEEDBACK_STOP_SECONDS
                ):
                    raise RuntimeError(
                        f"{side} hand feedback missing/stale (>1000 ms): "
                        f"age_ms={age * 1000:.3f}, connected={hand['connected']}, "
                        f"angle_valid={hand['angle_valid']}"
                    )
            observed = min(self.body_time, *self.image_times.values())
            return self.body, self.camera, observed


def _encode(value):
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise ValueError("Object arrays are not supported")
    return msgpack_numpy.encode(value)


def _decode(value):
    if value.get(b"nd", value.get("nd")) and value.get(b"kind", value.get("kind")) in (b"O", "O"):
        raise ValueError("Object arrays are not supported")
    if value.get("__ModalityConfig__") or value.get("__ModalityConfig_class__"):
        payload = value["as_json"]
        return json.loads(payload) if isinstance(payload, str) else payload
    return msgpack_numpy.decode(value)


class PolicyClient:
    """Small Python 3.10 client for official GR00T endpoints; no torch dependency.

    A client/socket belongs to one worker thread. Requests are never retried;
    timeout recovery creates a fresh REQ socket for a later explicit request.
    """

    def __init__(self, host="127.0.0.1", port=5550, timeout_ms=1000):
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.address)

    def call(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(msgpack.packb(request, default=_encode))
            raw = self.socket.recv()
        except zmq.Again as exc:
            self._connect()
            raise TimeoutError("GR00T request timed out; request was not retried") from exc
        if raw == b"ERROR":
            raise RuntimeError("GR00T server returned ERROR")
        response = msgpack.unpackb(raw, raw=False, object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def get_action(self, observation):
        # The official server dispatches handler(**data), not handler(data).
        return self.call("get_action", {"observation": observation, "options": None})


@dataclass
class Prediction:
    epoch: int
    request_id: int
    observed: float
    started: float
    completed: float
    actions: dict | None = None
    error: str | None = None


class PolicyWorker:
    def __init__(self, host, port, timeout_ms=1000):
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.busy = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(host, port, timeout_ms), daemon=True)
        self.thread.start()

    def submit(self, epoch, request_id, observation, observed):
        if self.busy.is_set():
            return False
        self.busy.set()
        try:
            self.requests.put_nowait((epoch, request_id, observation, observed))
            return True
        except queue.Full:
            self.busy.clear()
            return False

    def _run(self, host, port, timeout_ms):
        client = PolicyClient(host, port, timeout_ms)
        try:
            while not self.stop.is_set():
                try:
                    epoch, request_id, obs, observed = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                result = Prediction(epoch, request_id, observed, time.monotonic(), 0)
                try:
                    result.actions = client.get_action(obs)[0]
                except Exception as exc:
                    result.error = str(exc)
                result.completed = time.monotonic()
                # One request in flight; main loop must consume before resubmit.
                self.results.put(result)
        finally:
            client.close()

    def take(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy.clear()
        return result

    def close(self):
        self.stop.set()
        # Unblock a finished request whose result was not consumed at shutdown.
        self.take()
        self.thread.join(timeout=2)


class ActionTimeline:
    def __init__(self):
        self.epoch = 0
        self.chunk = None
        self.last_cursor = None

    def invalidate(self):
        self.epoch += 1
        self.chunk = None
        self.last_cursor = None

    def accept(self, prediction, now):
        if prediction.epoch != self.epoch:
            return False
        if prediction.error:
            raise RuntimeError(prediction.error)
        if not 0 <= now - prediction.observed < HORIZON / CONTROL_HZ:
            raise RuntimeError("Prediction expired before arrival (800 ms window)")
        actions = prediction.actions
        if set(actions) != {"motion_token", "hand"}:
            raise ValueError(f"Unexpected action keys: {list(actions)}")
        tokens = array(actions["motion_token"], (1, HORIZON, 64), "motion_token")
        hands = array(actions["hand"], (1, HORIZON, 2), "hand")
        if np.abs(tokens).max() > 1.25:
            raise ValueError("Motion token exceeds existing SONIC bound 1.25")
        prediction.actions = {"motion_token": tokens[0], "hand": hands[0]}
        self.chunk = prediction
        return True

    def step(self, now):
        if self.chunk is None:
            return None
        index = int(math.floor((now - self.chunk.observed) * CONTROL_HZ))
        if not 0 <= index < HORIZON:
            raise RuntimeError("Action sequence exhausted; refusing to repeat last action")
        cursor = (self.chunk.request_id, index)
        if cursor == self.last_cursor:
            return None
        self.last_cursor = cursor
        token = self.chunk.actions["motion_token"][index]
        continuous_hand = self.chunk.actions["hand"][index]
        return dict(
            token=token,
            hand=(continuous_hand >= 0.5).astype(np.int64),
            raw_hand=continuous_hand,
            request_id=cursor[0],
            action_index=index,
            valid_until=self.chunk.observed + HORIZON / CONTROL_HZ,
        )


def pack_action(token, sequence):
    token = array(token, (64,), "token_state")
    return pack_pose_message(
        {"token_state": token[None], "frame_index": np.array([sequence], dtype=np.int64)}, topic="pose", version=4
    )
