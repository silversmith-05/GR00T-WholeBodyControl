"""Inspire wire/schema helpers. No SDK import and no hardware connections here."""
from __future__ import annotations

import numpy as np
import copy

from gear_sonic.utils.teleop.inspire_hand_controller import (
    ANGLES, HAND_SCHEMA_VERSION, PRESET_REVISION, CLOSE_ANGLES, validate_close_angles,
)

HAND_METADATA = {
    "backend": "inspire", "model": "RH56E2-T1", "schema_version": HAND_SCHEMA_VERSION,
    "preset_revision": PRESET_REVISION,
    "preset_source": "SDK custom five-finger presets; deeper closure starting point for approximately 40 mm hard ball; uncalibrated",
    "sdk": "inspire-rh56e2==0.3.0", "transport": "Modbus TCP", "unit_id": 255,
    "byte_layout": "packed_little", "tactile_order": "big",
    "action_key": "action.hand", "order": ["left", "right"],
    "thumb_rotation": {
        "action_key": "action.thumb_rotation", "order": ["left", "right"],
        "unit": "device scale 0..1000; -1 unknown; NOT radians or degrees",
        "mapping": {"X": "left decrease", "Y": "left increase", "A": "right decrease", "B": "right increase"},
        "gesture": "short click on release; single key held >=0.6s enables slow steps; release/chord/grip stops new steps",
        "baseline": "fresh angle_act[5] after each reset; selected without writing",
        "limits": "per-frame hand.thumb_min/max; protocol bounds by default, not calibrated limits",
        "step": "per-frame hand.thumb_step; independently configured per hand",
        "hold_rate": "per-frame hand.thumb_hold_rate, 1..50 scale units/s; at most 10 small steps/s; no backlog/catch-up",
        "hold_feedback_lead": "held-key targets may lead measured rotation by at most 10 scale units in the requested direction; wait for fresh feedback otherwise",
        "hold_target": "held-key action.thumb_rotation advances only on register ACK; in-flight intent in hand.write_thumb_rotation; pending frames masked",
        "hold_cancel": "hand.thumb_hold_cancel_count counts hold preparation cancelled before any angle write; previous angle write result retained; not a motion fault",
        "composition": "release/close first five channels plus action.thumb_rotation as channel six",
    },
    "binary_meaning": {"0": "放开 / release", "1": "闭合 / close", "-1": "unknown; invalid"},
    "angle_order": ["little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"],
    "angle_unit": "device scale (0..1000); NOT radians or degrees",
    "release": list(ANGLES[0]),
    "close": list(ANGLES[1]), "speed": [200] * 6, "force": [200] * 6,
    "grasp": "fixed binary five-finger targets, simultaneous bend command; thumb rotation independently retained. "
             "No trigger interpolation, measured completion or ball-size-to-angle calibration assumed. "
             "The close template first five values are fixed for this dataset; channel six is overridden by rotation action.",
    "write_status": {"0": "none", "1": "pending", "2": "registers confirmed, NOT motion completion",
                     "3": "unconfirmed; never retried", "4": "failed", "5": "cancelled; may be partial"},
    "at_target": "all six measured angles within 20 scale units of target; only when at_target_valid",
    "timing": "Unix seconds and workstation monotonic seconds; latest hand snapshot at frame acquisition",
    "training": "action.hand and action.thumb_rotation contain intent on EVERY frame; "
                "both are action modalities. No Dex3 hand state/action fields. "
                "Filter hand.training_valid; exclude discarded_episode_indices. Pending writes are "
                "masked, while invalid input/feedback, disconnects or write faults discard the episode.",
    "capture_time": "observation.capture_time: body workstation receive time, ego_view, left_wrist, "
                    "right_wrist camera source timestamps (Unix seconds); missing=-1 and validity=false. "
                    "NX/workstation clock offsets must be accounted for during offline alignment.",
    "feedback": "angle_act only; last good values retained with angle_valid=false on failure/staleness",
}


def normalize_hand_metadata(metadata):
    """Preserve schema-4 compatibility across a runtime default-rate change."""
    result = copy.deepcopy(metadata)
    thumb = result.get("thumb_rotation")
    rate_description = HAND_METADATA["thumb_rotation"]["hold_rate"]
    legacy_description = rate_description.replace("; at most", "; default 20, at most")
    if isinstance(thumb, dict) and thumb.get("hold_rate") == legacy_description:
        thumb["hold_rate"] = rate_description
    return result


def hand_metadata(close_angles=CLOSE_ANGLES):
    result = copy.deepcopy(HAND_METADATA)
    result["close"] = [*validate_close_angles(close_angles), 339]
    return result

# key -> (dtype, shape, names); arrays of length 2 are always left, right.
HAND_FEATURES = {
    "action.hand": ("float32", 2, ["left", "right"]),
    "action.thumb_rotation": ("float32", 2, ["left", "right"]),
    "hand.angle_act": ("float32", 12, [f"{side}_{joint}" for side in ("left", "right")
                                             for joint in HAND_METADATA["angle_order"]]),
    "hand.angle_target": ("float32", 12, [f"{side}_{joint}" for side in ("left", "right")
                                             for joint in HAND_METADATA["angle_order"]]),
    "hand.close_angles": ("int64", 10, [f"{side}_{joint}" for side in ("left", "right")
                                             for joint in HAND_METADATA["angle_order"][:5]]),
    **{f"hand.{key}": (dtype, 2, ["left", "right"]) for key, dtype in {
        "trigger": "float32", "connected": "bool", "input_valid": "bool", "armed": "bool",
        "target_valid": "bool", "angle_valid": "bool", "at_target": "bool", "at_target_valid": "bool",
        "input_time": "float64", "input_monotonic": "float64", "angle_time": "float64",
        "angle_monotonic": "float64", "angle_elapsed_ms": "float32", "command_id": "int64",
        "write_id": "int64", "write_target": "int64", "write_status": "int64", "write_time": "float64",
        "write_current": "bool", "fault_count": "int64",
        "thumb_target_valid": "bool", "write_thumb_rotation": "int64",
        "thumb_step": "int64", "thumb_min": "int64", "thumb_max": "int64",
        "thumb_hold_rate": "float32", "thumb_hold_active": "bool", "thumb_hold_cancel_count": "int64",
    }.items()},
    **{f"hand.{key}": (dtype, 1, [key]) for key, dtype in {
        "frame_time": "float64", "frame_monotonic": "float64", "published_time": "float64",
        "published_monotonic": "float64", "publication_valid": "bool", "training_valid": "bool",
        "episode_fault": "bool",
    }.items()},
}
ACTION_FIELDS = {"action.hand": "target", "action.thumb_rotation": "thumb_rotation"}


def hand_features():
    return {key: dict(dtype=dtype, shape=(size,), names=names)
            for key, (dtype, size, names) in HAND_FEATURES.items()}


def validate_snapshot(snapshot):
    if (snapshot.get("schema_version") != HAND_SCHEMA_VERSION or snapshot.get("backend") != "inspire"
            or snapshot.get("model") != "RH56E2-T1" or len(snapshot.get("hands", [])) != 2):
        raise ValueError("Incompatible Inspire hand feedback schema")
    for key in ("published_time", "published_monotonic"):
        if not np.isfinite(snapshot[key]):
            raise ValueError("Invalid publication timestamp")
    for hand in snapshot["hands"]:
        if len(hand["angle"]) != 6 or hand["target"] not in (-1, 0, 1):
            raise ValueError("Invalid Inspire dimensions/target")
        validate_close_angles(hand["close_angles"])
        if (len(hand["angle_target"]) != 6 or not np.isfinite(hand["angle_target"]).all()
                or any(v < -1 or v > 1000 for v in hand["angle_target"])):
            raise ValueError("Invalid composed hand angle target")
        if hand["target_valid"]:
            bends = hand["close_angles"] if hand["target"] == 1 else list(ANGLES[0][:5])
            if hand["target"] not in (0,1) or hand["angle_target"] != [*bends, hand["thumb_rotation"]]:
                raise ValueError("Hand angle target differs from binary/rotation action")
        for key, (dtype, size, _) in HAND_FEATURES.items():
            field = ACTION_FIELDS.get(key, key.removeprefix("hand."))
            if size == 2:
                value = hand[field]
                if not np.isfinite(value) or (dtype == "bool" and type(value) is not bool):
                    raise ValueError(f"Invalid {key}")
        if not np.isfinite(hand["angle"]).all() or hand["write_status"] not in range(6):
            raise ValueError("Invalid angles/write status")
        for key in ("thumb_rotation", "write_thumb_rotation", "thumb_step", "thumb_min", "thumb_max"):
            if type(hand[key]) is not int:
                raise ValueError(f"Invalid integer {key}")
        if (not 0 <= hand["thumb_min"] < hand["thumb_max"] <= 1000
                or not 1 <= hand["thumb_step"] <= 1000
                or not 1 <= hand["thumb_hold_rate"] <= 50
                or not -1 <= hand["thumb_rotation"] <= 1000
                or not -1 <= hand["write_thumb_rotation"] <= 1000
                or (hand["thumb_target_valid"]
                    and not hand["thumb_min"] <= hand["thumb_rotation"] <= hand["thumb_max"])):
            raise ValueError("Invalid thumb rotation target/configuration")


def snapshot_to_frame(snapshot, *, frame_time, frame_monotonic, max_age=0.5):
    """Apply freshness again at the consumer; a dead publisher cannot look valid."""
    frame = {key: np.zeros(size, dtype=dtype) for key, (dtype, size, _) in HAND_FEATURES.items()}
    frame["action.hand"][:] = -1
    frame["action.thumb_rotation"][:] = -1
    frame["hand.angle_act"][:] = -1
    frame["hand.angle_target"][:] = -1
    frame["hand.trigger"][:] = -1
    frame["hand.write_target"][:] = -1
    frame["hand.write_thumb_rotation"][:] = -1
    frame["hand.frame_time"][0], frame["hand.frame_monotonic"][0] = frame_time, frame_monotonic
    if snapshot is not None:
        validate_snapshot(snapshot)
        fresh = 0 <= frame_monotonic - snapshot["published_monotonic"] <= max_age
        for key in ("published_time", "published_monotonic"):
            frame[f"hand.{key}"][0] = snapshot[key]
        frame["hand.publication_valid"][0] = fresh
        for i, hand in enumerate(snapshot["hands"]):
            frame["hand.angle_act"][i * 6:(i + 1) * 6] = hand["angle"]
            frame["hand.angle_target"][i * 6:(i + 1) * 6] = hand["angle_target"]
            frame["hand.close_angles"][i * 5:(i + 1) * 5] = hand["close_angles"]
            for key, (_, size, _) in HAND_FEATURES.items():
                if size == 2:
                    frame[key][i] = hand[ACTION_FIELDS.get(key, key.removeprefix("hand."))]
            input_fresh = 0 <= frame_monotonic - hand["input_monotonic"] <= 0.25
            angle_fresh = 0 <= frame_monotonic - hand["angle_monotonic"] <= max_age
            for field in ("connected", "input_valid", "target_valid", "thumb_target_valid",
                          "angle_valid", "at_target_valid", "write_current", "thumb_hold_active"):
                frame[f"hand.{field}"][i] &= fresh
            frame["hand.input_valid"][i] &= input_fresh
            frame["hand.angle_valid"][i] &= angle_fresh and hand["connected"]
            frame["hand.at_target_valid"][i] &= (frame["hand.angle_valid"][i]
                and frame["hand.target_valid"][i] and frame["hand.thumb_target_valid"][i])
            frame["hand.write_current"][i] &= (hand["write_status"] == 2
                and hand["write_id"] == hand["command_id"] and hand["write_target"] == hand["target"]
                and hand["write_thumb_rotation"] == hand["thumb_rotation"]
                and frame["hand.target_valid"][i] and frame["hand.thumb_target_valid"][i])
    healthy = all(frame[f"hand.{key}"].all() for key in
                  ("publication_valid", "connected", "input_valid", "target_valid", "thumb_target_valid", "angle_valid"))
    fault = not healthy or bool(np.isin(frame["hand.write_status"], [3, 4, 5]).any())
    frame["hand.episode_fault"][0] = fault
    frame["hand.training_valid"][0] = not fault and frame["hand.write_current"].all()
    return frame
