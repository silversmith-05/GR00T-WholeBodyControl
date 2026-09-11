"""Exporter-owned recording status shared with local browser preview processes."""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time


def default_recording_status_path(camera_host, camera_port):
    source = f"{camera_host.strip().lower()}:{camera_port}".encode()
    digest = hashlib.sha256(source).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"sonic-recording-{os.getuid()}-{digest}.json"


def read_recording_status(path, stale_after=3.0):
    """Missing, malformed or expired heartbeats must never imply idle/recording."""
    unavailable = {"available": False, "state": "unavailable", "recording": None,
                   "discarded": None, "episode_index": None, "result": None}
    try:
        with Path(path).open() as stream:
            data = json.load(stream)
        age = time.time() - data["updated_at"]
        if (data.get("version") != 1 or not math.isfinite(age) or not -1 <= age <= stale_after
                or data.get("state") not in {"idle", "recording", "saving", "error"}
                or not isinstance(data.get("discarded"), bool)
                or (data.get("episode_index") is not None
                    and (type(data["episode_index"]) is not int or data["episode_index"] < 0))):
            return unavailable
        return {**data, "available": True, "recording": data["state"] == "recording"}
    except (OSError, ValueError, TypeError, KeyError):
        return unavailable


class RecordingStatusPublisher:
    """Atomic heartbeat writes run off the recording loop, including during saves."""

    def __init__(self, path, dataset):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._stopped = threading.Event()
        self._warned = False
        self._snapshot = {"version": 1, "dataset": dataset, "state": "idle",
                          "episode_index": None, "discarded": False,
                          "discard_reason": None, "result": None}
        self._thread = threading.Thread(target=self._run, name="recording-status", daemon=True)
        self._thread.start()

    def update(self, **changes):
        with self._lock:
            if all(self._snapshot.get(key) == value for key, value in changes.items()):
                return
            self._snapshot.update(changes)
        self._changed.set()

    def _write(self):
        temporary = None
        try:
            with self._lock:
                snapshot = {**self._snapshot, "updated_at": time.time()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(snapshot, stream)
            os.replace(temporary, self.path)
            self._warned = False
        except OSError as exc:
            if not self._warned:
                logging.warning("Cannot publish browser recording status: %s", exc)
                self._warned = True
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _run(self):
        while not self._stopped.is_set():
            self._changed.clear()
            self._write()
            self._changed.wait(0.25)

    def close(self):
        self._stopped.set()
        self._changed.set()
        self._thread.join()
        self.update(state="stopped")
        self._write()
