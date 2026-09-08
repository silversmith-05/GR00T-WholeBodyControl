"""Read-only ZMQ to HTTP MJPEG relay, shared by all connected browsers."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorClient


PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SONIC Camera Preview</title>
<style>
body { margin:0; background:#10151e; color:#e5edf7; font:16px system-ui,sans-serif; }
main { max-width:1440px; margin:auto; padding:24px; }
h1 { font-size:24px; margin:0 0 10px; }
#source { color:#9aacc5; overflow-wrap:anywhere; }
#status { padding:12px 0; }
#cameras { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.camera { background:#203044; padding:6px 12px; border-radius:8px; }
.stale { color:#ffc477; }
img { display:block; width:100%; border-radius:10px; background:#080c12; }
img.offline { opacity:.3; }
small { display:block; color:#9aacc5; margin-top:14px; }
</style>
<main><h1>SONIC Camera Preview</h1><div id="source"></div>
<div id="status" role="status">Waiting for camera frames...</div><div id="cameras"></div>
<img id="preview" class="offline" alt="Live camera views">
<small>All camera views share one live stream. Preview does not record video.</small></main>
<script>
const preview = document.getElementById('preview');
let streaming = false;
preview.onerror = () => { streaming = false; };
async function update() {
  try {
    const response = await fetch('/status', {cache:'no-store', signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('Status unavailable');
    const data = await response.json();
    document.getElementById('source').textContent = data.source;
    document.getElementById('status').textContent = data.online
      ? 'Live' : (data.error || 'Waiting for camera frames / camera disconnected');
    preview.classList.toggle('offline', !data.online);
    if (data.online && !streaming) {
      preview.src = '/stream?t=' + Date.now();
      streaming = true;
    }
    if (!data.online) streaming = false;
    document.getElementById('cameras').replaceChildren(...data.cameras.map(camera => {
      const tag = document.createElement('span');
      tag.className = 'camera' + (camera.stale ? ' stale' : '');
      tag.textContent = camera.name + (camera.stale ? ' (stale)' : '');
      return tag;
    }));
  } catch (error) {
    document.getElementById('status').textContent = 'Preview server disconnected. Retrying...';
    preview.classList.add('offline');
    streaming = false;
  } finally { setTimeout(update, 1000); }
}
update();
</script></html>"""


class CameraRelay:
    """One subscriber/encoder; HTTP threads only read the latest encoded mosaic."""

    stale_after = 3.0

    def __init__(self, camera_host="localhost", camera_port=5555, fps=15, width=640):
        self.camera_host = camera_host
        self.camera_port = camera_port
        self.fps = fps
        self.width = width
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.jpeg = None
        self.sequence = 0
        self.received = {}
        self.error = None
        self.thread = threading.Thread(target=self._run, name="camera-relay", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join()

    def status(self):
        with self.condition:
            now = time.monotonic()
            cameras = [
                {"name": name, "age_seconds": round(now - received, 2),
                 "stale": now - received > self.stale_after}
                for name, received in sorted(self.received.items())
            ]
            return {
                "source": f"tcp://{self.camera_host}:{self.camera_port}",
                "online": any(not camera["stale"] for camera in cameras),
                "cameras": cameras,
                "error": self.error,
            }

    def wait_frame(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous or self.stopped.is_set(), timeout=1.0
            )
            if self.sequence == previous or self.stopped.is_set():
                return previous, None
            return self.sequence, self.jpeg

    def _run(self):
        # A ZMQ socket is created, used and closed exclusively on this thread.
        client = SensorClient()
        images = {}
        received = {}
        next_encode = 0.0
        try:
            client.start_client(self.camera_host, self.camera_port)
            while not self.stopped.is_set():
                try:
                    message = client.receive_message_nonblocking(timeout_ms=100)
                    now = time.monotonic()
                    if message is None or now < next_encode:
                        continue
                    sample = ImageMessageSchema.deserialize(message)
                    updated = False
                    for name, rgb in sample.images.items():
                        if not isinstance(rgb, np.ndarray) or rgb.size == 0:
                            continue
                        if rgb.ndim == 2:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                        else:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        height, width = bgr.shape[:2]
                        if width > self.width:
                            bgr = cv2.resize(bgr, (self.width, max(1, height * self.width // width)))
                        images[name] = bgr
                        received[name] = now
                        updated = True
                    if not updated:
                        continue
                    jpeg = self._mosaic(images, received, now)
                    with self.condition:
                        self.jpeg = jpeg
                        self.sequence += 1
                        self.received = received.copy()
                        self.error = None
                        self.condition.notify_all()
                    next_encode = time.monotonic() + 1.0 / self.fps
                except Exception as exc:
                    # A bad frame must not stop preview or prevent reconnection.
                    with self.condition:
                        self.error = f"Camera receive error: {exc}"
                    if self.stopped.wait(0.1):
                        break
        except Exception as exc:
            with self.condition:
                self.error = f"Camera connection error: {exc}"
        finally:
            if hasattr(client, "socket"):
                client.stop_client()

    def _mosaic(self, images, received, now):
        names = sorted(images)
        columns = min(3, math.ceil(math.sqrt(len(names))))
        tile_width = max(img.shape[1] for img in images.values())
        tile_height = max(img.shape[0] for img in images.values()) + 32
        canvas = np.zeros((math.ceil(len(names) / columns) * tile_height,
                           columns * tile_width, 3), dtype=np.uint8)
        for index, name in enumerate(names):
            x, y = index % columns * tile_width, index // columns * tile_height
            img = images[name]
            stale = now - received[name] > self.stale_after
            if stale:
                img = (img * 0.3).astype(np.uint8)
            height, width = img.shape[:2]
            canvas[y + 32:y + 32 + height, x:x + width] = img
            cv2.putText(canvas, name + (" [STALE]" if stale else ""), (x + 8, y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255) if stale else (220, 230, 240), 1)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("Cannot encode camera preview")
        return encoded.tobytes()


class CameraWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, relay):
        self.relay = relay
        super().__init__(address, CameraWebHandler)


class CameraWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Status polling and stream reconnects should not fill the tmux log.

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._respond(PAGE, "text/html; charset=utf-8")
            elif path == "/status":
                self._respond(json.dumps(self.server.relay.status()).encode(), "application/json")
            elif path == "/stream":
                self._stream()
            else:
                self.send_error(404)
        except OSError:
            pass  # Browser disconnected or is too slow; never block the subscriber.

    def _respond(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        relay = self.server.relay
        if not relay.status()["online"]:
            self._respond(b"Waiting for camera frames", "text/plain", 503)
            return
        self.connection.settimeout(5.0)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        previous = -1
        while not relay.stopped.is_set():
            previous, jpeg = relay.wait_frame(previous)
            if jpeg is None:
                if relay.stopped.is_set():
                    break
                # Keep established browser streams alive through camera outages.
                # A cached-frame heartbeat also detects disconnected clients; the
                # page's independent status poll marks these frames as stale.
                with relay.condition:
                    jpeg = relay.jpeg
                if jpeg is None:
                    continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"--frame--\r\n")
