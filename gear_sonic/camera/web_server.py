"""Read-only ZMQ to HTTP MJPEG relay, shared by all connected browsers."""

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorClient


PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SONIC Camera Preview</title>
<style>
* { box-sizing:border-box; }
html, body { width:100%; height:100%; margin:0; overflow:hidden; }
body { background:#10151e; color:#e5edf7; font:14px system-ui,sans-serif; }
main { width:100%; height:100vh; height:100dvh; padding:12px 16px; display:grid;
  grid-template-rows:auto auto auto auto minmax(0,1fr) auto; gap:8px; }
header { display:flex; align-items:center; justify-content:space-between; gap:12px; min-width:0; }
h1 { font-size:20px; margin:0; min-width:0; }
#source, #status { font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#source { color:#9aacc5; }
fieldset { min-width:0; border:1px solid #34465d; border-radius:10px; padding:8px 10px; margin:0; }
#cameras { display:flex; flex-wrap:wrap; gap:6px; }
.camera { display:flex; align-items:center; gap:8px; background:#203044;
  min-width:0; max-width:100%; padding:5px 10px; border-radius:8px; cursor:pointer; }
.camera span { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
input { accent-color:#79bfff; width:16px; height:16px; flex-shrink:0; }
button { color:#e5edf7; background:#203044; border:1px solid #547192;
  border-radius:6px; padding:5px 10px; margin:0 8px 8px 0; cursor:pointer; }
header button { flex-shrink:0; margin:0; }
button:disabled { opacity:.5; cursor:default; }
#layout-controls { display:inline-flex; align-items:center; gap:8px; margin-bottom:8px; }
#layout-controls button { margin:0; }
select { color:#e5edf7; background:#203044; border:1px solid #547192; border-radius:6px; padding:4px; }
.stale { color:#ffc477; }
#preview { grid-row:5; display:grid; gap:8px; width:100%; height:100%; min-width:0; min-height:0; }
#preview[hidden], #stream-source { display:none; }
.view-card { display:grid; grid-template-rows:auto minmax(0,1fr); min-width:0; min-height:0;
  overflow:hidden; background:#080c12; border:1px solid #34465d; border-radius:8px; }
.view-handle { width:100%; min-width:0; margin:0; padding:5px 8px; border:0; border-radius:0;
  text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  cursor:grab; touch-action:none; user-select:none; }
.view-handle:focus-visible { outline:2px solid #79bfff; outline-offset:-2px; }
.view-card.dragging { opacity:.6; }
.view-card.drop-target { outline:2px solid #79bfff; outline-offset:-2px; }
.view-card canvas { width:100%; height:100%; min-width:0; min-height:0; object-fit:contain; }
.view-card.offline canvas, #preview.offline canvas { opacity:.3; }
small { grid-row:6; color:#9aacc5; font-size:11px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.viewer-expanded main { padding:0; }
.viewer-expanded header { position:fixed; top:12px; right:12px; z-index:2; }
.viewer-expanded header h1, .viewer-expanded #source,
.viewer-expanded fieldset, .viewer-expanded small { display:none; }
.viewer-expanded #preview { position:fixed; inset:0; width:100%; height:100vh;
  height:100dvh; padding:6px; background:#080c12; z-index:1; }
.viewer-expanded #status { position:fixed; bottom:12px; left:12px; z-index:2;
  max-width:calc(100% - 24px); padding:6px 10px; border-radius:6px; background:#10151ee6; }
@media (max-height:500px), (max-width:600px) {
  main { padding:6px 8px; gap:4px; }
  h1 { font-size:17px; }
  fieldset { padding:4px 6px; }
  .camera { padding:3px 6px; font-size:12px; }
  button { padding:3px 8px; margin-bottom:4px; }
  header button { margin:0; }
  #preview { gap:4px; }
  .view-handle { padding:3px 5px; font-size:12px; }
}
</style>
<main><header><h1>SONIC Camera Preview</h1>
<button id="expand-view" type="button" aria-pressed="false" disabled>Expand view</button></header>
<div id="source"></div>
<div id="status" role="status">Waiting for camera frames...</div>
<fieldset><legend>Camera views</legend>
<button id="select-all" type="button">Select all</button>
<button id="select-none" type="button">Clear selection</button>
<span id="layout-controls"><label for="columns">Columns</label>
<select id="columns"><option value="0">Auto</option><option value="1">1</option>
<option value="2">2</option><option value="3">3</option><option value="4">4</option></select>
<button id="reset-layout" type="button">Reset layout</button></span>
<div id="cameras"></div></fieldset>
<div id="preview" class="offline" aria-label="Camera windows" hidden></div>
<small>Drag a view's title to rearrange, or focus it and use arrow keys. Layout is saved in this browser.</small></main>
<img id="stream-source" alt="" hidden>
<script>
const preview = document.getElementById('preview');
const source = document.getElementById('stream-source');
const expandView = document.getElementById('expand-view');
const columns = document.getElementById('columns');
function setExpanded(expanded) {
  document.body.classList.toggle('viewer-expanded', expanded);
  expandView.textContent = expanded ? 'Exit expanded view' : 'Expand view';
  expandView.setAttribute('aria-pressed', String(expanded));
}
expandView.onclick = () => setExpanded(!document.body.classList.contains('viewer-expanded'));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && document.body.classList.contains('viewer-expanded')) {
    setExpanded(false);
    expandView.focus();
  }
});
const controls = new Map();
const cards = new Map();
let latest = null;
let selection = null; // null means all views, including newly discovered cameras.
let storageKey = null;
let layoutKey = null;
let viewOrder = [];
let columnPreference = 0;
let gridColumns = 1;
let drag = null;
let streaming = false;
let streamKey = '';
let sourceNames = [];
let sourceReady = false;
source.onload = () => { sourceReady = true; };
source.onerror = () => { streaming = false; sourceReady = false; };
function orderedNames(names) {
  return [...viewOrder.filter(name => names.includes(name)), ...names.filter(name => !viewOrder.includes(name))];
}
function saveLayout() {
  try { if (layoutKey) localStorage.setItem(layoutKey, JSON.stringify({order:viewOrder,columns:columnPreference})); }
  catch (_) {}
}
function layoutGrid() {
  const count = preview.children.length;
  if (!count) return;
  let bestArea = -1;
  gridColumns = Math.min(columnPreference || 1, count);
  if (!columnPreference) {
    for (let cols = 1; cols <= Math.min(4, count); cols++) {
      const rows = Math.ceil(count / cols);
      const width = Math.max(1, (preview.clientWidth - 8 * (cols - 1)) / cols);
      const height = Math.max(1, (preview.clientHeight - 8 * (rows - 1)) / rows - 28);
      const area = Math.min(width, height * 4/3) * Math.min(height, width * 3/4);
      if (area > bestArea) { bestArea = area; gridColumns = cols; }
    }
  }
  preview.style.gridTemplateColumns = `repeat(${gridColumns}, minmax(0, 1fr))`;
  preview.style.gridTemplateRows = `repeat(${Math.ceil(count / gridColumns)}, minmax(0, 1fr))`;
}
new ResizeObserver(layoutGrid).observe(preview);
columns.onchange = () => { columnPreference = Number(columns.value); saveLayout(); layoutGrid(); };
document.getElementById('reset-layout').onclick = () => {
  viewOrder = []; columnPreference = 0; columns.value = '0'; saveLayout(); render();
};
function moveView(name, target) {
  const allNames = orderedNames(latest.cameras.map(camera => camera.name));
  const visible = allNames.filter(item => selection === null || selection.includes(item));
  const from = visible.indexOf(name), to = visible.indexOf(target);
  if (from < 0 || to < 0 || from === to) return;
  visible.splice(to, 0, visible.splice(from, 1)[0]);
  let index = 0;
  viewOrder = allNames.map(item => visible.includes(item) ? visible[index++] : item);
  saveLayout(); render(); cards.get(name)?.handle.focus();
}
function finishDrag(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  const previous = drag;
  drag = null;
  for (const card of cards.values()) card.element.classList.remove('dragging', 'drop-target');
  if (event.type === 'pointerup' && previous.active && previous.target) moveView(previous.name, previous.target);
}
function makeCard(name) {
  const element = document.createElement('article');
  element.className = 'view-card'; element.dataset.camera = name;
  const handle = document.createElement('button');
  handle.type = 'button'; handle.className = 'view-handle';
  handle.setAttribute('aria-label', 'Move ' + name);
  handle.title = 'Drag to rearrange. Arrow keys also move this view.';
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', name + ' live camera');
  element.append(handle, canvas);
  const card = {element,handle,canvas,context:canvas.getContext('2d')};
  handle.onpointerdown = event => {
    if (event.button !== 0) return;
    drag = {name,pointerId:event.pointerId,x:event.clientX,y:event.clientY,active:false,target:null};
    handle.setPointerCapture(event.pointerId);
  };
  handle.onpointermove = event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (Math.hypot(event.clientX-drag.x,event.clientY-drag.y) > 6) drag.active = true;
    if (!drag.active) return;
    element.classList.add('dragging');
    const target = document.elementFromPoint(event.clientX,event.clientY)?.closest('.view-card');
    drag.target = target?.dataset.camera ?? null;
    for (const item of cards.values()) item.element.classList.toggle('drop-target', item.element === target);
  };
  handle.onpointerup = finishDrag;
  handle.onpointercancel = finishDrag;
  handle.onlostpointercapture = finishDrag;
  handle.onkeydown = event => {
    const steps = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-gridColumns,ArrowDown:gridColumns};
    if (!(event.key in steps)) return;
    event.preventDefault();
    const names = [...preview.children].map(item => item.dataset.camera);
    const target = names[names.indexOf(name) + steps[event.key]];
    if (target) moveView(name, target);
  };
  return card;
}
let lastPaint = 0;
function paintFrames(now) {
  requestAnimationFrame(paintFrames);
  if (!sourceReady || !source.naturalWidth || !sourceNames.length || now-lastPaint < 1000/15) return;
  lastPaint = now;
  // One MJPEG connection supplies all cards, avoiding browser per-host connection limits.
  const cols = Math.min(3, Math.ceil(Math.sqrt(sourceNames.length)));
  const rows = Math.ceil(sourceNames.length / cols);
  const width = source.naturalWidth / cols;
  const height = source.naturalHeight / rows;
  sourceNames.forEach((name,index) => {
    const card = cards.get(name);
    if (!card || height <= 32) return;
    if (card.canvas.width !== width || card.canvas.height !== height-32) {
      card.canvas.width = width; card.canvas.height = height-32;
    }
    card.context.drawImage(source, index%cols*width, Math.floor(index/cols)*height+32,
      width, height-32, 0, 0, width, height-32);
  });
}
requestAnimationFrame(paintFrames);
function choose(names) {
  selection = names;
  try { if (storageKey) localStorage.setItem(storageKey, JSON.stringify(selection)); } catch (_) {}
  render();
}
document.getElementById('select-all').onclick = () => choose(null);
document.getElementById('select-none').onclick = () => choose([]);
function render() {
  if (!latest) return;
  const selected = latest.cameras.filter(camera => selection === null || selection.includes(camera.name));
  const names = selected.map(camera => camera.name);
  const ordered = orderedNames(names);
  expandView.disabled = !names.length;
  if (!names.length) setExpanded(false);
  const online = selected.some(camera => !camera.stale);
  document.getElementById('source').textContent = latest.source;
  document.getElementById('status').textContent = !latest.cameras.length
    ? (latest.error || 'Waiting for camera frames...')
    : !names.length ? 'Select at least one camera view.'
    : online ? 'Live - ' + names.length + '/' + latest.cameras.length + ' views'
    : (latest.error || 'Selected cameras disconnected / frames stale');
  preview.hidden = !names.length;
  preview.classList.toggle('offline', !online);
  for (const [name,card] of cards) {
    if (!names.includes(name)) { card.element.remove(); cards.delete(name); }
  }
  ordered.forEach((name,index) => {
    let card = cards.get(name);
    if (!card) { card = makeCard(name); cards.set(name,card); }
    const camera = selected.find(item => item.name === name);
    card.handle.textContent = ':: ' + name + (camera.stale ? ' (stale)' : '');
    card.element.classList.toggle('offline', camera.stale);
    if (preview.children[index] !== card.element) preview.insertBefore(card.element, preview.children[index] || null);
  });
  layoutGrid();
  const key = JSON.stringify(names);
  if (!names.length || key !== streamKey) {
    source.removeAttribute('src');
    sourceReady = false;
    streaming = false;
    streamKey = key;
  }
  if (online && !streaming) {
    const query = new URLSearchParams();
    names.forEach(name => query.append('camera', name));
    query.set('t', Date.now());
    sourceNames = names;
    source.src = '/stream?' + query.toString();
    streaming = true;
  }
  if (!online) streaming = false;
  for (const camera of latest.cameras) {
    let control = controls.get(camera.name);
    if (!control) {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      const text = document.createElement('span');
      checkbox.type = 'checkbox';
      checkbox.onchange = () => {
        const chosen = latest.cameras.filter(item => selection === null || selection.includes(item.name));
        const next = new Set(chosen.map(item => item.name));
        if (checkbox.checked) next.add(camera.name); else next.delete(camera.name);
        choose([...next]);
      };
      label.append(checkbox, text);
      document.getElementById('cameras').append(label);
      control = {label, checkbox, text};
      controls.set(camera.name, control);
    }
    control.label.className = 'camera' + (camera.stale ? ' stale' : '');
    control.checkbox.checked = names.includes(camera.name);
    control.text.textContent = camera.name + (camera.stale ? ' (stale)' : '');
  }
}
async function update() {
  try {
    const response = await fetch('/status', {cache:'no-store', signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('Status unavailable');
    latest = await response.json();
    if (!storageKey) {
      storageKey = 'sonic-camera-views:' + latest.source;
      layoutKey = 'sonic-camera-layout:' + latest.source;
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey));
        if (Array.isArray(saved) && saved.every(name => typeof name === 'string')) selection = saved;
      } catch (_) {}
      try {
        const saved = JSON.parse(localStorage.getItem(layoutKey));
        if (Array.isArray(saved?.order) && saved.order.every(name => typeof name === 'string')) {
          viewOrder = [...new Set(saved.order)];
        }
        if (Number.isInteger(saved?.columns) && saved.columns >= 0 && saved.columns <= 4) {
          columnPreference = saved.columns; columns.value = String(columnPreference);
        }
      } catch (_) {}
    }
    render();
  } catch (error) {
    document.getElementById('status').textContent = 'Preview server disconnected. Retrying...';
    preview.classList.add('offline');
    streaming = false;
  } finally { setTimeout(update, 1000); }
}
update();
</script></html>"""


class CameraRelay:
    """One subscriber; browsers with the same view selection share encoded mosaics."""

    stale_after = 3.0

    def __init__(self, camera_host="localhost", camera_port=5555, fps=15, width=640):
        self.camera_host = camera_host
        self.camera_port = camera_port
        self.fps = fps
        self.width = width
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.jpeg = None
        self.images = {}
        self.sequence = 0
        self.received = {}
        self.error = None
        self.view_cache = OrderedDict()
        self.encode_lock = threading.Lock()
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

    def wait_frame(self, previous, cameras=None):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous or self.stopped.is_set(), timeout=1.0
            )
            if self.sequence == previous or self.stopped.is_set():
                return previous, None
            sequence = self.sequence
            if cameras is None:
                return sequence, self.jpeg
            # Snapshot references under the lock; arrays are replaced, never mutated.
            images = {name: self.images[name] for name in cameras if name in self.images}
            received = self.received.copy()
        if not images:
            return sequence, None
        key = tuple(sorted(images))
        # Serialize encoders for selected views without blocking the ZMQ receiver.
        with self.encode_lock:
            cached = self.view_cache.get(key)
            if cached is not None and cached[0] == sequence:
                self.view_cache.move_to_end(key)
                return cached
            result = sequence, self._mosaic(images, received, time.monotonic())
            self.view_cache[key] = result
            self.view_cache.move_to_end(key)
            if len(self.view_cache) > 16:
                self.view_cache.popitem(last=False)
            return result

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
                        self.images = images.copy()
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
        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        cameras = tuple(sorted(set(query["camera"]))) if "camera" in query else None
        if cameras is not None:
            if not all(cameras):
                self._respond(b"Camera selection must not be empty", "text/plain", 400)
                return
            available = {camera["name"] for camera in relay.status()["cameras"]}
            if not set(cameras) <= available:
                self._respond(b"Unknown camera view", "text/plain", 404)
                return
        if not relay.status()["online"]:
            self._respond(b"Waiting for camera frames", "text/plain", 503)
            return
        self.connection.settimeout(5.0)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        previous = -1
        last_jpeg = None
        while not relay.stopped.is_set():
            previous, jpeg = relay.wait_frame(previous, cameras)
            if jpeg is None:
                if relay.stopped.is_set():
                    break
                # Keep established browser streams alive through camera outages.
                # A cached-frame heartbeat also detects disconnected clients; the
                # page's independent status poll marks these frames as stale.
                jpeg = last_jpeg
                if jpeg is None:
                    continue
            last_jpeg = jpeg
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"--frame--\r\n")
