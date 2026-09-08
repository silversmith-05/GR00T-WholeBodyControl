"""Standalone camera browser preview; also started by launch_data_collection.py.

Usage (from repo root):
    python gear_sonic/scripts/run_camera_web.py --camera-host 192.168.123.164

Open http://localhost:8080. Use --host 0.0.0.0 for access from other computers.
"""

import argparse
import os
from pathlib import Path
import signal
import sys


def _bootstrap_venv():
    try:
        import cv2  # noqa: F401
        import msgpack_numpy  # noqa: F401
        import zmq  # noqa: F401
        import gear_sonic  # noqa: F401
    except ImportError:
        python = Path(__file__).resolve().parents[2] / ".venv_data_collection/bin/python"
        if python.exists() and Path(sys.prefix) != python.parent.parent:
            os.execv(str(python), [str(python), *sys.argv])
        raise SystemExit("Camera dependencies missing. Run: bash install_scripts/install_data_collection.sh")


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _port(value):
    number = _positive_int(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("must be in 1..65535")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-host", default="localhost", help="ZMQ camera server host")
    parser.add_argument("--camera-port", type=_port, default=5555, help="ZMQ camera server port (default: 5555)")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=_port, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--fps", type=_positive_int, default=15, help="Maximum preview FPS (default: 15)")
    parser.add_argument("--width", type=_positive_int, default=640, help="Maximum tile width (default: 640)")
    args = parser.parse_args(argv)

    _bootstrap_venv()
    from gear_sonic.camera.web_server import CameraRelay, CameraWebServer

    relay = CameraRelay(args.camera_host, args.camera_port, args.fps, args.width)
    try:
        server = CameraWebServer((args.host, args.port), relay)
    except OSError as exc:
        parser.exit(1, f"Cannot start camera web preview on {args.host}:{args.port}: {exc}\n"
                    "Choose another --port or stop the existing preview service.\n")

    def stop(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"Camera source: tcp://{args.camera_host}:{args.camera_port}", flush=True)
    print(f"Browser preview: http://{display_host}:{args.port}", flush=True)
    if args.host == "0.0.0.0":
        print(f"Other computers: http://<this computer's IP>:{args.port}", flush=True)
    print("Waiting for camera frames. Ctrl+C stops the preview.", flush=True)
    try:
        with server:
            relay.start()
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping camera web preview.", flush=True)
    finally:
        relay.close()


if __name__ == "__main__":
    main()
