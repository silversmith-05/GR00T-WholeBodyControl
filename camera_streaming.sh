source .venv/bin/activate

python3 ~/GR00T-WholeBodyControl/gear_sonic/scripts/run_camera_web.py \
  --camera-host 192.168.123.164 \
  --host 0.0.0.0 \
  --port 8080