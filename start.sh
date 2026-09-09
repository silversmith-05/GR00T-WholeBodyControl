source .venv/bin/activate
cd /home/unitree/GR00T-WholeBodyControl

python3 /home/unitree/GR00T-WholeBodyControl/gear_sonic/scripts/launch_data_collection.py \
  --hand-backend inspire \
  --enable-hand-control \
  --inspire-left-ip 192.168.123.211 \
  --inspire-right-ip 192.168.123.210 \
  --inspire-port 6000 \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --task-prompt "pick up the ball" \
  --dataset-name inspire_ball_v4 \
  --inspire-close-angles 250 250 250 250 300