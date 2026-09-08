```
# bash install_scripts/install_pico.sh

source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

```
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

```
source .venv_teleop/bin/activate

# With full visualization (recommended for first run):
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl

# Without visualization (for headless / onboard practice):
# python gear_sonic/scripts/pico_manager_thread_server.py --manager
```