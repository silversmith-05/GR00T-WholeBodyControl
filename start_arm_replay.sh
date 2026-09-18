#!/usr/bin/env bash
set -euo pipefail
REPLAY_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$REPLAY_REPO/gear_sonic/scripts/launch_arm_replay.py" "$@"
