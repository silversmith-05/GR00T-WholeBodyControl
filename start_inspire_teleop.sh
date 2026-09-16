#!/usr/bin/env bash
set -euo pipefail
TELEOP_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$TELEOP_REPO/gear_sonic/scripts/launch_inspire_teleop.py" "$@"
