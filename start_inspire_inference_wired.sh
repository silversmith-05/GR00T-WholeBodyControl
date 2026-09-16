#!/usr/bin/env bash
set -euo pipefail
INSP_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$INSP_REPO/.venv_teleop/bin/python" "$INSP_REPO/gear_sonic/scripts/launch_inspire_inference.py" "$@"
