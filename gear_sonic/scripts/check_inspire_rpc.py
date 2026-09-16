"""Verify the lightweight client against a real local policy using recorded inputs only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.inference.inspire_bridge import PolicyClient, array, validate_modality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recorded-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--requests", type=int, default=30)
    p.add_argument("--budget-ms", type=float, default=320)
    args = p.parse_args()
    if args.requests < 1 or not np.isfinite(args.budget_ms) or args.budget_ms <= 0:
        raise ValueError("requests and budget-ms must be positive")
    if (args.output_dir / "rpc_report.json").exists():
        raise FileExistsError(args.output_dir / "rpc_report.json")
    obs = {
        "state": {},
        "video": {},
        "language": json.loads((args.recorded_dir / "recorded_language.json").read_text()),
    }
    with np.load(args.recorded_dir / "recorded_observation.npz", allow_pickle=False) as tensors:
        for key in tensors.files:
            modality, name = key.split(".", 1)
            obs[modality][name] = tensors[key]
    client = PolicyClient(args.host, args.port, timeout_ms=30000)
    latencies, warmup = [], []
    try:
        client.call("ping")
        validate_modality(client.call("get_modality_config"))
        for idx in range(args.requests + 3):
            start = time.monotonic()
            actions, _ = client.get_action(obs)
            duration = time.monotonic() - start
            array(actions["motion_token"], (1, 40, 64), "motion_token")
            array(actions["hand"], (1, 40, 2), "hand")
            (warmup if idx < 3 else latencies).append(duration * 1000)
        report = dict(
            passed=True,
            requests=args.requests,
            hardware_commands_sent=False,
            shapes={key: list(value.shape) for key, value in actions.items()},
            warmup_ms=warmup,
            latency_ms=latencies,
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            meets_320ms_budget=bool(np.percentile(latencies, 95) < 320),
            budget_ms=args.budget_ms,
            meets_budget=bool(np.percentile(latencies, 95) < args.budget_ms),
            binary_hand_values=np.unique(actions["hand"] >= 0.5).astype(int).tolist(),
        )
    finally:
        client.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rpc_report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
