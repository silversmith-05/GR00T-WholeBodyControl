"""Run with Isaac-GR00T's Python 3.12. Recorded data only; no robot connections."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gr00t-repo", type=Path, default=Path("/home/pku/workspace/Isaac-GR00T"))
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--initial-episode", type=int, default=0)
    p.add_argument("--initial-frame", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--export-observation", action="store_true")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int)
    args = p.parse_args()
    sys.path.insert(0, str(args.gr00t_repo))
    from examples.G1Inspire.g1_inspire_config import ACTION_LAYOUT, STATE_LAYOUT, config_dict
    import numpy as np
    import pyarrow.parquet as pq

    model = args.model_path.resolve()
    config = json.loads((model / "processor_config.json").read_text())["processor_kwargs"]
    selected = json.loads(json.dumps(config["modality_configs"]["new_embodiment"]))
    # Processor JSON uses enum names; config_dict uses enum values.
    for action in selected["action"]["action_configs"]:
        for key in ("rep", "type", "format"):
            action[key] = action[key].lower()
    if selected != config_dict():
        raise ValueError("Checkpoint modality does not match G1 Inspire")
    for key, expected in (("max_state_dim", 132), ("max_action_dim", 132), ("max_action_horizon", 40)):
        if config[key] != expected:
            raise ValueError(f"Unexpected processor {key}")
    if not config["use_percentiles"]:
        raise ValueError("Checkpoint does not use training percentile normalization")
    files = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    from safetensors import safe_open

    for shard in sorted(set(files.values())):
        with safe_open(model / shard, framework="numpy") as stream:
            if set(stream.keys()) != {k for k, v in files.items() if v == shard}:
                raise ValueError(f"Checkpoint shard/index mismatch: {shard}")
    if (model / "snapshot_manifest.json").exists():
        manifest = json.loads((model / "snapshot_manifest.json").read_text())
        for filename, size in manifest["files"].items():
            if (model / filename).stat().st_size != size:
                raise ValueError(f"Snapshot file size mismatch: {filename}")
    for name in ("config.json", "statistics.json", "embodiment_id.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    train = args.dataset_root / "train"
    statistics = json.loads((model / "statistics.json").read_text())["new_embodiment"]
    data_stats = json.loads((train / "meta/stats.json").read_text())
    for modality, layout in (("state", STATE_LAYOUT), ("action", ACTION_LAYOUT)):
        for key, (column, start, end) in layout.items():
            for field, value in statistics[modality][key].items():
                np.testing.assert_allclose(
                    value, np.asarray(data_stats[column][field])[start:end], rtol=1e-5, atol=1e-6
                )
    info = json.loads((train / "meta/info.json").read_text())
    source = train / info["data_path"].format(
        episode_index=args.initial_episode, episode_chunk=args.initial_episode // info["chunks_size"]
    )
    table = pq.read_table(source, columns=["action.motion_token"])
    if not 0 <= args.initial_frame < len(table):
        raise ValueError("Initial frame is outside the selected episode")
    token = table["action.motion_token"][args.initial_frame].as_py()
    if len(token) != 64 or not np.isfinite(token).all() or np.max(np.abs(token)) > 1.25:
        raise ValueError("Invalid initial motion token")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = dict(
        token=token,
        source=dict(
            dataset=str(train.resolve()),
            episode=args.initial_episode,
            frame=args.initial_frame,
            parquet=str(source.resolve()),
            parquet_sha256=sha(source),
        ),
    )
    (args.output_dir / "initial_token.json").write_text(json.dumps(initial, indent=2) + "\n")
    report = dict(
        passed=True,
        model_path=str(model),
        configuration=config_dict(),
        statistics_sha256=sha(model / "statistics.json"),
        metadata_sha256={
            name: sha(model / name) for name in ("config.json", "processor_config.json", "embodiment_id.json")
        },
        gpu="not_run",
        hardware_commands_sent=False,
    )
    if args.gpu or args.server_port or args.export_observation:
        from examples.G1Inspire.g1_inspire_config import G1_INSPIRE_CONFIG
        from examples.G1Inspire.run import first_observation, load_policy
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        obs = first_observation(LeRobotEpisodeLoader(train, G1_INSPIRE_CONFIG))
        if args.export_observation:
            tensors = {
                f"{modality}.{key}": value
                for modality in ("state", "video")
                for key, value in obs[modality].items()
            }
            np.savez_compressed(args.output_dir / "recorded_observation.npz", **tensors)
            (args.output_dir / "recorded_language.json").write_text(json.dumps(obs["language"]))
    if args.gpu or args.server_port:
        if args.server_port:
            from gr00t.policy.server_client import PolicyClient

            policy = PolicyClient(host=args.server_host, port=args.server_port, timeout_ms=30000)
        else:
            policy = load_policy(model, args.dataset_root, "cuda:0")
        latencies = []
        for _ in range(3):
            started = time.monotonic()
            actions, _ = policy.get_action(obs)
            latencies.append(time.monotonic() - started)
            for key, shape in (("motion_token", (1, 40, 64)), ("hand", (1, 40, 2))):
                if np.asarray(actions[key]).shape != shape or not np.isfinite(actions[key]).all():
                    raise ValueError(f"Invalid model output: {key}")
        report.update(
            gpu="passed",
            output_shapes={k: list(np.asarray(v).shape) for k, v in actions.items()},
            latency_ms=[v * 1000 for v in latencies],
            binary_hand_values=np.unique(np.asarray(actions["hand"]) >= 0.5).astype(int).tolist(),
        )
        if args.server_port:
            policy.close()
    (args.output_dir / "checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
