"""Merge complete non-discard Inspire episodes into verified LeRobot v2.1 datasets.

Run with the data-collection environment. Only episode_index and index change;
hand flags, SMPL, timestamps and video bytes are preserved. No hardware is used.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.scripts.process_dataset import (
    build_stale_mask,
    get_parquet_path,
    get_video_paths,
)


META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "episodes_stats.jsonl")
RENUMBERED_COLUMNS = ("episode_index", "index")
PRESERVED_COUNTS = ("hand_training_valid_false_frames", "smpl_cleaning_candidate_frames")


@dataclass
class MergeInspireConfig:
    dataset_path: list[Path] = field(default_factory=list)
    """Input LeRobot v2.1 directories, in chronological batch order."""
    output_path: Path | None = None
    """New bundle directory containing all/, train/, eval/ and a merge report."""
    eval_count: int = 8
    """Number of whole episodes held out, allocated proportionally across batches."""
    split_seed: int = 42
    """Seed used by numpy.default_rng for within-batch sampling."""


@dataclass
class Episode:
    source: Path
    meta: dict
    stats: dict
    parquet: Path
    videos: dict[str, Path]
    preserved_counts: dict[str, int]

    @property
    def identity(self) -> tuple[str, int]:
        return str(self.source), self.meta["episode_index"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {path}")
    return path


def indexed_rows(rows: list[dict], key: str, path: Path) -> dict[int, dict]:
    result = {}
    for row in rows:
        index = row[key]
        if type(index) is not int or index < 0 or index in result:
            raise ValueError(f"Invalid or duplicate {key}={index!r}: {path}")
        result[index] = row
    return result


def array(table: pa.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist())


def preservation_counts(table: pa.Table) -> dict[str, int]:
    # Report diagnostics only. Neither flag is a reason to remove a frame.
    return {
        "hand_training_valid_false_frames": int((~array(table, "hand.training_valid").astype(bool)).sum()),
        "smpl_cleaning_candidate_frames": int(build_stale_mask(array(table, "teleop.smpl_pose")).sum()),
    }


def numeric_stats(values: np.ndarray) -> dict:
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


def check_video(path: Path, length: int, fps: float):
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        if stream.frames != length or stream.average_rate is None or float(stream.average_rate) != fps:
            raise ValueError(f"Video frame count/FPS mismatch: {path} ({stream.frames} frames, {stream.average_rate} FPS)")


def preflight(paths: list[Path]):
    """Read and validate every retained source before creating output files."""
    reference = None
    groups = []
    hashes: dict[Path, str] = {}
    batches = []
    arrow_schema = None

    def remember(path: Path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty source file: {path}")
        hashes[path] = sha256(path)

    for root in paths:
        for filename in META_FILES:
            remember(contained_path(root, root / "meta" / filename))
        info = read_json(root / "meta/info.json")
        modality = read_json(root / "meta/modality.json")
        tasks_by_id = indexed_rows(read_jsonl(root / "meta/tasks.jsonl"), "task_index", root)
        tasks = [tasks_by_id[k] for k in sorted(tasks_by_id)]
        if not tasks:
            raise ValueError(f"No task metadata: {root}")
        if info.get("codebase_version") != "v2.1" or info.get("script_config", {}).get("hand", {}).get("backend") != "inspire":
            raise ValueError(f"Expected an Inspire LeRobot v2.1 dataset: {root}")
        if info.get("fps", 0) <= 0 or info.get("chunks_size", 0) <= 0:
            raise ValueError(f"Invalid FPS or chunks_size: {root}")
        comparison = {key: info.get(key) for key in (
            "codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path", "video_keys", "features", "script_config",
        )}
        comparison.update(modality=modality, tasks=tasks)
        if reference is None:
            reference = (info, modality, tasks, comparison)
        else:
            differences = [key for key in comparison if comparison[key] != reference[3][key]]
            if differences:
                raise ValueError(f"Dataset configuration mismatch in {root}: {', '.join(differences)}")

        episodes = indexed_rows(read_jsonl(root / "meta/episodes.jsonl"), "episode_index", root)
        stats = indexed_rows(read_jsonl(root / "meta/episodes_stats.jsonl"), "episode_index", root)
        discarded = set(info.get("discarded_episode_indices", []))
        if not discarded.issubset(episodes):
            raise ValueError(f"Discarded episode missing from episode metadata: {root}")
        if info["total_episodes"] != len(episodes) or info["total_frames"] != sum(ep["length"] for ep in episodes.values()):
            raise ValueError(f"Source totals disagree with episodes.jsonl: {root}")
        group = []
        for old_index in sorted(set(episodes) - discarded):
            meta = episodes[old_index]
            length = meta["length"]
            if type(length) is not int or length <= 0:
                raise ValueError(f"Invalid episode length: {root}, episode {old_index}")
            if old_index not in stats:
                raise ValueError(f"Missing episode statistics: {root}, episode {old_index}")
            parquet = contained_path(root, get_parquet_path(root, info, old_index))
            videos = {key: contained_path(root, path) for key, path in get_video_paths(root, info, old_index).items()}
            if not videos:
                raise ValueError(f"No camera features: {root}")
            remember(parquet)
            table = pq.read_table(parquet)
            if table.num_rows != length:
                raise ValueError(f"Parquet length disagrees with episode metadata: {parquet}")
            expected_columns = {key for key, spec in info["features"].items() if spec["dtype"] not in ("video", "image")}
            if set(table.column_names) != expected_columns:
                raise ValueError(f"Parquet columns disagree with features: {parquet}")
            if arrow_schema is None:
                arrow_schema = table.schema
            elif not arrow_schema.equals(table.schema, check_metadata=False):
                raise ValueError(f"Parquet schema mismatch: {parquet}")
            if not np.array_equal(array(table, "frame_index").reshape(-1), np.arange(length)):
                raise ValueError(f"Non-contiguous source frame_index: {parquet}")
            if not np.all(array(table, "episode_index") == old_index):
                raise ValueError(f"Source episode_index disagrees with filename: {parquet}")
            timestamps = array(table, "timestamp").reshape(-1)
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"Invalid or non-increasing timestamps: {parquet}")
            if not set(array(table, "task_index").reshape(-1).tolist()).issubset(tasks_by_id):
                raise ValueError(f"Unknown task_index: {parquet}")
            ep_stats = stats[old_index]["stats"]
            missing_stats = {key for key in expected_columns if info["features"][key]["dtype"] != "string"} - ep_stats.keys()
            if missing_stats:
                raise ValueError(f"Missing feature statistics in {parquet}: {sorted(missing_stats)}")
            for key in expected_columns & ep_stats.keys():
                if ep_stats[key].get("count") != [length]:
                    raise ValueError(f"Statistics count mismatch for {key}: {parquet}")
            for video in videos.values():
                remember(video)
                check_video(video, length, info["fps"])
            group.append(Episode(root, meta, ep_stats, parquet, videos, preservation_counts(table)))
        groups.append(group)
        batches.append({
            "source_dataset": str(root), "source_episodes": len(episodes),
            "discarded_episode_indices": sorted(discarded), "retained_episode_indices": [ep.meta["episode_index"] for ep in group],
            "retained_frames": sum(ep.meta["length"] for ep in group),
        })
    return reference[:3], groups, hashes, batches


def choose_evaluation(groups: list[list[Episode]], count: int, seed: int) -> set[tuple[str, int]]:
    """Largest-remainder allocation, with input order breaking allocation ties."""
    sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    if not 0 < count < sizes.sum():
        raise ValueError("eval_count must be positive and smaller than the number of retained episodes")
    quotas = sizes * (count / int(sizes.sum()))
    allocations = np.floor(quotas).astype(int)
    for index in sorted(range(len(groups)), key=lambda j: (-(quotas[j] - allocations[j]), j))[:count - int(allocations.sum())]:
        allocations[index] += 1
    rng = np.random.default_rng(seed)
    result = set()
    for group, allocation in zip(groups, allocations):
        if allocation:
            indices = rng.choice([ep.meta["episode_index"] for ep in group], size=int(allocation), replace=False)
            result.update((str(group[0].source), int(index)) for index in indices)
    return result


def renumber(table: pa.Table, episode_index: int, frame_start: int) -> pa.Table:
    for key, values in (
        ("episode_index", np.full(table.num_rows, episode_index, dtype=np.int64)),
        ("index", np.arange(frame_start, frame_start + table.num_rows, dtype=np.int64)),
    ):
        position = table.schema.get_field_index(key)
        table = table.set_column(position, table.schema.field(position), pa.array(values, type=table.schema.field(position).type))
    return table


def write_dataset(root: Path, episodes: list[Episode], reference: tuple, hashes: dict[Path, str], evaluation: set, merged_indices: dict, name: str):
    reference_info, modality, tasks = reference
    info = copy.deepcopy(reference_info)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    episode_rows, stats_rows, sources = [], [], []
    frame_start = 0
    for new_index, ep in enumerate(episodes):
        table = renumber(pq.read_table(ep.parquet), new_index, frame_start)
        dest = contained_path(root, get_parquet_path(root, info, new_index))
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        for key, path in get_video_paths(root, info, new_index).items():
            path = contained_path(root, path)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ep.videos[key], path)
        meta = copy.deepcopy(ep.meta)
        meta["episode_index"] = new_index
        episode_rows.append(meta)
        stats = copy.deepcopy(ep.stats)
        for key in RENUMBERED_COLUMNS:
            stats[key] = numeric_stats(array(table, key).reshape(-1))
        stats_rows.append({"episode_index": new_index, "stats": stats})
        sources.append({
            "episode_index": new_index, "merged_episode_index": merged_indices[ep.identity],
            "source_dataset": str(ep.source), "source_episode_index": ep.meta["episode_index"],
            "split": "eval" if ep.identity in evaluation else "train", "length": table.num_rows,
            "source_parquet_sha256": hashes[ep.parquet],
            "source_video_sha256": {key: hashes[path] for key, path in ep.videos.items()},
        })
        frame_start += table.num_rows
    split_name = {"all": "all", "train": "train", "eval": "validation"}[name]
    info.update(total_episodes=len(episodes), total_frames=frame_start, total_tasks=len(tasks),
                total_videos=sum(len(ep.videos) for ep in episodes),
                total_chunks=(len(episodes) - 1) // info["chunks_size"] + 1,
                splits={split_name: f"0:{len(episodes)}"})
    info.pop("discarded_episode_indices", None)
    write_json(meta_dir / "info.json", info)
    write_json(meta_dir / "modality.json", modality)
    write_jsonl(meta_dir / "tasks.jsonl", tasks)
    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows)
    write_jsonl(meta_dir / "source_episodes.jsonl", sources)


def verify_dataset(root: Path, episodes: list[Episode], reference: tuple, hashes: dict[Path, str], evaluation: set, merged_indices: dict, name: str) -> dict:
    """Verify actual written contents, not just the in-memory export inputs."""
    info = read_json(root / "meta/info.json")
    rows = read_jsonl(root / "meta/episodes.jsonl")
    stats = read_jsonl(root / "meta/episodes_stats.jsonl")
    sources = read_jsonl(root / "meta/source_episodes.jsonl")
    if not len(episodes) == len(rows) == len(stats) == len(sources):
        raise ValueError(f"Output metadata episode count mismatch: {root}")
    expected_info = copy.deepcopy(reference[0])
    expected_info.pop("discarded_episode_indices", None)
    expected_frames = sum(ep.meta["length"] for ep in episodes)
    expected_videos = sum(len(ep.videos) for ep in episodes)
    split_name = {"all": "all", "train": "train", "eval": "validation"}[name]
    expected_info.update(total_episodes=len(episodes), total_frames=expected_frames,
                         total_videos=expected_videos, total_tasks=len(reference[2]),
                         total_chunks=(len(episodes) - 1) // expected_info["chunks_size"] + 1,
                         splits={split_name: f"0:{len(episodes)}"})
    if info != expected_info or read_json(root / "meta/modality.json") != reference[1] or read_jsonl(root / "meta/tasks.jsonl") != reference[2]:
        raise ValueError(f"Output configuration mismatch: {root}")
    counts = dict.fromkeys(PRESERVED_COUNTS, 0)
    frame_start = 0
    for new_index, ep in enumerate(episodes):
        original = pq.read_table(ep.parquet)
        output = pq.read_table(get_parquet_path(root, info, new_index))
        expected_meta = {**ep.meta, "episode_index": new_index}
        if rows[new_index] != expected_meta or not output.schema.equals(original.schema, check_metadata=True):
            raise ValueError(f"Output episode metadata/schema mismatch: {root}, episode {new_index}")
        if output.num_rows != original.num_rows:
            raise ValueError(f"Output length mismatch: {root}, episode {new_index}")
        for key in original.column_names:
            if key not in RENUMBERED_COLUMNS and not original[key].equals(output[key]):
                raise ValueError(f"Preserved column changed: {root}, episode {new_index}, {key}")
        if not np.all(array(output, "episode_index") == new_index) or not np.array_equal(array(output, "index").reshape(-1), np.arange(frame_start, frame_start + output.num_rows)):
            raise ValueError(f"Incorrect output indices: {root}, episode {new_index}")
        expected_stats = copy.deepcopy(ep.stats)
        for key in RENUMBERED_COLUMNS:
            expected_stats[key] = numeric_stats(array(output, key).reshape(-1))
        if stats[new_index] != {"episode_index": new_index, "stats": expected_stats}:
            raise ValueError(f"Output statistics mismatch: {root}, episode {new_index}")
        expected_source = {
            "episode_index": new_index, "merged_episode_index": merged_indices[ep.identity],
            "source_dataset": str(ep.source), "source_episode_index": ep.meta["episode_index"],
            "split": "eval" if ep.identity in evaluation else "train", "length": output.num_rows,
            "source_parquet_sha256": hashes[ep.parquet],
            "source_video_sha256": {key: hashes[path] for key, path in ep.videos.items()},
        }
        if sources[new_index] != expected_source:
            raise ValueError(f"Source mapping mismatch: {root}, episode {new_index}")
        for key, path in get_video_paths(root, info, new_index).items():
            if sha256(path) != hashes[ep.videos[key]]:
                raise ValueError(f"Video bytes changed: {path}")
        preserved = preservation_counts(output)
        if preserved != ep.preserved_counts:
            raise ValueError(f"Preserved diagnostic counts changed: {root}, episode {new_index}")
        for key in counts:
            counts[key] += preserved[key]
        frame_start += output.num_rows
    if len(list((root / "data").rglob("*.parquet"))) != len(episodes) or len(list((root / "videos").rglob("*.mp4"))) != expected_videos:
        raise ValueError(f"Output file count mismatch: {root}")
    # Use the installed LeRobot v2.1 implementation, without invoking its
    # constructor's download fallback or constructing any robot/video dataset.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    metadata = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    metadata.root = root
    metadata.repo_id = f"local/{name}"
    metadata.load_metadata()
    if set(metadata.episodes) != set(range(len(episodes))) or set(metadata.episodes_stats) != set(metadata.episodes):
        raise ValueError(f"LeRobot metadata load mismatch: {root}")
    return {
        "episodes": len(episodes), "frames": frame_start, "videos": expected_videos, **counts,
        "parquet_columns_verified": True, "video_sha256_verified": True,
        "source_mapping_verified": True, "lerobot_v2_1_metadata_loaded": True,
    }


def merge_datasets(config: MergeInspireConfig) -> dict:
    if not config.dataset_path or config.output_path is None:
        raise ValueError("--dataset-path and --output-path are required")
    paths = [Path(path).resolve() for path in config.dataset_path]
    output = Path(config.output_path).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    output = output.resolve()
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source dataset directories")
    for path in paths:
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"Output and source directories overlap: {path}")
    reference, groups, hashes, batches = preflight(paths)
    evaluation = choose_evaluation(groups, config.eval_count, config.split_seed)
    all_episodes = [ep for group in groups for ep in group]
    datasets = {
        "all": all_episodes,
        "train": [ep for ep in all_episodes if ep.identity not in evaluation],
        "eval": [ep for ep in all_episodes if ep.identity in evaluation],
    }
    merged_indices = {ep.identity: index for index, ep in enumerate(all_episodes)}
    train_ids = {ep.identity for ep in datasets["train"]}
    eval_ids = {ep.identity for ep in datasets["eval"]}
    if train_ids & eval_ids or train_ids | eval_ids != set(merged_indices):
        raise ValueError("Training/evaluation partition does not cover the retained episodes exactly")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        results = {}
        for name, episodes in datasets.items():
            print(f"Writing and verifying {name}: {len(episodes)} episodes", flush=True)
            write_dataset(staging / name, episodes, reference, hashes, evaluation, merged_indices, name)
            results[name] = verify_dataset(staging / name, episodes, reference, hashes, evaluation, merged_indices, name)
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise ValueError(f"Source changed during export; output will not be published: {path}")
        report = {
            "format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output), "source_batches": batches,
            "policy": {
                "remove_discarded_episodes": True, "remove_frames": False,
                "remove_stale_smpl": False, "filter_hand_training_valid": False,
                "renumbered_columns": list(RENUMBERED_COLUMNS), "preserve_timestamps": True,
                "video_copy": "byte-for-byte copy; no re-encoding",
                "preserve_thumb_rotation": True, "gr00t_stats_generated": False,
            },
            "split": {
                "seed": config.split_seed, "eval_count": config.eval_count,
                "allocation": "proportional largest remainder; ties follow input batch order",
                "sampling": "numpy.default_rng(seed).choice(sorted source episode IDs, replace=False), in input batch order",
                "numpy_version": np.__version__, "unit": "complete source episode",
                "evaluation_episodes": [{"source_dataset": str(ep.source), "source_episode_index": ep.meta["episode_index"]} for ep in datasets["eval"]],
                "disjoint_and_complete": True,
            },
            "datasets": results,
            "verification": {
                "passed": True, "source_files_verified_unchanged": len(hashes),
                "source_sha256": {str(path): digest for path, digest in hashes.items()},
            },
            "training_note": "Train only from train/. Generate GR00T stats.json from train/ after configuring the 66-D actions. Evaluate eval/ with training/checkpoint normalization. all/ is the complete archive. No training was started.",
        }
        write_json(staging / "merge_report.json", report)
        (staging / "README.md").write_text(
            "# Inspire merged dataset\n\n"
            "- `all/`: complete non-discard archive.\n"
            "- `train/`: training input.\n"
            "- `eval/`: held-out complete episodes.\n\n"
            "Original timestamps, action values, SMPL and hand validity flags are preserved. "
            "Only episode_index and global index are renumbered. Videos are copied without re-encoding.\n\n"
            "See merge_report.json for counts, split selection and checksums; each dataset has "
            "meta/source_episodes.jsonl for provenance. Generate GR00T statistics from train/ only "
            "after configuring the 66-D actions. Use training/checkpoint normalization for evaluation.\n",
            encoding="utf-8",
        )
        if os.path.lexists(output):
            raise FileExistsError(f"Output appeared during export; refusing to overwrite: {output}")
        staging.rename(output)
        print(f"Verified bundle published: {output}", flush=True)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    merge_datasets(tyro.cli(MergeInspireConfig))
