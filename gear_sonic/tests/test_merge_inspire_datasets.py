"""Offline merge tests with small Parquet files and generated H.264 videos."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear_sonic.scripts import merge_inspire_datasets as merge


def make_video(path, length):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=50)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(length):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_dataset(root, count=3, discarded=(1,)):
    (root / "meta").mkdir(parents=True)
    scalar_features = {key: {"dtype": dtype, "shape": [1], "names": None} for key, dtype in (
        ("episode_index", "int64"), ("index", "int64"), ("frame_index", "int64"),
        ("timestamp", "float32"), ("task_index", "int64"), ("hand.training_valid", "bool"),
    )}
    vectors = {"action.hand": 2, "action.thumb_rotation": 2, "action.motion_token": 64,
               "hand.angle_act": 12, "teleop.smpl_pose": 63}
    features = {**scalar_features, **{key: {"dtype": "float32", "shape": [size], "names": None} for key, size in vectors.items()}}
    features["observation.images.ego_view"] = {"dtype": "video", "shape": [32, 32, 3], "names": ["height", "width", "channel"]}
    info = {
        "codebase_version": "v2.1", "robot_type": None, "fps": 50, "chunks_size": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features, "script_config": {"hand": {"backend": "inspire", "release": [850] * 4}},
        "total_episodes": count, "total_frames": sum(5 + i for i in range(count)),
        "total_videos": count, "total_tasks": 1, "total_chunks": (count - 1) // 2 + 1,
        "splits": {"train": f"0:{count}"}, "discarded_episode_indices": list(discarded),
    }
    episode_rows, stats_rows = [], []
    start = 100  # Deliberately different source global indices to test statistics.
    for index in range(count):
        length = 5 + index
        values = {
            "episode_index": np.full(length, index, dtype=np.int64),
            "index": np.arange(start, start + length, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": (np.arange(length) / 50).astype(np.float32),
            "task_index": np.zeros(length, dtype=np.int64),
            "hand.training_valid": np.array([True, False] + [True] * (length - 2)),
            "action.hand": np.asarray([[0, 0], [0, 1]] + [[0, 1]] * (length - 2), dtype=np.float32),
            "action.thumb_rotation": np.tile(np.array([3, 0], dtype=np.float32), (length, 1)),
            "action.motion_token": np.arange(length * 64, dtype=np.float32).reshape(length, 64),
            "hand.angle_act": np.full((length, 12), 100, dtype=np.float32),
            "teleop.smpl_pose": np.ones((length, 63), dtype=np.float32),
        }
        values["teleop.smpl_pose"][1] = 0
        columns = {key: pa.array(value, type=None) if value.ndim == 1 else pa.array(value.tolist(), type=pa.list_(pa.float32())) for key, value in values.items()}
        table = pa.table(columns).replace_schema_metadata({b"source_format": b"fixture"})
        parquet = merge.get_parquet_path(root, info, index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        for path in merge.get_video_paths(root, info, index).values():
            make_video(path, length)
        episode_rows.append({"episode_index": index, "length": length, "tasks": ["pick up the ball"]})
        stats = {}
        for key, value in values.items():
            stats[key] = {name: np.atleast_1d(operation(value, axis=0)).tolist() for name, operation in (
                ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))}
            stats[key]["count"] = [length]
        stats_rows.append({"episode_index": index, "stats": stats})
        start += length
    merge.write_json(root / "meta/info.json", info)
    merge.write_json(root / "meta/modality.json", {
        "state": {"hand": {"start": 0, "end": 12, "original_key": "hand.angle_act"}},
        "action": {"hand": {"start": 0, "end": 2, "original_key": "action.hand"}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    })
    merge.write_jsonl(root / "meta/episodes.jsonl", episode_rows)
    merge.write_jsonl(root / "meta/episodes_stats.jsonl", stats_rows)
    merge.write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick up the ball"}])


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = [self.root / "batch_a", self.root / "batch_b"]
        for path in self.sources:
            make_dataset(path)
        self.output = self.root / "merged"
        self.config = merge.MergeInspireConfig(self.sources, self.output, eval_count=2, split_seed=42)

    def test_complete_export_preserves_frames_bytes_and_rebuilds_indices(self):
        original_hashes = {path: merge.sha256(path) for root in self.sources for path in root.rglob("*") if path.is_file()}
        report = merge.merge_datasets(self.config)
        self.assertEqual(report["datasets"]["all"]["episodes"], 4)
        self.assertEqual(report["datasets"]["all"]["frames"], 24)
        self.assertEqual(report["datasets"]["all"]["hand_training_valid_false_frames"], 4)
        self.assertEqual(report["datasets"]["all"]["smpl_cleaning_candidate_frames"], 8)
        self.assertEqual(report["datasets"]["all"]["videos"], 4)
        all_info = merge.read_json(self.output / "all/meta/info.json")
        self.assertEqual(all_info["total_chunks"], 2)
        self.assertEqual(all_info["splits"], {"all": "0:4"})
        self.assertNotIn("discarded_episode_indices", all_info)
        identities = {}
        for name in ("all", "train", "eval"):
            root = self.output / name
            entries = merge.read_jsonl(root / "meta/source_episodes.jsonl")
            identities[name] = {(e["source_dataset"], e["source_episode_index"]) for e in entries}
            info = merge.read_json(root / "meta/info.json")
            stats = merge.read_jsonl(root / "meta/episodes_stats.jsonl")
            offset = 0
            for entry in entries:
                old_info = merge.read_json(Path(entry["source_dataset"]) / "meta/info.json")
                original = pq.read_table(merge.get_parquet_path(Path(entry["source_dataset"]), old_info, entry["source_episode_index"]))
                new_index = entry["episode_index"]
                written = pq.read_table(merge.get_parquet_path(root, info, new_index))
                for key in original.column_names:
                    if key not in merge.RENUMBERED_COLUMNS:
                        self.assertTrue(original[key].equals(written[key]), key)
                self.assertEqual(written["index"].to_pylist(), list(range(offset, offset + entry["length"])))
                self.assertEqual(stats[new_index]["stats"]["index"]["min"], [offset])
                self.assertEqual(stats[new_index]["stats"]["episode_index"]["mean"], [float(new_index)])
                self.assertEqual(stats[new_index]["stats"]["episode_index"]["std"], [0.0])
                offset += entry["length"]
                for key, path in merge.get_video_paths(root, info, new_index).items():
                    self.assertEqual(merge.sha256(path), entry["source_video_sha256"][key])
            self.assertTrue(report["datasets"][name]["lerobot_v2_1_metadata_loaded"])
            self.assertFalse((root / "meta/stats.json").exists())
        self.assertFalse(identities["train"] & identities["eval"])
        self.assertEqual(identities["train"] | identities["eval"], identities["all"])
        for path, digest in original_hashes.items():
            self.assertEqual(merge.sha256(path), digest)
        self.assertFalse(list(self.root.glob(".merged.tmp-*")))

    def test_config_task_and_modality_mismatches_fail_before_output(self):
        for filename, change in (
            ("info.json", lambda value: value["script_config"]["hand"].update(release=[700] * 4)),
            ("modality.json", lambda value: value["action"]["hand"].update(end=1)),
            ("info.json", lambda value: value.update(fps=30)),
        ):
            with self.subTest(filename=filename):
                path = self.sources[1] / "meta" / filename
                original = path.read_bytes()
                value = merge.read_json(path)
                change(value)
                merge.write_json(path, value)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    merge.merge_datasets(self.config)
                self.assertFalse(self.output.exists())
                path.write_bytes(original)
        task_path = self.sources[1] / "meta/tasks.jsonl"
        merge.write_jsonl(task_path, [{"task_index": 0, "task": "different task"}])
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            merge.merge_datasets(self.config)

    def test_missing_video_fails_before_output(self):
        next((self.sources[0] / "videos").rglob("episode_000000.mp4")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty source file"):
            merge.merge_datasets(self.config)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_touched(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            merge.merge_datasets(self.config)
        self.assertEqual(marker.read_text(), "keep")

    def test_copy_failure_leaves_no_published_or_partial_bundle(self):
        with patch.object(merge.shutil, "copy2", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                merge.merge_datasets(self.config)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".merged.tmp-*")))

    def test_corrupted_copy_is_detected_before_publication(self):
        def corrupt_copy(source, target):
            Path(target).write_bytes(b"corrupted")
        with patch.object(merge.shutil, "copy2", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "Video bytes changed"):
                merge.merge_datasets(self.config)
        self.assertFalse(self.output.exists())

    def test_invalid_split_duplicate_sources_and_overlap(self):
        for count in (0, 4, 5):
            config = copy.copy(self.config)
            config.eval_count = count
            with self.assertRaisesRegex(ValueError, "eval_count"):
                merge.merge_datasets(config)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge.merge_datasets(merge.MergeInspireConfig([self.sources[0]] * 2, self.output, 1))
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge.merge_datasets(merge.MergeInspireConfig(self.sources, self.sources[0] / "nested", 1))

    def test_seed_42_reproduces_agreed_real_episode_selection(self):
        indices = [
            [2, 3, 4, 8, 11, 12],
            [0, 2, 9, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 69, 70, 71, 72, 75, 77, 78],
            [2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30],
        ]
        groups = [[merge.Episode(Path(f"batch{batch}"), {"episode_index": index}, {}, Path(), {}, {}) for index in group] for batch, group in enumerate(indices)]
        expected = {("batch0", 2), ("batch1", 37), ("batch1", 50), ("batch1", 57), ("batch1", 67), ("batch1", 77), ("batch2", 17), ("batch2", 30)}
        self.assertEqual(merge.choose_evaluation(groups, 8, 42), expected)


if __name__ == "__main__":
    unittest.main()
