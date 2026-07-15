from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mtg_kernel_rl.artifacts import (
    canonical_json_bytes,
    read_json_file,
    rebuild_derived_caches,
    require_new_or_empty_dir,
    sha256_bytes,
    write_json_atomic,
)
from mtg_kernel_rl.artifact_io import MAX_SMALL_JSON_BYTES, read_authoritative_json, validate_training_json_privacy


class ArtifactTest(unittest.TestCase):
    def test_canonical_json_atomic_roundtrip_and_strict_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            path = tmp / "a.json"
            value = {"b": [2, 3], "a": {"x": "y"}}
            digest = write_json_atomic(path, value)
            self.assertEqual(digest, sha256_bytes(canonical_json_bytes(value)))
            self.assertEqual(read_json_file(path), value)
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json_file(path)
            with self.assertRaises(ValueError):
                canonical_json_bytes({"bad": float("nan")})

    def test_fresh_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            require_new_or_empty_dir(tmp / "new")
            occupied = tmp / "occupied"
            occupied.mkdir()
            (occupied / "x").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_new_or_empty_dir(occupied)

    def test_derived_caches_rebuild_from_update_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            records = [
                {
                    "schema": "kernel_rl_train_update_record/v2",
                    "run_digest": "r",
                    "update": 0,
                    "parent_head": None,
                    "episode_start": 0,
                    "episode_count": 0,
                    "episode_end_exclusive": 0,
                    "optimizer_step": False,
                    "learner_decision_count": 0,
                    "loss": {"policy_sum_hex": None, "value_sum_hex": None, "loss_hex": None},
                    "episode_summaries": [],
                    "post_update_logical_sha256": "h0",
                },
                {
                    "schema": "kernel_rl_train_update_record/v2",
                    "run_digest": "r",
                    "update": 1,
                    "parent_head": "h0",
                    "episode_start": 0,
                    "episode_count": 1,
                    "episode_end_exclusive": 1,
                    "optimizer_step": True,
                    "learner_decision_count": 2,
                    "loss": {"policy_sum_hex": "0x1.0p+0", "value_sum_hex": "0x1.0p+0", "loss_hex": "0x1.0p+0"},
                    "episode_summaries": [
                        {
                            "schema": "kernel_rl_train_episode_summary/v2",
                            "episode": 0,
                            "env_seed": 1,
                            "learner_seat": "p0",
                            "terminal_outcome": "p0_win",
                            "winner": "p0",
                            "learner_return": 1,
                            "decision_count": 2,
                            "learner_decision_count": 2,
                            "opponent_decision_count": 0,
                            "trajectory_digest": "d",
                        }
                    ],
                    "post_update_logical_sha256": "h1",
                },
            ]
            latest = {"schema": "kernel_rl_train_latest/v2", "update": 1, "run_digest": "r", "head": "head"}
            rebuild_derived_caches(tmp, records, latest)
            self.assertIn('"episode":0', (tmp / "episodes.jsonl").read_text(encoding="utf-8"))
            summary = read_json_file(tmp / "summary.json")
            self.assertEqual(summary["generations"], 2)
            self.assertEqual(summary["completed_training_updates"], 1)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["learner_wins"], 1)
            self.assertEqual(summary["optimizer_steps"], 1)

    def test_authoritative_json_rejects_oversized_noncanonical_duplicate_and_huge_int(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            path = tmp / "latest.json"
            path.write_bytes(b'{"schema":"x","schema":"y"}\n')
            with self.assertRaises(ValueError):
                read_authoritative_json(path, "latest")
            path.write_bytes(b'{ "schema":"x" }\n')
            with self.assertRaises(ValueError):
                read_authoritative_json(path, "latest")
            path.write_bytes(b'{"n":' + b"9" * 200 + b"}\n")
            with self.assertRaises(ValueError):
                read_json_file(path)
            path.write_bytes(b'{"x":"' + b"a" * (MAX_SMALL_JSON_BYTES + 1) + b'"}\n')
            with self.assertRaises(ValueError):
                read_authoritative_json(path, "latest")

    def test_privacy_scan_rejects_cross_platform_absolute_paths_without_version_false_positives(self) -> None:
        positives = [
            "/home/jack/mage",
            "prefix /tmp/run/root",
            "C:\\Users\\Jack\\IdeaProjects\\mage",
            "\\Users\\Jack\\IdeaProjects\\mage",
            "\\\\server\\share\\run",
            "\\\\?\\C:\\Users\\Jack",
            "file:///C:/Users/Jack/run",
        ]
        for value in positives:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_training_json_privacy({"metadata": value})
        negatives = [
            "terminal_reinforce_value/v1",
            "https://example.test/path",
            "loss = a / b",
            "b48d972b8f2fc56c330c815223c7cb7ef663a2cc45072a203a13e3f00b253f61",
            "train-learner-action/base_seed/episode_index",
        ]
        for value in negatives:
            with self.subTest(value=value):
                validate_training_json_privacy({"metadata": value})


if __name__ == "__main__":
    unittest.main()
