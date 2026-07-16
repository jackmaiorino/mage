from __future__ import annotations

import os
import hashlib
import subprocess
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
from mtg_kernel_rl.path_safety import atomic_quarantine, mkdir_no_follow


def _tree_snapshot(root: Path) -> dict[str, tuple[str, int, str | None, int, int, int]]:
    out: dict[str, tuple[str, int, str | None, int, int, int]] = {}
    for path in sorted([root, *root.rglob("*")], key=lambda item: str(item.relative_to(root.parent))):
        rel = str(path.relative_to(root))
        st = path.lstat()
        if path.is_symlink():
            kind = "link"
            digest = None
            size = 0
        elif path.is_dir():
            kind = "dir"
            digest = None
            size = 0
        else:
            kind = "file"
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            size = len(data)
        out[rel] = (kind, size, digest, int(st.st_mtime_ns), int(getattr(st, "st_dev", 0)), int(getattr(st, "st_ino", 0)))
    return out


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
            "/",
            "/a",
            "/data/run",
            "/gpfs/project/run",
            "/lustre/project/run",
            "/srv/mage/run",
            "/@root/secret",
            "/数据/run",
            "/Applications/My App/run",
            "prefix /data/run/root",
            "artifact=/@root/secret",
            "artifact=/ home/jack",
            "artifact=/\thome",
            "artifact= / home/jack",
            "artifact:/数据/run",
            '["/srv/run"]',
            "/home/jack/mage",
            "prefix /tmp/run/root",
            "C:\\Users\\Jack\\IdeaProjects\\mage",
            "\\Users\\Jack\\IdeaProjects\\mage",
            "\\secret",
            "artifact=\\ secret",
            "artifact=\\\thome",
            "artifact= \\ secret",
            "value=\\secret",
            "\\\\server\\share\\run",
            "x='\\\\server\\share\\run'",
            "\\\\.\\C:\\Users\\Jack",
            "\\\\?\\C:\\Users\\Jack",
            "file:///C:/Users/Jack/run",
            "https://example.test/path diagnostic=/home/jack/run",
            "https://example.test/path;diagnostic=/home/jack/run",
            "https://example.test/path?diagnostic=/home/jack/run",
            "https://x;diagnostic=C:\\Users\\Jack",
            "https://exa|mple.test/path",
            "https://exa|mple.test/path?diagnostic=/home/jack/run",
            "https://[::1/path",
            "https://[::gg]/path",
            "https://[::1]:70000/path",
            "https://example.test:/path",
            "https://example.test:bad/path",
            "https://example.test:99999/path",
            "https://-example.test/path",
            "https://example-.test/path",
            "https://example..test/path",
            "https://example[.]test/path",
            "https://example.test\\path",
            "diagnostic\t/home/jack",
            "note;/data",
            "note; / home/jack",
            "note;\u0301/home/jack",
            "note;\u0301\u0302/home/jack",
            "note\u200b/data",
            "note\uff1b/\u6570\u636e",
            "note;\\\\server",
            "note;\\Device\\HarddiskVolume1\\secret",
            "diagnostic=#/home/jack",
            "/|secret",
        ]
        for value in positives:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_training_json_privacy({"metadata": value})
                with self.assertRaises(ValueError):
                    validate_training_json_privacy({value: {"nested": value}})
        negatives = [
            "terminal_reinforce_value/v1",
            "http://example.test",
            "https://example.test/path",
            "https://[2001:db8::1]:443/path",
            "loss = a / b",
            "b48d972b8f2fc56c330c815223c7cb7ef663a2cc45072a203a13e3f00b253f61",
            "train-learner-action/base_seed/episode_index",
            "schema#/properties/run",
            "ordinary prose with / separated words",
            "namespace\u0301/component",
            "namespace\u0301\u0302/component",
        ]
        for value in negatives:
            with self.subTest(value=value):
                validate_training_json_privacy({"metadata": value})

    def test_quarantine_and_mkdir_no_follow_preserve_external_link_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            root = tmp / "root"
            outside = tmp / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "nested").mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"unchanged")
            (outside / "nested" / "sentinel2.txt").write_bytes(b"unchanged2")

            def assert_sentinel() -> None:
                self.assertEqual(_tree_snapshot(outside), outside_snapshot)

            outside_snapshot = _tree_snapshot(outside)

            victim_bad_reason = root / "victim-bad-reason.txt"
            victim_bad_reason.write_bytes(b"x")
            for reason in ("", ".", "..", "bad/path", "bad\\path", "C:bad", "\\\\server\\share", "原因"):
                with self.subTest(reason=reason):
                    with self.assertRaises(ValueError):
                        atomic_quarantine(root, victim_bad_reason, reason)
                    self.assertTrue(victim_bad_reason.exists())
                    assert_sentinel()

            if hasattr(os, "symlink"):
                link = root / "link"
                try:
                    os.symlink(outside, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    link = None  # type: ignore[assignment]
                if link is not None:
                    with self.assertRaises(ValueError):
                        mkdir_no_follow(link / "nested", parents=True, exist_ok=True)
                    assert_sentinel()
                    victim = root / "victim.txt"
                    victim.write_bytes(b"x")
                    q = root / ".quarantine"
                    os.symlink(outside, q, target_is_directory=True)
                    with self.assertRaises(ValueError):
                        atomic_quarantine(root, victim, "reason")
                    self.assertTrue(victim.exists())
                    assert_sentinel()
                    q.unlink()

            if os.name == "nt":
                junction = root / "junction"
                completed = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail("Windows junction coverage could not create mklink /J")
                with self.assertRaises(ValueError):
                    mkdir_no_follow(junction / "nested", parents=True, exist_ok=True)
                assert_sentinel()
                os.rmdir(junction)
                victim = root / "victim-junction.txt"
                victim.write_bytes(b"x")
                q = root / ".quarantine"
                completed = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(q), str(outside)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail("Windows quarantine junction coverage could not create mklink /J")
                with self.assertRaises(ValueError):
                    atomic_quarantine(root, victim, "reason")
                self.assertTrue(victim.exists())
                assert_sentinel()


if __name__ == "__main__":
    unittest.main()
