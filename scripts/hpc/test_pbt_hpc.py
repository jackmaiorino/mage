import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_PATH = REPO_ROOT / "scripts/hpc/run_spy_pbt_native.py"
TARGETS_PATH = REPO_ROOT / "scripts/hpc/generate_prometheus_targets.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self, pid: int = 1234, return_code=None):
        self.pid = pid
        self._return_code = return_code

    def poll(self):
        return self._return_code


class NativeOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_module("run_spy_pbt_native_test", NATIVE_PATH)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.registry_path = self.temp_path / "registry.json"
        self.registry_path.write_text("[]", encoding="utf-8")
        env = {
            "MAGE_MVN_CMD": "echo",
            "REGISTRY_PATH": str(self.registry_path),
            "ORCH_RUN_ID": "test-run",
            "MAGE_DB_DIR": str(self.temp_path / "db-root"),
        }
        self.env_patch = mock.patch.dict(os.environ, env, clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def make_orchestrator(self):
        orch = self.native.NativeOrchestrator()
        reports_root = self.temp_path / "reports" / "runs" / orch.run_id
        compat_root = self.temp_path / "reports"
        orch.repo_root = self.temp_path
        orch.compat_reports_root = compat_root
        orch.runs_root = compat_root / "runs"
        orch.reports_root = reports_root
        orch.trainer_logs_dir = reports_root / "trainers"
        orch.shared_gpu_logs_dir = reports_root / "gpu_hosts"
        orch.generated_decklists_dir = reports_root / "generated_decklists"
        orch.pbt_state_path = reports_root / "pbt_state.json"
        orch.orchestrator_status_path = reports_root / "orchestrator_status.json"
        orch.compat_pbt_state_path = compat_root / "pbt_state.json"
        orch.compat_orchestrator_status_path = compat_root / "orchestrator_status.json"
        orch.latest_run_path = compat_root / "latest_run.json"
        orch.db_root = self.temp_path / "db-root"
        return orch

    def test_load_profiles_respects_mode_over_train_enabled(self):
        payload = [
            {
                "profile": "FrozenProfile",
                "active": True,
                "train_enabled": True,
                "mode": "frozen",
                "priority": 1,
            },
            {
                "profile": "TrainableProfile",
                "active": True,
                "train_enabled": True,
                "priority": 2,
            },
        ]
        self.registry_path.write_text(json.dumps(payload), encoding="utf-8")
        orch = self.make_orchestrator()

        rows = orch.load_profiles()
        frozen = next(row for row in rows if row["profile"] == "FrozenProfile")
        trainable = next(row for row in rows if row["profile"] == "TrainableProfile")

        self.assertEqual("frozen", frozen["mode"])
        self.assertFalse(frozen["train_enabled"])
        self.assertTrue(trainable["train_enabled"])

    def test_stage_and_restore_profile_model_replacement(self):
        orch = self.make_orchestrator()
        winner_models = orch.profile_models_dir("Winner")
        loser_models = orch.profile_models_dir("Loser")
        winner_models.mkdir(parents=True, exist_ok=True)
        loser_models.mkdir(parents=True, exist_ok=True)

        winner_latest = winner_models / "model_latest.pt"
        loser_latest = loser_models / "model_latest.pt"
        loser_model = loser_models / "model.pt"
        winner_latest.write_text("winner", encoding="utf-8")
        loser_latest.write_text("loser-latest", encoding="utf-8")
        loser_model.write_text("loser-model", encoding="utf-8")

        restore_info = orch.stage_profile_model_replacement("Winner", "Loser")
        self.assertEqual("winner", loser_latest.read_text(encoding="utf-8"))
        self.assertEqual("winner", loser_model.read_text(encoding="utf-8"))

        orch.restore_profile_model_files(restore_info)
        self.assertEqual("loser-latest", loser_latest.read_text(encoding="utf-8"))
        self.assertEqual("loser-model", loser_model.read_text(encoding="utf-8"))

    def test_write_orchestrator_status_includes_run_scoped_fields(self):
        orch = self.make_orchestrator()
        orch.active_profiles = [
            {
                "profile": "ProfileA",
                "mode": "",
                "train_enabled": True,
                "population_group": "grp",
                "priority": 1,
                "target_winrate": 0.6,
                "deck_path": "deckA",
            },
            {
                "profile": "ProfileB",
                "mode": "frozen",
                "train_enabled": False,
                "population_group": "grp",
                "priority": 2,
                "target_winrate": 0.6,
                "deck_path": "deckB",
            },
        ]
        orch.selected_profiles = [orch.active_profiles[0]]
        orch.shared_gpu_hosts = {
            0: {
                "gpu_id": "0",
                "port": 26100,
                "metrics_port": 27100,
                "process": FakeProcess(pid=4321, return_code=None),
                "stdout_log": str(self.temp_path / "gpu.stdout.log"),
                "stderr_log": str(self.temp_path / "gpu.stderr.log"),
            }
        }
        trainer_state = types.SimpleNamespace(
            process=FakeProcess(pid=1234, return_code=None),
            restart_count=1,
            consecutive_failures=0,
            last_restart_reason="",
            launched_at_utc="2026-03-07T00:00:00Z",
            metrics_port=9100,
            py4j_base_port=25000,
            opponent_decklist=self.temp_path / "opp.decklist.txt",
            env={"RL_AGENT_DECK_LIST": "deckA"},
            completed=False,
        )
        orch.trainers = {"ProfileA": trainer_state}
        snapshots = {
            "ProfileA": {
                "episode": 10,
                "rolling_current": 0.2,
                "rolling_avg": 0.15,
                "sample_count": 10,
                "baseline_wr": 0.0,
                "target_winrate": 0.6,
                "promoted": False,
                "train_enabled": True,
                "mode": "",
            },
            "ProfileB": {
                "episode": 0,
                "rolling_current": None,
                "rolling_avg": None,
                "sample_count": 0,
                "baseline_wr": 0.0,
                "target_winrate": 0.6,
                "promoted": False,
                "train_enabled": False,
                "mode": "frozen",
            },
        }

        orch.write_orchestrator_status(snapshots, note="training_concurrent_1")
        payload = json.loads(orch.orchestrator_status_path.read_text(encoding="utf-8"))

        self.assertEqual("test-run", payload["run_id"])
        self.assertEqual(str(orch.reports_root), payload["paths"]["reports_root"])
        self.assertEqual(["ProfileA"], payload["selected_profiles"])
        self.assertEqual(1, len(payload["shared_gpu_hosts"]))
        self.assertEqual(["ProfileA"], [row["profile"] for row in payload["profile_snapshots"]])
        self.assertTrue(orch.compat_orchestrator_status_path.exists())
        self.assertTrue(orch.latest_run_path.exists())


class GeneratePrometheusTargetsTests(unittest.TestCase):
    def test_emits_trainer_and_gpu_host_targets_with_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            status_path = temp_path / "status.json"
            output_path = temp_path / "targets.json"
            status_path.write_text(
                json.dumps(
                    {
                        "run_id": "job-42",
                        "note": "training_concurrent_1",
                        "updated_at_utc": "2026-03-07T00:00:00Z",
                        "trainers": [
                            {"profile": "ProfileA", "running": True, "metrics_port": 9100},
                        ],
                        "shared_gpu_hosts": [
                            {"slot": 0, "gpu_id": "0", "running": True, "metrics_port": 27100},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(TARGETS_PATH),
                    "--status",
                    str(status_path),
                    "--output",
                    str(output_path),
                    "--host",
                    "compute-node",
                    "--strict",
                ],
                check=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            labels = {(row["labels"]["kind"], row["targets"][0]): row["labels"] for row in payload}
            self.assertIn(("trainer", "compute-node:9100"), labels)
            self.assertIn(("gpu_host", "compute-node:27100"), labels)
            self.assertEqual("job-42", labels[("trainer", "compute-node:9100")]["run_id"])
            self.assertEqual("job-42", labels[("gpu_host", "compute-node:27100")]["run_id"])


if __name__ == "__main__":
    unittest.main()
