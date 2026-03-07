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
SATURATION_PATH = REPO_ROOT / "scripts/hpc/spy_saturation.py"
AVAILABILITY_PATH = REPO_ROOT / "scripts/hpc/slurm_availability.py"


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


class SpySaturationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saturation = load_module("spy_saturation_test", SATURATION_PATH)

    def test_build_experiment_rows_enables_throughput_mode(self):
        args = types.SimpleNamespace(
            train_profiles="2",
            cpus_per_task="64",
            runner_oversubscription_factor="8",
            infer_workers="1",
            gres="gpu:a100:2",
            trainer_start_wave_size=0,
            extra_export=["FOO=bar"],
            job_prefix="spy-sat",
            cpu_headroom=0,
            trainer_start_stagger_seconds=45,
            gpu_service_startup_timeout_seconds=120,
            py_bridge_connect_retries=60,
            py_bridge_connect_retry_delay_ms=2000,
            total_episodes=1000000,
            stall_restart_minutes=45,
            game_log_frequency=0,
            metrics_port_base=None,
            gpu_service_port_base=None,
            gpu_service_metrics_port_base=None,
            throughput_mode=True,
        )

        rows = self.saturation.build_experiment_rows(args, Path("/tmp/rl-runtime.tar.gz"))

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("spy-sat-p2-c64-o8-g2", row["label"])
        self.assertEqual("1", row["exports"]["INFER_WORKERS"])
        self.assertEqual("1000000000", row["exports"]["PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT"])
        self.assertEqual("1000000", row["exports"]["EVAL_EVERY_MINUTES"])
        self.assertEqual("bar", row["exports"]["FOO"])

    def test_summarize_job_aggregates_job_scoped_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            job_id = "4242"
            jobs_root = (
                repo_root
                / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
                / job_id
            )
            runs_root = (
                repo_root
                / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs"
                / job_id
            )
            trainers_dir = runs_root / "trainers"
            trainers_dir.mkdir(parents=True, exist_ok=True)
            jobs_root.mkdir(parents=True, exist_ok=True)

            status_payload = {
                "run_id": job_id,
                "note": "training_concurrent_2",
                "updated_at_utc": "2026-03-07T00:00:00Z",
                "selected_profiles": ["ProfileA", "ProfileB"],
                "trainers": [
                    {"profile": "ProfileA", "running": True, "metrics_port": 19100},
                    {"profile": "ProfileB", "running": True, "metrics_port": 19101},
                ],
                "selected_profile_snapshots": [
                    {"profile": "ProfileA", "episode": 100, "rolling_current": 0.05},
                    {"profile": "ProfileB", "episode": 80, "rolling_current": 0.02},
                ],
            }
            (runs_root / "orchestrator_status.json").write_text(
                json.dumps(status_payload),
                encoding="utf-8",
            )
            (jobs_root / "telemetry.log").write_text(
                "\n".join(
                    [
                        "===== 2026-03-07T00:00:00Z =====",
                        "host=compute-node",
                        "cpu_total=64 cpu_headroom=0 runner_oversubscription_factor=8 target_total_runners=512 runners_per_profile=256",
                        "2026/03/07 00:00:00.000, NVIDIA A100, 0, 80, 40, 20480, 40960, 60",
                        "2026/03/07 00:00:30.000, NVIDIA A100, 0, 60, 35, 18432, 40960, 58",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (trainers_dir / "ProfileA.stdout.log").write_text(
                "\n".join(
                    [
                        "progress (run=10, 0.500 eps/s)",
                        "progress (run=20, 0.700 eps/s)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (trainers_dir / "ProfileB.stdout.log").write_text(
                "\n".join(
                    [
                        "progress (run=10, 0.200 eps/s)",
                        "progress (run=20, 0.400 eps/s)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = self.saturation.summarize_job(
                job_id=job_id,
                record={
                    "label": "spy-sat-p2-c64-o8-g1",
                    "config": {
                        "train_profiles": 2,
                        "runner_oversubscription_factor": 8.0,
                        "infer_workers": 1,
                        "cpu_headroom": 0,
                    },
                    "sbatch": {"cpus_per_task": 64},
                    "exports": {},
                },
                repo_root=repo_root,
                heartbeat_window=3,
            )

            self.assertEqual(2, summary["selected_count"])
            self.assertEqual(180, summary["total_episode"])
            self.assertAlmostEqual(0.9, summary["heartbeat_eps_per_s"], places=3)
            self.assertAlmostEqual(70.0, summary["telemetry"]["gpu_util_avg"], places=3)
            self.assertAlmostEqual(19.0, summary["telemetry"]["gpu_mem_used_gb_avg"], places=3)
            self.assertAlmostEqual(0.035, summary["rolling_current_avg"], places=3)


class SlurmAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.availability = load_module("slurm_availability_test", AVAILABILITY_PATH)

    def test_parse_and_aggregate_rows_by_partition_and_type(self):
        sample = "\n".join(
            [
                "gpu-a100*|idle|128|515000|gpu:a100:4(S:1,3,5,7)|7-00:00:00|up",
                "gpu-a100|mixed|128|515000|gpu:a100:4(S:1,3,5,7)|7-00:00:00|up",
                "gpu-h100|allocated|128|515000|gpu:h100:4(S:1,3,5,7)|7-00:00:00|up",
                "cpu|idle|64|257000|(null)|7-00:00:00|up",
            ]
        )

        rows = self.availability.parse_sinfo_node_rows(sample)
        by_partition, by_type = self.availability.aggregate_rows(rows)

        gpu_a100 = next(row for row in by_partition if row["label"] == "gpu-a100")
        cpu = next(row for row in by_partition if row["label"] == "cpu")
        gpu_a100_type = next(row for row in by_type if row["label"] == "gpu-a100")

        self.assertEqual("gpu-a100", rows[0]["partition"])
        self.assertEqual("gpu-a100", rows[0]["type"])
        self.assertEqual(4, rows[0]["gpu_count"])

        self.assertEqual(2, gpu_a100["nodes_total"])
        self.assertEqual(1, gpu_a100["nodes_idle"])
        self.assertEqual(1, gpu_a100["nodes_mix"])
        self.assertEqual(8, gpu_a100["gpu_total"])
        self.assertEqual(4, gpu_a100["gpu_idle_est"])
        self.assertEqual(4, gpu_a100["gpu_mixed"])
        self.assertEqual(128, gpu_a100["cpu_idle_est"])

        self.assertEqual(1, cpu["nodes_idle"])
        self.assertEqual(0, cpu["gpu_total"])
        self.assertEqual(64, cpu["cpu_idle_est"])

        self.assertEqual(["gpu-a100"], gpu_a100_type["partitions"])
        self.assertEqual(4, gpu_a100_type["gpu_idle_est"])


if __name__ == "__main__":
    unittest.main()
