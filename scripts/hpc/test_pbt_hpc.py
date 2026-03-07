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
GPU_CORE_PATH = REPO_ROOT / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py"
GPU_HOST_PATH = REPO_ROOT / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py"


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

    def test_parse_utc_timestamp_supports_python36_safe_zulu_format(self):
        stamp = self.saturation.parse_utc_timestamp("2026-03-07T15:28:20Z")

        self.assertIsNotNone(stamp)
        self.assertEqual(2026, stamp.year)
        self.assertEqual(3, stamp.month)
        self.assertEqual(7, stamp.day)
        self.assertEqual(15, stamp.hour)
        self.assertEqual(28, stamp.minute)
        self.assertEqual(20, stamp.second)

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
                        "cpu_usage_pct=72.0",
                        "load1=32.0 load5=28.0 load15=20.0 tasks_running=180 tasks_total=900",
                        "2026/03/07 00:00:00.000, NVIDIA A100, 0, 80, 40, 20480, 40960, 60",
                        "===== 2026-03-07T00:00:30Z =====",
                        "host=compute-node",
                        "cpu_total=64 cpu_headroom=0 runner_oversubscription_factor=8 target_total_runners=512 runners_per_profile=256",
                        "cpu_usage_pct=68.0",
                        "load1=30.0 load5=27.0 load15=19.0 tasks_running=170 tasks_total=880",
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
            self.assertAlmostEqual(6.0, summary["episodes_per_sec"], places=3)
            self.assertAlmostEqual(0.0, summary["updates_per_sec"], places=3)
            self.assertEqual(64, summary["cpus_per_task"])
            self.assertAlmostEqual(8.0, summary["runner_oversubscription_factor"], places=3)
            self.assertAlmostEqual(70.0, summary["telemetry"]["gpu_util_avg"], places=3)
            self.assertAlmostEqual(70.0, summary["telemetry"]["cpu_util_avg"], places=3)
            self.assertAlmostEqual(71.8, summary["telemetry"]["cpu_util_p95"], places=3)
            self.assertAlmostEqual(0.484375, summary["telemetry"]["load1_per_cpu_avg"], places=6)
            self.assertAlmostEqual(19.0, summary["telemetry"]["gpu_mem_used_gb_avg"], places=3)
            self.assertAlmostEqual(0.035, summary["rolling_current_avg"], places=3)

    def test_summarize_job_uses_final_probe_metrics_for_finished_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            job_id = "18430000"
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
                "selected_profiles": ["ProfileA"],
                "trainers": [{"profile": "ProfileA", "running": False, "metrics_port": 19100}],
                "selected_profile_snapshots": [{"profile": "ProfileA", "episode": 90, "rolling_current": 0.03}],
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
                        "cpu_total=64 cpu_headroom=0 runner_oversubscription_factor=16 target_total_runners=1024 runners_per_profile=1024",
                        "cpu_usage_pct=55.0",
                        "2026/03/07 00:00:00.000, NVIDIA H100, 0, 50, 20, 10240, 81920, 52",
                        "===== 2026-03-07T00:00:30Z =====",
                        "host=compute-node",
                        "cpu_total=64 cpu_headroom=0 runner_oversubscription_factor=16 target_total_runners=1024 runners_per_profile=1024",
                        "cpu_usage_pct=57.0",
                        "2026/03/07 00:00:30.000, NVIDIA H100, 0, 60, 22, 12288, 81920, 54",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (jobs_root / "final_probe_metrics.txt").write_text(
                "\n".join(
                    [
                        "source=final_snapshot",
                        "hostname=gpu-a6-4.zaratan.umd.edu",
                        "episodes_completed_total=240",
                        "training_updates_total=60",
                        "active_episodes_total=512",
                        "train_batches_total=15",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = self.saturation.summarize_job(
                job_id=job_id,
                record={
                    "label": "spy-sat-finished",
                    "config": {
                        "train_profiles": 1,
                        "runner_oversubscription_factor": 16.0,
                    },
                    "sbatch": {"cpus_per_task": 64},
                    "exports": {},
                },
                repo_root=repo_root,
                heartbeat_window=3,
            )

            self.assertAlmostEqual(8.0, summary["episodes_per_sec"], places=3)
            self.assertAlmostEqual(2.5, summary["updates_per_sec"], places=3)
            self.assertAlmostEqual(512.0, summary["active_episodes"], places=3)
            self.assertAlmostEqual(240.0, summary["counter_episode_total"], places=3)
            self.assertAlmostEqual(75.0, summary["counter_update_total"], places=3)

    def test_discover_local_job_records_finds_numeric_job_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            jobs_root = (
                repo_root
                / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
            )
            runs_root = (
                repo_root
                / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs"
            )
            (jobs_root / "111").mkdir(parents=True, exist_ok=True)
            (jobs_root / "111" / "telemetry.log").write_text("host=compute-node\n", encoding="utf-8")
            (jobs_root / "notes").mkdir(parents=True, exist_ok=True)
            (runs_root / "222").mkdir(parents=True, exist_ok=True)
            (runs_root / "222" / "orchestrator_status.json").write_text("{}", encoding="utf-8")
            (jobs_root / "222").mkdir(parents=True, exist_ok=True)

            records = self.saturation.discover_local_job_records(repo_root=repo_root, username="")

            self.assertEqual(["111", "222"], [row["job_id"] for row in records])

    def test_discover_current_slurm_job_records_uses_squeue(self):
        with mock.patch.object(self.saturation.subprocess, "check_output", return_value="18429954\n18429955\nnot-a-job\n18429954\n"):
            records = self.saturation.discover_current_slurm_job_records(username="jmaior")

        self.assertEqual(["18429954", "18429955"], [row["job_id"] for row in records])


class SlurmAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.availability = load_module("slurm_availability_test", AVAILABILITY_PATH)

    def test_parse_and_aggregate_rows_by_partition_and_type(self):
        sample = "\n".join(
            [
                "node-h100-1|gpu,gpu-h100|idle|128|515000|gpu:h100:4(S:1,3,5,7)|7-00:00:00|up",
                "node-h100-2|gpu,gpu-h100|mixed|128|515000|gpu:h100:4(S:1,3,5,7)|7-00:00:00|up",
                "node-v100-1|gpu,gpu-v100|idle|40|257000|gpu:v100:4|7-00:00:00|up",
                "node-a100-1|gpu-a100*|mixed|128|515000|gpu:a100:4(S:1,3,5,7)|7-00:00:00|up",
                "node-a100-2|gpu-a100|allocated|128|515000|gpu:a100:4(S:1,3,5,7)|7-00:00:00|up",
                "node-cpu-1|cpu|idle|64|257000|(null)|7-00:00:00|up",
            ]
        )

        rows = self.availability.parse_sinfo_node_rows(sample)
        by_partition, by_type = self.availability.aggregate_rows(rows)

        gpu_a100 = next(row for row in by_partition if row["label"] == "gpu-a100")
        cpu = next(row for row in by_partition if row["label"] == "cpu")
        gpu_partition = next(row for row in by_partition if row["label"] == "gpu")
        gpu_a100_type = next(row for row in by_type if row["label"] == "gpu-a100")
        gpu_h100_type = next(row for row in by_type if row["label"] == "gpu-h100")
        gpu_v100_type = next(row for row in by_type if row["label"] == "gpu-v100")

        first_h100 = next(row for row in rows if row["node"] == "node-h100-1" and row["partition"] == "gpu")
        self.assertEqual("gpu-h100", first_h100["type"])
        self.assertEqual(4, first_h100["gpu_count"])

        self.assertEqual(2, gpu_a100["nodes_total"])
        self.assertEqual(0, gpu_a100["nodes_idle"])
        self.assertEqual(1, gpu_a100["nodes_mix"])
        self.assertEqual(1, gpu_a100["nodes_alloc"])
        self.assertEqual(8, gpu_a100["gpu_total"])
        self.assertEqual(0, gpu_a100["gpu_idle_est"])
        self.assertEqual(4, gpu_a100["gpu_mixed"])
        self.assertEqual(["gpu-a100"], gpu_a100["types"])

        self.assertEqual(1, cpu["nodes_idle"])
        self.assertEqual(0, cpu["gpu_total"])
        self.assertEqual(64, cpu["cpu_idle_est"])

        self.assertEqual(["gpu-h100", "gpu-v100"], gpu_partition["types"])
        self.assertEqual(2, gpu_partition["nodes_idle"])
        self.assertEqual(1, gpu_partition["nodes_mix"])

        self.assertEqual(["gpu-a100"], gpu_a100_type["partitions"])
        self.assertEqual(0, gpu_a100_type["gpu_idle_est"])
        self.assertEqual(2, gpu_h100_type["nodes_total"])
        self.assertEqual(4, gpu_h100_type["gpu_idle_est"])
        self.assertEqual(4, gpu_h100_type["gpu_mixed"])
        self.assertEqual(["gpu", "gpu-h100"], gpu_h100_type["partitions"])
        self.assertEqual(1, gpu_v100_type["nodes_total"])
        self.assertEqual(4, gpu_v100_type["gpu_idle_est"])


class SharedGpuCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entry_module = types.ModuleType("py4j_entry_point")

        class StubPythonEntryPoint:
            def __init__(self):
                self.model = None
                self.optimizer = None
                self.initialize_calls = 0
                self.train_calls = 0
                self.saved_paths = []

            def initializeModel(self):
                self.initialize_calls += 1
                self.model = object()
                self.optimizer = object()

            def trainCandidatesMultiFlat(self, *args, **kwargs):
                self.train_calls += 1
                return True

            def saveLatestModelAtomic(self, path=None):
                self.saved_paths.append(path)
                return True

        entry_module.PythonEntryPoint = StubPythonEntryPoint
        cls._entry_patch = mock.patch.dict(sys.modules, {"py4j_entry_point": entry_module})
        cls._entry_patch.start()
        cls.core = load_module("gpu_service_core_test", GPU_CORE_PATH)

    @classmethod
    def tearDownClass(cls):
        cls._entry_patch.stop()

    def test_learner_context_initializes_model_eagerly(self):
        context = self.core.ProfileContext("ProfileA", {}, role="learner")

        self.assertEqual("learner", context.role)
        self.assertEqual(1, context.entry.initialize_calls)
        self.assertIsNotNone(context.entry.model)
        self.assertIsNotNone(context.entry.optimizer)

    def test_train_batch_reinitializes_if_model_was_cleared(self):
        context = self.core.ProfileContext("ProfileA", {}, role="learner")
        context.entry.model = None
        context.entry.optimizer = None

        ok = context.train_batch(*([b""] * 14), 1, 1, 1, 1, 1)

        self.assertTrue(ok)
        self.assertEqual(2, context.entry.initialize_calls)
        self.assertEqual(1, context.entry.train_calls)


class SharedGpuHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_core = types.ModuleType("gpu_service_core")

        class StubProfileContext:
            def __init__(self, profile_id, headers, role="inference"):
                self.profile_id = profile_id
                self.headers = headers
                self.role = role
                self.save_latest_calls = 0
                self.reload_latest_calls = 0
                self.train_batch_calls = 0

            def save_latest_model_atomic(self, path=None):
                self.save_latest_calls += 1
                return True

            def reload_latest_model_if_newer(self, path=None):
                self.reload_latest_calls += 1
                return True

            def train_batch(self, *args, **kwargs):
                self.train_batch_calls += 1
                return True

        stub_core.ProfileContext = StubProfileContext
        stub_core.feature_array_from_bytes = lambda data: data
        stub_core.merge_segments = lambda segments: segments[0] if segments else []
        cls._gpu_core_patch = mock.patch.dict(sys.modules, {"gpu_service_core": stub_core})
        cls._gpu_core_patch.start()
        cls.host = load_module("gpu_service_host_test", GPU_HOST_PATH)

    @classmethod
    def tearDownClass(cls):
        cls._gpu_core_patch.stop()

    def test_register_creates_separate_inference_and_learner_contexts(self):
        shared_host = self.host.SharedGpuHost()

        state = shared_host.get_or_create_profile({"profile_id": "ProfileA"})

        self.assertIsNot(state.infer_context, state.learner_context)
        self.assertEqual("inference", state.infer_context.role)
        self.assertEqual("learner", state.learner_context.role)

    def test_score_and_train_lanes_can_both_dispatch_when_ready(self):
        shared_host = self.host.SharedGpuHost()
        shared_host.batch_timeout_s = 0.2
        shared_host.train_batch_timeout_s = 0.1

        state = self.host.ProfileState(infer_context=object(), learner_context=object())
        now = 1000.0
        state.pending_scores.append(
            self.host.ScoreTask(
                session=types.SimpleNamespace(reply=lambda *args, **kwargs: None),
                request_id=1,
                profile_id="ProfileA",
                headers={"batch_size": "1"},
                segments=[],
                enqueued_at=now - 0.25,
            )
        )
        state.pending_trains.append(
            self.host.TrainTask(
                profile_id="ProfileA",
                headers={"batch_size": "8"},
                segments=[],
                enqueued_at=now - 0.15,
            )
        )
        shared_host._profiles = {"ProfileA": state}

        with mock.patch.object(self.host.time, "monotonic", return_value=now):
            score_work, score_sleep = shared_host._select_score_work_locked()
            train_work, train_sleep = shared_host._select_train_work_locked()

        self.assertIsNotNone(score_work)
        self.assertIsNotNone(train_work)
        self.assertEqual(0.0, score_sleep)
        self.assertEqual(0.0, train_sleep)

    def test_publish_and_reload_counters_use_separate_contexts(self):
        shared_host = self.host.SharedGpuHost()
        state = shared_host.get_or_create_profile({"profile_id": "ProfileA"})

        with mock.patch.object(self.host.time, "monotonic", return_value=1000.0):
            shared_host._maybe_publish_latest_model(state)
            shared_host._maybe_reload_inference_model(state)

        self.assertEqual(1, shared_host._model_publishes)
        self.assertEqual(1, shared_host._model_reloads)
        self.assertEqual(1, state.learner_context.save_latest_calls)
        self.assertEqual(0, state.infer_context.save_latest_calls)
        self.assertEqual(1, state.infer_context.reload_latest_calls)
        self.assertEqual(0, state.learner_context.reload_latest_calls)

    def test_train_batch_counts_even_if_publish_latest_fails(self):
        shared_host = self.host.SharedGpuHost()
        state = shared_host.get_or_create_profile({"profile_id": "ProfileA"})
        state.learner_context.save_latest_model_atomic = mock.Mock(side_effect=RuntimeError("publish failed"))

        task = self.host.TrainTask(
            profile_id="ProfileA",
            headers={
                "batch_size": "2",
                "seq_len": "1",
                "d_model": "1",
                "max_candidates": "1",
                "cand_feat_dim": "1",
            },
            segments=[b""] * 14,
            enqueued_at=1000.0,
        )

        with mock.patch.object(self.host.time, "monotonic", side_effect=[1000.0, 1000.5]):
            shared_host._run_train_batch(state, [task])

        self.assertEqual(1, shared_host._train_batches)
        self.assertEqual(1, shared_host._train_failures)
        self.assertEqual(1, state.learner_context.train_batch_calls)
        self.assertIn("publish failed", shared_host._last_error)


if __name__ == "__main__":
    unittest.main()
