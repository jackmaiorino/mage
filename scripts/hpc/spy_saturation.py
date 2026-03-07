#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pwd  # type: ignore
except Exception:  # pragma: no cover
    pwd = None


REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_ROOT = REPO_ROOT / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
RUNS_ROOT = REPO_ROOT / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs"
DEFAULT_SWEEP_ROOT = REPO_ROOT / "local-training/hpc/saturation_runs"
HEARTBEAT_RE = re.compile(r"\((?:run=(\d+),\s*)?([0-9]*\.?[0-9]+)\s+eps/s\)")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def split_csv_values(raw: str, cast) -> List[Any]:
    values: List[Any] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        values.append(cast(token))
    return values


def parse_extra_exports(items: Sequence[str]) -> Dict[str, str]:
    exports: Dict[str, str] = {}
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected KEY=VALUE for --extra-export, got: {item}")
        key, value = token.split("=", 1)
        name = key.strip()
        if not name:
            raise ValueError(f"missing export name in: {item}")
        exports[name] = value
    return exports


def current_username() -> str:
    for name in ("USER", "USERNAME"):
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    if pwd is not None:
        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            pass
    return ""


def discover_current_slurm_job_records(username: Optional[str] = None) -> List[Dict[str, Any]]:
    user = str(username or current_username()).strip()
    if not user:
        return []
    try:
        output = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%i"],
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    seen = set()
    for raw_line in output.splitlines():
        job_id = str(raw_line).strip()
        if not job_id or not job_id.isdigit() or job_id in seen:
            continue
        seen.add(job_id)
        rows.append(
            {
                "job_id": job_id,
                "label": job_id,
                "config": {},
                "sbatch": {},
                "exports": {},
            }
        )
    return rows


def sanitize_label(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "run"


def label_number(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return sanitize_label(str(value))
    if number.is_integer():
        return str(int(number))
    return str(number).replace(".", "p")


def parse_gpu_count(gres: str) -> Optional[int]:
    match = re.search(r":(\d+)\s*$", str(gres).strip())
    if not match:
        return None
    try:
        count = int(match.group(1))
    except Exception:
        return None
    return count if count > 0 else None


def find_latest_bundle(bundle_path: Optional[str], bundle_dir: str) -> Path:
    if bundle_path:
        path = Path(bundle_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"bundle not found: {path}")
        return path
    search_root = Path(bundle_dir).expanduser().resolve()
    matches = sorted(search_root.glob("rl-runtime-*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"no runtime bundle found under {search_root}")
    return matches[0]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, pct)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values)) / float(len(values))


def shell_join(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_experiment_rows(args: argparse.Namespace, bundle: Path) -> List[Dict[str, Any]]:
    profile_counts = split_csv_values(args.train_profiles, int)
    cpu_counts = split_csv_values(args.cpus_per_task, int)
    oversub_factors = split_csv_values(args.runner_oversubscription_factor, float)
    infer_workers = split_csv_values(args.infer_workers, int)
    if not profile_counts:
        raise ValueError("at least one train profile count is required")
    if not cpu_counts:
        raise ValueError("at least one cpus-per-task value is required")
    if not oversub_factors:
        raise ValueError("at least one runner oversubscription factor is required")
    if not infer_workers:
        raise ValueError("at least one infer workers value is required")

    gpu_count = parse_gpu_count(args.gres)
    trainer_start_wave_size = args.trainer_start_wave_size or max(1, gpu_count or 1)
    extra_exports = parse_extra_exports(args.extra_export)
    rows: List[Dict[str, Any]] = []
    seen_labels: Dict[str, int] = {}

    for train_profiles, cpus_per_task, oversub_factor, infer_worker_count in itertools.product(
        profile_counts,
        cpu_counts,
        oversub_factors,
        infer_workers,
    ):
        label_parts = [
            args.job_prefix,
            f"p{train_profiles}",
            f"c{cpus_per_task}",
            f"o{label_number(oversub_factor)}",
        ]
        if gpu_count is not None:
            label_parts.append(f"g{gpu_count}")
        if infer_worker_count != 1:
            label_parts.append(f"iw{infer_worker_count}")
        label = sanitize_label("-".join(label_parts))
        label_count = seen_labels.get(label, 0) + 1
        seen_labels[label] = label_count
        if label_count > 1:
            label = f"{label}-n{label_count}"

        exports: Dict[str, str] = {
            "HPC_NATIVE_ORCH": "1",
            "PY_SERVICE_MODE": "shared_gpu",
            "MAGE_RL_RUNTIME_TARBALL": str(bundle),
            "TRAIN_PROFILES": str(train_profiles),
            "CPU_HEADROOM": str(args.cpu_headroom),
            "RUNNER_OVERSUBSCRIPTION_FACTOR": str(oversub_factor),
            "TRAINER_START_STAGGER_SECONDS": str(args.trainer_start_stagger_seconds),
            "TRAINER_START_WAVE_SIZE": str(trainer_start_wave_size),
            "GPU_SERVICE_STARTUP_TIMEOUT_SECONDS": str(args.gpu_service_startup_timeout_seconds),
            "PY_BRIDGE_CONNECT_RETRIES": str(args.py_bridge_connect_retries),
            "PY_BRIDGE_CONNECT_RETRY_DELAY_MS": str(args.py_bridge_connect_retry_delay_ms),
            "TOTAL_EPISODES": str(args.total_episodes),
            "STALL_RESTART_MINUTES": str(args.stall_restart_minutes),
            "GAME_LOG_FREQUENCY": str(args.game_log_frequency),
            "INFER_WORKERS": str(infer_worker_count),
            "SATURATION_EXPERIMENT_LABEL": label,
        }
        if args.metrics_port_base is not None:
            exports["METRICS_PORT_BASE"] = str(args.metrics_port_base)
        if args.gpu_service_port_base is not None:
            exports["GPU_SERVICE_PORT_BASE"] = str(args.gpu_service_port_base)
        if args.gpu_service_metrics_port_base is not None:
            exports["GPU_SERVICE_METRICS_PORT_BASE"] = str(args.gpu_service_metrics_port_base)
        if args.throughput_mode:
            exports.update(
                {
                    "PBT_EXPLOIT_INTERVAL_MINUTES": "1000000",
                    "PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT": "1000000000",
                    "PBT_MIN_EPISODE_DELTA_PER_PROFILE": "1000000000",
                    "PBT_TIME_FALLBACK_MIN_EPISODE_DELTA": "1000000000",
                    "PBT_MIN_WINNER_GAP": "1.0",
                    "PBT_MIN_WINNER_WINRATE": "1.0",
                    "EVAL_EVERY_MINUTES": "1000000",
                }
            )
        exports.update(extra_exports)
        rows.append(
            {
                "label": label,
                "train_profiles": train_profiles,
                "cpus_per_task": cpus_per_task,
                "runner_oversubscription_factor": oversub_factor,
                "infer_workers": infer_worker_count,
                "trainer_start_wave_size": trainer_start_wave_size,
                "requested_gpu_count": gpu_count,
                "exports": exports,
            }
        )
    return rows


def manifest_output_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser().resolve()
    tag = sanitize_label(args.tag)
    return Path(args.sweep_root).expanduser().resolve() / f"{timestamp_slug()}_{tag}" / "manifest.json"


def build_sbatch_command(args: argparse.Namespace, label: str, cpus_per_task: int, export_names: Iterable[str]) -> List[str]:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        sanitize_label(label)[:120],
        "--partition",
        args.partition,
        "--gres",
        args.gres,
        "--cpus-per-task",
        str(cpus_per_task),
        "--mem",
        args.mem,
        "--time",
        args.time,
        "--export",
        "ALL," + ",".join(sorted(export_names)),
    ]
    if args.account:
        command.extend(["--account", args.account])
    if args.qos:
        command.extend(["--qos", args.qos])
    if args.constraint:
        command.extend(["--constraint", args.constraint])
    command.append("scripts/hpc/submit_spy_pbt.slurm")
    return command


def submit_experiments(args: argparse.Namespace) -> int:
    bundle = find_latest_bundle(args.bundle, args.bundle_dir)
    manifest_path = manifest_output_path(args)
    experiments = build_experiment_rows(args, bundle)
    manifest: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "repo_root": str(REPO_ROOT),
        "tag": args.tag,
        "bundle": str(bundle),
        "throughput_mode": bool(args.throughput_mode),
        "jobs": [],
    }
    atomic_write_json(manifest_path, manifest)

    print(f"bundle={bundle}")
    print(f"manifest={manifest_path}")
    print(f"planned_jobs={len(experiments)}")
    for experiment in experiments:
        exports = dict(experiment["exports"])
        command = build_sbatch_command(
            args=args,
            label=str(experiment["label"]),
            cpus_per_task=int(experiment["cpus_per_task"]),
            export_names=exports.keys(),
        )
        record: Dict[str, Any] = {
            "label": experiment["label"],
            "submitted_at_utc": "",
            "job_id": "",
            "status": "planned" if args.dry_run else "submitted",
            "sbatch": {
                "partition": args.partition,
                "gres": args.gres,
                "cpus_per_task": int(experiment["cpus_per_task"]),
                "mem": args.mem,
                "time": args.time,
                "account": args.account or "",
                "qos": args.qos or "",
                "constraint": args.constraint or "",
                "job_name": sanitize_label(str(experiment["label"]))[:120],
            },
            "config": {
                "train_profiles": int(experiment["train_profiles"]),
                "runner_oversubscription_factor": float(experiment["runner_oversubscription_factor"]),
                "cpu_headroom": int(args.cpu_headroom),
                "infer_workers": int(experiment["infer_workers"]),
                "trainer_start_stagger_seconds": int(args.trainer_start_stagger_seconds),
                "trainer_start_wave_size": int(experiment["trainer_start_wave_size"]),
                "requested_gpu_count": experiment.get("requested_gpu_count"),
            },
            "exports": exports,
            "command": command,
        }
        if args.dry_run:
            print(f"dry_run label={record['label']} command={shell_join(command)}")
        else:
            env = os.environ.copy()
            env.update(exports)
            try:
                result = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                details = []
                if getattr(exc, "stdout", ""):
                    details.append(str(exc.stdout).strip())
                if getattr(exc, "stderr", ""):
                    details.append(str(exc.stderr).strip())
                detail_text = "\n".join(item for item in details if item)
                if not detail_text:
                    detail_text = str(exc)
                raise SystemExit(
                    "sbatch failed for label={} command={}\n{}".format(
                        record["label"],
                        shell_join(command),
                        detail_text,
                    )
                )
            record["job_id"] = result.stdout.strip()
            record["submitted_at_utc"] = now_utc()
            print(
                f"submitted label={record['label']} job_id={record['job_id']} "
                f"cpus={record['sbatch']['cpus_per_task']} train_profiles={record['config']['train_profiles']} "
                f"oversub={record['config']['runner_oversubscription_factor']}"
            )
        manifest["jobs"].append(record)
        atomic_write_json(manifest_path, manifest)

    print()
    print("next_steps:")
    if args.dry_run:
        print("  rerun the same command without --dry-run to submit this sweep")
    else:
        print(f"  python3 scripts/hpc/spy_saturation.py summarize --manifest {shlex.quote(str(manifest_path))}")
        print(f"  bash scripts/hpc/observe_job.sh <job_id>")
    return 0


def read_tail_bytes(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    if size <= 0:
        return ""
    read_size = min(size, max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(size - read_size)
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def recent_heartbeat_rates(log_path: Path, limit: int) -> List[float]:
    text = read_tail_bytes(log_path)
    if not text:
        return []
    matches = HEARTBEAT_RE.findall(text)
    values: List[float] = []
    for _run_ep, rate in matches[-limit:]:
        try:
            values.append(float(rate))
        except Exception:
            continue
    return values


def aggregate_heartbeat_rates(trainers_dir: Path, selected_profiles: Sequence[str], limit: int) -> Dict[str, Any]:
    profile_rows: List[Dict[str, Any]] = []
    total_mean = 0.0
    for profile in selected_profiles:
        rates = recent_heartbeat_rates(trainers_dir / f"{profile}.stdout.log", limit)
        if not rates:
            rates = recent_heartbeat_rates(trainers_dir / f"{profile}.stderr.log", limit)
        if not rates:
            profile_rows.append({"profile": profile, "heartbeat_mean": 0.0, "samples": 0})
            continue
        heartbeat_mean = safe_mean(rates)
        total_mean += heartbeat_mean
        profile_rows.append({"profile": profile, "heartbeat_mean": heartbeat_mean, "samples": len(rates)})
    return {
        "profiles": profile_rows,
        "aggregate_mean_eps_per_s": total_mean,
        "profile_samples": sum(int(row["samples"]) for row in profile_rows),
    }


def parse_telemetry_log(path: Path) -> Dict[str, Any]:
    gpu_utils: List[float] = []
    gpu_mem_utils: List[float] = []
    gpu_mem_used_gb: List[float] = []
    gpu_temp_c: List[float] = []
    cpu_utils: List[float] = []
    load1_values: List[float] = []
    load5_values: List[float] = []
    load15_values: List[float] = []
    tasks_running_values: List[float] = []
    cpu_total = 0
    runner_oversubscription_factor = 0.0
    if not path.exists():
        return {
            "sample_count": 0,
            "cpu_sample_count": 0,
            "cpu_total": 0,
            "cpu_util_avg": 0.0,
            "cpu_util_p50": 0.0,
            "cpu_util_p95": 0.0,
            "load1_avg": 0.0,
            "load5_avg": 0.0,
            "load15_avg": 0.0,
            "load1_per_cpu_avg": 0.0,
            "tasks_running_avg": 0.0,
            "runner_oversubscription_factor": 0.0,
            "gpu_util_avg": 0.0,
            "gpu_util_p50": 0.0,
            "gpu_util_p95": 0.0,
            "gpu_mem_util_avg": 0.0,
            "gpu_mem_used_gb_avg": 0.0,
            "gpu_mem_used_gb_max": 0.0,
            "gpu_temp_c_avg": 0.0,
        }
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=") or line.startswith("host="):
            continue
        if line.startswith("cpu_total="):
            cpu_match = re.search(r"\bcpu_total=(\d+)", line)
            oversub_match = re.search(r"\brunner_oversubscription_factor=([0-9]*\.?[0-9]+)", line)
            if cpu_match:
                cpu_total = parse_int(cpu_match.group(1), cpu_total)
            if oversub_match:
                runner_oversubscription_factor = parse_float(oversub_match.group(1), runner_oversubscription_factor)
            continue
        if line.startswith("cpu_usage_pct="):
            value = line.split("=", 1)[1].strip()
            try:
                cpu_utils.append(float(value))
            except Exception:
                pass
            continue
        if line.startswith("load1="):
            load1_match = re.search(r"\bload1=([0-9]*\.?[0-9]+)", line)
            load5_match = re.search(r"\bload5=([0-9]*\.?[0-9]+)", line)
            load15_match = re.search(r"\bload15=([0-9]*\.?[0-9]+)", line)
            tasks_running_match = re.search(r"\btasks_running=(\d+)", line)
            if load1_match:
                load1_values.append(parse_float(load1_match.group(1), 0.0))
            if load5_match:
                load5_values.append(parse_float(load5_match.group(1), 0.0))
            if load15_match:
                load15_values.append(parse_float(load15_match.group(1), 0.0))
            if tasks_running_match:
                tasks_running_values.append(parse_float(tasks_running_match.group(1), 0.0))
            continue
        if line.startswith("MemTotal:") or line.startswith("MemAvailable:") or line == "nvidia-smi unavailable":
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            util_gpu = float(parts[3])
            util_mem = float(parts[4])
            mem_used_gb = float(parts[5]) / 1024.0
            temp_c = float(parts[7])
        except Exception:
            continue
        gpu_utils.append(util_gpu)
        gpu_mem_utils.append(util_mem)
        gpu_mem_used_gb.append(mem_used_gb)
        gpu_temp_c.append(temp_c)
    return {
        "sample_count": len(gpu_utils),
        "cpu_sample_count": len(cpu_utils),
        "cpu_total": cpu_total,
        "cpu_util_avg": safe_mean(cpu_utils),
        "cpu_util_p50": percentile(cpu_utils, 0.50),
        "cpu_util_p95": percentile(cpu_utils, 0.95),
        "load1_avg": safe_mean(load1_values),
        "load5_avg": safe_mean(load5_values),
        "load15_avg": safe_mean(load15_values),
        "load1_per_cpu_avg": (safe_mean(load1_values) / float(cpu_total)) if cpu_total > 0 else 0.0,
        "tasks_running_avg": safe_mean(tasks_running_values),
        "runner_oversubscription_factor": runner_oversubscription_factor,
        "gpu_util_avg": safe_mean(gpu_utils),
        "gpu_util_p50": percentile(gpu_utils, 0.50),
        "gpu_util_p95": percentile(gpu_utils, 0.95),
        "gpu_mem_util_avg": safe_mean(gpu_mem_utils),
        "gpu_mem_used_gb_avg": safe_mean(gpu_mem_used_gb),
        "gpu_mem_used_gb_max": max(gpu_mem_used_gb) if gpu_mem_used_gb else 0.0,
        "gpu_temp_c_avg": safe_mean(gpu_temp_c),
    }


def summarize_job(
    job_id: str,
    record: Optional[Dict[str, Any]] = None,
    repo_root: Path = REPO_ROOT,
    heartbeat_window: int = 5,
) -> Dict[str, Any]:
    jobs_root = repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
    runs_root = repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs"
    job_dir = jobs_root / str(job_id)
    run_dir = runs_root / str(job_id)
    status_path = run_dir / "orchestrator_status.json"
    status = load_json(status_path) or {}
    selected_profiles = [str(p) for p in status.get("selected_profiles", []) if str(p).strip()]
    if not selected_profiles:
        selected_profiles = sorted(
            {
                path.name[: -len(".stdout.log")]
                for path in (run_dir / "trainers").glob("*.stdout.log")
                if path.name.endswith(".stdout.log")
            }
        )
    trainer_rows = [row for row in status.get("trainers", []) if isinstance(row, dict)]
    snapshot_rows = [row for row in status.get("selected_profile_snapshots", []) if isinstance(row, dict)]

    telemetry = parse_telemetry_log(job_dir / "telemetry.log")
    heartbeat = aggregate_heartbeat_rates(run_dir / "trainers", selected_profiles, heartbeat_window)
    rolling_values: List[float] = []
    total_episode = 0
    for row in snapshot_rows:
        try:
            total_episode += int(row.get("episode", 0))
        except Exception:
            pass
        try:
            if row.get("rolling_current") is not None:
                rolling_values.append(float(row.get("rolling_current")))
        except Exception:
            pass

    config = dict(record.get("config", {})) if record else {}
    sbatch = dict(record.get("sbatch", {})) if record else {}
    exports = dict(record.get("exports", {})) if record else {}

    effective_cpus = int(sbatch.get("cpus_per_task", 0) or 0)
    if effective_cpus <= 0:
        effective_cpus = int(telemetry.get("cpu_total", 0) or 0)
    effective_oversub = float(config.get("runner_oversubscription_factor", 0.0) or 0.0)
    if effective_oversub <= 0:
        effective_oversub = float(telemetry.get("runner_oversubscription_factor", 0.0) or 0.0)

    return {
        "label": str(record.get("label", job_id)) if record else str(job_id),
        "job_id": str(job_id),
        "run_id": str(status.get("run_id", job_id)),
        "status_note": str(status.get("note", "")),
        "updated_at_utc": str(status.get("updated_at_utc", "")),
        "selected_profiles": selected_profiles,
        "selected_count": len(selected_profiles),
        "running_trainers": sum(1 for row in trainer_rows if bool(row.get("running", False))),
        "requested_train_profiles": int(config.get("train_profiles", len(selected_profiles) or 0)),
        "cpus_per_task": effective_cpus,
        "requested_gpu_count": int(config.get("requested_gpu_count", 0) or 0),
        "runner_oversubscription_factor": effective_oversub,
        "infer_workers": int(config.get("infer_workers", 0) or 0),
        "cpu_headroom": int(config.get("cpu_headroom", 0) or 0),
        "telemetry": telemetry,
        "heartbeat": heartbeat,
        "heartbeat_eps_per_s": float(heartbeat["aggregate_mean_eps_per_s"]),
        "heartbeat_profile_samples": int(heartbeat["profile_samples"]),
        "total_episode": total_episode,
        "rolling_current_avg": safe_mean(rolling_values),
        "rolling_current_max": max(rolling_values) if rolling_values else 0.0,
        "exports": exports,
    }


def discover_local_job_records(repo_root: Path = REPO_ROOT, username: Optional[str] = None) -> List[Dict[str, Any]]:
    jobs_root = repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
    runs_root = repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/runs"
    target_user = str(username or current_username()).strip()
    rows: List[Dict[str, Any]] = []
    if not jobs_root.exists():
        return rows
    for child in sorted(jobs_root.iterdir()):
        if not child.is_dir():
            continue
        job_id = child.name.strip()
        if not job_id.isdigit():
            continue
        try:
            stat_info = child.stat()
        except Exception:
            continue
        if target_user:
            owner_name = ""
            if pwd is not None:
                try:
                    owner_name = pwd.getpwuid(stat_info.st_uid).pw_name
                except Exception:
                    owner_name = ""
            if owner_name and owner_name != target_user:
                continue
        run_dir = runs_root / job_id
        if not (child / "telemetry.log").exists() and not (run_dir / "orchestrator_status.json").exists():
            continue
        rows.append(
            {
                "job_id": job_id,
                "label": job_id,
                "config": {},
                "sbatch": {},
                "exports": {},
            }
        )
    return rows


def print_summary_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        ("label", 22),
        ("job", 10),
        ("p", 3),
        ("cpu", 5),
        ("ovr", 5),
        ("hb_g/s", 8),
        ("gpu_avg", 8),
        ("gpu_p95", 8),
        ("cpu_avg", 8),
        ("cpu_p95", 8),
        ("mem_gb", 8),
        ("ep", 8),
        ("wr_avg", 7),
        ("samp", 5),
    ]

    def trim(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "+"

    def fmt(value: Any, width: int, decimals: int = 0) -> str:
        if value is None:
            return "-".rjust(width)
        if isinstance(value, float):
            return f"{value:>{width}.{decimals}f}"
        return trim(str(value), width).rjust(width)

    print(" ".join(name.rjust(width) for name, width in headers))
    for row in rows:
        telemetry = row.get("telemetry", {})
        values = [
            trim(str(row.get("label", "")), 22).ljust(22),
            trim(str(row.get("job_id", "")), 10).rjust(10),
            fmt(row.get("requested_train_profiles"), 3),
            fmt(row.get("cpus_per_task"), 5),
            fmt(row.get("runner_oversubscription_factor"), 5, 1),
            fmt(row.get("heartbeat_eps_per_s"), 8, 3),
            fmt(float(telemetry.get("gpu_util_avg", 0.0)), 8, 1),
            fmt(float(telemetry.get("gpu_util_p95", 0.0)), 8, 1),
            fmt(float(telemetry.get("cpu_util_avg", 0.0)), 8, 1),
            fmt(float(telemetry.get("cpu_util_p95", 0.0)), 8, 1),
            fmt(float(telemetry.get("gpu_mem_used_gb_avg", 0.0)), 8, 1),
            fmt(row.get("total_episode"), 8),
            fmt(row.get("rolling_current_avg"), 7, 3),
            fmt(int(telemetry.get("sample_count", 0)), 5),
        ]
        print(" ".join(values))


def summarize_experiments(args: argparse.Namespace) -> int:
    manifest = load_json(Path(args.manifest).expanduser().resolve()) if args.manifest else None
    records: List[Dict[str, Any]] = []
    if manifest and isinstance(manifest.get("jobs"), list):
        records.extend([row for row in manifest["jobs"] if isinstance(row, dict)])
    for job_id in args.job_id:
        records.append({"job_id": str(job_id), "label": str(job_id), "config": {}, "sbatch": {}, "exports": {}})
    if not records:
        if args.all_local:
            records.extend(discover_local_job_records())
        else:
            records.extend(discover_current_slurm_job_records())
            if not records:
                records.extend(discover_local_job_records())
    deduped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        job_id = str(record.get("job_id", "")).strip()
        if job_id:
            deduped[job_id] = record
    if not deduped:
        raise SystemExit("no submitted jobs found; pass --manifest or --job-id, or run from the checkout that contains your job reports")

    rows = [summarize_job(job_id, record=record, heartbeat_window=args.heartbeat_window) for job_id, record in deduped.items()]
    rows.sort(key=lambda row: (-float(row.get(args.sort_by, 0.0) or 0.0), str(row.get("label", ""))))

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_summary_table(rows)
        print()
        print("notes:")
        print("  hb_g/s is the aggregate mean of the last heartbeat eps/s values from the job-scoped trainer logs.")
        print("  gpu_* and cpu_* columns come from the job-scoped telemetry log sampled during the run.")
        print("  Use higher hb_g/s together with higher gpu_avg/gpu_p95 and cpu_avg/cpu_p95 to find the best single-node saturation point.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit and summarize single-node Spy saturation experiments.")
    subparsers = parser.add_subparsers(dest="command")
    try:
        subparsers.required = True
    except Exception:
        pass

    submit = subparsers.add_parser("submit", help="submit a sweep of single-node saturation jobs")
    submit.add_argument("--bundle", default="", help="path to runtime tarball; defaults to newest bundle in bundle-dir")
    submit.add_argument("--bundle-dir", default=str(REPO_ROOT / "local-training/hpc/bundles"))
    submit.add_argument("--manifest", default="", help="optional output manifest path")
    submit.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    submit.add_argument("--tag", default="single-node")
    submit.add_argument("--job-prefix", default="spy-sat")
    submit.add_argument("--partition", default="gpu-a100")
    submit.add_argument("--gres", default="gpu:a100:1")
    submit.add_argument("--mem", default="64G")
    submit.add_argument("--time", default="01:30:00")
    submit.add_argument("--account", default="")
    submit.add_argument("--qos", default="")
    submit.add_argument("--constraint", default="")
    submit.add_argument("--train-profiles", default="2,4")
    submit.add_argument("--cpus-per-task", default="32,64,96")
    submit.add_argument("--runner-oversubscription-factor", default="20,24")
    submit.add_argument("--infer-workers", default="1")
    submit.add_argument("--cpu-headroom", type=int, default=0)
    submit.add_argument("--trainer-start-stagger-seconds", type=int, default=45)
    submit.add_argument("--trainer-start-wave-size", type=int, default=0)
    submit.add_argument("--gpu-service-startup-timeout-seconds", type=int, default=120)
    submit.add_argument("--py-bridge-connect-retries", type=int, default=60)
    submit.add_argument("--py-bridge-connect-retry-delay-ms", type=int, default=2000)
    submit.add_argument("--total-episodes", type=int, default=1_000_000)
    submit.add_argument("--stall-restart-minutes", type=int, default=45)
    submit.add_argument("--game-log-frequency", type=int, default=0)
    submit.add_argument("--metrics-port-base", type=int, default=None)
    submit.add_argument("--gpu-service-port-base", type=int, default=None)
    submit.add_argument("--gpu-service-metrics-port-base", type=int, default=None)
    submit.add_argument("--extra-export", action="append", default=[])
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--throughput-mode", dest="throughput_mode", action="store_true")
    submit.add_argument("--disable-throughput-mode", dest="throughput_mode", action="store_false")
    submit.set_defaults(throughput_mode=True, func=submit_experiments)

    summarize = subparsers.add_parser("summarize", help="summarize completed or running saturation jobs")
    summarize.add_argument("--manifest", default="")
    summarize.add_argument("--job-id", action="append", default=[])
    summarize.add_argument("--all-local", action="store_true", help="summarize all local job reports from this checkout instead of current Slurm jobs")
    summarize.add_argument("--heartbeat-window", type=int, default=5)
    summarize.add_argument("--sort-by", default="heartbeat_eps_per_s")
    summarize.add_argument("--json", action="store_true")
    summarize.set_defaults(func=summarize_experiments)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
