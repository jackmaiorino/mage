#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Prometheus file_sd targets from native orchestrator_status.json"
    )
    parser.add_argument(
        "--status",
        default="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator/orchestrator_status.json",
        help="Path to orchestrator_status.json (absolute or relative to --repo-root)",
    )
    parser.add_argument(
        "--output",
        default="monitoring/file_sd/mage_hpc_targets.json",
        help="Output Prometheus file_sd JSON path (absolute or relative to --repo-root)",
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help="Optional repo root used to resolve relative --status/--output paths",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host used for metrics targets (default: localhost)",
    )
    parser.add_argument(
        "--job-label",
        default="mage-hpc-trainer",
        help="Value of the Prometheus 'job' label for generated targets",
    )
    parser.add_argument(
        "--include-stopped",
        action="store_true",
        help="Include trainers that are not currently marked running",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of silently returning no targets when status JSON is missing or invalid",
    )
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return base / p


def load_json(path: Path, strict: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"status JSON not found: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        if strict:
            raise RuntimeError(f"failed to parse status JSON {path}: {exc}")
        return {}


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    status_path = resolve(repo_root, args.status)
    output_path = resolve(repo_root, args.output)

    try:
        payload = load_json(status_path, strict=args.strict)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    trainers = payload.get("trainers", [])
    if not isinstance(trainers, list):
        trainers = []
    shared_gpu_hosts = payload.get("shared_gpu_hosts", [])
    if not isinstance(shared_gpu_hosts, list):
        shared_gpu_hosts = []

    generated: List[Dict[str, Any]] = []
    seen_targets = set()
    note = str(payload.get("note", "")).strip()
    updated_at = str(payload.get("updated_at_utc", "")).strip()
    run_id = str(payload.get("run_id", "")).strip()

    def add_target(target: str, labels: Dict[str, str]) -> None:
        if target in seen_targets:
            return
        seen_targets.add(target)
        generated.append({"targets": [target], "labels": labels})

    for row in trainers:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if not profile:
            continue
        try:
            port = int(row.get("metrics_port", 0))
        except Exception:
            port = 0
        if port <= 0:
            continue
        running = bool(row.get("running", False))
        if (not args.include_stopped) and (not running):
            continue

        target = f"{args.host}:{port}"
        labels = {
            "job": args.job_label,
            "profile": profile,
            "kind": "trainer",
            "running": "1" if running else "0",
            "source": "native_orchestrator_status",
        }
        if note:
            labels["orch_note"] = note
        if updated_at:
            labels["orch_updated_at_utc"] = updated_at
        if run_id:
            labels["run_id"] = run_id
        add_target(target, labels)

    for row in shared_gpu_hosts:
        if not isinstance(row, dict):
            continue
        try:
            port = int(row.get("metrics_port", 0))
        except Exception:
            port = 0
        if port <= 0:
            continue
        running = bool(row.get("running", False))
        if (not args.include_stopped) and (not running):
            continue
        slot = str(row.get("slot", "")).strip()
        gpu_id = str(row.get("gpu_id", "")).strip()
        target = f"{args.host}:{port}"
        labels = {
            "job": args.job_label,
            "kind": "gpu_host",
            "running": "1" if running else "0",
            "source": "native_orchestrator_status",
        }
        if slot:
            labels["slot"] = slot
        if gpu_id:
            labels["gpu_id"] = gpu_id
        if note:
            labels["orch_note"] = note
        if updated_at:
            labels["orch_updated_at_utc"] = updated_at
        if run_id:
            labels["run_id"] = run_id
        add_target(target, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    temp_path.replace(output_path)

    print(f"status={status_path}")
    print(f"output={output_path}")
    print(f"targets={len(generated)}")
    if generated:
        labels = []
        for item in generated:
            row_labels = item.get("labels", {})
            kind = str(row_labels.get("kind", "")).strip()
            if "profile" in row_labels:
                labels.append(str(row_labels["profile"]))
            elif kind == "gpu_host" and "slot" in row_labels:
                labels.append(f"gpu_slot_{row_labels['slot']}")
        if labels:
            print("profiles=" + ",".join(labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
