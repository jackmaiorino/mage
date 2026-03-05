#!/usr/bin/env python3
import argparse
import json
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
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return base / p


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    status_path = resolve(repo_root, args.status)
    output_path = resolve(repo_root, args.output)

    payload = load_json(status_path)
    trainers = payload.get("trainers", [])
    if not isinstance(trainers, list):
        trainers = []

    generated: List[Dict[str, Any]] = []
    seen_targets = set()
    note = str(payload.get("note", "")).strip()
    updated_at = str(payload.get("updated_at_utc", "")).strip()

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
        if target in seen_targets:
            continue
        seen_targets.add(target)

        labels = {
            "job": args.job_label,
            "profile": profile,
            "running": "1" if running else "0",
            "source": "native_orchestrator_status",
        }
        if note:
            labels["orch_note"] = note
        if updated_at:
            labels["orch_updated_at_utc"] = updated_at

        generated.append({"targets": [target], "labels": labels})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")

    print(f"status={status_path}")
    print(f"output={output_path}")
    print(f"targets={len(generated)}")
    if generated:
        print("profiles=" + ",".join(item["labels"]["profile"] for item in generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
