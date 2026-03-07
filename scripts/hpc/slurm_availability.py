#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SINFO_FORMAT = "%P|%T|%c|%m|%G|%l|%a"


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


def normalize_partition_name(partition: str) -> str:
    text = str(partition).strip()
    while text.endswith("*"):
        text = text[:-1]
    return text


def normalize_state(state: str) -> str:
    text = str(state).strip().lower()
    if not text:
        return "unknown"
    text = text.split("+", 1)[0]
    text = text.split("~", 1)[0]
    text = text.split("#", 1)[0]
    text = text.rstrip("*")
    return text


def state_bucket(state: str) -> str:
    text = normalize_state(state)
    if text in ("idle",):
        return "idle"
    if text in ("mixed", "mix"):
        return "mix"
    if text in ("allocated", "alloc", "completing", "comp"):
        return "alloc"
    if text in ("drain", "draining", "drained", "fail", "failing", "maint", "maintenance"):
        return "drain"
    if text in ("down", "downing", "reboot", "reboot_issued", "power_down", "powered_down"):
        return "down"
    return "other"


def gpu_count_from_gres(gres: str) -> int:
    text = str(gres).strip()
    if not text or text in ("(null)", "N/A", "none"):
        return 0
    total = 0
    for match in re.finditer(r"(?:^|,)gpu(?::[^:,()]+)*:(\d+)", text, re.IGNORECASE):
        total += parse_int(match.group(1), 0)
    return total


def partition_type_for(partition: str, gres: str) -> str:
    name = normalize_partition_name(partition).lower()
    gres_text = str(gres).lower()
    if "gpu:h100" in gres_text or "h100" in name:
        return "gpu-h100"
    if "gpu:a100" in gres_text or "a100" in name:
        return "gpu-a100"
    if "gpu:v100" in gres_text or "v100" in name:
        return "gpu-v100"
    if re.search(r"(?:^|,)gpu(?::|,|$)", gres_text) or name.startswith("gpu"):
        return "gpu"
    return name.split("-", 1)[0] if "-" in name else name


def parse_sinfo_node_rows(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split("|")]
        if len(parts) != 7:
            continue
        partition, state, cpus, mem_mb, gres, time_limit, avail = parts
        partition_name = normalize_partition_name(partition)
        rows.append(
            {
                "partition": partition_name,
                "type": partition_type_for(partition_name, gres),
                "state": normalize_state(state),
                "bucket": state_bucket(state),
                "cpus": parse_int(cpus, 0),
                "memory_gb": parse_float(mem_mb, 0.0) / 1024.0,
                "gpu_count": gpu_count_from_gres(gres),
                "gres": gres,
                "time_limit": time_limit,
                "avail": avail,
            }
        )
    return rows


def blank_summary(key: str, label: str, type_name: str, avail: str, time_limit: str) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": type_name,
        "avail": avail,
        "time_limit": time_limit,
        "partitions": set(),
        "nodes_total": 0,
        "nodes_idle": 0,
        "nodes_mix": 0,
        "nodes_alloc": 0,
        "nodes_drain": 0,
        "nodes_down": 0,
        "nodes_other": 0,
        "cpu_total": 0,
        "cpu_idle_est": 0,
        "mem_total_gb": 0.0,
        "mem_idle_est_gb": 0.0,
        "gpu_total": 0,
        "gpu_idle_est": 0,
        "gpu_mixed": 0,
    }


def apply_row(summary: Dict[str, Any], row: Dict[str, Any]) -> None:
    summary["partitions"].add(str(row.get("partition", "")))
    summary["nodes_total"] += 1
    summary["cpu_total"] += int(row.get("cpus", 0))
    summary["mem_total_gb"] += float(row.get("memory_gb", 0.0))
    summary["gpu_total"] += int(row.get("gpu_count", 0))

    bucket = str(row.get("bucket", "other"))
    if bucket == "idle":
        summary["nodes_idle"] += 1
        summary["cpu_idle_est"] += int(row.get("cpus", 0))
        summary["mem_idle_est_gb"] += float(row.get("memory_gb", 0.0))
        summary["gpu_idle_est"] += int(row.get("gpu_count", 0))
    elif bucket == "mix":
        summary["nodes_mix"] += 1
        summary["gpu_mixed"] += int(row.get("gpu_count", 0))
    elif bucket == "alloc":
        summary["nodes_alloc"] += 1
    elif bucket == "drain":
        summary["nodes_drain"] += 1
    elif bucket == "down":
        summary["nodes_down"] += 1
    else:
        summary["nodes_other"] += 1


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_partition: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        partition = str(row.get("partition", "")).strip()
        if not partition:
            continue
        type_name = str(row.get("type", "")).strip() or partition
        avail = str(row.get("avail", "")).strip()
        time_limit = str(row.get("time_limit", "")).strip()

        partition_summary = by_partition.get(partition)
        if partition_summary is None:
            partition_summary = blank_summary(partition, partition, type_name, avail, time_limit)
            by_partition[partition] = partition_summary
        apply_row(partition_summary, row)

        type_summary = by_type.get(type_name)
        if type_summary is None:
            type_summary = blank_summary(type_name, type_name, type_name, avail, time_limit)
            by_type[type_name] = type_summary
        apply_row(type_summary, row)

    return finalize_summaries(by_partition.values()), finalize_summaries(by_type.values())


def finalize_summaries(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        summary = dict(row)
        summary["partitions"] = sorted(p for p in row.get("partitions", set()) if str(p).strip())
        result.append(summary)
    result.sort(
        key=lambda item: (
            -int(item.get("gpu_idle_est", 0)),
            -int(item.get("nodes_idle", 0)),
            -int(item.get("cpu_idle_est", 0)),
            str(item.get("label", "")),
        )
    )
    return result


def fetch_sinfo_rows() -> List[Dict[str, Any]]:
    output = subprocess.check_output(
        ["sinfo", "-N", "-h", "-o", SINFO_FORMAT],
        universal_newlines=True,
        stderr=subprocess.STDOUT,
    )
    return parse_sinfo_node_rows(output)


def table(headers: Sequence[Tuple[str, int]], rows: Sequence[Sequence[Any]]) -> str:
    def render(value: Any, width: int, align_left: bool = False) -> str:
        text = str(value)
        if len(text) > width:
            text = text[: width - 1] + "+"
        return text.ljust(width) if align_left else text.rjust(width)

    lines: List[str] = []
    header_line = " ".join(render(name, width, True) for name, width in headers)
    lines.append(header_line)
    for row in rows:
        rendered: List[str] = []
        for idx, value in enumerate(row):
            width = headers[idx][1]
            align_left = idx < 3
            rendered.append(render(value, width, align_left))
        lines.append(" ".join(rendered))
    return "\n".join(lines)


def partition_table_rows(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    result: List[List[Any]] = []
    for row in rows:
        result.append(
            [
                row.get("label", ""),
                row.get("type", ""),
                row.get("avail", ""),
                int(row.get("nodes_idle", 0)),
                int(row.get("nodes_mix", 0)),
                int(row.get("nodes_alloc", 0)),
                int(row.get("nodes_drain", 0)) + int(row.get("nodes_down", 0)),
                int(row.get("gpu_idle_est", 0)),
                int(row.get("gpu_mixed", 0)),
                int(row.get("gpu_total", 0)),
                int(row.get("cpu_idle_est", 0)),
                int(row.get("cpu_total", 0)),
            ]
        )
    return result


def type_table_rows(rows: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    result: List[List[Any]] = []
    for row in rows:
        result.append(
            [
                row.get("label", ""),
                ",".join(row.get("partitions", [])),
                row.get("avail", ""),
                int(row.get("nodes_idle", 0)),
                int(row.get("nodes_mix", 0)),
                int(row.get("nodes_alloc", 0)),
                int(row.get("gpu_idle_est", 0)),
                int(row.get("gpu_mixed", 0)),
                int(row.get("gpu_total", 0)),
                int(row.get("cpu_idle_est", 0)),
                int(row.get("cpu_total", 0)),
            ]
        )
    return result


def print_text(by_partition: Sequence[Dict[str, Any]], by_type: Sequence[Dict[str, Any]]) -> None:
    partition_headers = [
        ("partition", 18),
        ("type", 12),
        ("avail", 6),
        ("idle_n", 6),
        ("mix_n", 5),
        ("alloc_n", 7),
        ("down_n", 6),
        ("gpu_idle", 8),
        ("gpu_mix", 7),
        ("gpu_tot", 7),
        ("cpu_idle", 8),
        ("cpu_tot", 7),
    ]
    type_headers = [
        ("type", 12),
        ("partitions", 28),
        ("avail", 6),
        ("idle_n", 6),
        ("mix_n", 5),
        ("alloc_n", 7),
        ("gpu_idle", 8),
        ("gpu_mix", 7),
        ("gpu_tot", 7),
        ("cpu_idle", 8),
        ("cpu_tot", 7),
    ]

    print("By Partition")
    print(table(partition_headers, partition_table_rows(by_partition)))
    print()
    print("By Type")
    print(table(type_headers, type_table_rows(by_type)))
    print()
    print("Notes")
    print("  cpu_idle and gpu_idle are conservative estimates from fully idle nodes only.")
    print("  gpu_mix shows GPUs that sit on mixed nodes and may or may not be practically available.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show clear Slurm availability by partition and partition type.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text tables")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = fetch_sinfo_rows()
    except subprocess.CalledProcessError as exc:
        message = exc.output.strip() if getattr(exc, "output", "") else str(exc)
        print("failed to query sinfo: {}".format(message), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("failed to query sinfo: sinfo not found on PATH", file=sys.stderr)
        return 1

    by_partition, by_type = aggregate_rows(rows)
    if args.json:
        print(
            json.dumps(
                {
                    "by_partition": by_partition,
                    "by_type": by_type,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_text(by_partition, by_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
