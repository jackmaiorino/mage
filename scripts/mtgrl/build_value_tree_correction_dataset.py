#!/usr/bin/env python3
"""Build a weighted supervised dataset from value-tree corrections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ADMISSION_GATE = "value_tree_source_loss_best_win_delta"
DATASET_VERSION = "weighted_checkpoint_correction_v1"

CSV_FIELDS = [
    "example_id",
    "dataset_version",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_count",
    "source_indices",
    "source_texts",
    "target_indices",
    "target_texts",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "label_weight",
    "source_win_rate",
    "source_loss_rate",
    "source_terminal_rate",
    "target_win_rate",
    "target_loss_rate",
    "target_terminal_rate",
    "delta_win_rate",
    "importance_score",
    "classification",
    "actions_evaluated",
    "total_rollouts",
    "terminal_rollouts",
    "source_outcomes",
    "target_outcomes",
    "input_run_id",
    "manifest_row_number",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume value_tree_corrections.csv and emit weighted first-action "
            "correction examples with checkpoint/value-tree provenance."
        )
    )
    parser.add_argument("--manifest", required=True, help="value_tree_corrections.csv path.")
    parser.add_argument("--output-dir", required=True, help="Directory for dataset outputs.")
    parser.add_argument(
        "--expect-examples",
        type=int,
        default=None,
        help="Fail if the generated example count differs from this value.",
    )
    parser.add_argument(
        "--allow-missing-snapshots",
        action="store_true",
        help="Do not fail when a manifest snapshot path is not present on local disk.",
    )
    parser.add_argument(
        "--min-label-weight",
        type=float,
        default=0.0,
        help="Reject admitted manifest rows below this weight.",
    )
    parser.add_argument("--write-readme", action="store_true", help="Write a generated README.md.")
    return parser.parse_args(argv)


def as_bool(row: Dict[str, str], field: str) -> bool:
    return (row.get(field) or "").strip().lower() == "true"


def as_int(row: Dict[str, str], field: str) -> int:
    value = (row.get(field) or "").strip()
    if not value:
        return 0
    return int(value)


def as_float(row: Dict[str, str], field: str) -> float:
    value = (row.get(field) or "").strip()
    if not value:
        return 0.0
    return float(value)


def has_value(row: Dict[str, str], field: str) -> bool:
    return bool((row.get(field) or "").strip())


def parse_indices(value: str) -> List[int]:
    text = (value or "").strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split("|"):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def read_csv(path: Path) -> List[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(index, row) for index, row in enumerate(reader, start=2)]


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def local_snapshot_exists(snapshot_path: str, manifest_path: Path) -> bool:
    path = Path(snapshot_path)
    if path.is_file():
        return True
    if (manifest_path.parent / path).is_file():
        return True
    return False


def outcomes_terminal(outcomes: str) -> bool:
    parts = [part.strip() for part in (outcomes or "").split("|") if part.strip()]
    return bool(parts) and all(part.startswith("terminal_") for part in parts)


def manifest_rejections(row: Dict[str, str], manifest_path: Path, args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []

    if (row.get("manifest_status") or "").strip() != "admitted":
        reasons.append("manifest_status_not_admitted")
    if (row.get("admission_gate") or "").strip() != ADMISSION_GATE:
        reasons.append("admission_gate_mismatch")

    for field in ("snapshot_path", "candidate_hash", "state_hash", "rng_state_hash"):
        if not has_value(row, field):
            reasons.append(f"missing_{field}")

    if has_value(row, "snapshot_path") and not args.allow_missing_snapshots:
        if not local_snapshot_exists(row["snapshot_path"], manifest_path):
            reasons.append("snapshot_file_missing")

    candidate_count = as_int(row, "candidate_count")
    if candidate_count < 2:
        reasons.append("candidate_count_lt_2")

    source_indices = parse_indices(row.get("source_indices", ""))
    target_indices = parse_indices(row.get("target_indices", ""))
    if not source_indices or not has_value(row, "source_texts"):
        reasons.append("missing_source_choice")
    if not target_indices or not has_value(row, "target_texts"):
        reasons.append("missing_target_choice")
    if source_indices == target_indices:
        reasons.append("source_and_target_same_indices")
    if any(index < 0 or index >= candidate_count for index in source_indices + target_indices):
        reasons.append("choice_index_out_of_range")

    if not as_bool(row, "reentry_a_matched"):
        reasons.append("reentry_a_not_matched")
    if not as_bool(row, "reentry_b_matched"):
        reasons.append("reentry_b_not_matched")

    candidate_hash = (row.get("candidate_hash") or "").strip()
    state_hash = (row.get("state_hash") or "").strip()
    if (row.get("reentry_a_candidate_hash") or "").strip() != candidate_hash:
        reasons.append("reentry_a_candidate_hash_mismatch")
    if (row.get("reentry_b_candidate_hash") or "").strip() != candidate_hash:
        reasons.append("reentry_b_candidate_hash_mismatch")
    if (row.get("reentry_a_state_hash") or "").strip() != state_hash:
        reasons.append("reentry_a_state_hash_mismatch")
    if (row.get("reentry_b_state_hash") or "").strip() != state_hash:
        reasons.append("reentry_b_state_hash_mismatch")

    label_weight = as_float(row, "label_weight")
    if label_weight <= 0.0:
        reasons.append("label_weight_not_positive")
    if label_weight > 1.0:
        reasons.append("label_weight_gt_1")
    if label_weight < args.min_label_weight:
        reasons.append("label_weight_below_threshold")

    if as_float(row, "source_loss_rate") <= 0.0:
        reasons.append("source_loss_rate_not_positive")
    if as_float(row, "source_terminal_rate") <= 0.0:
        reasons.append("source_terminal_rate_not_positive")
    if as_float(row, "target_win_rate") <= 0.0:
        reasons.append("target_win_rate_not_positive")
    if as_float(row, "target_terminal_rate") <= 0.0:
        reasons.append("target_terminal_rate_not_positive")
    if as_float(row, "delta_win_rate") <= 0.0:
        reasons.append("delta_win_rate_not_positive")
    if as_float(row, "importance_score") <= 0.0:
        reasons.append("importance_score_not_positive")

    if not outcomes_terminal(row.get("source_outcomes", "")):
        reasons.append("source_outcomes_not_terminal")
    if not outcomes_terminal(row.get("target_outcomes", "")):
        reasons.append("target_outcomes_not_terminal")

    return reasons


def make_example_id(row: Dict[str, str]) -> str:
    seed = "|".join(
        [
            row.get("correction_id", ""),
            row.get("snapshot_path", ""),
            row.get("candidate_hash", ""),
            row.get("state_hash", ""),
            row.get("rng_state_hash", ""),
            row.get("source_indices", ""),
            row.get("target_indices", ""),
            row.get("label_weight", ""),
            row.get("classification", ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def build_example(row: Dict[str, str]) -> Dict[str, object]:
    source_indices = parse_indices(row["source_indices"])
    target_indices = parse_indices(row["target_indices"])
    return {
        "dataset_version": DATASET_VERSION,
        "example_id": make_example_id(row),
        "task": "weighted_first_action_correction",
        "label": {
            "target_indices": target_indices,
            "target_texts": row.get("target_texts", ""),
            "target_win_rate": as_float(row, "target_win_rate"),
            "target_loss_rate": as_float(row, "target_loss_rate"),
            "target_terminal_rate": as_float(row, "target_terminal_rate"),
            "target_outcomes": row.get("target_outcomes", ""),
            "weight": as_float(row, "label_weight"),
        },
        "negative": {
            "source_indices": source_indices,
            "source_texts": row.get("source_texts", ""),
            "source_win_rate": as_float(row, "source_win_rate"),
            "source_loss_rate": as_float(row, "source_loss_rate"),
            "source_terminal_rate": as_float(row, "source_terminal_rate"),
            "source_outcomes": row.get("source_outcomes", ""),
        },
        "decision": {
            "snapshot_path": row.get("snapshot_path", ""),
            "ordinal": as_int(row, "ordinal"),
            "decision_number": as_int(row, "decision_number"),
            "action_type": row.get("action_type", ""),
            "candidate_count": as_int(row, "candidate_count"),
            "candidate_hash": row.get("candidate_hash", ""),
            "state_hash": row.get("state_hash", ""),
            "rng_state_hash": row.get("rng_state_hash", ""),
        },
        "value_tree": {
            "classification": row.get("classification", ""),
            "delta_win_rate": as_float(row, "delta_win_rate"),
            "importance_score": as_float(row, "importance_score"),
            "actions_evaluated": as_int(row, "actions_evaluated"),
            "total_rollouts": as_int(row, "total_rollouts"),
            "terminal_rollouts": as_int(row, "terminal_rollouts"),
        },
        "proof": {
            "admission_gate": row.get("admission_gate", ""),
            "reentry_a_matched": as_bool(row, "reentry_a_matched"),
            "reentry_b_matched": as_bool(row, "reentry_b_matched"),
            "reentry_a_candidate_hash": row.get("reentry_a_candidate_hash", ""),
            "reentry_b_candidate_hash": row.get("reentry_b_candidate_hash", ""),
            "reentry_a_state_hash": row.get("reentry_a_state_hash", ""),
            "reentry_b_state_hash": row.get("reentry_b_state_hash", ""),
            "reentry_a_reason": row.get("reentry_a_reason", ""),
            "reentry_b_reason": row.get("reentry_b_reason", ""),
        },
        "provenance": {
            "manifest_correction_id": row.get("correction_id", ""),
            "manifest_input_run_id": row.get("input_run_id", ""),
            "manifest_summary_csv": row.get("summary_csv", ""),
            "manifest_action_csv": row.get("action_csv", ""),
            "manifest_row_number": row.get("row_number", ""),
        },
    }


def csv_row(example: Dict[str, object]) -> Dict[str, str]:
    decision = example["decision"]  # type: ignore[index]
    label = example["label"]  # type: ignore[index]
    negative = example["negative"]  # type: ignore[index]
    value_tree = example["value_tree"]  # type: ignore[index]
    provenance = example["provenance"]  # type: ignore[index]
    return {
        "example_id": str(example["example_id"]),
        "dataset_version": str(example["dataset_version"]),
        "snapshot_path": str(decision["snapshot_path"]),
        "ordinal": str(decision["ordinal"]),
        "decision_number": str(decision["decision_number"]),
        "action_type": str(decision["action_type"]),
        "candidate_count": str(decision["candidate_count"]),
        "source_indices": "|".join(str(value) for value in negative["source_indices"]),
        "source_texts": str(negative["source_texts"]),
        "target_indices": "|".join(str(value) for value in label["target_indices"]),
        "target_texts": str(label["target_texts"]),
        "candidate_hash": str(decision["candidate_hash"]),
        "state_hash": str(decision["state_hash"]),
        "rng_state_hash": str(decision["rng_state_hash"]),
        "label_weight": f"{float(label['weight']):.6f}",
        "source_win_rate": f"{float(negative['source_win_rate']):.6f}",
        "source_loss_rate": f"{float(negative['source_loss_rate']):.6f}",
        "source_terminal_rate": f"{float(negative['source_terminal_rate']):.6f}",
        "target_win_rate": f"{float(label['target_win_rate']):.6f}",
        "target_loss_rate": f"{float(label['target_loss_rate']):.6f}",
        "target_terminal_rate": f"{float(label['target_terminal_rate']):.6f}",
        "delta_win_rate": f"{float(value_tree['delta_win_rate']):.6f}",
        "importance_score": f"{float(value_tree['importance_score']):.6f}",
        "classification": str(value_tree["classification"]),
        "actions_evaluated": str(value_tree["actions_evaluated"]),
        "total_rollouts": str(value_tree["total_rollouts"]),
        "terminal_rollouts": str(value_tree["terminal_rollouts"]),
        "source_outcomes": str(negative["source_outcomes"]),
        "target_outcomes": str(label["target_outcomes"]),
        "input_run_id": str(provenance["manifest_input_run_id"]),
        "manifest_row_number": str(provenance["manifest_row_number"]),
    }


def write_readme(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Weighted Value-Tree Correction Dataset",
        "",
        "Generated by `scripts/mtgrl/build_value_tree_correction_dataset.py`.",
        "",
        f"- Dataset version: `{DATASET_VERSION}`",
        f"- Admission gate: `{ADMISSION_GATE}`",
        f"- Examples: `{summary['examples']}`",
        f"- Manifest rows: `{summary['manifest_rows']}`",
        "",
        "Each example labels a first action with a scalar weight derived from the",
        "value-tree importance score. Weight 1.0 means the sampled target won every",
        "terminal rollout while the source lost; fractional weights preserve weaker",
        "but still admitted win-rate deltas.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    rows = read_csv(manifest_path)
    if not rows:
        print("manifest has zero rows", file=sys.stderr)
        return 2

    examples: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    error_reason_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    seen_ids = set()

    for row_number, row in rows:
        classification_counts[(row.get("classification") or "").strip() or "<missing>"] += 1
        reasons = manifest_rejections(row, manifest_path, args)
        if reasons:
            error_reason_counts.update(reasons)
            errors.append(
                {
                    "row_number": row_number,
                    "correction_id": row.get("correction_id", ""),
                    "snapshot_path": row.get("snapshot_path", ""),
                    "reasons": reasons,
                }
            )
            continue
        example = build_example(row)
        example_id = example["example_id"]
        if example_id in seen_ids:
            errors.append(
                {
                    "row_number": row_number,
                    "correction_id": row.get("correction_id", ""),
                    "snapshot_path": row.get("snapshot_path", ""),
                    "reasons": ["duplicate_example_id"],
                }
            )
            error_reason_counts.update(["duplicate_example_id"])
            continue
        seen_ids.add(example_id)
        examples.append(example)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_version": DATASET_VERSION,
        "admission_gate": ADMISSION_GATE,
        "manifest": str(manifest_path),
        "manifest_rows": len(rows),
        "examples": len(examples),
        "errors": len(errors),
        "classification_counts": dict(sorted(classification_counts.items())),
        "error_reason_counts": dict(sorted(error_reason_counts.items())),
        "error_rows": errors,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "value_tree_correction_examples.jsonl", examples)
    write_csv(output_dir / "value_tree_correction_examples.csv", CSV_FIELDS, [csv_row(example) for example in examples])
    if args.write_readme:
        write_readme(output_dir / "README.md", summary)

    stdout_summary = dict(summary)
    stdout_summary.pop("error_rows", None)
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))

    if errors:
        return 3
    if not examples:
        return 4
    if args.expect_examples is not None and len(examples) != args.expect_examples:
        print(
            f"expected {args.expect_examples} examples, got {len(examples)}",
            file=sys.stderr,
        )
        return 5
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return build(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
